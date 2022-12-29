# Copyright (c) 2022 Darren Erik Vengroff
"""
Utilities for loading census data.

This module relies on the US Census API, which
it wraps in a pythonic manner.
"""

import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd

import censusdis.geography as cgeo
import censusdis.maps as cmap
from censusdis.impl.exceptions import CensusApiException
from censusdis.impl.fetch import data_from_url
from censusdis.impl.varcache import VariableCache
from censusdis.impl.varsource.censusapi import CensusApiVariableSource

# This is the type we can accept for geographic
# filters. When provided, these filters are either
# single values as a string, or, if multivalued,
# then an iterable containing all the values allowed
# by the filter.
GeoFilterType = Optional[Union[str, Iterable[str]]]


def _gf2s(geo_filter: GeoFilterType) -> Optional[str]:
    """
    Utility to convert a filter to a string.

    For the Census API, multiple values are encoded
    in a single comma separated string.
    """
    if geo_filter is None or isinstance(geo_filter, str):
        return geo_filter
    return ",".join(geo_filter)


_MAX_FIELDS_PER_DOWNLOAD = 50


__dw_strategy_metrics = {"merge": 0, "concat": 0}
"""
Counters for how often we use each strategy for wide tables.
"""


def _download_wide_strategy_metrics() -> Dict[str, int]:
    """
    Metrics on which strategies have been used for wide tables.

    Returns
    -------
        A dictionary of metrics on how often each strategy has
        been used.
    """
    return dict(**__dw_strategy_metrics)


def _download_concat(
    dataset: str,
    year: int,
    fields: List[str],
    *,
    key: Optional[str],
    census_variables: "VariableCache",
    with_geometry: bool = False,
    **kwargs: cgeo.InSpecType,
) -> pd.DataFrame:
    """
    Download data in groups of columns and concatenate the results together.

    The reason for this function is that the API will only return a maximum
    of 50 columns per query. This function downloads wider data 50 columns
    at a time and concatenates them.

    Parameters
    ----------
    dataset
        The dataset to download from. For example `acs/acs5` or
        `dec/pl`.
    year
        The year to download data for.
    download_variables
        The census variables to download, for example `["NAME", "B01001_001E"]`.
    with_geometry
        If `True` a :py:class:`gpd.GeoDataFrame` will be returned and each row
        will have a geometry that is a cartographic boundary suitable for platting
        a map. See https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2020.html
        for details of the shapefiles that will be downloaded on your behalf to
        generate these boundaries.
    api_key
        An optional API key. If you don't have or don't use a key, the number
        of calls you can make will be limited.
    variable_cache
        A cache of metadata about variables.
    kwargs
        A specification of the geometry that we want data for.

    Returns
    -------
        The full results of the query with all columns.

    """
    # Divide the fields into groups.
    field_groups = [
        # black and flake8 disagree about the whitespace before ':' here...
        fields[start : start + _MAX_FIELDS_PER_DOWNLOAD]  # noqa: 203
        for start in range(0, len(fields), _MAX_FIELDS_PER_DOWNLOAD)
    ]

    if len(field_groups) < 2:
        raise ValueError(
            "_download_concat expects to be called with at least "
            f"{_MAX_FIELDS_PER_DOWNLOAD + 1} variables. With fewer,"
            "use download instead."
        )

    # Get the data for each chunk.
    dfs = [
        download(
            dataset,
            year,
            field_group,
            api_key=key,
            variable_cache=census_variables,
            with_geometry=with_geometry and (ii == 0),
            **kwargs,
        )
        for ii, field_group in enumerate(field_groups)
    ]

    # What fields came back in the first df but were not
    # requested? These are the ones that will be duplicated
    # in the later dfs.
    extra_variables = [f for f in dfs[0].columns if f not in set(field_groups[0])]

    # If we put in the geometry column, it's not part of the merge
    # key.
    if with_geometry:
        extra_variables = [f for f in extra_variables if f != "geometry"]

    # Verify that, as much as we can tell from the other columns,
    # that things line up properly. For some cases like area-based
    # counts where each row represents a unique geography, we could
    # do repeated merges instead of the concat below, but for survey
    # data where each row is representative of a fraction of the population
    # they are not unique and an inner join will blow up the size and
    # give us results that do not make sense. But we have to go with the
    # merge strategy if the rows are in different order.

    rows0 = len(dfs[0].index)
    extra_variables_match = True

    for df in dfs[1:]:
        extra_variables_match = (
            rows0 == len(df.index)
            and (dfs[0][extra_variables] == df[extra_variables]).all().all()
        )
        if not extra_variables_match:
            # At least one difference. So we will have to
            # see if we can fall back on the merge strategy.
            break

    if extra_variables_match:
        # Having done the verification, we can concatenate all
        # the data together.

        __dw_strategy_metrics["concat"] = __dw_strategy_metrics["concat"] + 1

        df_data = pd.concat(
            [dfs[0]] + [df.drop(extra_variables, axis="columns") for df in dfs[1:]],
            axis="columns",
        )
    else:
        # This is the kind of join that only works when the values
        # in extra_variables are unique per row, like in acs/acs5 and
        # similar count-based data sets. But all the rows have to be
        # unique. If they are not all unique then the merge will
        # generate cross products that we almost certainly not intended.
        for df in dfs:
            if len(df.value_counts(extra_variables, sort=False)) != len(df.index):
                raise CensusApiException(
                    "Unexpected data misalignment across multiple API calls. "
                    "Extra variables don't all match but values are duplicated."
                )

        # Now we can do the merging strategy.
        __dw_strategy_metrics["merge"] = __dw_strategy_metrics["merge"] + 1

        df_data = dfs[0]

        for df_right in dfs[1:]:
            df_data = df_data.merge(df_right, on=extra_variables)

    return df_data


__shapefile_root: Union[str, None] = None
__shapefile_readers: Dict[int, cmap.ShapeReader] = {}


def set_shapefile_path(shapefile_path: Union[str, None]) -> None:
    """
    Set the path to the directory to cache shapefiles.

    This is where we will cache shapefiles downloaded when
    `with_geometry=True` is passed to :py:func:`~download`.

    Parameters
    ----------
    shapefile_path
        The path to use for caching shapefiles.
    """
    global __shapefile_root

    __shapefile_root = shapefile_path


def get_shapefile_path() -> Union[str, None]:
    """
    Get the path to the directory to cache shapefiles.

    This is where we will cache shapefiles downloaded when
    `with_geometry=True` is passed to :py:func:`~download`.

    Returns
    -------
        The path to use for caching shapefiles.
    """
    global __shapefile_root

    return __shapefile_root


def __shapefile_reader(year: int):
    reader = __shapefile_readers.get(year, None)

    if reader is None:
        reader = cmap.ShapeReader(
            __shapefile_root,
            year,
        )

        __shapefile_readers[year] = reader

    return reader


_GEO_QUERY_FROM_DATA_QUERY_INNER_GEO: Dict[
    str, Tuple[Optional[str], str, List[str], List[str]]
] = {
    # innermost geo: ( shapefile_scope, shapefile_geo_name, df_on, gdf_on )
    "region": ("us", "region", ["REGION"], ["REGIONCE"]),
    "division": ("us", "division", ["DIVISION"], ["DIVISIONCE"]),
    "combined statistical area": (
        "us",
        "csa",
        ["COMBINED_STATISTICAL_AREA"],
        ["CSAFP"],
    ),
    "metropolitan statistical area/micropolitan statistical area": (
        "us",
        "cbsa",
        ["METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA"],
        ["CBSAFP"],
    ),
    "state": ("us", "state", ["STATE"], ["STATEFP"]),
    "county": ("us", "county", ["STATE", "COUNTY"], ["STATEFP", "COUNTYFP"]),
    # For these, the shapefiles are at the state level, so `None`
    # indicates that we have to fill it in based on the geometry
    # being queried.
    "tract": (
        None,
        "tract",
        ["STATE", "COUNTY", "TRACT"],
        ["STATEFP", "COUNTYFP", "TRACTCE"],
    ),
    "block group": (
        None,
        "bg",
        ["STATE", "COUNTY", "TRACT", "BLOCK_GROUP"],
        ["STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE"],
    ),
}
"""
Helper map for the _with_geometry case.

A map from the innermost level of a geometry specification
to the arguments we need to pass to `get_cb_shapefile`
to get the right shapefile for the geography and the columns
we need to join the data and shapefile on.
"""


def _add_geography(
    df_data: pd.DataFrame, year: int, shapefile_scope: str, geo_level: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Add geography to data.

    Parameters
    ----------
    df_data
        The data we downloaded from the census API
    year
        The year for which to fetch geometries. We need this
        because they change over time.
    shapefile_scope
        The scope of the shapefile. This is typically either a state
        such as `STATE_NJ` or the string `"us"`.
    geo_level
        The geography level we want to add.

    Returns
    -------
        A GeoDataFrame with the original data and an
        added geometry column for each row.
    """

    if geo_level not in _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO:
        raise CensusApiException(
            "The with_geometry=True flag is only allowed if the "
            f"geometry for the data to be loaded ('{geo_level}') is one of "
            f"{[geo for geo in _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO.keys()]}."
        )

    (
        query_shapefile_scope,
        shapefile_geo_level,
        df_on,
        gdf_on,
    ) = _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO[geo_level]

    # If the query spec has a hard-coded value then we use it.
    if query_shapefile_scope is not None:
        shapefile_scope = query_shapefile_scope

    gdf_shapefile = __shapefile_reader(year).read_cb_shapefile(
        shapefile_scope,
        shapefile_geo_level,
    )

    gdf_data = (
        gdf_shapefile[gdf_on + ["geometry"]]
        .merge(df_data, how="right", left_on=gdf_on, right_on=df_on)
        .drop(gdf_on, axis="columns")
    )

    # Rearrange columns so geometry is at the end.
    gdf_data = gdf_data[
        [col for col in gdf_data.columns if col != "geometry"] + ["geometry"]
    ]

    return gdf_data


def infer_geo_level(df: pd.DataFrame) -> str:
    """
    Infer the geography level based on columns names.

    Parameters
    ----------
    df
        A dataframe of variables with one or more columns that
        can be used to infer what geometry level the rows represent.

        For example, if the column `"STATE"` exists, we could infer that
        the data in on a state by state basis. But if there are
        columns for both `"STATE"` and `"COUNTY"`, the data is probably
        at the county level.

        If, on the other hand, there is a `"COUNTY" column but not a
        `"STATE"` column, then there is some ambiguity. The data
        probably corresponds to counties, but the same county ID can
        exist in multiple states, so we will raise a
        :py:class:`~CensusApiException` with an error message expalining
        the situation.

        If there is no match, we will also raise an exception. Again we
        do this, rather than for example, returning `None`, so that we
        can provide an informative error message about the likely cause
        and what to do about it.

        This function is not often called directly, but rather from
        :py:func:`~add_inferred_geography`, which infers the geography
        level and then adds a `geometry` column containing the appropriate
        geography for each row.

    Returns
    -------
        The name of the geography level.
    """
    match_key = None
    match_on_len = 0
    partial_match_keys = []

    for k, (_, _, df_on, _) in _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO.items():
        if all(col in df.columns for col in df_on):
            # Full match. We want the longest full match
            # we find.
            if match_key is None or len(df_on) > match_on_len:
                match_key = k
        elif df_on[-1] in df.columns:
            # Partial match. This could result in us
            # not getting what we expect. Like if we
            # have STATE and TRACT, but not COUNTY, we will
            # get a partial match on [STATE, COUNTY. TRACT]
            # and a full match on [STATE]. We probably did
            # not mean to match [STATE], we just lost the
            # COUNTY column somewhere along the way.
            partial_match_keys.append(k)

    if match_key is None:
        raise CensusApiException(
            f"Unable to infer geometry. Was not able to locate any of the "
            "known sets of columns "
            f"{tuple(df_on for _, _, df_on, _ in _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO.values())} "
            f"in the columns {list(df.columns)}."
        )

    if partial_match_keys:
        raise CensusApiException(
            f"Unable to infer geometry. Geometry matched {match_key} on columns "
            f"{_GEO_QUERY_FROM_DATA_QUERY_INNER_GEO[match_key][2]} "
            "but also partially matched one or more candidates "
            f"{tuple(_GEO_QUERY_FROM_DATA_QUERY_INNER_GEO[k][2] for k in partial_match_keys)}. "
            "Partial matches are usually unintended. Either add columns to allow a "
            "full match or rename some columns to prevent the undesired partial match."
        )

    return match_key


def add_inferred_geography(df: pd.DataFrame, year: int) -> gpd.GeoDataFrame:
    """
    Infer the geography level of the given dataframe and
    add geometry to each row for that level.

    See Also
    --------
        :py:ref:`~infer_geo_level` for more on how inference is done.

    Parameters
    ----------
    df
        A dataframe of variables with one or more columns that
        can be used to infer what geometry level the rows represent.
    year
        The year for which to fetch geometries. We need this
        because they change over time.

    Returns
    -------
        A geo data frame containing the original data augmented with
        the appropriate geometry for each row.
    """

    geo_level = infer_geo_level(df)

    shapefile_scope = _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO[geo_level][0]

    if shapefile_scope is not None:
        # The scope is the same across the board.
        gdf = _add_geography(df, year, shapefile_scope, geo_level)
        return gdf

    # We have to group by different values of the shapefile
    # scope from the appropriate column and add the right
    # geography to each group.
    shapefile_scope_column = _GEO_QUERY_FROM_DATA_QUERY_INNER_GEO[geo_level][2][0]

    df_with_geo = (
        df.groupby(shapefile_scope_column, group_keys=False)
        .apply(lambda g: _add_geography(g, year, g.name, geo_level))
        .reset_index(drop=True)
    )

    gdf = gpd.GeoDataFrame(df_with_geo)

    return gdf


def download_detail(
    dataset: str,
    year: int,
    download_variables: Iterable[str],
    *,
    with_geometry: bool = False,
    api_key: Optional[str] = None,
    variable_cache: Optional["VariableCache"] = None,
    **kwargs: cgeo.InSpecType,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Deprecated version of :py:ref:`~download`; use `download` instead.

    Same functionality but under the old name. Back in the pre-history
    of `censusdis`, this function started life as a way to download
    ACS detail tables. It evolved significantly since then and does much
    more now. Hence the name change.

    This function will disappear completely in a future version.
    """
    warnings.warn(
        "censusdis.data.download_detail is deprecated. "
        "Please use censusdis.data.download instead.",
        DeprecationWarning,
        2,
    )
    return download(
        dataset,
        year,
        download_variables,
        with_geometry=with_geometry,
        api_key=api_key,
        variable_cache=variable_cache,
        **kwargs,
    )


def download(
    dataset: str,
    year: int,
    download_variables: Iterable[str],
    *,
    with_geometry: bool = False,
    api_key: Optional[str] = None,
    variable_cache: Optional["VariableCache"] = None,
    **kwargs: cgeo.InSpecType,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Download data from the US Census API.

    This is the main API for downloading US Census data with the
    `censusdis` package. There are many examples of how to use
    this in the demo notebooks provided with the package at
    https://github.com/vengroff/censusdis/tree/main/notebooks.

    Parameters
    ----------
    dataset
        The dataset to download from. For example `acs/acs5` or
        `dec/pl`.
    year
        The year to download data for.
    download_variables
        The census variables to download, for example `["NAME", "B01001_001E"]`.
    with_geometry
        If `True` a :py:class:`gpd.GeoDataFrame` will be returned and each row
        will have a geometry that is a cartographic boundary suitable for platting
        a map. See https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2020.html
        for details of the shapefiles that will be downloaded on your behalf to
        generate these boundaries.
    api_key
        An optional API key. If you don't have or don't use a key, the number
        of calls you can make will be limited.
    variable_cache
        A cache of metadata about variables.
    kwargs
        A specification of the geometry that we want data for.

    Returns
    -------
        A :py:class:`~pd.DataFrame` containing the requested US Census data.
    """
    if variable_cache is None:
        variable_cache = variables

    # The side effect here is to prime the cache.
    cgeo.geo_path_snake_specs(dataset, year)

    # In case they came to us in py format, as kwargs often do.
    kwargs = {
        cgeo.path_component_from_snake(dataset, year, k): v for k, v in kwargs.items()
    }

    if not isinstance(download_variables, list):
        download_variables = list(download_variables)

    # Special case if we are trying to get too many fields.
    if len(download_variables) > _MAX_FIELDS_PER_DOWNLOAD:
        return _download_concat(
            dataset,
            year,
            download_variables,
            key=api_key,
            census_variables=variable_cache,
            with_geometry=with_geometry,
            **kwargs,
        )

    # Prefetch all the types before we load the data.
    # That way we fail fast if a field is not known.
    for variable in download_variables:
        try:
            variable_cache.get(dataset, year, variable)
        except Exception:
            census_url = CensusApiVariableSource.url(
                dataset, year, variable, response_format="html"
            )
            census_variables_url = CensusApiVariableSource.variables_url(
                dataset, year, response_format="html"
            )

            raise CensusApiException(
                f"Unable to get metadata on the variable {variable} from the "
                f"dataset {dataset} for year {year} from the census API. "
                f"Check the census URL for the variable ({census_url}) to ensure it exists. "
                f"If not found, check {census_variables_url} for all variables in the dataset."
            )

    # If we were given a list, join it together into
    # a comma-separated string.
    string_kwargs = {k: _gf2s(v) for k, v in kwargs.items()}

    url, params, bound_path = census_table_url(
        dataset, year, download_variables, api_key=api_key, **string_kwargs
    )
    df_data = data_from_url(url, params)

    for field in download_variables:
        # predicateType does not exist in some older data sets like acs/acs3
        # So in that case we just go with what we got in the JSON. But if we
        # have it try to set the type.
        if "predicateType" in variable_cache.get(dataset, year, field):
            field_type = variable_cache.get(dataset, year, field)["predicateType"]

            if field_type == "int":
                if df_data[field].isnull().any():
                    # Some Census data sets put in null in int fields.
                    # We have to go with a float to make this a NaN.
                    # Int has no representation for NaN or None.
                    df_data[field] = df_data[field].astype(float, errors="ignore")
                else:
                    try:
                        df_data[field] = df_data[field].astype(int)
                    except ValueError:
                        # Sometimes census metadata says int, but they
                        # put in float values anyway, so fall back on
                        # trying to get them as floats.
                        df_data[field] = df_data[field].astype(float, errors="ignore")
            elif field_type == "float":
                df_data[field] = df_data[field].astype(float)
            elif field_type == "string":
                pass
            else:
                # Leave it as an object?
                pass

    download_variables_upper = [dv.upper() for dv in download_variables]

    # Put the geo fields (STATE, COUNTY, etc...) that came back up front.
    df_data = df_data[
        [col for col in df_data.columns if col not in download_variables_upper]
        + download_variables_upper
    ]

    if with_geometry:
        # We need to get the geometry and merge it in.
        geo_level = bound_path.path_spec.path[-1]
        shapefile_scope = bound_path.bindings[bound_path.path_spec.path[0]]

        gdf_data = _add_geography(df_data, year, shapefile_scope, geo_level)
        return gdf_data

    return df_data


def census_table_url(
    dataset: str,
    year: int,
    download_variables: Iterable[str],
    *,
    api_key: Optional[str] = None,
    **kwargs: cgeo.InSpecType,
) -> Tuple[str, Mapping[str, str], cgeo.BoundGeographyPath]:
    """
    Construct the URL to download data from the U.S. Census API.

    Parameters
    ----------
    dataset
        The dataset to download from. For example `acs/acs5` or
        `dec/pl`.
    year
        The year to download data for.
    download_variables
        The census variables to download, for example `["NAME", "B01001_001E"]`.
    api_key
        An optional API key. If you don't have or don't use a key, the number
        of calls you can make will be limited.
    kwargs
        A specification of the geometry that we want data for.

    Returns
    -------
        The URL, parameters and bound path.

    """
    bound_path = cgeo.PathSpec.partial_prefix_match(dataset, year, **kwargs)

    if bound_path is None:
        raise CensusApiException(
            f"Unable to match the geography specification {kwargs}.\n"
            f"Supported geographies for dataset='{dataset}' in year={year} are:\n"
            + "\n".join(
                f"{path_spec}"
                for path_spec in cgeo.geo_path_snake_specs(dataset, year).values()
            )
        )

    query_spec = cgeo.CensusGeographyQuerySpec(
        dataset, year, list(download_variables), bound_path, api_key=api_key
    )

    url, params = query_spec.table_url()

    return url, params, bound_path


variables = VariableCache()
