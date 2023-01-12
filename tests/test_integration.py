"""
These are integration tests because they require access to the remote census API.

Most of the functionality can be unit tested elsewhere or with mocks, but
these tests actually call the census API itself to cover the bits of code
immediately around those calls.
"""
import os.path
import tempfile
import unittest

import geopandas
import geopandas as gpd
import numpy as np
import pandas as pd

import censusdis.data as ced
import censusdis.impl.exceptions
import censusdis.impl.varsource.censusapi
import censusdis.maps as cem
import censusdis.values as cev
from censusdis.states import (
    ALL_STATES_AND_DC,
    STATE_CA,
    STATE_GA,
    STATE_NJ,
    STATE_NY,
    TERRITORY_PR,
)

if __name__ == "__main__":
    unittest.main()


class DownloadTestCase(unittest.TestCase):
    """
    Test the full download capability.

    This is more an integration test than a unit test, which
    calls the census api and downloads real data. We are mainly
    just testing that we can do this without error. We're not that
    concerned with the data that comes back.
    """

    PATH_PREFIX = "test_integration_shapefiles_"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up our shapefile path once at class load time."""
        ced.set_shapefile_path(tempfile.mkdtemp(prefix=cls.PATH_PREFIX))

    def setUp(self) -> None:
        """Set up before each test."""
        self._variable_source = (
            censusdis.impl.varsource.censusapi.CensusApiVariableSource()
        )
        self._dataset = "acs/acs5"
        self._year = 2020
        self._group_name = "B19001"
        self._name = f"{self._group_name}_001E"

    def test_path(self):
        """Are we using the right cache path for shapefiles?"""
        path = ced.get_shapefile_path()
        filename = os.path.basename(path)
        self.assertTrue(filename.startswith(self.PATH_PREFIX))

    def test_download(self):
        """Download just a couple of variables."""

        df = ced.download(
            self._dataset, self._year, ["NAME", self._name], state=STATE_NJ, county="*"
        )

        self.assertEqual((21, 4), df.shape)

        self.assertEqual(["STATE", "COUNTY", "NAME", "B19001_001E"], list(df.columns))

    def test_download_detail(self):
        """Use the deprecated API and assert it warns."""

        with self.assertWarns(DeprecationWarning):
            df = ced.download_detail(
                self._dataset,
                self._year,
                ["NAME", self._name],
                state=STATE_NJ,
                county="*",
            )

        self.assertEqual((21, 4), df.shape)

        self.assertEqual(["STATE", "COUNTY", "NAME", "B19001_001E"], list(df.columns))

    def test_bad_variable(self):
        """Try to download a variable that does not exist."""

        with self.assertRaises(censusdis.impl.exceptions.CensusApiException) as cm:
            ced.download(
                self._dataset,
                self._year,
                ["NAME", "I_DONT_EXIST"],
                state=STATE_NJ,
                county="*",
            )

        self.assertIn(
            "https://api.census.gov/data/2020/acs/acs5/variables/I_DONT_EXIST.html",
            str(cm.exception),
        )
        self.assertIn(
            "https://api.census.gov/data/2020/acs/acs5/variables.html",
            str(cm.exception),
        )

    def test_download_with_bad_values(self):
        """Test replacing bad values with NaN."""
        median_income_variable = "B19013_001E"

        df_raw = ced.download(
            self._dataset,
            self._year,
            ["NAME", median_income_variable],
            state=STATE_GA,
            tract="*",
        )

        df_nan = ced.download(
            self._dataset,
            self._year,
            ["NAME", median_income_variable],
            state=STATE_GA,
            tract="*",
            set_to_nan=cev.ALL_SPECIAL_VALUES,
        )

        self.assertEqual(df_raw.shape, df_nan.shape)

        self.assertTrue(
            (
                df_raw[median_income_variable] == cev.INSUFFICIENT_SAMPLE_OBSERVATIONS
            ).any()
        )
        self.assertFalse(
            (
                df_nan[median_income_variable] == cev.INSUFFICIENT_SAMPLE_OBSERVATIONS
            ).any()
        )

        self.assertFalse(df_raw[median_income_variable].isna().any())
        self.assertTrue(df_nan[median_income_variable].isna().any())

        self.assertTrue(
            (
                (df_raw[median_income_variable] == cev.INSUFFICIENT_SAMPLE_OBSERVATIONS)
                == df_nan[median_income_variable].isna()
            ).all(),
            "All locations that were cev.INSUFFICIENT_SAMPLE_OBSERVATIONS should be NaN.",
        )

    def test_download_with_geometry_county(self):
        """Download at the county level with geometry."""

        gdf = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            with_geometry=True,
            state=STATE_NJ,
            county="*",
        )

        self.assertIsInstance(gdf, geopandas.GeoDataFrame)

        self.assertEqual((21, 5), gdf.shape)

        self.assertEqual(
            ["STATE", "COUNTY", "NAME", "B19001_001E", "geometry"], list(gdf.columns)
        )

    def test_download_with_geometry_state(self):
        """Download at the county level with geometry."""

        gdf = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            with_geometry=True,
            state="*",
        )

        self.assertIsInstance(gdf, geopandas.GeoDataFrame)

        self.assertEqual((52, 4), gdf.shape)

        self.assertEqual(
            ["STATE", "NAME", "B19001_001E", "geometry"], list(gdf.columns)
        )

    def test_download_with_geometry_tract(self):
        """Download at the county level with geometry."""

        gdf = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            with_geometry=True,
            state=STATE_NJ,
            county="001",
            tract="*",
        )

        self.assertIsInstance(gdf, geopandas.GeoDataFrame)

        self.assertEqual((74, 6), gdf.shape)

        self.assertEqual(
            ["STATE", "COUNTY", "TRACT", "NAME", "B19001_001E", "geometry"],
            list(gdf.columns),
        )

    def test_download_with_geometry_block_group(self):
        """Download at the county level with geometry."""

        gdf = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            with_geometry=True,
            state=STATE_NJ,
            county="001",
            block_group="*",
        )

        self.assertIsInstance(gdf, geopandas.GeoDataFrame)

        self.assertEqual((194, 7), gdf.shape)

        self.assertEqual(
            [
                "STATE",
                "COUNTY",
                "TRACT",
                "BLOCK_GROUP",
                "NAME",
                "B19001_001E",
                "geometry",
            ],
            list(gdf.columns),
        )

    def test_download_with_geometry_cousub(self):
        """Download at the county level with geometry."""

        with self.assertRaises(
            censusdis.impl.exceptions.CensusApiException
        ) as assertion:
            ced.download(
                self._dataset,
                self._year,
                ["NAME", self._name],
                with_geometry=True,
                state=STATE_NJ,
                county_subdivision="*",
            )

        self.assertTrue(
            str(assertion.exception).startswith(
                "The with_geometry=True flag is only allowed if"
            )
        )

        # But it is OK without geometry.

        df = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            with_geometry=False,
            state=STATE_NJ,
            county_subdivision="*",
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIsInstance(df, gpd.GeoDataFrame)

        self.assertEqual((570, 5), df.shape)

    def test_multi_state(self):
        """
        Test the case where the only geo argument has multiple values.

        As in `state=[STATE_NJ, STATE_NY]`, vs. the more general
        `state="*"`.
        """

        df = ced.download(
            self._dataset, self._year, ["NAME", self._name], state=[STATE_NJ, STATE_NY]
        )

        self.assertEqual((2, 3), df.shape)

        self.assertIn(STATE_NJ, list(df["STATE"]))
        self.assertIn(STATE_NY, list(df["STATE"]))

    def test_multi_state_county(self):
        """
        Test the case where the first geo argument has multiple values.

        As in `state=[STATE_NJ, STATE_NY]`, vs. the more general
        `state="*"` and `county="*"`.
        """

        df = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            state=[STATE_NJ, STATE_NY],
            county="*",
        )

        self.assertEqual((83, 4), df.shape)

        df_51 = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            state=ALL_STATES_AND_DC,
            county="*",
        )

        df_star = ced.download(
            self._dataset, self._year, ["NAME", self._name], state="*", county="*"
        )

        # This will get us some outside the 51.
        self.assertLess(len(df_51.index), len(df_star.index))

        df_star = df_star[df_star.STATE.isin(ALL_STATES_AND_DC)]

        self.assertEqual(len(df_51.index), len(df_star.index))

    def test_multi_state_tract(self):
        """
        Test the case where the first geo argument has multiple values.

        As in `state=[STATE_NJ, STATE_NY]`, vs. the more general
        `state="*"` and `tract="*"`.

        In this test we also skip a level.
        """

        df = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            state=[STATE_NJ, STATE_NY],
            tract="*",
        )

        self.assertEqual((7592, 5), df.shape)

        self.assertIn("STATE", df.columns)
        self.assertIn("COUNTY", df.columns)
        self.assertIn("TRACT", df.columns)
        self.assertIn("NAME", df.columns)
        self.assertIn(self._name, df.columns)

    def test_multi_state_bg(self):
        """
        Test the case where the first geo argument has multiple values.

        As in `state=[STATE_NJ, STATE_NY]`, vs. the more general
        `state="*"` and `tract="*"`.

        In this test we also skip two levels.
        """

        df = ced.download(
            self._dataset,
            self._year,
            ["NAME", self._name],
            state=[STATE_NJ, STATE_NY],
            block_group="*",
        )

        self.assertEqual((22669, 6), df.shape)

        self.assertIn("STATE", df.columns)
        self.assertIn("COUNTY", df.columns)
        self.assertIn("TRACT", df.columns)
        self.assertIn("BLOCK_GROUP", df.columns)
        self.assertIn("NAME", df.columns)
        self.assertIn(self._name, df.columns)

    def test_get_all_groups(self):
        """Test get_all_groups and related functionality."""
        df_groups = ced.variables.all_groups(self._dataset, self._year)

        group = df_groups.iloc[0]["GROUP"]

        df_variables = ced.variables.all_variables(self._dataset, self._year, group)

        self.assertEqual((49, 7), df_variables.shape)

        self.assertEqual(
            [
                "YEAR",
                "DATASET",
                "GROUP",
                "VARIABLE",
                "LABEL",
                "SUGGESTED_WEIGHT",
                "VALUES",
            ],
            list(df_variables.columns),
        )

        self.assertTrue((df_variables["GROUP"] == group).all())

    def test_get_all_datasets(self):
        """Test getting all data sets."""
        df_datasets = ced.variables.all_data_sets()

        df_datasets_for_year = ced.variables.all_data_sets(year=self._year)

        self.assertEqual(list(df_datasets.columns), list(df_datasets_for_year.columns))

        # There are more total than in the one year.
        self.assertGreater(len(df_datasets.index), len(df_datasets_for_year.index))

        # Everything in the year is in the year.
        self.assertTrue((df_datasets_for_year["YEAR"] == self._year).all())

        # Everything in the year is in the df of all datasets.
        df_both = df_datasets_for_year.merge(
            df_datasets,
            on=list(df_datasets.columns),
        )

        self.assertEqual(df_both.shape, df_datasets_for_year.shape)

        self.assertTrue((df_both == df_datasets_for_year).all().all())


class DownloadWideTestCase(unittest.TestCase):
    """
    Test downloading wide tables.

    For these cases, we have to make multiple calls to the census
    API and then merge or concatenate the results that come back,
    depending on the details of the scenario.
    """

    def test_wide_merge(self):
        """
        Download a really wide set of variables.

        The goal is to trigger a call to
        `_download_multiple`. This version is for a scenario
        that will trigger the merge strategy because each
        sub-query will have rows with a unique geogrpahic key.
        """

        dataset = "acs/acs1/spp"
        year = 2019
        group = "S0201"

        variables = ced.variables.group_variables(
            dataset, year, group, skip_annotations=False
        )

        self.assertGreater(len(variables), ced._MAX_VARIABLES_PER_DOWNLOAD)

        metrics_0 = ced._download_wide_strategy_metrics()

        df = ced.download(dataset, year, variables, state="*")

        metrics_1 = ced._download_wide_strategy_metrics()

        metrics_diff = {k: v - metrics_0[k] for k, v in metrics_1.items()}

        self.assertEqual(1, metrics_diff["merge"])
        self.assertEqual(0, metrics_diff["concat"])

        self.assertEqual((51, 1 + len(variables)), df.shape)

        columns = set(df.columns)
        for variable in variables:
            self.assertIn(variable, columns)

    def test_wide_concat(self):
        """
        Download a really wide set of variables.

        The goal is to trigger a call to
        `_download_multiple`. This version is for a scenario
        that will trigger the concat strategy because the
        query column (state in this case) is not unique.
        """
        dataset = "cps/basic/nov"
        year = 2020

        variables = ced.variables.group_variables(dataset, year, None)

        self.assertEqual(389, len(variables))
        self.assertGreater(len(variables), ced._MAX_VARIABLES_PER_DOWNLOAD)

        metrics_0 = ced._download_wide_strategy_metrics()

        with self.assertLogs(ced.__name__, level="INFO") as cm:
            df = ced.download(dataset, year, variables, state=STATE_NJ)

        # Make sure we got the log message.

        self.assertTrue(
            any(
                message.startswith(
                    "INFO:censusdis.data:Using the concat strategy, which is not guaranteed reliable if "
                )
                for message in cm.output
            )
        )

        metrics_1 = ced._download_wide_strategy_metrics()

        metrics_diff = {k: v - metrics_0[k] for k, v in metrics_1.items()}

        self.assertEqual(1, metrics_diff["concat"])
        self.assertEqual(0, metrics_diff["merge"])

        # One column per variable, plus state. Lots of rows.
        self.assertEqual((2109, len(variables) + 1), df.shape)

        # All the same state.
        self.assertTrue((df["STATE"] == STATE_NJ).all())

        # One column per variable plus state.
        columns = set(df.columns)
        for variable in ["STATE"] + variables:
            self.assertIn(variable, columns)


class DownloadGroupTestCase(unittest.TestCase):
    """
    Test downloading by group.

    This is more an integration test than a unit test, which
    calls the census api and downloads real data. Similar to
    :py:class:`~DownloadTestCase` but with a focus on using
    the `group` and `leaves_of_group` args.
    """

    def setUp(self) -> None:
        """Set up before each test."""
        self._variable_source = (
            censusdis.impl.varsource.censusapi.CensusApiVariableSource()
        )
        self._dataset = "acs/acs5"
        self._year = 2019
        self._group_name_0 = "B19001"
        self._group_name_1 = "B03002"

    def test_group(self):
        """Download the whole group."""

        df_group = ced.download(
            self._dataset,
            self._year,
            group=self._group_name_0,
            state=STATE_NJ,
            county="*",
        )

        # Make sure we got the variables we expected.
        group_variables = ced.variables.group_variables(
            self._dataset, self._year, self._group_name_0
        )

        self.assertEqual(["STATE", "COUNTY"] + group_variables, list(df_group.columns))

    def test_leaves_of_group(self):
        """Download the leaves of the group."""

        df_leaves = ced.download(
            self._dataset,
            self._year,
            leaves_of_group=self._group_name_0,
            state=STATE_NJ,
            county="*",
        )

        # Make sure we got the variables we expected.
        leaf_variables = ced.variables.group_leaves(
            self._dataset, self._year, self._group_name_0
        )

        self.assertEqual(["STATE", "COUNTY"] + leaf_variables, list(df_leaves.columns))

    def test_group_plus(self):
        """Download the whole group plus another variable."""

        extra_variable = "NAME"

        df_group = ced.download(
            self._dataset,
            self._year,
            [extra_variable, extra_variable],  # should dedup here also
            group=self._group_name_0,
            state=STATE_NJ,
            county="*",
        )

        # Make sure we got the variables we expected.
        group_variables = ced.variables.group_variables(
            self._dataset, self._year, self._group_name_0
        )

        self.assertEqual(
            ["STATE", "COUNTY", extra_variable] + group_variables,
            list(df_group.columns),
        )

    def test_leaves_of_group_plus(self):
        """Download the leaves of the group plus another variable."""

        extra_variable = "NAME"

        df_leaves = ced.download(
            self._dataset,
            self._year,
            extra_variable,
            leaves_of_group=self._group_name_0,
            state=STATE_NJ,
            county="*",
        )

        # Make sure we got the variables we expected.
        leaf_variables = ced.variables.group_leaves(
            self._dataset, self._year, self._group_name_0
        )

        self.assertEqual(
            ["STATE", "COUNTY", extra_variable] + leaf_variables,
            list(df_leaves.columns),
        )

    def test_group_with_dups(self):
        """Test the case where some variables are double specified."""

        group_variables = ced.variables.group_variables(
            self._dataset, self._year, self._group_name_0
        )

        some_group_variables = [
            group_variables[ii] for ii in range(0, len(group_variables), 2)
        ]
        # Be sure we are requesting a few twice.
        self.assertGreater(len(some_group_variables), 2)

        df_leaves = ced.download(
            self._dataset,
            self._year,
            some_group_variables,
            group=self._group_name_0,
            state=STATE_NJ,
            county="*",
        )

        # Should be no dups. State and county are the two added.
        self.assertEqual(len(df_leaves.columns), 2 + len(group_variables))

        # Make sure we got the variables we expected.
        returned_variables = set(df_leaves.columns)

        self.assertEqual(len(returned_variables), len(df_leaves.columns))

        # State county and the extra are the two added.
        self.assertEqual(len(returned_variables), 2 + len(group_variables))

        for variable in group_variables:
            self.assertIn(variable, returned_variables)

        self.assertIn("STATE", returned_variables)
        self.assertIn("COUNTY", returned_variables)

    def test_leaves_of_group_with_dups(self):
        """Test the case where some variables are double specified."""

        extra_variable = "NAME"
        leaf_variables = ced.variables.group_leaves(
            self._dataset, self._year, self._group_name_0
        )

        some_leaf_variables = [
            leaf_variables[ii] for ii in range(0, len(leaf_variables), 2)
        ]
        # Be sure we are requesting a few twice.
        self.assertGreater(len(some_leaf_variables), 2)

        df_leaves = ced.download(
            self._dataset,
            self._year,
            [extra_variable] + some_leaf_variables,
            leaves_of_group=self._group_name_0,
            state=STATE_NJ,
            county="*",
        )

        # Should be no dups.
        # State, county, and the extra are the three added.
        self.assertEqual(len(df_leaves.columns), 3 + len(leaf_variables))

        # Make sure we got the variables we expected.
        returned_variables = set(df_leaves.columns)

        self.assertEqual(len(returned_variables), len(df_leaves.columns))

        # State county and the extra are the three added.
        self.assertEqual(len(returned_variables), 3 + len(leaf_variables))

        for variable in leaf_variables:
            self.assertIn(variable, returned_variables)

        self.assertIn("STATE", returned_variables)
        self.assertIn("COUNTY", returned_variables)
        self.assertIn(extra_variable, returned_variables)

    def test_multiple_groups(self):
        """Download from more than one group."""

        df_group = ced.download(
            self._dataset,
            self._year,
            group=[self._group_name_0, self._group_name_1],
            leaves_of_group=[self._group_name_0],
            state=STATE_NJ,
            county="*",
        )

        # Make sure we got the variables we expected.
        group_variables = ced.variables.group_variables(
            self._dataset, self._year, self._group_name_0
        ) + ced.variables.group_variables(self._dataset, self._year, self._group_name_1)

        self.assertEqual(["STATE", "COUNTY"] + group_variables, list(df_group.columns))


class AcsSubjectTestCase(unittest.TestCase):
    """
    Test on ACS Subject Data that includes null in an int field.
    """

    def setUp(self) -> None:
        """Set up before each test."""
        self._dataset = "acs/acs5/profile"
        self._year = 2021
        self._variable_name = "DP02_0001E"

    def test_states_with_null_in_pr(self):
        df = ced.download(
            self._dataset, self._year, ["NAME", self._variable_name], state="*"
        )

        self.assertEqual((52, 3), df.shape)

        # The API returns a null for PR but numbers for all others.
        # We have to convert to a float to represent this even though
        # the census metadata says the variable is an int.
        self.assertEqual(np.float64, df[self._variable_name].dtype)

        self.assertFalse(
            df[df.STATE != TERRITORY_PR][self._variable_name].isnull().any()
        )
        self.assertTrue(
            df[df.STATE == TERRITORY_PR][self._variable_name].isnull().all()
        )


class ShapefileTestCase(unittest.TestCase):

    PATH_PREFIX = "test_integration_shapefiles_"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up our shapefile path once at class load time."""
        cls.shapefile_path = tempfile.mkdtemp(prefix=cls.PATH_PREFIX)

    def setUp(self) -> None:
        """Set up before each test."""
        self._year = 2019
        self.reader = cem.ShapeReader(self.shapefile_path, self._year)

    def test_county_shapefile(self):
        gdf_counties = self.reader.read_shapefile("us", "county")

        self.assertIsInstance(gdf_counties, gpd.GeoDataFrame)

        self.assertEqual((3233, 18), gdf_counties.shape)

    def test_county_cb_shapefile(self):
        gdf_counties = self.reader.read_cb_shapefile("us", "county")

        self.assertIsInstance(gdf_counties, gpd.GeoDataFrame)

        self.assertEqual((3233, 10), gdf_counties.shape)

    def test_puma_shapefile(self):
        gdf_puma = self.reader.read_cb_shapefile("us", "puma")

        self.assertIsInstance(gdf_puma, gpd.GeoDataFrame)

        self.assertEqual((2380, 9), gdf_puma.shape)

    def test_2010_shapefile(self):
        # Override the normal setup for 2010.
        self._year = 2010
        self.reader = cem.ShapeReader(self.shapefile_path, self._year)

        gdf_counties = self.reader.read_shapefile("us", "county")

        self.assertIsInstance(gdf_counties, gpd.GeoDataFrame)

        self.assertEqual((3221, 18), gdf_counties.shape)

    def test_2010_cb_shapefile(self):
        # Override the normal setup for 2010.
        self._year = 2010
        self.reader = cem.ShapeReader(self.shapefile_path, self._year)

        gdf_counties = self.reader.read_cb_shapefile("us", "county")

        self.assertIsInstance(gdf_counties, gpd.GeoDataFrame)

        self.assertEqual((3221, 7), gdf_counties.shape)


class AddInferredGeographyTestCase(unittest.TestCase):
    """Test our ability to add inferred geometry."""

    PATH_PREFIX = "test_integration_shapefiles_"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up our shapefile path once at class load time."""
        cls.shapefile_path = tempfile.mkdtemp(prefix=cls.PATH_PREFIX)

    def setUp(self) -> None:
        """Set up before each test."""
        self._year = 2020
        self.reader = cem.ShapeReader(self.shapefile_path, self._year)

    def test_state(self):
        """Test that we can infer state geometries."""
        df_state = pd.DataFrame(
            [[STATE_NJ, 0.5, 0.6], [STATE_CA, 0.1, 0.2]],
            columns=["STATE", "metric1", "metric2"],
        )

        gdf_inferred = ced.add_inferred_geography(df_state, self._year)

        self._assert_data_unchanged_in_inference(df_state, gdf_inferred)

        # Now get the state shapefile directly and see if we
        # inferred the right geometries.

        gdf_state = self.reader.read_cb_shapefile("us", "state")

        for state in df_state["STATE"]:
            self.assertTrue(
                gdf_state[gdf_state.STATEFP == state].geometry.iloc[0],
                gdf_inferred[gdf_inferred.STATE == state].geometry.iloc[0],
            )

    def test_county(self):
        """Test that we can infer a county geometries."""
        df_county = pd.DataFrame(
            [[STATE_NJ, "011", 0.5, 0.6], [STATE_NJ, "013", 0.1, 0.2]],
            columns=["STATE", "COUNTY", "metric1", "metric2"],
        )

        gdf_inferred = ced.add_inferred_geography(df_county, self._year)

        self._assert_data_unchanged_in_inference(df_county, gdf_inferred)

        # Now get the county shapefile directly and see if we
        # inferred the right geometries.

        gdf_county = self.reader.read_cb_shapefile("us", "county")

        for row in df_county[["STATE", "COUNTY"]].itertuples():
            state, county = row.STATE, row.COUNTY
            self.assertEqual(
                gdf_county[
                    (gdf_county.STATEFP == state) & (gdf_county.COUNTYFP == county)
                ].geometry.iloc[0],
                gdf_inferred[
                    (gdf_inferred.STATE == state) & (gdf_inferred.COUNTY == county)
                ].geometry.iloc[0],
            )

    def test_tract(self):
        """Test that we can infer census tract geometries."""
        df_tract = pd.DataFrame(
            [
                [STATE_NJ, "013", "019000", 0.1, 0.2],
                [STATE_NJ, "013", "019100", 0.3, 0.4],
                [STATE_NJ, "013", "019200", 0.5, 0.6],
                [STATE_NY, "061", "021600", 1.0, 1.1],
                [STATE_NY, "061", "021800", 1.2, 1.3],
            ],
            columns=["STATE", "COUNTY", "TRACT", "metric1", "metric2"],
        )

        gdf_inferred = ced.add_inferred_geography(df_tract, self._year)

        self._assert_data_unchanged_in_inference(df_tract, gdf_inferred)

        # Now get the county shapefiles directly and see if we
        # inferred the right geometries.

        gdf_tract_nj = self.reader.read_cb_shapefile(STATE_NJ, "tract")
        gdf_tract_ny = self.reader.read_cb_shapefile(STATE_NY, "tract")

        gdf_tract = gdf_tract_nj.append(gdf_tract_ny)

        for row in df_tract[["STATE", "COUNTY", "TRACT"]].itertuples():
            state, county, tract = row.STATE, row.COUNTY, row.TRACT
            self.assertEqual(
                gdf_tract[
                    (gdf_tract.STATEFP == state)
                    & (gdf_tract.COUNTYFP == county)
                    & (gdf_tract.TRACTCE == tract)
                ].geometry.iloc[0],
                gdf_inferred[
                    (gdf_inferred.STATE == state)
                    & (gdf_inferred.COUNTY == county)
                    & (gdf_inferred.TRACT == tract)
                ].geometry.iloc[0],
            )

    def _assert_data_unchanged_in_inference(self, df, gdf_inferred):
        """
        Assert that when we added a geometry column nothing else changed.

        Parameters
        ----------
        df
            The original df.
        gdf_inferred
            The gdf with inferred geometry.
        """
        # We should have the name number of rows and one new column
        # for geometry.
        old_shape = df.shape
        new_shape = gdf_inferred.shape

        self.assertEqual(old_shape[0], new_shape[0])
        self.assertEqual(old_shape[1] + 1, new_shape[1])

        self.assertTrue(all(col in gdf_inferred.columns for col in df.columns))
        self.assertIn("geometry", gdf_inferred.columns)

        # Make sure all the data we had stayed where it was.
        self.assertTrue((gdf_inferred[df.columns] == df).all().all())
