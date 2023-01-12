# Copyright (c) 2022 Darren Erik Vengroff
"""
Special estimate values that may be returned from the U.S. Census API.

See https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html
"""

INSUFFICIENT_SAMPLE_OBSERVATIONS = -666666666
"""
Insufficient number of sample observations.

The estimate could not be computed because there were an insufficient number of sample observations.
For a ratio of medians estimate, one or both of the median estimates falls in the lowest interval or
highest interval of an open-ended distribution. The estimate could not be computed because there were
an insufficient number of sample observations. For a ratio of medians estimate, one or both of the
median estimates falls in the lowest interval or highest interval of an open-ended distribution. For
a 5-year median estimate, the margin of error associated with a median was larger than the median itself.
"""

INSUFFICIENT_SAMPLES_IN_GEOGRAPHY = -999999999
"""
Insufficient samples in geography.

The estimate or margin of error cannot be displayed because there were an insufficient number of sample
cases in the selected geographic area.
"""

NOT_APPLICABLE_OR_NOT_AVAILABLE = -888888888
"""
Not applicable or not available.

The estimate or margin of error is not applicable or not available.
"""

INSUFFICIENT_SAMPLE_OBSERVATIONS_FOR_MARGIN_OF_ERROR = -222222222
"""
Insufficient sample observations for margin of error.

The margin of error could not be computed because there were an insufficient number of sample observations.
"""

MEDIAN_IN_OPEN_INTERVAL = -333333333
"""
Median falls in an open-ended interval.

The margin of error could not be computed because the median falls in the lowest interval or highest interval
of an open-ended distribution.
"""

MARGIN_OF_ERROR_NOT_APPROPRIATE = -555555555
"""
Margin of error not appropriate.

A margin of error is not appropriate because the corresponding estimate is controlled to an independent
population or housing estimate. Effectively, the corresponding estimate has no sampling error and the margin
of error may be treated as zero.
"""

ALL_SPECIAL_VALUES = (
    INSUFFICIENT_SAMPLE_OBSERVATIONS,
    INSUFFICIENT_SAMPLES_IN_GEOGRAPHY,
    NOT_APPLICABLE_OR_NOT_AVAILABLE,
    INSUFFICIENT_SAMPLE_OBSERVATIONS_FOR_MARGIN_OF_ERROR,
    MEDIAN_IN_OPEN_INTERVAL,
    MARGIN_OF_ERROR_NOT_APPROPRIATE,
)
