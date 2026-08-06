import numpy as np

from hybrid_qnn_xrdct.physics import effective_sf, scale_factors_to_weight_fraction


def test_weight_fraction_sums_to_one():
    sf = effective_sf(np.array([1.0, 0.5, 0.0, 0.2]), np.array([7e-10, 8e-9, 1e-9, 4e-10]))
    result = scale_factors_to_weight_fraction(sf, np.array([100.0, 50.0, 60.0, 40.0]), np.array([293.619, 96.086, 107.926, 254.184]))
    assert np.isclose(result.sum(), 1.0)
    assert result[2] == 0.0

