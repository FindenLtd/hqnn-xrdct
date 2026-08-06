import numpy as np

from hybrid_qnn_xrdct.data import sample_coefficients


def test_sampling_is_sparse_and_in_coefficient_domain():
    config = {
        "phases": ["a", "b", "c", "d"],
        "dataset": {
            "phase_count_probability": [0.03, 0.62, 0.30, 0.05],
            "phase_selection_probability": [0.25] * 4,
            "anchor_coefficient": [0.1, 1.0],
            "strong_coefficient": [0.02, 1.0],
            "ghost_coefficient": [0.0, 0.02],
            "strong_nonanchor_probability": 0.75,
        },
    }
    rng = np.random.default_rng(7)
    samples = np.vstack([sample_coefficients(rng, config) for _ in range(2000)])
    counts = (samples > 0).sum(axis=1)
    assert set(np.unique(counts)) <= {1, 2, 3, 4}
    assert np.all((samples >= 0) & (samples <= 1))

