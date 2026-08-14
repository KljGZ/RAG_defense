import numpy as np

from rgrd.attribution import compute_role_map


def test_total_variation_is_bounded_and_reaches_extremes() -> None:
    same = compute_role_map([1, 2, 3], [1, 2, 3])
    disjoint = compute_role_map([1, 0], [0, 1])
    assert same.t_rg == 0.0
    assert disjoint.t_rg == 1.0
    rng = np.random.default_rng(9)
    for _ in range(100):
        result = compute_role_map(rng.normal(size=8), rng.normal(size=8))
        assert 0.0 <= result.t_rg <= 1.0
