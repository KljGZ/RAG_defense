import numpy as np

from rgrd.attribution import compute_role_map


def test_positive_effects_form_role_distributions() -> None:
    role = compute_role_map([3.0, -2.0, 1.0], [0.0, 3.0, 1.0])
    assert role.role_identifiable
    assert np.isclose(role.p_retrieval.sum(), 1.0)
    assert np.isclose(role.p_generation.sum(), 1.0)
    assert role.roles == ("anchor-like", "payload-like", "mixed")


def test_noise_floor_prevents_spurious_normalization() -> None:
    role = compute_role_map(
        [1e-10, 0.0],
        [1e-10, 0.0],
        retrieval_noise_floor=1e-8,
        generation_noise_floor=1e-8,
    )
    assert not role.role_identifiable
    assert role.t_rg == 0.0
