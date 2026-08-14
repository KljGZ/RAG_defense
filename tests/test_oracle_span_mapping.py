from rgrd.attribution import aggregate_overlapping_span_effects, compute_role_map, oracle_mass
from rgrd.schema import CharRange


def test_oracle_mass_uses_character_overlap_after_view_aggregation() -> None:
    atoms = aggregate_overlapping_span_effects(
        [
            (CharRange(start=0, end=4), 4.0, 0.0),
            (CharRange(start=2, end=6), 2.0, 4.0),
        ]
    )
    role = compute_role_map(
        [atom.retrieval_effect for atom in atoms],
        [atom.generation_effect for atom in atoms],
    )
    ranges = [atom.char_range for atom in atoms]
    anchor = oracle_mass(ranges, role.p_retrieval, [CharRange(start=0, end=2)])
    payload = oracle_mass(ranges, role.p_generation, [CharRange(start=4, end=6)])
    assert anchor > 0
    assert payload > 0
    assert anchor <= 1.0
    assert payload <= 1.0
    assert sum(atom.char_range.length for atom in atoms) == 6
