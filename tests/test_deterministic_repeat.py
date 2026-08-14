from rgrd.attribution import compute_role_map
from rgrd.ingestion import WhitespaceOffsetTokenizer, chunk_source
from rgrd.schema import SourceDocument
from rgrd.statistics import paired_sign_flip_test


def test_same_seed_and_inputs_repeat_exactly() -> None:
    source = SourceDocument(source_doc_id="repeat", text="one two three four five six")
    first_chunks = chunk_source(source, WhitespaceOffsetTokenizer(), chunk_size=3, overlap=1)
    second_chunks = chunk_source(source, WhitespaceOffsetTokenizer(), chunk_size=3, overlap=1)
    assert [chunk.model_dump() for chunk in first_chunks] == [
        chunk.model_dump() for chunk in second_chunks
    ]
    first_role = compute_role_map([1, 2, 0], [0, 1, 3])
    second_role = compute_role_map([1, 2, 0], [0, 1, 3])
    assert first_role.t_rg == second_role.t_rg
    assert paired_sign_flip_test([1, 2, 3], permutations=500, seed=7) == paired_sign_flip_test(
        [1, 2, 3], permutations=500, seed=7
    )
