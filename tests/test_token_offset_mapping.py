from rgrd.ingestion import WhitespaceOffsetTokenizer, build_prompt_token_map


def test_generator_prompt_offset_mapping_is_exact() -> None:
    chunk = "retrieval anchor payload fact"
    prompt = f"Question: q\nContext 0: {chunk}\nContext 1: other\nFINAL_ANSWER:"
    mapping = build_prompt_token_map(prompt, chunk, WhitespaceOffsetTokenizer())
    assert prompt[mapping.chunk_chars_in_prompt.start : mapping.chunk_chars_in_prompt.end] == chunk
    assert len(mapping.chunk_token_indices) == 4
    recovered = [chunk[span.start : span.end] for span in mapping.token_char_ranges_in_chunk]
    assert recovered == ["retrieval", "anchor", "payload", "fact"]
