from rgrd.ingestion import WhitespaceOffsetTokenizer, chunk_source
from rgrd.schema import CharRange, SourceDocument


def test_source_chunk_lineage_round_trip() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    source = SourceDocument(source_doc_id="doc-1", text=text)
    anchor = CharRange(start=text.index("beta"), end=text.index("gamma") + len("gamma"))
    payload = CharRange(start=text.index("zeta"), end=text.rindex("eta") + len("eta"))
    chunks = chunk_source(
        source,
        WhitespaceOffsetTokenizer(),
        chunk_size=4,
        overlap=1,
        anchor_ranges=[anchor],
        payload_ranges=[payload],
    )
    assert len(chunks) == 3
    for chunk in chunks:
        assert source.text[chunk.source_chars.start : chunk.source_chars.end] == chunk.text
        for source_span, chunk_span in zip(
            chunk.anchor_ranges_source, chunk.anchor_ranges_chunk, strict=True
        ):
            assert (
                source.text[source_span.start : source_span.end]
                == chunk.text[chunk_span.start : chunk_span.end]
            )
        for source_span, chunk_span in zip(
            chunk.payload_ranges_source, chunk.payload_ranges_chunk, strict=True
        ):
            assert (
                source.text[source_span.start : source_span.end]
                == chunk.text[chunk_span.start : chunk_span.end]
            )
