from types import SimpleNamespace

from rgrd.ingestion import WhitespaceOffsetTokenizer
from rgrd.schema import CharRange, ChunkLineage, TokenRange
from rgrd.v01.donors import (
    DeterministicDonorSampler,
    replace_oracle_groups,
    token_length,
)
from rgrd.v01.engine import donor_interventions


class _FakeIndex:
    def __init__(self) -> None:
        self.rows = [
            SimpleNamespace(
                faiss_id=index,
                chunk=SimpleNamespace(
                    chunk_id=f"chunk-{index}",
                    text=" ".join(f"d{index}t{token}" for token in range(40)),
                ),
                source=SimpleNamespace(source_doc_id=f"doc-{index}"),
            )
            for index in range(64)
        ]
        self.chunk_count = len(self.rows)

    def fetch_chunks(self, identifiers):
        return [self.rows[index] for index in identifiers]


class _DoubleTokenCount:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(2 * len(text.split())))}


def test_donors_are_exact_length_distinct_and_reproducible() -> None:
    tokenizer = WhitespaceOffsetTokenizer()
    sampler = DeterministicDonorSampler(_FakeIndex(), tokenizer, seed=11)
    kwargs = dict(
        sample_id="sample",
        anchor_lengths=[3],
        payload_lengths=[5],
        original_text="a b c middle p q r s t",
        anchor_ranges=[CharRange(start=0, end=5)],
        payload_ranges=[CharRange(start=13, end=22)],
        excluded_source_ids={"doc-0"},
        forbidden_texts=["forbidden query"],
    )
    first = sampler.sample_pairs(**kwargs)
    second = sampler.sample_pairs(**kwargs)
    assert first == second
    sources = [segment.source_doc_id for pair in first for segment in (*pair.anchor, *pair.payload)]
    assert len(sources) == len(set(sources)) == 16
    assert all(token_length(tokenizer, pair.anchor[0].text) == 3 for pair in first)
    assert all(token_length(tokenizer, pair.payload[0].text) == 5 for pair in first)
    constrained = sampler.sample_pairs(
        **kwargs,
        anchor_minimum_requirements=[[("retriever", _DoubleTokenCount(), 6)]],
        payload_minimum_requirements=[[("retriever", _DoubleTokenCount(), 10)]],
    )
    assert all(
        segment.model_token_lengths["retriever"] >= 2 * segment.token_length
        for pair in constrained
        for segment in (*pair.anchor, *pair.payload)
    )
    original_length = token_length(tokenizer, kwargs["original_text"])
    for pair in first:
        replaced = replace_oracle_groups(
            kwargs["original_text"],
            anchor_ranges=kwargs["anchor_ranges"],
            payload_ranges=kwargs["payload_ranges"],
            anchor_replacements=[pair.anchor[0].text],
            payload_replacements=[pair.payload[0].text],
        )
        assert token_length(tokenizer, replaced) == original_length

    chunk = ChunkLineage(
        chunk_id="poison",
        source_doc_id="poison-source",
        text=kwargs["original_text"],
        source_chars=CharRange(start=0, end=len(kwargs["original_text"])),
        source_tokens=TokenRange(start=0, end=9),
        anchor_ranges_source=kwargs["anchor_ranges"],
        payload_ranges_source=kwargs["payload_ranges"],
        anchor_ranges_chunk=kwargs["anchor_ranges"],
        payload_ranges_chunk=kwargs["payload_ranges"],
        chunker_name="test",
        chunker_hash="test",
    )
    interventions = donor_interventions(chunk, first[0])
    expected_spans = kwargs["anchor_ranges"] + kwargs["payload_ranges"]
    assert all(spans == expected_spans for spans, _ in interventions.values())
    assert [value is None for value in interventions["empty"][1]] == [False, False]
    assert [value is None for value in interventions["anchor"][1]] == [True, False]
    assert [value is None for value in interventions["payload"][1]] == [False, True]
    assert [value is None for value in interventions["both"][1]] == [True, True]


def test_coalition_replacement_changes_only_oracle_ranges() -> None:
    text = "left AAA middle PPP right"
    anchor = [CharRange(start=5, end=8)]
    payload = [CharRange(start=16, end=19)]
    observed = replace_oracle_groups(
        text,
        anchor_ranges=anchor,
        payload_ranges=payload,
        anchor_replacements=["XXX"],
        payload_replacements=["YYY"],
    )
    assert observed == "left XXX middle YYY right"
