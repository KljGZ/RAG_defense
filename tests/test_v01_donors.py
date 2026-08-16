from types import SimpleNamespace

from rgrd.ingestion import WhitespaceOffsetTokenizer
from rgrd.schema import CharRange
from rgrd.v01.donors import (
    DeterministicDonorSampler,
    replace_oracle_groups,
    token_length,
)


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
