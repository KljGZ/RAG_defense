import pytest

from rgrd.attribution import attribute_retrieval
from rgrd.retrieval import freeze_thresholds, pipeline_margin


def test_thresholds_remain_frozen_for_hidden_scores() -> None:
    frozen = freeze_thresholds(
        [10.0, 9.0, 8.0, 7.0, 6.0],
        [3.0, 2.0, 1.0],
        dense_top_b=3,
        rerank_top_k=2,
    )
    assert frozen.dense_threshold == 8.0
    assert frozen.rerank_threshold == 2.0
    full = pipeline_margin(9.0, 2.5, frozen)
    results = attribute_retrieval(
        ["span"],
        full_dense_score=9.0,
        full_rerank_score=2.5,
        hidden_score=lambda _: (1.0, 2.4),
        frozen=frozen,
    )
    assert results[0].full_margin == full
    assert results[0].hidden_margin == pipeline_margin(1.0, 2.4, frozen)
    assert results[0].effect == pytest.approx(full - results[0].hidden_margin)
