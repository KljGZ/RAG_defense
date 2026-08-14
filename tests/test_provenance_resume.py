import json

from rgrd.experiments.resume import prepare_jsonl_resume


def test_resume_discards_cross_commit_and_duplicate_rows(tmp_path) -> None:
    output = tmp_path / "worker.jsonl"
    current = {
        "schema_version": 2,
        "detector_code_commit": "current",
        "model_revisions": {"generator": "fixed"},
        "pipeline_config_sha256": "a" * 64,
    }
    rows = [
        {**current, "sample_id": "keep", "value": 1},
        {**current, "sample_id": "keep", "value": 2},
        {**current, "sample_id": "drop", "detector_code_commit": "old"},
    ]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    completed = prepare_jsonl_resume(
        output,
        key_fields=("sample_id",),
        expected_provenance=current,
    )
    assert completed == {("keep",)}
    kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1
    assert kept[0]["value"] == 1
