import json
import sys
from types import ModuleType

from rgrd.attacks.joint_gcg import run_official_worker


def test_joint_worker_freezes_models_and_persists_early_success(tmp_path, monkeypatch) -> None:
    class FakeModel:
        frozen = False

        def requires_grad_(self, enabled: bool):
            self.frozen = not enabled
            return self

    class FakeCore:
        def __init__(self, log_path: str):
            self.log_path = log_path
            self.tag = "rag_v2/cluster_0"
            self.verbose = 1
            self.fake_corpus = "optimized text"
            self.adv_tag = " adversarial tag"

        def step(self, epoch: int, eval_only: bool = False):
            self.log_data_json = {"epoch": epoch, "eval_only": eval_only}
            return True

    attack_module = ModuleType("attack_rag")

    def load_model(*_args, **_kwargs):
        return FakeModel(), object()

    def attack_joint(**keywords):
        model, _ = attack_module.load_model("fixture")
        assert model.frozen
        assert FakeCore(str(tmp_path / "output/log")).step(0)

    attack_module.load_model = load_model
    attack_module.attack_joint = attack_joint
    rag_package = ModuleType("rag")
    rag_package.__path__ = []
    poison_module = ModuleType("rag.poisionedrag")
    poison_module.PoisionedRAGJointAttackCore = FakeCore
    monkeypatch.setitem(sys.modules, "attack_rag", attack_module)
    monkeypatch.setitem(sys.modules, "rag", rag_package)
    monkeypatch.setitem(sys.modules, "rag.poisionedrag", poison_module)

    official = tmp_path / "official"
    assets = tmp_path / "assets"
    output = tmp_path / "output"
    official.mkdir()
    assets.mkdir()
    (assets / "task_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "t"}]}), encoding="utf-8"
    )
    transfer = tmp_path / "transfer.npy"
    corpus = tmp_path / "corpus.jsonl"
    retrieval = tmp_path / "retrieval.json"
    for path in (transfer, corpus, retrieval):
        path.write_bytes(b"fixture")

    result = run_official_worker(
        official_root=official,
        assets=assets,
        output=output,
        retriever=tmp_path / "retriever",
        generator=tmp_path / "generator",
        transfer_matrix=transfer,
        corpus=corpus,
        retrieval_results=retrieval,
        epochs=1,
        n_samples=2,
        topk=1,
        tag_length=2,
    )
    terminal = json.loads((output / "log/rag_v2/cluster_0/0.json").read_text(encoding="utf-8"))
    assert terminal["fake_corpus"] == "optimized text"
    assert terminal["terminal_success"] is True
    assert result["tasks"] == 1
    assert attack_module.load_model is load_model
    assert FakeCore.step.__name__ == "step"
