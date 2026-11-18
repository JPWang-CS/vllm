import json
import sys
from pathlib import Path

import pytest

from fullnet.assets import assets
from fullnet.config import load, resolve
from fullnet.setup import configure


@pytest.fixture
def project(tmp_path):
    directory = tmp_path / "models/public-model"
    directory.mkdir(parents=True)
    (directory / "model.yaml").write_text('name: public-model\nfamily: qwen3\nplatforms: [a3, a5]\ndownload:\n  provider: huggingface\n  repo_id: owner/model\n')
    return tmp_path


def test_download_plan_writes_nothing(project):
    result = assets(project, "public-model", "plan")
    assert result["repo_id"] == "owner/model"
    assert not (project / "assets").exists()


def test_missing_weight_shard_rejected(project):
    target = project / "assets/weights/public-model"
    target.mkdir(parents=True)
    (target / "config.json").write_text('{}')
    (target / "model-01.safetensors").write_text('fixture, not real weights')
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-01.safetensors", "b": "model-02.safetensors"}}))
    with pytest.raises(ValueError, match="missing weight shard"):
        assets(project, "public-model", "check")


def test_configure_separates_a3_and_a5_without_changing_environment(project):
    paths = []
    for platform in ("a3", "a5"):
        result = configure(project, "public-model", platform, sys.executable, "readonly-weights", "dataset.jsonl", "metrics.py", [0], 16384)
        c = resolve(result["config"], project)
        assert c["evaluation"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert c["evaluation"]["minimum_score"] is None
        paths.append(result["config"])
    assert paths[0] != paths[1]
    with pytest.raises(FileExistsError):
        configure(project, "public-model", "a3", sys.executable, "weights", "data", "metrics", [0], 16384)


def test_conflicting_changes_fail_before_execution(project):
    change = project / "models/public-model/changes/op"
    change.mkdir(parents=True)
    (change / "change.yaml").write_text('platforms: [a3]\nconflicts: [op]\nverify_argv: [python, verify.py]\n')
    path = configure(project, "public-model", "a3", sys.executable, "weights", "data", "metrics", [0], 16384)["config"]
    config = load(path)
    config["changes"] = ["op"]
    import yaml
    Path(path).write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="conflicting"):
        resolve(path, project)


def test_export_is_separate_and_preserves_source(project):
    from fullnet.export import export_run
    run = project / "models/public-model/experiments/test"
    run.mkdir(parents=True)
    (run / "results.json").write_text('{"scope":"fixture"}')
    destination = project / "export"
    export_run(project, run, destination)
    assert (destination / "results.json").read_text() == (run / "results.json").read_text()
    with pytest.raises(ValueError, match="separate"):
        export_run(project, run, run / "nested-export")


def test_safetensors_structure_without_loading_weights(tmp_path):
    import struct
    from fullnet.assets import check_safetensors
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path = tmp_path / "weights.safetensors"
    content = struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1., 2.)
    path.write_bytes(content)
    assert check_safetensors(path) == 1
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="exceeds"):
        check_safetensors(path)


def test_lfs_pointer_is_not_a_weight_file(tmp_path):
    from fullnet.assets import check_safetensors
    path = tmp_path / "weights.safetensors"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid example\n")
    with pytest.raises(ValueError, match="header"):
        check_safetensors(path)
