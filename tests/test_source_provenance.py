"""Source freezes use only this repository and explicitly supplied receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from harnessmm import source_provenance as provenance
from scripts import run_chisel_component_quality as quality
from scripts import run_searchqa_residual_meta_campaign as searchqa


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Source fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "source.py").write_text("# synthetic source\n")
    git(root, "add", "source.py")
    git(root, "commit", "-m", "Synthetic source fixture")
    git(root, "remote", "add", "origin", "git@github.com:YuhuaJiang2002/HarnessMM.git")
    return root


@pytest.mark.parametrize("remote", [
    "git@github.com:example/HarnessMM.git",
    "ssh://git@github.com/example/HarnessMM.git",
    "https://github.com/example/HarnessMM.git",
    "https://github.com/example/HarnessMM/",
])
def test_remote_identity_supports_checkout_transports(remote):
    assert provenance.repository_from_remote(remote) == "example/HarnessMM"


@pytest.mark.parametrize("remote", ["/tmp/another-checkout", "file:///tmp/repo", "not-a-remote"])
def test_remote_identity_rejects_local_paths(remote):
    with pytest.raises(provenance.SourceProvenanceError):
        provenance.repository_from_remote(remote)


def test_both_runners_capture_their_own_checkout(checkout, tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "ROOT", checkout)
    monkeypatch.setattr(searchqa, "ROOT", checkout)
    state = searchqa._git_state()
    result = quality.git_provenance(tmp_path / "run", None)
    assert state["repository"] == result["git_repository"] == provenance.DEFAULT_REPOSITORY
    assert state["head"] == result["git_head"] == git(checkout, "rev-parse", "HEAD")
    assert state["branch"] == result["git_branch"] == "main"
    assert result["git_state_source"] == "current_working_tree"
    assert state["tracked_status_sha256"] == hashlib.sha256(b"").hexdigest()

    (checkout / "source.py").write_text("# changed fixture\n")
    assert searchqa._git_state()["tracked_status_sha256"] != state["tracked_status_sha256"]


def test_detached_checkout_and_fork_keep_actual_identity(checkout, monkeypatch):
    git(checkout, "remote", "set-url", "origin", "git@github.com:example/HarnessMM.git")
    git(checkout, "checkout", "--detach", "HEAD")
    monkeypatch.setattr(searchqa, "ROOT", checkout)
    state = searchqa._git_state()
    assert state["branch"] == "HEAD"
    assert state["repository"] == "example/HarnessMM"


def test_source_archive_cannot_capture_an_enclosing_repository(checkout, monkeypatch):
    archive = checkout / "unpacked-release"
    archive.mkdir()
    monkeypatch.setattr(searchqa, "ROOT", archive)
    with pytest.raises(provenance.SourceProvenanceError, match="own Git checkout"):
        searchqa._git_state()


def test_receipt_cli_reads_the_requested_checkout(checkout):
    result = subprocess.run(
        [sys.executable, "-m", "harnessmm.source_provenance", "--root", str(checkout)],
        check=True, capture_output=True, text=True,
    )
    state = json.loads(result.stdout)
    assert state["head"] == git(checkout, "rev-parse", "HEAD")
    assert state["repository"] == provenance.DEFAULT_REPOSITORY


def test_quality_requires_explicit_config_before_initialization(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        quality.main([
            "init", "--experiment-dir", str(tmp_path / "run"),
            "--data-manifest", "fixture-data.json", "--service-manifest", "fixture-service.json",
        ])
    assert error.value.code == 2
    assert "--config" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_quality_accepts_a_harnessmm_receipt_without_a_git_checkout(
    checkout, tmp_path, monkeypatch,
):
    state = provenance.capture_git_state(checkout)
    receipt = tmp_path / "git-state.json"
    content = json.dumps(state, indent=2).encode() + b"\n"
    receipt.write_bytes(content)
    archive = tmp_path / "source-only"
    archive.mkdir()
    monkeypatch.setattr(quality, "ROOT", archive)
    output = tmp_path / "run"
    result = quality.git_provenance(output, receipt)
    assert result["git_repository"] == provenance.DEFAULT_REPOSITORY
    assert result["source_git_state"] == state
    assert result["source_git_state_sha256"] == hashlib.sha256(content).hexdigest()
    assert (output / "source-git-state.json").read_bytes() == content


@pytest.mark.parametrize(("field", "bad_value"), [
    ("repository", "unrelated/OtherProject"),
    ("head", "invalid"),
    ("tracked_status_sha256", "0" * 63),
    ("branch", "   "),
    ("captured_at_utc", "2026-11-26T00:00:00"),
    ("captured_at_utc", None),
])
def test_quality_rejects_invalid_or_foreign_receipts_before_writing(
    checkout, tmp_path, monkeypatch, field, bad_value,
):
    state = provenance.capture_git_state(checkout)
    state[field] = bad_value
    receipt = tmp_path / "invalid.json"
    receipt.write_text(json.dumps(state))
    monkeypatch.setattr(quality, "ROOT", checkout)
    output = tmp_path / "run"
    with pytest.raises(quality.ProtocolError):
        quality.git_provenance(output, receipt)
    assert not output.exists()


def freeze_args(tmp_path, source_state=None):
    """Build a freeze-only fixture; no dataset, GPU or provider call is used."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    prompts = inputs / "prompts"
    prompts.mkdir()
    for variant in searchqa.VARIANTS:
        (prompts / f"{variant}.system.txt").write_text(f"Synthetic {variant} prompt\n")
    (inputs / "train.parquet").write_bytes(b"synthetic bytes, never materialized")
    (inputs / "remainder.json").write_text(json.dumps([f"fixture-{n}" for n in range(4)]))
    service = {
        "schema_version": searchqa.SERVICE_SCHEMA,
        "status": "READY", "hostname": "fixture-worker",
        "request_model_id": "fixture-model", "expected_response_model_id": "fixture-model",
        "repository": "fixture/model", "revision": "0" * 40,
        "endpoints": [{
            "base_url": "http://127.0.0.1:1", "physical_gpu_index": 0,
            "gpu_uuid": "fixture-gpu", "port": 1, "service_dir": str(inputs),
            "supervisor_pid": 1, "server_pid": 2,
        }],
    }
    service["canonical_sha256"] = searchqa.canonical_hash(service)
    (inputs / "service.json").write_text(json.dumps(service))
    argv = [
        "freeze", "--experiment-dir", str(tmp_path / "run"),
        "--train-parquet", str(inputs / "train.parquet"),
        "--remainder-ids", str(inputs / "remainder.json"),
        "--prompt-source-dir", str(prompts), "--service-manifest", str(inputs / "service.json"),
        "--salt", "synthetic-fixture", "--discovery-tasks", "1", "--calibration-tasks", "1",
        "--confirmation-tasks", "1", "--reserve-tasks", "1",
    ]
    if source_state is not None:
        argv.extend(["--source-git-state", str(source_state)])
    return searchqa.parse_args(argv)


@pytest.mark.parametrize("source_only", [False, True])
def test_searchqa_freeze_binds_local_helper_and_validates_the_source_receipt(
    checkout, tmp_path, monkeypatch, source_only,
):
    receipt = None
    root = checkout
    if source_only:
        receipt = tmp_path / "source-git.json"
        receipt.write_text(json.dumps(provenance.capture_git_state(checkout)))
        root = tmp_path / "source-only"
        root.mkdir()
    monkeypatch.setattr(searchqa, "ROOT", root)
    args = freeze_args(tmp_path, receipt)
    assert searchqa.freeze(args) == 0
    protocol = searchqa.load_protocol(args.experiment_dir)
    assert protocol["source_state"]["repository"] == provenance.DEFAULT_REPOSITORY
    frozen = args.experiment_dir / "source_snapshot/harnessmm/source_provenance.py"
    assert frozen.read_bytes() == Path(provenance.__file__).read_bytes()
    frozen.write_text("# tampered synthetic snapshot\n")
    with pytest.raises(RuntimeError, match="provenance helper differs"):
        searchqa.load_protocol(args.experiment_dir)


def test_searchqa_rejects_foreign_source_state_before_creating_a_run(
    checkout, tmp_path, monkeypatch,
):
    state = provenance.capture_git_state(checkout)
    state["repository"] = "unrelated/OtherProject"
    receipt = tmp_path / "foreign-git.json"
    receipt.write_text(json.dumps(state))
    monkeypatch.setattr(searchqa, "ROOT", checkout)
    args = freeze_args(tmp_path, receipt)
    with pytest.raises(provenance.SourceProvenanceError, match="repository differs"):
        searchqa.freeze(args)
    assert not args.experiment_dir.exists()
