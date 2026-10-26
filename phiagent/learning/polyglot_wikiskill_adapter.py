"""Budget-matched Polyglot adapter for the WikiSkill reimplementation.

This adapter deliberately reuses the frozen DGM Polyglot task agent and
evaluator.  It adds only the paper-specified full skill injection hook.  The
official 225-task test is never selected here; callers must provide explicit
evolution and validation task IDs from the frozen 50-task development subset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Sequence


@dataclass(frozen=True)
class PolyglotSplit:
    train_items: tuple[str, ...]
    val_items: tuple[str, ...]
    test_items: tuple[str, ...] = ()

    def get_train_size(self) -> int:
        return len(self.train_items)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_text(path: Path, limit: int | None = None) -> str:
    if not path.is_file():
        return ""
    value = path.read_text(encoding="utf-8", errors="replace")
    return value if limit is None else value[:limit]


class PolyglotWikiSkillAdapter:
    """Expose the frozen DGM Polyglot harness through the WikiSkill API."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        train_ids: Sequence[str],
        val_ids: Sequence[str],
        seed: int,
        workers: int,
        same_model_id: str,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.seed = int(seed)
        self.workers = int(workers)
        self.same_model_id = str(same_model_id)
        self.split = PolyglotSplit(tuple(train_ids), tuple(val_ids))
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"Polyglot source root is missing: {self.source_root}")
        if not self.split.train_items or not self.split.val_items:
            raise ValueError("Polyglot train and validation splits must be non-empty")
        all_ids = self.split.train_items + self.split.val_items
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("Polyglot evolution and validation IDs overlap")
        if self.workers < 1:
            raise ValueError("workers must be positive")

        metadata_path = self.source_root / "polyglot/polyglot_benchmark_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metadata_by_id = {str(row["instance_id"]): row for row in metadata}
        missing = sorted(set(all_ids) - set(self.metadata_by_id))
        if missing:
            raise ValueError(f"unknown frozen Polyglot task IDs: {missing}")

    def setup(self, _: dict[str, Any] | None = None) -> None:
        return None

    def get_dataloader(self) -> PolyglotSplit:
        return self.split

    def build_eval_env(self, _size: int, split_name: str, _seed: int) -> list[str]:
        if split_name != "valid_seen":
            raise ValueError(f"unsupported Polyglot evaluation split: {split_name}")
        return list(self.split.val_items)

    def build_train_env(self, size: int, seed: int) -> list[str]:
        if int(size) != len(self.split.train_items):
            raise ValueError("WikiSkill Polyglot uses paper-faithful full-batch training")
        items = list(self.split.train_items)
        random.Random(int(seed)).shuffle(items)
        return items

    def _load_harness(self):
        source = str(self.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        return importlib.import_module("polyglot.harness").harness

    def _validate_selected(self, items: Sequence[str]) -> list[str]:
        selected = list(items)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Polyglot rollout task IDs must be unique and non-empty")
        allowed = set(self.split.train_items) | set(self.split.val_items)
        if not set(selected) <= allowed:
            raise PermissionError("rollout attempted to open a sealed Polyglot task")
        return selected

    def _rows_from_task_root(
        self,
        selected: Sequence[str],
        task_root: Path,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        transport_markers = (
            "connection refused",
            "connection reset",
            "read timed out",
            "connect timeout",
            "openai api error",
            "error in get_response",
            "no route to host",
            "temporary failure in name resolution",
        )
        for task_id in selected:
            result_path = task_root / f"{task_id}.json"
            payload: dict[str, Any]
            if result_path.is_file():
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
                payload = loaded if isinstance(loaded, dict) else {}
            else:
                payload = {}
            chat = _read_text(task_root / f"{task_id}.md")
            docker_log = _read_text(task_root / f"{task_id}_docker.log", 50000)
            eval_log = _read_text(task_root / f"{task_id}_eval.md", 50000)
            combined_lower = (chat + "\n" + docker_log).lower()
            transport_failure = next(
                (marker for marker in transport_markers if marker in combined_lower),
                "",
            )
            eval_result = str(payload.get("eval_result") or "missing")
            harness_success = bool(payload.get("success"))
            infrastructure_invalid = (
                not result_path.is_file()
                or not harness_success
                or eval_result in {"missing", "incomplete", "error"}
                or bool(transport_failure)
            )
            reason = ""
            if not result_path.is_file():
                reason = "missing task result artifact"
            elif transport_failure:
                reason = f"model transport marker: {transport_failure}"
            elif not harness_success or eval_result in {"incomplete", "error"}:
                reason = f"harness failure: success={harness_success}, eval={eval_result}"
            metadata = self.metadata_by_id[task_id]
            rows.append(
                {
                    "id": task_id,
                    "task_type": str(metadata.get("language") or "unknown"),
                    "hard": eval_result == "resolved" and not infrastructure_invalid,
                    "eval_result": eval_result,
                    "harness_success": harness_success,
                    "infrastructure_invalid": infrastructure_invalid,
                    "infrastructure_reason": reason,
                    "predicted_answer": eval_result,
                    "ground_truth": "resolved",
                    "model_patch": str(payload.get("model_patch") or ""),
                    "chat_history": chat,
                    "evaluation_log": eval_log,
                    "docker_log_excerpt": docker_log,
                }
            )
        return rows

    def rows_from_cached_rollout(
        self,
        items: Sequence[str],
        shared_cache_root: str | Path,
        output_dir: str | Path,
    ) -> list[dict[str, Any]]:
        """Reuse immutable initial-agent outcomes without another model call."""

        selected = self._validate_selected(items)
        cache_root = Path(shared_cache_root).resolve()
        if _read_text(cache_root / "status.txt").strip() != "COMPLETE":
            raise RuntimeError("shared Polyglot initial cache is incomplete")
        cached_ids = json.loads((cache_root / "task-list.json").read_text())
        if len(cached_ids) != 50 or len(set(cached_ids)) != 50:
            raise RuntimeError("shared Polyglot cache does not contain 50 unique tasks")
        if not set(selected) <= set(str(value) for value in cached_ids):
            raise RuntimeError("shared Polyglot cache is missing selected validation tasks")

        source_task_root = cache_root / "initial/predictions/initial_0"
        if not source_task_root.is_dir():
            raise FileNotFoundError(
                f"shared Polyglot task artifacts are missing: {source_task_root}"
            )
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        evidence_root = root / "reused-initial_0"
        evidence_root.mkdir()
        copied: list[dict[str, str]] = []
        for task_id in selected:
            for suffix in (".json", ".md", "_docker.log", "_eval.md"):
                source = source_task_root / f"{task_id}{suffix}"
                if not source.is_file():
                    continue
                destination = evidence_root / source.name
                shutil.copy2(source, destination)
                copied.append(
                    {
                        "task_id": task_id,
                        "file": destination.name,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    }
                )
        (root / "selected-task-ids.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (root / "shared-baseline-reuse.json").write_text(
            json.dumps(
                {
                    "source_cache_root": str(cache_root),
                    "source_task_root": str(source_task_root),
                    "model_calls_executed": 0,
                    "selected_task_count": len(selected),
                    "copied_artifacts": copied,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return self._rows_from_task_root(selected, evidence_root)

    def rollout(
        self,
        items: Sequence[str],
        active_skills: str,
        output_dir: str,
    ) -> list[dict[str, Any]]:
        selected = self._validate_selected(items)

        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        (root / "selected-task-ids.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (root / "active-skills.sha256").write_text(
            _sha256_text(active_skills) + "\n", encoding="utf-8"
        )
        (root / "active-skills.md").write_text(active_skills, encoding="utf-8")

        previous = os.environ.get("PHIAGENT_ACTIVE_SKILLS")
        os.environ["PHIAGENT_ACTIVE_SKILLS"] = active_skills
        model_name = "wikiskill-" + root.parent.name + "-" + root.name
        try:
            harness = self._load_harness()
            harness(
                dataset_path=str(
                    self.source_root / "polyglot/polyglot_benchmark_metadata.json"
                ),
                test_task_list=selected,
                num_samples=-1,
                max_workers=self.workers,
                model_name_or_path=model_name,
                model_patch_paths=None,
                num_evals=1,
                num_evals_parallel=1,
                pred_dname=root,
                output_dir=root,
                root_dir=str(self.source_root),
            )
        finally:
            if previous is None:
                os.environ.pop("PHIAGENT_ACTIVE_SKILLS", None)
            else:
                os.environ["PHIAGENT_ACTIVE_SKILLS"] = previous

        task_root = root / f"{model_name}_0"
        return self._rows_from_task_root(selected, task_root)


__all__ = ["PolyglotSplit", "PolyglotWikiSkillAdapter"]
