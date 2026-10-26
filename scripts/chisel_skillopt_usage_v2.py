"""Optional, durable qwen attempt accounting; stdlib only, no model imports."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit
import uuid

ENV = "PHIAGENT_SKILLOPT_USAGE_DB"


class IncompleteRun(BaseException):
    """Bypass upstream Exception-to-look/merge fallbacks after a fatal guard."""


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@contextmanager
def _db(path: str):
    connection = None
    try:
        target = Path(path).absolute()
        if target.is_symlink() or target.resolve() != target:
            raise IncompleteRun("usage ledger must not traverse symlinks")
        connection = sqlite3.connect(target.as_uri() + "?mode=rw",
                                     uri=True, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except (sqlite3.Error, OSError) as error:
        raise IncompleteRun(f"usage ledger unavailable: {type(error).__name__}") from error
    finally:
        if connection is not None:
            connection.close()


def initialize(path: Path, profile: dict, *, hard_ceiling_total_tokens: int | None = None) -> dict:
    """Create once, before an authorized run. No implicit resume or zero-usage fallback."""
    for role in ("target", "optimizer"):
        spec = profile["roles"][role]
        url = urlsplit(spec.get("base_url") or "")
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("freeze both credential-free role service endpoints before initialization")
        if url.query or url.fragment:
            raise ValueError("endpoint query/fragment must not contain transport secrets")
        if spec["context_tokens"] <= spec["max_tokens"] or spec["seed"] != 42:
            raise ValueError("invalid frozen context/output/seed contract")
    if hard_ceiling_total_tokens is not None and (
        type(hard_ceiling_total_tokens) is not int or hard_ceiling_total_tokens <= 0
    ):
        raise ValueError("hard ceiling must be a positive token-reservation integer")
    if path.is_symlink() or path.parent.resolve() != path.parent.absolute():
        raise ValueError("ledger must not traverse symlinks")
    with path.open("xb"):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript("""
            CREATE TABLE run (
                id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL,
                profile_json TEXT NOT NULL, profile_sha256 TEXT NOT NULL,
                reserved_tokens INTEGER NOT NULL, ceiling_tokens INTEGER, abort_reason TEXT
            );
            CREATE TABLE calls (
                id TEXT PRIMARY KEY, role TEXT NOT NULL, stage TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, endpoint TEXT NOT NULL,
                retries INTEGER NOT NULL, status TEXT NOT NULL, started REAL NOT NULL
            );
            CREATE TABLE attempts (
                id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, attempt_index INTEGER NOT NULL,
                role TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
                started REAL NOT NULL, finished REAL, reserved_tokens INTEGER NOT NULL,
                prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                response_sha256 TEXT, error_type TEXT, error_sha256 TEXT,
                usage_status TEXT NOT NULL, UNIQUE(call_id, attempt_index)
            );
        """)
        connection.execute("INSERT INTO run VALUES(1,'OPEN',?,?,0,?,NULL)",
                           (json.dumps(profile, sort_keys=True), _hash(profile),
                            hard_ceiling_total_tokens))
        connection.commit()
    finally:
        connection.close()
    return {"schema_version": "phiagent-skillopt-usage/v2", "profile_sha256": _hash(profile),
            "database": str(path), "hard_ceiling_total_tokens": hard_ceiling_total_tokens}


def _abort(db, reason: str):
    db.execute("UPDATE run SET status='ABORT_INCOMPLETE', abort_reason=? WHERE id=1", (reason,))
    db.commit()
    raise IncompleteRun(reason)


def _open_run(db):
    run = db.execute("SELECT * FROM run WHERE id=1").fetchone()
    if run is None:
        raise IncompleteRun("uninitialized usage ledger")
    if run["status"] != "OPEN":
        raise IncompleteRun("run already ABORT/INCOMPLETE")
    return run


def begin_call(role: str, stage: str, payload: dict, retries: int, endpoint: str):
    path = os.environ.get(ENV)
    if not path:
        return None
    with _db(path) as db:
        run = _open_run(db)
        profile = json.loads(run["profile_json"])
        if role not in profile["roles"]:
            _abort(db, "unexpected provider role")
        spec = profile["roles"][role]
        expected_retries = 5 if role == "target" else 3
        allowed_stages = {"rollout"} if role == "target" else {
            "analyst", "merge", "ranking", "slow_update", "meta_skill",
        }
        if (payload.get("model") != spec["model"]
                or payload.get("seed") != spec["seed"]
                or payload.get("temperature") != spec["temperature"]
                or payload.get("chat_template_kwargs") != {"enable_thinking": False}
                or payload.get("max_tokens") != min(spec["max_tokens"], 16384)
                or payload.get("tools") or payload.get("tool_choice")
                or endpoint.rstrip("/") != spec["base_url"].rstrip("/")
                or retries != expected_retries or stage not in allowed_stages):
            _abort(db, "frozen role/stage/wire contract mismatch")
        call_id = uuid.uuid4().hex
        db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)",
                   (call_id, role, stage, _hash(payload), endpoint,
                    retries, "STARTED", time.time()))
    return {"path": path, "call_id": call_id, "role": role, "stage": stage}


def before_attempt(audit, attempt_index: int):
    if audit is None:
        return None
    with _db(audit["path"]) as db:
        run = _open_run(db)
        profile = json.loads(run["profile_json"])
        reserve = profile["roles"][audit["role"]]["context_tokens"]
        role_count = db.execute("SELECT COUNT(*) FROM attempts WHERE role=?",
                                (audit["role"],)).fetchone()[0]
        if role_count >= (138000 if audit["role"] == "target" else 114):
            _abort(db, "complete-loop maximum HTTP attempts exhausted")
        total = run["reserved_tokens"] + reserve
        if run["ceiling_tokens"] is not None and total > run["ceiling_tokens"]:
            _abort(db, "prospective token-cost ceiling reached")
        db.execute("UPDATE run SET reserved_tokens=? WHERE id=1", (total,))
        cursor = db.execute("""
            INSERT INTO attempts(call_id,attempt_index,role,stage,status,started,
                                 reserved_tokens,usage_status)
            VALUES(?,?,?,?,'STARTED',?,?,'UNKNOWN')
        """, (audit["call_id"], attempt_index, audit["role"], audit["stage"], time.time(), reserve))
        return cursor.lastrowid


def response_received(audit, attempt_id, response: dict):
    if audit is None:
        return
    usage = response.get("usage") if isinstance(response, dict) else None
    valid = isinstance(usage, dict) and all(
        type(usage.get(k)) is int and usage[k] >= 0
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if valid:
        valid = usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    with _db(audit["path"]) as db:
        db.execute("""
            UPDATE attempts SET status='RESPONSE',response_sha256=?,prompt_tokens=?,
                completion_tokens=?,total_tokens=?,usage_status=? WHERE id=? AND call_id=?
        """, (_hash(response), usage["prompt_tokens"] if valid else None,
              usage["completion_tokens"] if valid else None,
              usage["total_tokens"] if valid else None, "REPORTED" if valid else "UNKNOWN",
              attempt_id, audit["call_id"]))
        if valid:
            row = db.execute("SELECT reserved_tokens FROM attempts WHERE id=?",
                             (attempt_id,)).fetchone()
            profile = json.loads(db.execute("SELECT profile_json FROM run").fetchone()[0])
            cap = profile["roles"][audit["role"]]["max_tokens"]
            if usage["total_tokens"] > row["reserved_tokens"] or usage["completion_tokens"] > cap:
                _abort(db, "provider violated frozen context/output ceiling")


def succeeded(audit, attempt_id):
    if audit is None:
        return
    with _db(audit["path"]) as db:
        db.execute("UPDATE attempts SET status='SUCCEEDED',finished=? WHERE id=? AND call_id=?",
                   (time.time(), attempt_id, audit["call_id"]))
        db.execute("UPDATE calls SET status='SUCCEEDED' WHERE id=?", (audit["call_id"],))


def failed(audit, attempt_id, error: Exception):
    if audit is None:
        return
    with _db(audit["path"]) as db:
        db.execute("""
            UPDATE attempts SET status='FAILED',finished=?,error_type=?,error_sha256=?
            WHERE id=? AND call_id=?
        """, (time.time(), type(error).__name__, _hash(str(error)), attempt_id, audit["call_id"]))


def exhausted(audit):
    if audit is None:
        return
    with _db(audit["path"]) as db:
        db.execute("UPDATE calls SET status='EXHAUSTED' WHERE id=?", (audit["call_id"],))
        _abort(db, "provider retries exhausted; do not replace failure with look/merge success")


def report(path: Path) -> dict:
    with _db(str(path)) as db:
        run = dict(db.execute("SELECT * FROM run WHERE id=1").fetchone())
        rows = [dict(r) for r in db.execute("""
            SELECT role,stage,COUNT(*) AS attempts,
                SUM(status='SUCCEEDED') AS succeeded,SUM(status='FAILED') AS failed,
                SUM(status IN ('STARTED','RESPONSE')) AS unfinished,
                SUM(usage_status='UNKNOWN') AS unknown_usage,
                SUM(prompt_tokens) AS reported_prompt_tokens,
                SUM(completion_tokens) AS reported_completion_tokens,
                SUM(total_tokens) AS reported_total_tokens
            FROM attempts GROUP BY role,stage ORDER BY role,stage
        """)]
        unfinished_calls = db.execute(
            "SELECT COUNT(*) FROM calls WHERE status!='SUCCEEDED'").fetchone()[0]
    unknown = sum(r["unknown_usage"] for r in rows)
    unfinished = sum(r["unfinished"] for r in rows)
    observed_sum = sum(r["reported_total_tokens"] or 0 for r in rows)
    exact = unknown == 0 and unfinished == 0 and unfinished_calls == 0
    return {
        "schema_version": "phiagent-skillopt-usage-report/v2",
        "run_status": run["status"], "abort_reason": run["abort_reason"],
        "profile_sha256": run["profile_sha256"], "by_role_stage": rows,
        "unknown_usage_attempts": unknown, "unfinished_attempts": unfinished,
        "unfinished_calls": unfinished_calls,
        "observed_reported_token_sum": observed_sum,
        "exact_total_tokens": observed_sum if exact else None,
        "exact_usage_valid": exact,
        "reserved_token_upper": run["reserved_tokens"],
        "hard_ceiling_total_tokens": run["ceiling_tokens"],
        "usage_gate_passed": exact and run["status"] == "OPEN",
        "algorithm_complete": False,
        "completion_note": "Usage pass alone never proves four epochs or final selection.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize", "report"])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--hard-ceiling-total-tokens", type=int)
    args = parser.parse_args()
    if args.command == "initialize":
        if args.profile is None:
            parser.error("--profile is required for initialize")
        value = json.loads(args.profile.read_bytes())
        result = initialize(args.database, value.get("profile", value),
                            hard_ceiling_total_tokens=args.hard_ceiling_total_tokens)
    else:
        result = report(args.database)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
