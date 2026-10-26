#!/usr/bin/env python3
"""Protocol-only JSONL process boundary; see runner.py for the versioned public ABI."""

from __future__ import annotations

import contextlib
import json
import re
import signal
import sys
import threading
from typing import Any, Callable, TextIO

if __package__:
    from .runner import OfficialHarnessSession, canonical_json, validate_prompt
else:
    from runner import OfficialHarnessSession, canonical_json, validate_prompt


def serve(
    input_stream: TextIO, output_stream: TextIO,
    *, session_factory: Callable[..., Any] = OfficialHarnessSession,
) -> int:
    """Process one initialized scenario. Cancellation and shutdown are terminal."""
    session: Any = None
    active: threading.Thread | None = None
    busy = threading.Event()
    output_lock = threading.Lock()
    seen: set[str] = set()

    def reply(request: dict[str, Any], *, ok: bool, **payload: Any) -> None:
        value = {
            "protocol_version": 1, "id": request.get("id"), "op": request.get("op"),
            "ok": ok, **payload,
        }
        if session is not None:
            value = session.redact(value)
        with output_lock:
            output_stream.write(canonical_json(value) + "\n")
            output_stream.flush()

    def error(request: dict[str, Any], exc: Exception) -> None:
        reply(request, ok=False, error={"type": type(exc).__name__, "message": str(exc)})

    def run(request: dict[str, Any]) -> None:
        try:
            result = session.run(request["prompt"])
        except Exception as exc:
            busy.clear()
            error(request, exc)
        else:
            busy.clear()
            reply(request, ok=True, result=result)

    try:
        for line in input_stream:
            request: dict[str, Any] = {}
            try:
                try:
                    parsed = json.loads(line)
                except (ValueError, TypeError):
                    raise ValueError("invalid JSON request") from None
                if not isinstance(parsed, dict):
                    raise ValueError("request must be an object")
                identity, operation = parsed.get("id"), parsed.get("op")
                if not isinstance(identity, str) or not re.fullmatch(
                        r"[a-zA-Z0-9_.-]{1,120}", identity):
                    raise ValueError("request id must be a nonempty safe identifier")
                request = {"id": identity, "op": operation}
                if identity in seen:
                    raise ValueError("duplicate request id; native turns are never replayed")
                seen.add(identity)
                if ("protocol_version" in parsed and (
                        type(parsed["protocol_version"]) is not int
                        or parsed["protocol_version"] != 1)):
                    raise ValueError("unsupported protocol_version")
                fields = {
                    "init": {"job"}, "run": {"prompt"}, "cancel": set(), "shutdown": set(),
                }
                if not isinstance(operation, str) or operation not in fields:
                    raise ValueError("unsupported operation")
                if (set(parsed) - {"id", "op", "protocol_version"} != fields[operation]):
                    raise ValueError("request fields differ from the operation contract")
                request = parsed
                if operation == "init":
                    if session is not None:
                        raise ValueError("the worker already owns an initialized session")
                    session = session_factory(request["job"])
                    try:
                        session.start()
                    except Exception as exc:
                        error(request, exc)
                        return 1
                    reply(request, ok=True, session_id=session.session_id,
                          run_dir=str(session.run_dir), runtime_pid=session.runtime_pid)
                elif session is None:
                    raise ValueError("initialize the worker before submitting an operation")
                elif operation == "run":
                    if busy.is_set():
                        raise ValueError("a native turn is already active; wait for its result")
                    validate_prompt(request["prompt"])
                    if active is not None:
                        active.join()
                    busy.set()
                    active = threading.Thread(target=run, args=(request,),
                                              name="official-sdk-native-turn")
                    active.start()
                else:
                    result = session.cancel() if operation == "cancel" else session.close()
                    if active is not None:
                        active.join()
                    reply(request, ok=True, result=result)
                    return 0
            except Exception as exc:
                error(request, exc)
        return 0
    finally:
        if session is not None:
            session.close()
        if active is not None:
            active.join()


def main() -> int:
    protocol_stdout = sys.stdout

    def interrupted(signum: int, _frame: Any) -> None:
        signal.signal(signum, signal.SIG_IGN)
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            return serve(sys.stdin, protocol_stdout)
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
