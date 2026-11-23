"""Submit a structured skill choice to the Responses provider and retain receipts."""
import base64
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import uuid4


class Actor:
    def __init__(self, root, model, reasoning_effort):
        self.root = Path(root)
        self.root.mkdir()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.calls = []

    def call(self, prompt, images, schema, purpose):
        if len(images) != 1:
            raise ValueError("Expected exactly one native RGB frame")

        credential = Path(os.environ["HMMR_CREDENTIAL_FILE"])
        if credential.stat().st_mode & 0o077:
            raise ValueError("Credential file must be owner-only")
        secret = credential.read_text().strip()
        if not secret:
            raise ValueError("Credential file is empty")
        endpoint = os.environ.get("HMMR_PROVIDER_URL", "https://api.githubcopilot.com/responses")
        image = base64.b64encode(Path(images[0]).read_bytes()).decode("ascii")
        request = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "stream": False,
            "store": False,
            "max_output_tokens": 4096,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
            ]}],
            "text": {"format": {
                "type": "json_schema", "name": "robot_decision", "strict": True, "schema": schema,
            }},
        }

        call_id = uuid4().hex
        directory = self.root / call_id
        directory.mkdir()
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request))
        config = "\n".join([
            f"url = {json.dumps(endpoint)}",
            f"header = {json.dumps('Authorization: Bearer ' + secret)}",
            'header = "Content-Type: application/json"',
            'header = "Copilot-Integration-Id: copilot-developer-cli"',
            'header = "User-Agent: GithubCopilot/1.0.100"',
        ])
        started = perf_counter()
        result = subprocess.run(
            ["curl", "-q", "--silent", "--show-error", "--connect-timeout", "10",
             "--max-time", "600", "--config", "-", "--data-binary", f"@{request_path}",
             "--write-out", "\\n%{http_code}"],
            input=config.encode(), capture_output=True, check=False,
        )
        duration_ms = (perf_counter() - started) * 1000
        raw, _, status = result.stdout.rpartition(b"\n")
        echoed_secret = secret.encode() in raw
        if not echoed_secret:
            (directory / "response.json").write_bytes(raw)
        try:
            response = {} if echoed_secret else json.loads(raw)
        except (ValueError, UnicodeError):
            response = {}
        if not isinstance(response, dict):
            response = {}
        usage = response.get("usage")
        row = {
            "id": call_id, "purpose": purpose, "model": self.model,
            "served_model": response.get("model"), "reasoning_effort": self.reasoning_effort,
            "elapsed_ms": duration_ms, "returncode": result.returncode,
            "http_status": status.decode("ascii", errors="replace"), "usage": usage,
            "automatic_retry": False, "response_omitted": echoed_secret,
        }
        self.calls.append(row)
        with (self.root / "calls.jsonl").open("a") as ledger:
            ledger.write(json.dumps(row) + "\n")

        if echoed_secret:
            raise RuntimeError(f"Provider echoed the credential in request {call_id}")
        if result.returncode or status != b"200" or response.get("status") != "completed":
            raise RuntimeError(f"Model request {call_id} failed (HTTP {row['http_status']})")
        if not str(response.get("model", "")).startswith(self.model):
            raise RuntimeError(f"Served model mismatch in request {call_id}")
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        ):
            raise RuntimeError(f"Incomplete provider accounting for request {call_id}")
        answer = "".join(
            part["text"]
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        return json.loads(answer)
