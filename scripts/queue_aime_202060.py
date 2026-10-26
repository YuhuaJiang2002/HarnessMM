"""Local read-only orchestration; every preparer, model and scorer runs remotely."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

RUNS = (
    ("train0-v3", "train_baseline", 2, 29320, 0),
    ("train1-v2", "train_baseline", 7, 29370, 1),
    ("skillopt-v2", "skillopt", 5, 29350, None),
    ("autosaddler-v2", "autosaddler", 6, 29360, None),
)
PARENT = "/opt/phiagent/work/aime-comparison"
PARTITION = "/opt/phiagent/work/df-202060/20260928-partition-v2"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def remote(code, *, timeout=60):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "h200-2", "python3 -"],
        input=code, text=True, check=True, capture_output=True, timeout=timeout,
    )
    return json.loads(result.stdout)


def source_ready():
    return remote(f"""
import pathlib,json,hashlib
r=pathlib.Path({PARTITION!r})
p=r/'PREPARED.json'
if not p.exists():
 print(json.dumps({{'ready':False}}))
else:
 proof=json.loads(p.read_text())
 binding=next(x for x in proof['partitions'] if x['benchmark']=='aime2025')
 manifest=r/'aime2025/SPLIT_MANIFEST.json'
 actual=hashlib.sha256(manifest.read_bytes()).hexdigest()
 assert actual==binding['manifest_sha256']
 assert binding['counts']=={{'train':6,'validation':6,'test':18}}
 print(json.dumps({{'ready':True,'sha256':actual,'parent_receipt':proof}}))
""")


def capacity(destination):
    command = (
        "hostname; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu "
        "--format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory "
        "--format=csv,noheader; free -m; uptime; ps -eo user,pid,ppid,etimes,comm; lslocks --json"
    )
    record = {"at_utc": now()}
    for host in ("h200-1", "h200-2"):
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
            text=True, capture_output=True, timeout=60,
        )
        record[host] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if result.returncode:
            raise RuntimeError("Both-node capacity inspection failed: " + host)
    with (destination / "CAPACITY.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    return record


def launch(spec, bundle, sha256):
    suffix, mode, gpu, port, shard = spec
    name = "20260928-split202060-" + suffix
    command = [
        "python3", bundle + "/scripts/prepare_aime_202060.py",
        "--name", name, "--mode", mode, "--gpu", str(gpu), "--port", str(port),
        "--manifest-sha256", sha256, "--partition-root", PARTITION + "/aime2025", "--wait-for-gpu",
    ]
    if shard is not None:
        command.extend(("--shard", str(shard)))
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "h200-2", shlex.join(command)],
        text=True, capture_output=True, timeout=180,
    )
    return {
        "at_utc": now(), "root": PARENT + "/" + name, "mode": mode, "gpu": gpu,
        "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }


def status(roots):
    return remote(f"""
import pathlib,json,os
results=[]
for root in {roots!r}:
 r=pathlib.Path(root);value={{'root':root}}
 for name in ('LAUNCH.json','PROGRESS.json','PROVIDER_LEDGER.json','FAILURE.json','VERIFIED.json','RESULT.json','RETIREMENT.json'):
  p=r/name
  if p.exists():value[name]=json.loads(p.read_text())
 pid=value.get('LAUNCH.json',{{}}).get('pid')
 try:
  os.kill(pid,0)
  value['alive']=pathlib.Path('/proc',str(pid),'stat').read_text().split()[2]!='Z'
 except (OSError,TypeError):value['alive']=False
 value['gpu_allocated']=(r/'GPU_SELECTION.json').exists()
 value['provider_requests']=len(list((r/'provider').glob('call-*')))
 value['waiting_unused']=(value['alive'] and (r/'WAITING_FOR_GPU.json').exists()
  and not value['gpu_allocated'] and value['provider_requests']==0
  and not (r/'service/OWNER.json').exists() and not (r/'PROVIDER_LEDGER.json').exists())
 if value['gpu_allocated']:
  selected=json.loads((r/'GPU_SELECTION.json').read_text())
  value['gpu_index']=selected['index']
 results.append(value)
print(json.dumps(results))
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=PARENT + "/split202060-full-bundle-v2")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    claim = args.destination / "OWNER.json"
    with claim.open("x") as output:
        json.dump({"pid": os.getpid(), "at_utc": now(), "command": os.sys.argv}, output, indent=2)
    launches = None
    last_capacity = 0
    while True:
        try:
            if time.monotonic() - last_capacity >= 300:
                last_capacity = time.monotonic()
                capacity(args.destination)
            if launches is None:
                ready = source_ready()
                write(args.destination / "PARTITION_STATUS.json", {"at_utc": now(), **ready})
                if ready["ready"]:
                    capacity(args.destination)
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        launches = list(pool.map(
                            lambda spec: launch(spec, args.bundle, ready["sha256"]), RUNS,
                        ))
                    write(args.destination / "LAUNCHES.json", launches)
                    print(json.dumps({"at_utc": now(), "launches": launches}), flush=True)
                    if any(row["returncode"] for row in launches):
                        write(args.destination / "PREPARATION_BLOCKERS.json", launches)
            if launches is not None:
                rows = status([row["root"] for row in launches])
                write(args.destination / "LATEST.json", {"at_utc": now(), "runs": rows})
                with (args.destination / "EVENTS.jsonl").open("a") as output:
                    output.write(json.dumps({"at_utc": now(), "runs": rows}) + "\n")
                train = [row for row in rows if any(key in row["root"] for key in ("train0-", "train1-"))]
                if len(train) == 2 and all("VERIFIED.json" in row for row in train):
                    write(args.destination / "TRAIN_READY.json", {
                        "at_utc": now(), "roots": [row["root"] for row in train],
                        "next_task": "Derive constraint ledger exclusively from complete train6 baseline",
                    })
                if all(not row["alive"] for row in rows):
                    write(args.destination / "TERMINAL.json", {"at_utc": now(), "runs": rows})
                    break
            delay = min(120, max(1, last_capacity + 300 - time.monotonic()))
            time.sleep(delay)
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
            with (args.destination / "ERRORS.jsonl").open("a") as output:
                output.write(json.dumps({"at_utc": now(), "error": str(error)}) + "\n")
            time.sleep(120)


if __name__ == "__main__":
    main()
