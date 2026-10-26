#!/usr/bin/env python3
"""Present saved AA aggregates only; no task inputs, generation, or evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


LABELS = {
    "original_chisel": "Serial A",
    "parallel_fixed_a": "Concurrent 16 / fixed order A",
    "parallel_shuffled_a": "Concurrent 16 / shuffled order A",
    "parallel_shuffled_b": "Concurrent 16 / shuffled order B",
    "parallel_fixed_b": "Concurrent 16 / fixed order B",
    "serial_repeat": "Serial B",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_summary(campaign: Path) -> dict:
    result_path, config_path = campaign / "experiment/RESULT.json", campaign / "experiment/config.json"
    audit_path = campaign / "audit-v1/VERIFIED.json"
    result, config, audit = (json.loads(path.read_text()) for path in (result_path, config_path, audit_path))
    schedule_path = campaign / "audit-v1/CLIENT_SCHEDULE.json"
    if (
        config["schema_version"] != "phiagent-qa-aa-calibration/v1"
        or list(config["arms"]) != list(LABELS)
        or audit["status"] != "INDEPENDENT_AA_CALIBRATION_REPLAY_VERIFIED"
        or audit["result_sha256"] != sha(result_path) or audit["config_sha256"] != sha(config_path)
        or audit["client_schedule_sha256"] != sha(schedule_path)
        or audit["methods"] != result["methods"]
        or audit["repeatability"] != result["aa_repeatability"]
        or audit["target_gate"] != result["target_gate"]
        or any(value["paper_admitted"] is not False for value in (config, result, audit))
        or result["calls"]["unknown_cost_requests"] != 0 or result["calls"]["pending_http"] != 0
    ):
        raise ValueError("Complete hash-verified AA diagnostics are required")
    schedule = json.loads(schedule_path.read_text())
    repeatability = audit["repeatability"]
    rows = []
    for arm in config["arms"]:
        method = result["methods"][arm]
        successes = round(method["exact_match"] * method["tasks"] / 100)
        if abs(100 * successes / method["tasks"] - method["exact_match"]) > 1e-10:
            raise ValueError("AA EM does not match integer success counts")
        rows.append({
            "arm": arm, "tasks": method["tasks"], "successes": successes,
            "exact_match": method["exact_match"], "f1": method["f1"],
            "prompt_tokens": method["prompt_tokens"], "completion_tokens": method["completion_tokens"],
            "total_tokens": method["total_tokens"],
            **{key: value for key, value in schedule["phases"][arm].items() if key != "requests"},
        })
    pairs = []
    for pair in repeatability["all_fifteen_pairs"].values():
        pairs.append({
            "left": pair["left"], "right": pair["right"], "within_mode_repeat": pair["within_mode_repeat"],
            **{f"{key}_changed_tasks": value for key, value in pair["changed_tasks"].items()},
            "em_gains": pair["em_gains"], "em_losses": pair["em_losses"],
            "em_difference_pp": pair["em_difference_pp"], "f1_difference_pp": pair["f1_difference_pp"],
            "total_token_difference": pair["total_token_difference"],
        })
    return {
        "phases": rows, "pairs": pairs, "repeatability": repeatability,
        "accounting": audit["accounting"],
        "parallel_http_observed": schedule["parallel_http_observed_in_every_concurrent_phase"],
        "model": config["model"], "resource": {
            key: config["resource"][key] for key in ("selected_hostname", "physical_gpu_index", "gpu_uuid")
        },
        "runtime_batch_invariant": config["calibration"]["runtime_batch_invariant"],
        "scope": config["data"]["claim_boundary"],
        "evidence": {
            "result_sha256": sha(result_path), "verification_sha256": sha(audit_path),
            "config_sha256": sha(config_path), "schedule_sha256": sha(schedule_path),
        },
        "model_calls": 0, "evaluator_calls": 0, "benchmark_inputs_opened": False,
        "paper_admitted": False,
    }


def render(campaign: Path, output: Path) -> dict:
    summary = verified_summary(campaign)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("phases.csv", summary["phases"]), ("pairs.csv", summary["pairs"])):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "data.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    diagnostic = summary["repeatability"]
    changes = diagnostic["any_change_question_groups"]
    phase_rows = "".join(
        f'<tr><th>{html.escape(LABELS[row["arm"]])}</th><td>{row["successes"]}/{row["tasks"]}</td>'
        f'<td>{row["exact_match"]:.3f}%</td><td>{row["f1"]:.3f}%</td>'
        f'<td>{row["prompt_tokens"]:,}</td><td>{row["completion_tokens"]:,}</td>'
        f'<td>{row["total_tokens"]:,}</td><td>{row["peak_client_http_overlap"]}</td>'
        f'<td>{row["latency_p50_seconds"]:.3f}</td><td>{row["latency_p95_seconds"]:.3f}</td></tr>'
        for row in summary["phases"]
    )
    pair_rows = "".join(
        f'<tr><th>{html.escape(LABELS[row["left"]])} / {html.escape(LABELS[row["right"]])}</th>'
        f'<td>{row["raw_content_changed_tasks"]}</td><td>{row["extracted_answer_changed_tasks"]}</td>'
        f'<td>{row["em_gains"]}</td><td>{row["em_losses"]}</td><td>{row["f1_changed_tasks"]}</td>'
        f'<td>{row["completion_tokens_changed_tasks"]}</td></tr>'
        for row in summary["pairs"]
    )
    upper = diagnostic["zero_change_iid_one_sided_95_upper_bound"]
    sensitivity = (
        f'Even with zero observed answer changes, the one-sided 95% zero-event upper bound is '
        f'{100 * upper:.3f}% under iid question-group assumptions, not zero.'
        if upper is not None else
        'Observed variation refutes exact repeatability for these tested calls; its runtime cause is not isolated.'
    )
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Original CHISEL A/A calibration</title>
<style>body{{font:16px system-ui;line-height:1.6;color:#172238;background:#f3f6fa;margin:0}}
main{{max-width:1400px;margin:auto;padding:28px}}section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dce3ed;padding:10px;text-align:right}}
th:first-child{{text-align:left}}.scroll{{overflow-x:auto}}nav a{{margin-right:18px}}.boundary{{border-left:5px solid #b36b00}}
code{{overflow-wrap:anywhere}}</style><main><h1>Original CHISEL: A/A measurement calibration</h1>
<nav><a href="phases.csv">Six phases CSV</a><a href="pairs.csv">All 15 pairs CSV</a>
<a href="data.json">Verified data</a><a href="../audit-v1/VERIFIED.json">Raw audit</a></nav>
<section class="boundary"><strong>{html.escape(diagnostic["status"])}</strong>
<p>{html.escape(summary["scope"])}</p>
<p>{diagnostic["question_groups"]} unique question groups; six fresh requests per task.
The repeated requests are not six times as many independent observations. All phases use identical
Original request bytes, model revision, seed, complete context and output cap. No new policy is tested.</p>
<p>Groups with any change across all six runs: raw assistant text <strong>{changes["raw_content"]}</strong>;
extracted answer <strong>{changes["extracted_answer"]}</strong>; EM <strong>{changes["exact_match"]}</strong>;
F1 <strong>{changes["f1"]}</strong>; completion usage <strong>{changes["completion_tokens"]}</strong>.</p>
<p>{sensitivity}</p></section>
<section><h2>Frozen scheduling and measured overlap</h2>
<p>Execution order: serial A, fixed-order concurrent A, shuffled-order concurrent A,
shuffled-order concurrent B, fixed-order concurrent B, serial B.
Concurrent phases dispatch ordered groups of at most16requests and wait at a barrier.
Actual GPU microbatches are not controlled or observed. Cache state and elapsed time remain possible
contributors; differing responses do not prove a specific batch-kernel cause.</p>
<p>Concurrent HTTP overlap observed in every concurrent phase: {summary["parallel_http_observed"]}.
The retained runtime has VLLM_BATCH_INVARIANT={html.escape(summary["runtime_batch_invariant"])}.
No determinism flag or production default was changed.</p>
<div class="scroll"><table><thead><tr><th>Phase</th><th>Correct</th><th>Official EM</th><th>Official F1</th>
<th>Input tokens</th><th>Output tokens</th><th>Total tokens</th><th>Peak HTTP</th>
<th>Latency p50 s</th><th>Latency p95 s</th></tr></thead><tbody>{phase_rows}</tbody></table></div></section>
<section><h2>Every paired comparison</h2><p>Gains and losses are right minus left; unchanged totals can hide turnover.
Raw comparison excludes provider IDs and timestamps. These15comparisons are descriptive, not a search for
a favorable significance test.</p><div class="scroll"><table><thead><tr><th>Left / right</th><th>Raw changes</th>
<th>Answer changes</th><th>EM gains</th><th>EM losses</th><th>F1 changes</th><th>Output-token changes</th>
</tr></thead><tbody>{pair_rows}</tbody></table></div></section>
<section><h2>Cost and provenance</h2><p>{summary["accounting"]["real_requests"]:,} fresh task requests;
{summary["accounting"]["synthetic_requests"]} synthetic admission request;
{summary["accounting"]["all_in_acquisition_tokens"]:,} total acquisition tokens; unknown-cost requests:
{summary["accounting"]["unknown_cost_requests"]}. GPU allocation:
{summary["accounting"]["allocation_seconds"]:.2f} seconds, with audited retirement.</p>
<p>Model <code>{html.escape(summary["model"]["repository"])}</code>, revision
<code>{html.escape(summary["model"]["revision"])}</code>; seed {summary["model"]["request_seed"]}.
Host <code>{html.escape(summary["resource"]["selected_hostname"])}</code>,
physical GPU {summary["resource"]["physical_gpu_index"]}.</p>
<p>This report invokes no model or evaluator. No candidate-quality, public-benchmark or universal
repeatability claim follows from this calibration.</p></section></main></html>"""
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(content)
    manifest = {
        "status": "VERIFIED_AA_AGGREGATE_PRESENTATION",
        "evidence": summary["evidence"], "model_calls": 0, "evaluator_calls": 0,
        "benchmark_inputs_opened": False,
        "files_sha256": {name: sha(output / name) for name in ("phases.csv", "pairs.csv", "data.json", "index.html")},
    }
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.campaign, args.output), indent=2))


if __name__ == "__main__":
    main()
