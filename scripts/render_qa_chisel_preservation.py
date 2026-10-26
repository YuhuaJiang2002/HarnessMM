#!/usr/bin/env python3
"""Render only hash-verified saved aggregates; no tasks, models, or evaluators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from scripts.render_qa_capability_cost import LABELS as COMPARISON_LABELS
from scripts.render_qa_harness_mixture import FIELDS, sha


LABELS = {
    **COMPARISON_LABELS,
    "source_field_original": "Strict source field + fresh Original fallback",
    "light_original": "Original head + light rule compression",
}
CANDIDATES = ("source_field_original", "light_original")


def checked_summary(campaign: Path, audit_name: str = "audit-v1") -> dict:
    if Path(audit_name).name != audit_name or audit_name in {"", ".", ".."}:
        raise ValueError("A single campaign-relative audit directory is required")
    result_path = campaign / "experiment/RESULT.json"
    config_path = campaign / "experiment/config.json"
    audit_path = campaign / audit_name / "VERIFIED.json"
    result, config, audit = (
        json.loads(path.read_text()) for path in (result_path, config_path, audit_path)
    )
    candidates = config["candidate_arms"]
    expected = set(COMPARISON_LABELS) | set(candidates)
    if (
        config["schema_version"] != "phiagent-qa-chisel-preservation/v1"
        or audit["status"] != "INDEPENDENT_HARNESS_MIXTURE_REPLAY_VERIFIED"
        or audit["result_sha256"] != sha(result_path)
        or audit["config_sha256"] != sha(config_path)
        or audit["methods"] != result["methods"]
        or audit["target_gate"] != result["target_gate"]
        or candidates not in (list(CANDIDATES), [CANDIDATES[0]])
        or config["primary_reference"] != "original_chisel"
        or config["cost_reference"] != "original_chisel"
        or set(result["methods"]) != set(config["arms"])
        or len(config["arms"]) != len(expected) or set(config["arms"]) != expected
        or any(value["paper_admitted"] is not False for value in (result, config, audit))
    ):
        raise ValueError("Complete hash-bound preservation evidence is required")
    accounting, calls = result["execution_accounting"], result["calls"]
    if (
        calls["pending_http"] != 0 or calls["unknown_cost_requests"] != 0
        or accounting["reused_model_responses"] != 0
        or accounting["real_provider_requests"] != calls["provider_requests_completed"]
        or accounting["deterministic_source_outputs"] != calls["deterministic_outputs"]
        or accounting["logical_cells"] != calls["completed"]
        or calls["known_real_tokens"] != sum(row["total_tokens"] for row in result["methods"].values())
    ):
        raise ValueError("Fresh provider/native accounting is incomplete or unknown")
    original_tokens = result["methods"]["original_chisel"]["total_tokens"]
    rows = []
    for arm in config["arms"]:
        method = result["methods"][arm]
        pairs = result["paired_original_comparisons"][arm]
        successes = pairs["both_correct"] + pairs["gains"]
        if (
            sum(pairs[key] for key in ("both_correct", "both_wrong", "gains", "regressions")) != method["tasks"]
            or abs(successes / method["tasks"] * 100 - method["exact_match"]) > 1e-10
            or method["prompt_tokens"] + method["completion_tokens"] != method["total_tokens"]
        ):
            raise ValueError("Paired success counts, official EM, or additive tokens differ")
        rows.append({
            "arm": arm, **{key: method[key] for key in FIELDS},
            "successes": successes,
            "total_token_saving_percent": (1 - method["total_tokens"] / original_tokens) * 100,
            "em_regressions": pairs["regressions"], "em_gains": pairs["gains"],
            "f1_regressions": pairs["f1_regressions"],
        })
    return {
        "rows": rows, "target_gate": result["target_gate"],
        "candidate_arms": candidates,
        "light_readiness_blocker": config.get("preservation", {}).get("light_readiness_blocker"),
        "execution_accounting": accounting, "cost_accounting": result["cost_accounting"],
        "model": config["model"], "resource": {
            key: config["resource"][key]
            for key in ("selected_hostname", "physical_gpu_index", "gpu_uuid", "hardware_name")
        },
        "cohort_boundary": config["data"]["claim_boundary"],
        "question_groups": config["data"]["question_groups"],
        "measurement_identity": "Fresh native outputs and separately generated model requests for every fallback",
        "source_bindings": {
            "result_sha256": sha(result_path), "config_sha256": sha(config_path),
            "verification_sha256": sha(audit_path),
        },
        "paper_admitted": False, "production_defaults_changed": False,
    }


def render(campaign: Path, output: Path, audit_name: str = "audit-v1") -> dict:
    summary = checked_summary(campaign, audit_name)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    with (output / "data.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    body = []
    for row in summary["rows"]:
        body.append(
            f'<tr><th>{html.escape(LABELS[row["arm"]])}</th>'
            f'<td>{row["successes"]}/{row["tasks"]}</td><td>{row["exact_match"]:.3f}%</td>'
            f'<td>{row["f1"]:.3f}%</td><td>{row["prompt_tokens"]:,}</td>'
            f'<td>{row["completion_tokens"]:,}</td><td>{row["total_tokens"]:,}</td>'
            f'<td>{row["total_token_saving_percent"]:.2f}%</td>'
            f'<td>{row["em_regressions"]}</td><td>{row["em_gains"]}</td>'
            f'<td>{row["f1_regressions"]}</td></tr>'
        )
    gates = []
    for arm in summary["candidate_arms"]:
        gate = summary["target_gate"]["candidate_gates"][arm]
        gates.append(
            f'<li><strong>{html.escape(LABELS[arm])}</strong>: '
            f'Original-preservation point gate: {gate["point_target_passed"]}; '
            f'simultaneous numerical gate: {gate["statistical_target_passed"]}; '
            f'EM/F1 strictly above all compared methods: '
            f'{gate["quality_strictly_above_every_compared_method"]}; '
            f'tokens below AutoSaddler: {gate["tokens_strictly_below_autosaddler"]}.</li>'
        )
    accounting = summary["execution_accounting"]
    blocked = summary["light_readiness_blocker"]
    block_notice = (
        f'<p>Light-rule compression was not run: its frozen input-only saving was '
        f'{100 * blocked["observed_light_prompt_saving"]:.3f}%, below the unchanged 10% '
        f'readiness threshold. No light-model outcomes were collected. Only the independently '
        f'authorized source-field branch proceeds on another new cohort.</p>'
        if blocked is not None else ""
    )
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Original CHISEL preservation: fresh inference</title>
<style>
body{{font:16px system-ui,sans-serif;line-height:1.6;color:#172238;background:#f4f6fb;margin:0}}
main{{max-width:1450px;margin:auto;padding:32px}}h1{{line-height:1.2}}
section{{background:white;padding:20px;border-radius:12px;margin:20px 0}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:10px;text-align:right;border-bottom:1px solid #dbe1ec}}th:first-child{{text-align:left}}
nav a{{margin-right:20px}}.boundary{{border-left:5px solid #b66a00}}code{{overflow-wrap:anywhere}}
</style><main>
<h1>Original CHISEL: can we save tokens without losing quality?</h1>
<nav><a href="comparison.csv">All methods CSV</a><a href="data.json">Verified summary</a>
<a href="../{html.escape(audit_name)}/VERIFIED.json">Raw replay audit</a></nav>
<section class="boundary"><strong>PARTIAL: bounded training-source evidence, not a full benchmark claim.</strong>
<p>{html.escape(summary["cohort_boundary"])}</p>
<p>The source-field branch and light rules are tested separately. Original head, full source context,
model revision, and model output cap remain fixed. No combined policy or production default changed.</p>
{block_notice}</section>
<section><h2>Frozen acceptance gates</h2><ul>{"".join(gates)}</ul>
<p>Source fields require strict total-token savings; light rules require at least 10%.
Both require EM and F1 at least equal to Original. Quality ties are not superiority.
A non-significant difference is not proof of noninferiority, and aggregate ties can hide regressions.</p></section>
<section><h2>All {len(summary["rows"])} methods</h2><div class="scroll"><table><thead><tr>
<th>Method</th><th>Successes</th><th>Official EM</th><th>Official F1</th>
<th>Input tokens</th><th>Output tokens</th><th>Total tokens</th><th>Saved vs Original</th>
<th>EM losses</th><th>EM gains</th><th>F1 losses</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>
<p>Loss/gain counts are paired against Original on identical tasks, not changes in net success count.
All tasks and grouped duplicates are retained.</p></section>
<section><h2>Execution identity and cost</h2>
<p>{accounting["logical_cells"]:,} logical task/method cells:
{accounting["real_provider_requests"]:,} fresh provider requests and
{accounting["deterministic_source_outputs"]:,} recomputed native source-field outputs.
Reused model responses: {accounting["reused_model_responses"]}.</p>
<p>Native outputs are not fabricated provider completions. Their provider-token cost is zero because
no model request occurs; their execution time and original source spans are saved separately.
Method totals are selected-path costs, not the cost of acquiring the entire comparison matrix.</p>
<p>Whole study acquisition: {summary["cost_accounting"]["all_in_acquisition_tokens"]:,} provider tokens,
including the separately counted synthetic service check.</p>
<p>Model: <code>{html.escape(summary["model"]["repository"])}</code>;
revision: <code>{html.escape(summary["model"]["revision"])}</code>;
seed: {summary["model"]["request_seed"]}. Host:
<code>{html.escape(summary["resource"]["selected_hostname"])}</code>,
physical GPU {summary["resource"]["physical_gpu_index"]}.</p></section>
</main></html>"""
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(content)
    manifest = {
        "status": "VERIFIED_SAVED_AGGREGATE_PRESENTATION",
        "benchmark_inputs_opened": False, "model_calls": 0, "evaluator_calls": 0,
        "source_bindings": summary["source_bindings"],
        "files_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("comparison.csv", "data.json", "index.html")
        },
    }
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-name", default="audit-v1")
    args = parser.parse_args()
    print(json.dumps(render(args.campaign, args.output, args.audit_name), indent=2))


if __name__ == "__main__":
    main()
