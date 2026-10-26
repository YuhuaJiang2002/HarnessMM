#!/usr/bin/env python3
"""Present completed, independently verified QA aggregates; no benchmark execution."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from phiagent.evaluation.dual_quality_cost import VerifiedAggregate, benchmark_dual_gate
from scripts.render_qa_harness_mixture import FIELDS, aggregate_table, sha


LABELS = {
    "no_skill": "No Skill", "skillopt": "SkillOpt artifact", "autosaddler": "AutoSaddler artifact",
    "delete": "DELETE", "original_chisel": "Current Ours / Original CHISEL",
    "capability_routed": "Mandatory core + routed optional skills",
    "core_only": "Mandatory core only", "routed_without_core": "Routed optional skills only",
    "answer_only": "No Skill / answer-only control",
    "all_capabilities": "All core and optional skills",
    "count_matched_random": "Core + count-matched random skills",
    "legacy_nine_all": "Previous nine skills / full evidence",
}


def checked_values(campaign: Path, audit_name: str) -> tuple[dict, dict, list[dict], dict]:
    if Path(audit_name).name != audit_name or audit_name in {"", ".", ".."}:
        raise ValueError("a single relative audit directory is required")
    result_path, config_path = campaign / "experiment/RESULT.json", campaign / "experiment/config.json"
    verified_path = campaign / audit_name / "VERIFIED.json"
    result, config, verified = (
        json.loads(path.read_text()) for path in (result_path, config_path, verified_path)
    )
    if (
        config["schema_version"] not in {
            "phiagent-qa-capability-cost/v2", "phiagent-qa-capability-core/v3",
        }
        or verified["status"] != "INDEPENDENT_HARNESS_MIXTURE_REPLAY_VERIFIED"
        or verified["result_sha256"] != sha(result_path)
        or verified["config_sha256"] != sha(config_path)
        or verified["paper_admitted"] is not False or result["paper_admitted"] is not False
        or verified["methods"] != result["methods"]
        or set(result["methods"]) != set(config["arms"])
        or len(config["arms"]) != len(set(config["arms"]))
        or verified["target_gate"] != result["target_gate"]
        or verified["routing_replay"]["all_original_evidence_retained"] is not True
        or verified["routing_replay"]["all_original_output_caps_retained"] is not True
    ):
        raise ValueError("complete verified full-evidence QA evidence is required")
    calls = result["calls"]
    if (
        calls["unknown_cost_requests"] != 0 or calls["pending_http"] != 0
        or result["cost_accounting"]["router_provider_tokens"] != 0
        or calls["known_real_tokens"] != sum(method["total_tokens"] for method in result["methods"].values())
    ):
        raise ValueError("new study acquisition and zero-router-overhead accounting differ")
    ordered = {arm: result["methods"][arm] for arm in config["arms"]}
    candidate, reference = config["primary_candidate"], config["cost_reference"]
    rows, _ = aggregate_table(ordered, candidate=candidate, reference=reference, token_ratio_target=1)
    runs = [
        VerifiedAggregate(
            method=arm, benchmark="searchqa", cohort_sha256=result["data_manifest_sha256"],
            matched_protocol_sha256=sha(config_path), verification_sha256=sha(verified_path),
            tasks=value["tasks"], metrics={metric: value[metric] for metric in ("exact_match", "f1")},
            prompt_tokens=value["prompt_tokens"], completion_tokens=value["completion_tokens"],
            overhead_tokens=0, unknown_cost_requests=0,
        )
        for arm, value in ordered.items()
    ]
    gate = benchmark_dual_gate(
        runs, candidate=candidate, incumbent=config["primary_reference"], autosaddler=reference,
        metric_directions={"exact_match": "higher", "f1": "higher"},
        expected_quality_references=[arm for arm in config["arms"] if arm != candidate],
        full_official_protocol=False, independent_confirmation=False,
    )
    if (
        gate["observed_point_gate_passed"] != verified["target_gate"]["point_target_passed"]
        or gate["autosaddler_token_ratio"] != verified["target_gate"]["observed_total_token_ratio"]
    ):
        raise ValueError("presentation gate differs from the independent verifier")
    return config, verified, rows, gate


def render(campaign: Path, output: Path, *, audit_name: str = "audit-v1") -> None:
    config, verified, rows, gate = checked_values(campaign, audit_name)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm", *FIELDS, "total_token_ratio"))
        writer.writeheader()
        writer.writerows(rows)
    (output / "data.json").write_text(json.dumps({
        "rows": rows, "gate": gate, "independent_target_gate": verified["target_gate"],
        "scope": "Whole-group training-source development or replication, not full public evaluation",
    }, indent=2, sort_keys=True) + "\n")
    rendered = []
    for row in rows:
        label = html.escape(LABELS.get(row["arm"], row["arm"]))
        primary = row["arm"] == config["primary_candidate"]
        rendered.append(
            f'<tr class="{"primary" if primary else ""}"><th>{label}</th>'
            f'<td>{row["tasks"]}</td><td>{row["exact_match"]:.4f}</td><td>{row["f1"]:.4f}</td>'
            f'<td>{row["prompt_tokens"]:,}</td><td>{row["completion_tokens"]:,}</td>'
            f'<td>{row["total_tokens"]:,}</td><td>{100 * row["total_token_ratio"]:.2f}%</td></tr>'
        )
    headline = ("Observed joint point gate passes; confirmation is not established."
                if gate["observed_point_gate_passed"] else "The complete quality-up / token-down gate does not pass.")
    replication = config["schema_version"] == "phiagent-qa-capability-core/v3"
    title = ("Frozen mandatory core / disjoint replication"
             if replication else "Preserve evidence. Keep mandatory skills. Route only optional rules.")
    optional_description = (
        "The primary uses no optional skills. Routing, all-skill and matched random alternatives remain controls."
        if replication else
        "Up to two input-tagged rules; random controls match each task's optional count. No routing LLM calls."
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-evidence capability routing / verified quality-cost experiment</title>
<style>
body{{margin:0;background:#f4f7fb;color:#162c40;font:16px system-ui,sans-serif}}
main{{max-width:1220px;margin:auto;padding:32px 22px}}h1{{font-size:32px}}
p{{line-height:1.65}}.notice{{padding:20px;border-left:5px solid #b47a22;background:#fff5de}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}
.card{{background:white;padding:20px;border:1px solid #d8e3ec;border-radius:8px}}
.table{{overflow-x:auto;background:white}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}
th,td{{padding:12px;border-bottom:1px solid #e0e7ee;text-align:right}}th:first-child{{text-align:left}}
thead{{background:#eaf1f7;font-size:12px}}.primary{{background:#e8f5fc}}
a{{color:#165e91}}footer{{font-size:12px;line-height:1.6;color:#516578;margin-top:22px}}
@media(max-width:760px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p>PHIAGENT / SOURCE-PRESERVING CAPABILITY ROUTING</p>
<h1>{title}</h1>
<p><a href="comparison.csv">Complete CSV</a> &middot; <a href="data.json">Measured values and gates</a>
&middot; <a href="manifest.json">Evidence hashes</a></p>
<div class="notice"><strong>{headline}</strong><br>
Quality must be strictly higher than every displayed control, including current Ours.
Total input + output tokens must be strictly lower than AutoSaddler on the identical cohort.</div>
<div class="cards"><div class="card"><strong>All original evidence retained</strong>
<p>No 64-token context truncation. The full question/context and identical512-token output caps are kept.</p></div>
<div class="card"><strong>Mandatory capability skeleton</strong>
<p>Requested type, joint evidence, minimal unambiguous form and source wording remain active.</p></div>
<div class="card"><strong>Scoped optional skills</strong>
<p>{optional_description}</p></div></div>
<h2>Complete official-MRQA-scored training comparison</h2>
<div class="table"><table><thead><tr><th>Method</th><th>Tasks</th><th>EM (%)</th><th>F1 (%)</th>
<th>Prompt</th><th>Completion</th><th>Total tokens</th><th>% of AutoSaddler</th></tr></thead>
<tbody>{''.join(rendered)}</tbody></table></div>
<footer>This is a complete matched development matrix on disjoint official-training question groups,
not the full public development benchmark or fresh confirmation. Historical exposure elsewhere is not certified.
All baselines retain their fixed-artifact identities; no complete optimizer reproduction is claimed.
New study acquisition: {verified["real_provider_tokens"]:,} real + {verified["synthetic_provider_tokens"]:,}
synthetic tokens across all arms. Historical optimizer/training costs are not included. Incorrect and
length-stopped outputs remain charged. The failed prior aggressive-compression experiment is not overwritten.</footer>
</main></body></html>"""
    (output / "index.html").write_text(page)
    (output / "manifest.json").write_text(json.dumps({
        "result_sha256": verified["result_sha256"],
        "verification_sha256": sha(campaign / audit_name / "VERIFIED.json"),
        "renderer_sha256": sha(Path(__file__)),
        "files_sha256": {name: sha(output / name) for name in ("comparison.csv", "data.json", "index.html")},
        "benchmark_calls": 0, "model_calls": 0, "evaluator_calls": 0, "paper_admitted": False,
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-name", default="audit-v1")
    args = parser.parse_args()
    render(args.campaign, args.output, audit_name=args.audit_name)


if __name__ == "__main__":
    main()
