#!/usr/bin/env python3
"""Render verified saved aggregates only; never load tasks, models or a scorer."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path


FIELDS = (
    "tasks", "exact_match", "f1", "prompt_tokens", "completion_tokens", "total_tokens",
)
VERIFIED_STATUSES = frozenset({
    "VERIFIED", "VERIFIED_HARNESS_MIXTURE", "INDEPENDENT_HARNESS_MIXTURE_REPLAY_VERIFIED",
})
LABELS = {
    "no_skill": "No Skill",
    "skillopt": "SkillOpt / frozen artifact",
    "autosaddler": "AutoSaddler / frozen artifact",
    "delete": "DELETE / full context",
    "original_chisel": "Original CHISEL",
    "original_ours": "Original Ours",
    "ours_original": "Original Ours",
    "full_mixture": "MoH + MoS / full mixture",
    "without_harness_router": "Fixed micro harness + routed skills",
    "without_skill_router": "Routed harness + all skills",
    "without_both_routers": "Fixed micro harness + all skills",
    "random_mixture": "Count-matched random mixture",
    "fixed_micro_no_skills": "Fixed micro harness / no optional skills",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_table(
    methods: dict[str, dict], *, candidate: str, reference: str, token_ratio_target: float,
) -> tuple[list[dict], dict]:
    if not methods or candidate == reference or candidate not in methods or reference not in methods:
        raise ValueError("distinct measured candidate and reference methods are required")
    if (
        type(token_ratio_target) not in (int, float)
        or not math.isfinite(token_ratio_target) or not 0 < token_ratio_target <= 1
    ):
        raise ValueError("the target must be a finite fraction in (0, 1]")
    counts = set()
    for method in methods.values():
        for field in ("tasks", "prompt_tokens", "completion_tokens", "total_tokens"):
            if type(method[field]) is not int or method[field] < 0:
                raise ValueError("integer nonnegative task/token counts are required")
        if method["tasks"] == 0 or method["prompt_tokens"] == 0:
            raise ValueError("complete positive task and prompt counts are required")
        if method["total_tokens"] != method["prompt_tokens"] + method["completion_tokens"]:
            raise ValueError("prompt and completion token accounting is not additive")
        for field in ("exact_match", "f1"):
            if (
                type(method[field]) not in (int, float)
                or not math.isfinite(method[field]) or not 0 <= method[field] <= 100
            ):
                raise ValueError("finite official percentage metrics are required")
        counts.add(method["tasks"])
    if len(counts) != 1:
        raise ValueError("all rows must cover the same complete paired cohort")
    denominator = methods[reference]["total_tokens"]
    rows = [
        {"arm": arm, **{key: method[key] for key in FIELDS},
         "total_token_ratio": method["total_tokens"] / denominator}
        for arm, method in methods.items()
    ]
    current = methods[candidate]
    quality = all(current[key] >= methods[reference][key] for key in ("exact_match", "f1"))
    ratio = current["total_tokens"] / denominator
    threshold = Fraction(str(token_ratio_target))
    within_budget = current["total_tokens"] * threshold.denominator <= denominator * threshold.numerator
    return rows, {
        "candidate": candidate, "reference": reference,
        "total_token_ratio": ratio, "token_ratio_target": token_ratio_target,
        "observed_em_difference_pp": current["exact_match"] - methods[reference]["exact_match"],
        "observed_f1_difference_pp": current["f1"] - methods[reference]["f1"],
        "observed_quality_nonregression": quality,
        "point_target_passed": quality and within_budget,
        "statistical_guarantee_inferred_from_point_estimates": False,
        "paper_admitted": False,
    }


def render(
    campaign: Path, output: Path, *, candidate: str, reference: str,
    token_ratio_target: float = 0.1, audit_name: str = "audit-v1",
) -> None:
    if Path(audit_name).name != audit_name or audit_name in ("", ".", ".."):
        raise ValueError("audit name must be one campaign-relative directory")
    result_path = campaign / "experiment/RESULT.json"
    config_path = campaign / "experiment/config.json"
    verified_path = campaign / audit_name / "VERIFIED.json"
    result = json.loads(result_path.read_text())
    config = json.loads(config_path.read_text())
    verified = json.loads(verified_path.read_text())
    if (
        verified.get("status") not in VERIFIED_STATUSES
        or verified["result_sha256"] != sha(result_path)
        or verified["config_sha256"] != sha(config_path)
        or verified["paper_admitted"] is not False
        or result["paper_admitted"] is not False
        or config["primary_candidate"] != candidate
        or config["primary_reference"] != reference
        or config["mixture_router"]["token_ratio_target"] != token_ratio_target
        or len(config["arms"]) != len(set(config["arms"]))
        or set(config["arms"]) != set(result["methods"])
        or set(verified["methods"]) != set(result["methods"])
    ):
        raise ValueError("the complete independently verified development result is required")
    for arm, method in result["methods"].items():
        if any(method[field] != verified["methods"][arm][field] for field in FIELDS):
            raise ValueError(f"verified aggregate differs: {arm}")
    calls = result["calls"]
    if (
        any(type(calls[key]) is not int or calls[key] < 0 for key in (
            "unknown_cost_requests", "pending_http", "known_real_tokens", "synthetic_tokens",
        ))
        or calls["unknown_cost_requests"] != 0 or calls["pending_http"] != 0
    ):
        raise ValueError("unknown costs or pending calls cannot enter a completed table")
    rows, target = aggregate_table(
        result["methods"], candidate=candidate, reference=reference,
        token_ratio_target=token_ratio_target,
    )
    gate = verified["target_gate"]
    if (
        gate["primary_baseline_status"] != "BOUND"
        or gate["primary_reference"] != reference
        or gate["point_target_passed"] != target["point_target_passed"]
        or gate["observed_total_token_ratio"] != target["total_token_ratio"]
    ):
        raise ValueError("the reported target must use the independently verified primary reference")
    routing = verified["routing_replay"]["summary"]["methods"][candidate]
    route_counts = routing["selected_harnesses"]
    if (
        not route_counts
        or any(not isinstance(name, str) or type(count) is not int or count < 0
               for name, count in route_counts.items())
        or sum(route_counts.values()) != result["methods"][candidate]["tasks"]
        or routing["incumbent_fallbacks"] != route_counts.get("incumbent", 0)
    ):
        raise ValueError("the complete verified route histogram is required")
    routes_text = " / ".join(f"{html.escape(name)}: {count}"
                             for name, count in sorted(route_counts.items()))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "data.json").write_text(json.dumps({
        "rows": rows, "target": target, "calls": calls, "primary_route_counts": route_counts,
        "scope": "Paired training-source development only; not complete public confirmation.",
    }, indent=2, sort_keys=True) + "\n")
    largest = max(row["total_token_ratio"] for row in rows)
    body = []
    for row in rows:
        name = html.escape(LABELS.get(row["arm"], row["arm"]))
        badge = "PRIMARY" if row["arm"] == candidate else "REFERENCE" if row["arm"] == reference else ""
        body.append(
            f'<tr class="{"primary" if row["arm"] == candidate else ""}">'
            f'<th>{name}<small>{badge}</small></th><td>{row["tasks"]:,}</td>'
            f'<td>{row["exact_match"]:.4f}</td><td>{row["f1"]:.4f}</td>'
            f'<td>{row["prompt_tokens"]:,}</td><td>{row["completion_tokens"]:,}</td>'
            f'<td>{row["total_tokens"]:,}</td><td>{100 * row["total_token_ratio"]:.2f}%'
            f'<span class="bar"><i style="width:{100 * row["total_token_ratio"] / largest:.3f}%">'
            '</i></span></td></tr>'
        )
    headline = (
        "Point-estimate target met; full-benchmark no-loss claim not established."
        if target["point_target_passed"]
        else f"The {100 * token_ratio_target:g}% token / no-loss target is not met."
    )
    escaped_reference = html.escape(LABELS.get(reference, reference))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harness router: mixture of harnesses and skills</title>
<style>
:root{{color-scheme:light;font-family:system-ui,sans-serif;color:#15283b;background:#f4f7fb}}
body{{margin:0}}main{{max-width:1280px;margin:auto;padding:32px 24px}}
.eyebrow{{font-size:12px;letter-spacing:.18em;color:#496880}}h1{{font-size:34px;margin-bottom:12px}}
p{{line-height:1.65}}a{{color:#185d93}}nav{{display:flex;gap:24px}}
.banner{{padding:18px 22px;margin:24px 0;border-left:5px solid #c38123;background:#fff6e5}}
.flow,.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}
.card{{padding:18px;background:white;border:1px solid #dce4eb;border-radius:8px}}
.card strong{{display:block;font-size:19px;margin:6px 0}}.card p{{font-size:13px;margin:0}}
.table{{overflow-x:auto;background:white;border:1px solid #dce4eb;border-radius:8px}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}
th,td{{padding:13px 12px;text-align:right;border-bottom:1px solid #e5ebf0}}
th:first-child{{text-align:left}}thead{{font-size:12px;color:#496880;background:#edf3f7}}
small{{display:block;font-size:10px;letter-spacing:.1em;color:#1e6a85}}
.primary{{background:#edf7fb}}.bar{{display:block;width:120px;height:5px;background:#e3eaf0;margin-top:7px}}
.bar i{{display:block;height:100%;background:#3c85a3}}footer{{font-size:12px;color:#536d7e;margin-top:22px}}
@media(max-width:760px){{.flow,.metrics{{grid-template-columns:1fr 1fr}}h1{{font-size:26px}}}}
</style></head><body><main>
<div class="eyebrow">PHIAGENT / PROVISIONAL HIERARCHICAL ROUTING</div>
<h1>Mixture of harnesses &times; mixture of skills</h1>
<nav><a href="comparison.csv">All measured rows / CSV</a><a href="data.json">Values and scope</a>
<a href="manifest.json">Source hashes</a></nav>
<div class="banner"><strong>{headline}</strong><br>
Target: all selected-path prompt + completion tokens &le; {100 * token_ratio_target:.2f}% of
{escaped_reference}, without reducing official EM or F1. No task failures are removed from cost.</div>
<section class="flow">
<article class="card"><small>01 / ORIGINAL INPUT</small><strong>Question + source</strong>
<p>No answers, test labels or model predictions enter the router.</p></article>
<article class="card"><small>02 / MIXTURE OF SKILLS</small><strong>Sparse whole clauses</strong>
<p>Input tags activate complete source-derived skills. Dependencies stay intact.</p></article>
<article class="card"><small>03 / HARNESS ROUTER</small><strong>Minimum eligible budget</strong>
<p>Exact-token source views and output caps; uncertain inputs keep the original incumbent.</p></article>
<article class="card"><small>04 / FROZEN ACTOR</small><strong>Official EM + F1</strong>
<p>All matched arms, actual token receipts, source-span provenance and independent replay.</p></article>
</section>
<p>Frozen primary routes: <strong>{routes_text}</strong>.
Original-incumbent fallbacks: <strong>{routing["incumbent_fallbacks"]}</strong>.
These counts describe routing, not answer correctness.</p>
<section class="metrics">
<div class="card"><small>TOKENS REMAINING / REFERENCE</small>
<strong>{100 * target["total_token_ratio"]:.2f}%</strong><p>Goal &le; {100 * token_ratio_target:.2f}%</p></div>
<div class="card"><small>OBSERVED EM DIFFERENCE</small>
<strong>{target["observed_em_difference_pp"]:+.4f} pp</strong><p>Primary versus reference</p></div>
<div class="card"><small>OBSERVED F1 DIFFERENCE</small>
<strong>{target["observed_f1_difference_pp"]:+.4f} pp</strong><p>Primary versus reference</p></div>
<div class="card"><small>EVIDENCE BOUNDARY</small>
<strong>Development only</strong><p>No innovation or paper admission</p></div>
</section>
<h2>Complete measured comparison</h2>
<div class="table"><table><thead><tr><th>Method / ablation</th><th>Tasks</th><th>EM (%)</th>
<th>F1 (%)</th><th>Prompt</th><th>Completion</th><th>Total tokens</th><th>% of reference</th>
</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p>Observed ties are not a statistical proof of zero loss. Source-lexeme retention is a
routing heuristic, not a semantic support certificate. Baseline artifact identities are
not complete optimizer reproductions. This training-source cohort is not the full public benchmark.</p>
<footer>All values come from completed, independently verified saved aggregates. Rendering
makes no model or evaluator calls. New study acquisition across all arms:
{calls["known_real_tokens"]:,} real
provider tokens; synthetic admission: {calls["synthetic_tokens"]:,} tokens.
Historical training/optimizer costs of reused artifacts are outside these totals;
this is not an all-lifecycle cost claim. Unknown retry costs are never treated as zero.</footer>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "result_sha256": sha(result_path), "config_sha256": sha(config_path),
        "verification_sha256": sha(verified_path), "renderer_sha256": sha(Path(__file__)),
        "reference_arm": reference, "candidate_arm": candidate,
        "model_calls": 0, "evaluator_calls": 0, "benchmark_inputs_opened": False,
        "files_sha256": {name: sha(output / name) for name in ("comparison.csv", "data.json", "index.html")},
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", default="full_mixture")
    parser.add_argument("--audit-name", default="audit-v1")
    args = parser.parse_args()
    render(args.campaign, args.output, candidate=args.candidate, reference=args.reference,
           audit_name=args.audit_name)


if __name__ == "__main__":
    main()
