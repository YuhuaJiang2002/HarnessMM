# README figures

`paper-figure-1.png` is the Figure 1 image shared by the HarnessMM and HarnessMM-V papers.
The README uses `harness-overview-domain.png`, a copy with the feedback label
changed from “Visual signals” to “Domain signal”. The manuscript PNG is unchanged.

The ten-second opening animation (`rsi-results.gif`) starts with a 1.2-second project
title card, then shows frames 8–10 of the original demo: three RSI steps
at one second each. The original coding-results table follows for 2.6 seconds,
and the six-benchmark chart closes the loop for 3.2 seconds.
The cover identifies the author affiliations with the [Tsinghua University
emblem](https://commons.wikimedia.org/wiki/File:Tsinghua_University_Logo.svg)
and the [Microsoft symbol](https://commons.wikimedia.org/wiki/File:Microsoft_logo.svg)
beside the Microsoft Research Asia name.
Below Figure 1, `guarded-reuse-overview.gif` shows frames 1–7 followed by
the workflow overview. The compile-and-test sequence comes from the
[synthetic controller example](../../../examples/guarded_reuse.mjs). It shows
how an authored graph, a successful compile receipt and a substitution
certificate replace one model callback. The original static overview
(`guarded-reuse.png`) includes the method and three Qwen3.5-4B comparisons.

Rebuild the 1200 × 600 GIF and PNG with
`python docs/assets/readme/generate_teaser.py` using Pillow 13.0.0 and Arial
or DejaVu Sans.

Run `python docs/assets/readme/generate.py` from the repository root with
Matplotlib 3.11.2 and Pillow 13.0.0 installed. Each figure is exported as SVG
for the GitHub README and PNG for inspection. All figures have an opaque white
background and use the same navy, slate and teal palette.

`results.json` records the three Qwen3.5-4B coding comparisons shown in the
README and all four MBPP-sanitized ablation conditions. It identifies the
November 20, 2026 manuscript panels and the underlying result records.
The ablation entries include exact execution-token totals and certified
callback counts. Quantitative bar axes begin at zero.

`visual-comparison.json` separately records all ten complete MiniWoB++ visual
extension arms, including exact integer task and token totals. The new
`competitor-results` figure highlights HarnessMM-V against HarnessPro, SoL-Pi,
Codex Harness and DeepSeek Harness. Its savings are computed from exact
execution totals before rounding. The [visual protocol](../../VISUAL_COMPARISON.md)
retains the complete comparison, method adaptations and cost scope. These
visual results are separate from the HarnessMM coding paper data in `results.json`.

`overview` depicts EDG authorship, GDR receipt checking, CCS native execution,
the receipt loop, return to the slow actor, and the separate train/validation/test
lifecycle. Arrows show control flow.

## Visualization workflow reference

The figure workflow used the scientific-visualization skill from Scientific
Agent Skills: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M.
(2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents](https://doi.org/10.48550/arXiv.2609.00065).
