# README figures

`paper-figure-1.png` is the original Figure 1 from the HarnessMM paper,
copied without modification from `figures/harness_rsi_overview_user.png`
in the manuscript sources. It appears in the README’s “How it works” section.

Each frame of the homepage animation shows the execution tokens per task before
and after adding HarnessMM to a coding harness. The three comparisons use the
published savings in `results.json` and share a bar scale. The reduced-motion
PNG shows all three comparisons together. Rebuild both with
`python docs/assets/readme/generate_teaser.py` using Pillow and Arial or DejaVu Sans.

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
