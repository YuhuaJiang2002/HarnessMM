# README figures

Run `python docs/assets/readme/generate.py` from the repository root with
Matplotlib 3.11.2 and Pillow 13.0.0 installed. Each figure is exported as SVG
for the GitHub README and PNG for inspection. All figures have an opaque white
background and use the same navy, slate and teal palette.

`results.json` preserves the published rounded values already in the repository
README, with source sections and cost scope. The main comparison and ablation
are separate experiments. No new experiments or uncertainty estimates were
introduced; all quantitative bar axes begin at zero. The original data tables
and scope statements remain next to the images in the main README.

`overview` depicts EDG authorship, GDR receipt checking, CCS native execution,
the receipt loop, return to the slow actor, and the separate train/validation/test
lifecycle. Arrows describe control flow and do not encode measured rates.

## Visualization workflow reference

The figure workflow used the scientific-visualization skill from Scientific
Agent Skills: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M.
(2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents](https://doi.org/10.48550/arXiv.2609.00065).
