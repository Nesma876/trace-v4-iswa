# Required manuscript corrections

| Section / line | Original claim | Proposed revision | Evidence |
|---|---|---|---|
| Methods, inference (264–269) | Bootstrap described as 2,000 draws without configuration caveat | State that executed reporting used 2,000 draws despite the frozen JSON retaining an obsolete 1,000-replicate field; cite final provenance. | `INTERNAL_MODELLING.json:22`; both `bootstrap_provenance.json` files. |
| Methods, tests (282–287) | “It passed 102/102 automated tests” | “The synthetic/unit regression suite contained 102 tests and passed in the audited release environment; these tests do not reproduce the clinical experiments.” | Independent discovery of 102; tests are synthetic. |
| Results, Set B transportability (493–523) | Implies unchanged evaluation without integrity qualification | Add the E2 potential-unblinding disclosure: preflight accessed Set B during evaluator development, aborted during loading, and exposed no metrics; scientific adaptation cannot be ruled out as strongly as in a pristine lock. | `outputs/v4_external/analysis/evaluation_integrity.json`; `paper/jbi/SET_B_EXTERNAL_INTEGRITY_ADJUDICATION.md`. |
| Results, density | Reports final density conclusions without correction provenance | State that density reporting was corrected post hoc from immutable sufficient statistics to align point estimate and CI estimands, with no model inference rerun. | `outputs/v4_external/tables/E6_density_paired_ci.csv`. |
| Discussion, RSSM (562) | “plausible model-configuration or optimization concern” | Retain, but add that shared NLL/masking and exclusion of KL at validation were verified; causal mechanism remains unresolved. | shared `masked_training_nll`; five distinct hashes/logs. |
| Code availability (595–596) | Repository URL/access terms are placeholders and independent reproducibility is disclaimed | Insert immutable URL/tag/commit and enumerate excluded artifacts; do not use “fully reproducible” unless checkpoints/sufficient statistics become accessible. | No tags; key artifacts ignored. |
| Figures 1–5 | LaTeX references non-existent PDFs | Generate and commit the exact final PDFs from canonical tables, or change paths to committed final files after visual verification. | Repository-wide exact-name search found none. |
| Data/methods (125) | “fixed before locked-test evaluation” | Qualify which components were genuinely frozen and identify later reporting adjudications/corrections. | Historical and post-correction artifacts coexist. |

No numerical headline correction is required based on the retained canonical tables; values reconcile within published rounding precision.

