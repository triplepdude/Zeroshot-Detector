# Reference standard data (experiments branch)

Materials and raw outputs of the text-level reference standard
(`docs/experiments/PLAN.md`, `CODEBOOK.md`). Everything here was produced
**after** the plan and codebook were committed. The sample was drawn once, with
seed 20251128.

| File | Content |
|---|---|
| `strings.csv.gz` | All 28,490 strings of the population (one file per post, 1,027,433 frames): frame count, segmented `objects`, the original `pipeline_label` and `llm_label`, and the deterministic **rule reference** (categories, scores, primary, depiction, and the label under each reading), as committed before annotation. The toy rule was later corrected (DEVIATIONS); `zsd experiments report` recomputes the rule labels, and 14 strings (30 frames) differ from this file. |
| `design_records.csv` | The sample. One record per certainty (`C`) or screened-census (`S`) string, with weight = its frame count; one record per sampled tail (`T`) frame, with weight `N_h / n_h`. `fixed` = not resampled in the bootstrap (C, S, census strata). |
| `items.csv` | Everything shown to annotators. `kind`: `sample`; `human_consistency` (human-labelled strings outside the sample); `synthetic` (strings absent from the archive, with `expected` rule answers); `retest` (sample strings re-presented under a fresh id; `source_item_id` points to the original). |
| `batches.json` | Per configuration (`A`, `B`, `C`; `D`/`E` = extra passes for the 23 heaviest strings; `A1` = single-item calls): batches of up to 25 items, in the order and with the object order that were shown. |
| `instruction.md` | The exact instruction text (header + codebook v2). The prompt for a batch is this text + `"\n## Items\n\n"` + the batch's items as JSON (indent 1) + `"\n"` (`experiments.materials.prompt_for`). |
| `raw/workflow_*.json` | Raw outputs of the annotation runs: per batch, the configuration, attempts, validation problems, whether the single-item fallback was used, and every answer as returned. |
| `raw/tool_audit.json` | Tool calls found in the annotator transcripts. Only the structured-answer tool is allowed. |
| `MANIFEST.json` | Design parameters, counts and SHA-256 of the materials and the codebook. |

**Annotators.** Configurations A, C and E use the primary Claude model of the
session; B and D use a smaller Claude model tier. Specific model version strings
are deliberately not recorded in this repository. Annotators received only the
prompt (no tools, files or system labels), as verified by the transcript audit.
They are language models applying the authors' codebook, not human coders.
