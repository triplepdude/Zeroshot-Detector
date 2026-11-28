# Legacy code (original study, unchanged)

These are the scripts of the original study, moved here unchanged (`git mv`
preserves their history) so the published work can be traced. **They are not
maintained and are not meant to be run.** The main script does not parse as
committed; see `docs/audit.md` C1. Use the `zeroshot_detector` package instead:

| Legacy file | Purpose | Replacement |
|---|---|---|
| `pipeline/Recursive Florence 2 + Bart + Confidence.py` | Final pipeline: MinIO → frames → Florence-2 `<OD>` → keyword rule → BART-MNLI (τ = 0.8) → video label; copies videos into category folders | `zsd run --config configs/original_study.yaml` (exact behaviour) or `configs/default.yaml` (corrected) |
| `pipeline/iterations/florence2.py` | First experiment: Florence-2 on one local video, labels drawn on frames | — |
| `pipeline/iterations/Recursive Florence 2 +  bart-large.py` | Earlier pipeline (labels `non_military`/`unknown`, no keyword rule) | — |
| `pipeline/iterations/Recursive Florence 2 + bart-large + confidence thresholds.py` | Earlier batched variant with a 32-process pool (shares one GPU model across processes) | — |
| `post_processing/test.py` | Extracts pipeline/LLM mismatches; command-line labelling | `zsd analyze` (REPORT §4) |
| `post_processing/prioritize_and_automate.py` | Keyword "auto-correction" of mismatches; frequent mismatches | `zeroshot_detector.data.legacy_tables` (regenerated for provenance), REPORT §5 |
| `post_processing/flask_script.py`, `docker/` | Web app for human review of mismatches (shows both system labels) | `zsd review` (blind, image-based) |
| `post_processing/templates/index.html` | Earlier "image feedback" UI (recovered from a notebook checkpoint; no matching script) | — |
| `post_processing/l.py` | Counts human-labelled rows | `zsd analyze` (REPORT §1) |
| `random_file_picker.py` | Extracts up to 100 random annotated frames per category for spot checks | `zsd annotate sample` |
| `rm.sh` | Deletes local outputs | — |

The data files these scripts read and wrote are in [`../data`](../data) (see
its README for the mapping).
