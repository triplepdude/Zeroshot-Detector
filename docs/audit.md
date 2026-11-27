# Audit of the original code and data

This document lists the defects found while turning the original collection of
scripts (now in [`legacy/`](../legacy)) into the `zeroshot_detector` package.
For each defect it gives the evidence, the measured effect on the archived
data where one can be measured, and how the new code handles it. Section
numbers such as "REPORT §3" refer to the generated
[`results/REPORT.md`](../results/REPORT.md).

The main script is `legacy/pipeline/Recursive Florence 2 + Bart + Confidence.py`
(abbreviated **main** below; line numbers refer to that file).

Severity: **High** = changes reported results; **Medium** = threatens validity
or reproducibility; **Low** = robustness or hygiene.

## A. Frame-level classification

| # | Defect | Evidence | Effect on the archived data | Handling in the package |
|---|---|---|---|---|
| A1 | **Florence-2 labels glued together.** `batch_decode(..., skip_special_tokens=True)` removes the `<loc_*>` box tokens, which were the only separators between object labels. BART-MNLI therefore scored strings such as `human facemantrousers`. Boxes were discarded. | main:134; archived strings (e.g. `car tiretruck` = *car tire* + *truck*). | **High.** Non-words drive zero-shot decisions: `human facemantrousers` scores 0.85 for *weapons* and was labelled weapons (834 frames). Measured in REPORT §10. | `florence.parse_od` keeps labels and boxes; the corrected classifier joins labels with `", "`. `legacy=True` reproduces the original string for comparison. |
| A2 | **Keyword rule is a case-sensitive substring test.** `any(word in text.lower() for word in military_keywords)` with capitalised keywords. | main:104–107, 139. | **High.** 4 of 18 keywords (`APC`, `MRAP`, `Stryker`, `Bradley`) can never match. Only `tank` ever fired, and it produced **all** 52,060 `ground_vehicles` frames; BART-MNLI never assigned that class. Substring hits include `tank top` (clothing, 28 frames). | `rules.KeywordRule`: case-insensitive, whole-word, per object label, with an exclusion list (`tank top`, `fish tank`, …). `rules.legacy_keyword_match` reproduces the original. |
| A3 | Keyword categories are inconsistent: `artillery` and `howitzer` force `ground_vehicles` rather than `weapons`. | main:104–107. | None in the archive (these keywords never fired), but a taxonomy error. | Keywords and their target category are configurable; the default is unchanged for comparability. |
| A4 | Candidate labels are passed to the NLI model as identifiers (`ground_vehicles`, `nonmilitary`) inside the default hypothesis `This example is {}.`. "Ground vehicles" does not say *military*. | main:37, 143. | Contributes to the main pipeline/LLM disagreement: civilian `truck`/`car` frames (32,545) are `ground_vehicles` for the LLM and non-military for the pipeline. | `classifier.verbalizers` and `hypothesis_template` are configurable and recorded in the run manifest. |
| A5 | The 0.8 confidence threshold was not chosen on validation data. | main:144. | The threshold sensitivity analysis (REPORT §9) shows how the military share and agreement change with it. | Configurable; all scores are stored per frame so thresholds can be changed after the run. |
| A6 | Inconsistent counters: frames whose top label is `nonmilitary` with score ≥ 0.8 are not counted in `nonmilitary_count`, but below-threshold frames, invalid outputs and exceptions are. | main:141–163. | Only affects A9. | A single per-frame decision with an explicit `source` (`keyword`, `zero_shot`, `below_threshold`, `empty`). |

## B. Video-level aggregation

| # | Defect | Evidence | Effect | Handling |
|---|---|---|---|---|
| B1 | **Videos without decodable frames are labelled `weapons`.** `label_counts` is pre-filled with zeros, so `if label_counts` is always true and `max` over all-zero counts returns the first label. The video is then copied into the `weapons` folder. | main:172. | **High** for video counts: the 16 files without frames would be counted as weapons videos (REPORT §8). | `plurality` and `military_fraction` return *undetermined* (`None`) for empty videos. |
| B2 | **Ties go to military labels.** `max` breaks ties by label order (weapons → aircraft → maritime → ground_vehicles → nonmilitary). | main:172. | 193 videos tied between a military label and `nonmilitary`, and all 193 were labelled military. | `plurality` breaks ties towards `nonmilitary` and reports `tie=True`. |
| B3 | **The "more than half non-military" override almost never fires.** It compares below-threshold *sampled* frames with half of *all decoded* frames, about 30 times more. | main:174 (`frame_count` counts every decoded frame). | It could have fired in at most 67 files (those with one or two samples). | `legacy` reproduces it exactly, using decoded-frame counts in the pipeline. The corrected rules do not depend on it. |
| B4 | The video label depends heavily on the aggregation rule, which is not discussed. | — | The share of military videos ranges from 14.5% (plurality) to 35.1% (any military frame) on the same frame labels (REPORT §8). | Three documented rules; the analysis reports all of them. |

## C. Data collection and processing

| # | Defect | Evidence | Effect | Handling |
|---|---|---|---|---|
| C1 | **The committed script does not run**: `MINIO_ENDPOINT = #INPUT STRING HERE.` is a syntax error and `BUCKET_NAMES` is empty. | main:25–29. | Not reproducible as committed. | YAML configuration (`configs/`), validated on load. |
| C2 | **Duplicate posts processed twice.** The same post id appears in an unprefixed file and in a dated batch. | `data/frames.csv.gz`, REPORT §1. | 2,266 posts (72,518 frames, 6.6%) are exact duplicates and are double-counted in every frame and video statistic. | The analysis clusters by post; `annotate sample` removes duplicates; the duplicate rate is reported. |
| C3 | Models and libraries are not pinned. Microsoft's remote Florence-2 code fails with current `transformers` (SDPA detection and KV-cache errors). | main:45–56; reproduced in this repository. | The original environment cannot be recreated from the repository. | Model revisions are configurable and recorded in each manifest. `attn_implementation="eager"` and `use_cache=False` are used for the remote code; a `native` backend uses `transformers`' built-in Florence-2. `requirements-lock.txt` pins the environment used for re-derivation. |
| C4 | `CUDA_LAUNCH_BLOCKING=1` is set globally (a debugging flag that serialises GPU work); fp32 only; one frame per forward pass. | main:18, 44. | Slow. | Batched inference; dtype configurable. |
| C5 | Each video is read fully into memory and also written to a temporary file. The temporary file leaks when an exception occurs because `os.remove` is only on the success path, and its suffix is always `.mp4`. | main:86–91, 169. | Disk leak on failures. | `sources.MinioSource.open` streams to a temporary file that is always removed; frames are processed in chunks. |
| C6 | Resume state is `master_roster.txt`, re-read for every video (O(n²)). Failures are logged only to `processing.log` and silently retried next time. | main:61–81. | — | `videos.csv` records status and error per video. A run refuses to resume with a different configuration (config fingerprint). |
| C7 | The connection uses no TLS and no credentials (`MINIO_SECURE = False`). | main:26. | Data in transit is unencrypted. | `secure: true` by default; credentials come from environment variables, never from config files. |
| C8 | Sampling is by frame index (every 30th decoded frame), not time, so the sampling rate depends on each video's frame rate. The de-duplication threshold (L1 distance 5000 over a whole frame) only removes near-identical frames. | main:100, 119–121. | 3,023 samples (0.27%) were skipped. | `sampling.every_seconds` is available; the threshold is configurable and dropped samples are counted. |

## D. Validation and human review

| # | Defect | Evidence | Effect | Handling |
|---|---|---|---|---|
| D1 | **The LLM labelling step is not in the repository.** There is no code, model name, prompt, temperature or date for `predicted_category`. | `text_with_openAI_Validation.csv` has no producing script. | **Medium.** The second annotator cannot be reproduced or characterised. Its labels are treated as *agreement*, not ground truth. | Documented in `data/README.md` and the report caveats. |
| D2 | **The LLM saw the same Florence-2 text as the pipeline**, not the image. | Same column of strings. | The two "annotators" are not independent; errors made by Florence-2 are shared by both. | Agreement is reported as agreement. The annotation protocol (`zsd annotate`) labels images. |
| D3 | **Disagreements were mostly resolved by substring rules that always took the LLM's side.** Rules such as `truck → ground_vehicles` and `boat → maritime` were applied before human review. | `legacy/post_processing/prioritize_and_automate.py:13`; REPORT §5. | 43,418 of 56,858 disagreement frames (76%) were resolved automatically, all of them to the LLM label. The rules label every truck, boat or plane as military, and `jet ski` as aircraft. | Reported (REPORT §5); not used as ground truth. |
| D4 | **Human review covered a non-random subset.** Only frames the rules did not match were reviewed: exactly the cells *pipeline = weapons/maritime/aircraft, LLM = nonmilitary*. | `flask_script.py:15`; REPORT §6. | Human labels cannot estimate overall accuracy, recall, or any rate in other cells. | Reported per cell with that scope; `zsd annotate` implements a stratified random design with known inclusion probabilities. |
| D5 | **Reviewers saw both system labels** ("Original Label", "Predicted Category") and only the text, not the frame. | `templates/review_updated.html:20–21`. | Anchoring bias; the judged quantity is "what does this string mean", not "what is in the frame". | The new review app is blind to system labels and shows the frame image. |
| D6 | No annotator id, timestamp or repeat labelling. Decisions are keyed by text and copied to every matching row. | `flask_script.py:22–27`. | No inter-annotator agreement is possible. Label noise is visible: `paddle` → aircraft (260 frames, and labelled both *aircraft* and *maritime* within one session), `footwearpersonweapon` → ground_vehicles. | The new app stores `(sample_id, annotator, label, seconds, time)`; `annotate agreement` computes Cohen's and Fleiss' κ. |
| D7 | Security: the local app runs with `debug=True`, which exposes the Werkzeug debugger and allows code execution if reachable. The Docker variant binds `0.0.0.0:80`. | `flask_script.py:38`, `docker/flask_script.py:38`. | — | `zsd review` binds `127.0.0.1` and never runs in debug mode. |
| D8 | The Dockerfile cannot be built: it references a missing `requirements.txt` and a missing `flask_updated_mismatch_reviewer.py`. | `docker/Dockerfile:11,17`. | — | Superseded by the package (`pip install .[review]`). |
| D9 | Hand-edited copies of the data were committed as notebook checkpoints: one with a typo label `nonmilita222ry`, one with 5 silently changed labels, and one 28 MB truncated file. | `post_processing/.ipynb_checkpoints/`. | Risk of analysing the wrong file. | One canonical table with checksums. The original files are regenerated byte-identically (`zsd data verify`) and remain in git history. |
| D10 | `random_file_picker.py` draws an unseeded random frame per video and burns the Florence-2 text into the image. | `legacy/random_file_picker.py`. | Not reproducible, and the burned-in label would bias any evaluation that used the images. | `zsd annotate sample` is seeded and blind. |

## What could not be checked

* The paper linked from the README (hdl.handle.net/20.500.14216/1718) could not
  be retrieved from this environment, so this audit is based on the code and
  data only.
* No videos are in the repository, so Florence-2's output on the original
  frames cannot be re-derived. Everything downstream of the Florence-2 strings
  can be, and was (REPORT §2).
