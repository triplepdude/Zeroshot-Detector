# Zeroshot-Detector

Zero-shot detection of military content in short-form video. Sampled frames are
described by **Florence-2** object detection, and the descriptions are
classified by **BART-MNLI** zero-shot classification into *weapons*, *aircraft*,
*maritime*, *ground vehicles* and *non-military*. This repository contains:

* the code and archived outputs of an independent study that applied this
  pipeline to 30,958 TikTok videos (paper: <https://hdl.handle.net/20.500.14216/1718>);
* `zeroshot_detector`, a tested Python package that reimplements the original
  pipeline, with its decision logic reproduced exactly and verified against the
  archived labels (`configs/original_study.yaml`), and provides a corrected
  version (`configs/default.yaml`);
* a reproducible re-analysis of the archived data, with uncertainty estimates
  ([`results/REPORT.md`](results/REPORT.md));
* a blind, stratified annotation protocol for the image-level gold-standard
  evaluation the original study lacked;
* a pre-registered experiment that builds a text-level reference standard for
  the detected-object strings and uses it to compare the pipeline, the LLM
  labels and alternative classifiers ([below](#experiments-a-text-level-reference-standard)).

> **Status.** Pipeline labels were never compared with image-level ground
> truth. The archived data support statements about *reproducibility*,
> *agreement* with a second (LLM) annotator, and *sensitivity* to design
> choices, but not about accuracy or true prevalence. See
> [`docs/methodology.md`](docs/methodology.md) §4.

## Key findings from the archived data

All numbers are from [`results/REPORT.md`](results/REPORT.md). Intervals are 95%
cluster-bootstrap intervals that resample posts.

1. **The stored labels are reproducible.** Re-running the original decision rule (keyword rule, then BART-MNLI at threshold 0.8) on all 28,490 distinct Florence-2 strings reproduces the archived pipeline label for every one of them: 100% of 1,099,951 frames, including 111 strings whose score lies within 0.001 of the threshold.
2. **Agreement with the LLM labels is high per frame but moderate per distinct description.**
   Cohen's κ is 0.84 (0.83–0.85) over 1.1 M frames but 0.56 (0.55–0.58) over the
   28,490 distinct Florence-2 strings. The frame-level figure is carried by a few
   very frequent, easy strings: the top 10 cover 36% of frames.
3. **One keyword produced an entire class.** All 52,060 `ground_vehicles` frames
   come from the substring `tank`. BART-MNLI never assigned that class, 17 of the
   18 keywords never fired, and 4 can never match because of a case bug.
4. **Disagreements were mostly resolved in favour of the LLM, by rule.** 76% of
   disagreement frames were "auto-corrected" by substring rules (e.g. any
   `truck` → military ground vehicle), and every one of them received the LLM's
   label. Humans reviewed only the rest: frames the pipeline called
   weapons/maritime/aircraft and the LLM called non-military. Reviewers saw
   the text and both labels, not the image. Over distinct strings, they sided
   with the LLM 55% (51–58%) and with the pipeline 43% (39–47%) of the time.
5. **Estimated precision of military labels, optimistic by construction:**
   weapons 0.82 (0.80–0.84), maritime 0.74 (0.70–0.78). These estimates assume
   that frames where the pipeline and LLM agree are correct.
6. **The video-level conclusion depends more on the aggregation rule than on the
   classifier.** On the same frame labels, 14.5% of videos are military under
   plurality voting and 35.1% under "any military frame". The original rule also
   labels the 16 videos without frames as *weapons* and resolves all 193
   military/non-military ties as military.
7. **6.6% of frames are exact duplicates**: 2,266 posts were collected twice.
8. **Descriptive trend.** The share of videos labelled military rises from
   4% (2019) to 17% (2022), then falls to 14% (2024). Post dates are decoded
   from post ids. The video sampling is undocumented, so this describes the
   archive, not the platform.
9. **The separator bug matters, but fixing it is not enough.** Splitting the glued object labels and correcting the keyword rule changes the label of 10,227 frames (568 strings, among the 95% of frames re-scored). κ with the LLM rises from 0.847 to 0.864, and agreement with the adjudicated human labels from 61.8% to 78.1%. Some changes go the wrong way: `helmet + person + weapon` moves from weapons to non-military. Without the keyword rule, BART-MNLI would call 98% of the `tank` frames non-military. The study threshold (0.8) is close to, but not at, the agreement-maximising one (0.75; REPORT §9).

[`docs/audit.md`](docs/audit.md) lists every defect found in the original code,
with its evidence and measured effect.

## Experiments: a text-level reference standard

Every archived label (pipeline, LLM, human) is a deterministic function of the
Florence-2 object string. The experiments therefore build a documented
**reference label for strings** on a probability sample, which supports
design-weighted estimates for any classifier that reads the string. The plan
(with its analyses and predictions) and the codebook were committed before the
sample was drawn. They were written after the system labels, the earlier
disagreement tables and the archived human labels had been inspected, as the
plan states
([`docs/experiments/PLAN.md`](docs/experiments/PLAN.md),
[`CODEBOOK.md`](docs/experiments/CODEBOOK.md)); every departure from them is
logged in [`DEVIATIONS.md`](docs/experiments/DEVIATIONS.md). Full results:
[`results/experiments/REPORT.md`](results/experiments/REPORT.md).

* **Sample.** 1,108 strings carrying 85% of the population's frames: the 500
  most frequent strings; a census of the 167 rarer strings that both systems
  called non-military although their objects include a weapon or a watercraft;
  and 441 strings from a stratified random sample of the remaining frames
  (seed 20251128).
* **Annotators.** Language-model annotators applied the codebook blind (only
  the object list, no system labels, no tools; verified in every transcript), in
  three configurations (two model tiers, shuffled object order), with extra
  passes for the 23 heaviest strings, test–retest items, single-item calls and
  synthetic strings with known answers.
* **Construct.** Each vehicle category gets a 0–3 military score. Four
  pre-registered readings: *equipment* (any weapon or vehicle category at any
  score; the generic `vehicle` alone does not count), *type* (score ≥ 1),
  *lenient* (primary: at least plausibly military, score ≥ 2) and *strict*
  (score 3: inherently military objects such as tanks, or dual-use vehicles
  with a military cue). Weapons count under every reading.
* **Outcome.** *TLR agreement*: design-weighted agreement with this text-level
  reference. It is agreement with a codebook reading of the object list, **not
  accuracy against the video**. The annotators' own error rate against human
  judgement is not estimated, and their errors may be correlated with those of
  the original LLM labeller, which would favour the LLM labels.

Findings:

1. **The construct decides the result (RQ1).** Applied exactly to all
   1,027,433 frames, the codebook rules give a military share of 45.4%
   (equipment), 21.1% (lenient) or 6.4% (strict). The pipeline agrees with the
   lenient labels on 95.5% of frames and the LLM on 98.8%; under strict the
   ranking flips (88.1% vs 84.2%). Both predictions made before annotation were
   confirmed: the panel reproduces these rule-based values (P1), including the
   flip (P2).
2. **The panel applies the codebook almost mechanically.** Configurations agree
   on 99.9% of strings (Fleiss' κ 0.999); test–retest, single-item and synthetic
   checks agree 100%. The panel departs from the deterministic rule reference
   on 0.01% of frames (one string, `boatsail`). With these annotators the
   reference is, in effect, the codebook: its validity rests on the codebook's
   decisions, which are documented and can be changed.
3. **C1 (confirmatory): the LLM labels agree more with the lenient reference
   than the pipeline**: 98.8% vs 95.5%, a difference of 3.3 pp (95% interval
   3.1 to 3.4). Post-hoc exact analyses show that this rests on one codebook
   decision. 2.9 pp of the difference come from ground-vehicle strings with a
   `truck` and no `tank` (2.9% of frames), which the archive's auto-correction maps
   to military and the codebook scores as plausibly military (tier T2). If trucks
   are instead treated as everyday civilian vehicles (tier T1), the pipeline is
   ahead by 2.5 pp; treating every dual-use label (trucks, boats, aircraft) this
   way reproduces the strict reading. Smaller contributions: the pipeline misses
   most boats (maritime recall 41% over the population) and `handgun`, and the LLM labels call the
   bare string `weapon` (0.3% of frames) non-military.
4. **C2 (confirmatory): a stronger zero-shot model does not help.** The
   pre-registered DeBERTa-v3 NLI set-up ("military …" verbalizers, separated
   labels, keyword rule, arg-max) agrees with the lenient reference 17.2 pp
   *less* than the original pipeline (−18.1 to −16.3), and less than labelling
   everything non-military (78.3% vs 78.9%). Under arg-max it calls 22% of the
   reference's non-military frames military (e.g. `poster`, `vehicle`,
   `carwheel`). In the exploratory grid (2 models × 2 templates × 2 verbalizer
   sets × 2 input formats × thresholds, with and without the keyword rule) the
   best DeBERTa setting is 3.4 pp below the pipeline and the best BART setting
   (separated labels, keyword rule, threshold 0.55) 0.9 pp above it; nested
   selection gives 96.2% out of fold. A logistic regression on the bag of
   object labels reaches 99.9% when trained on the reference strings and 98.8%
   when trained on the archived LLM labels. Given the object list, the label is
   close to a deterministic function of the codebook, and zero-shot NLI is the
   weak link.
5. **RQ5 (exploratory): no evidence of a change around 24 Feb 2022 or
   7 Oct 2023.** Within batch 12, the only collection batch spanning the first
   date (it covers Sep 2021 – Jul 2022), the share of videos the pipeline labels
   military rises from 5% to 27% (LLM labels: 9% to 32%). The batch-adjusted
   step (a GLM with batch fixed effects and a month trend) is matched or
   exceeded in 27–31% of placebo months, and neither date passes the
   pre-registered placebo criterion.
6. **Independent verification.** Separate agents re-derived the headline
   numbers from the raw files without the project's analysis code, in two
   rounds. The first covered the re-analysis (dataset, κ, keyword rule,
   auto-correction, video aggregation, reproduction, human labels) and RQ1, the
   panel, C1/P2 and the batch-12 shares: all eleven claims reproduced (one only
   after correcting the wording of the claim itself). The second covered C2,
   the full RQ4 grid, nested selection and the logistic regression, confirmed
   that `zsd experiments report` reproduces `summary.json` byte for byte, and
   checked the report against every item of the plan. The re-implementation of
   the codebook found one bug in the rule reference (the toy rule; 30 frames
   under the equipment reading, 11 under lenient), and the completeness check
   found reporting gaps; both were fixed and logged, and no reported number
   changed at the precision shown.

```bash
zsd experiments report                              # results/experiments/ from data/reference/ and the stored grid scores (~2 min)
zsd experiments score-grid --grid-scores grid_new/  # re-score the zero-shot grid into an empty directory (about 1.5 h on 4 CPU cores)
```

## Repository layout

```
src/zeroshot_detector/     the package
  florence.py              Florence-2 inference and <OD> output parsing
  zero_shot.py             batched NLI zero-shot scorer (matches the HF pipeline to 2e-6)
  rules.py, classify.py    keyword rule + thresholded zero-shot decision (legacy and corrected)
  aggregate.py             video-level aggregation rules (legacy, plurality, military fraction)
  video.py, sources.py     frame sampling; local directory / MinIO sources
  pipeline.py, config.py   resumable, provenance-tracked runs (zsd run)
  data.py                  canonical data, byte-exact regeneration of the original CSVs
  segment.py               re-segmentation of the glued archived strings
  evaluation/              metrics (κ, F1, Wilson) and cluster bootstrap
  analysis/                the re-analysis behind results/ (zsd analyze)
  annotation.py, review/   gold-standard protocol and blind annotation app
  experiments/             text-level reference: codebook rules, design, adjudication, RQ1–RQ5 (zsd experiments)
configs/                   default.yaml (corrected), original_study.yaml (exact reproduction)
data/                      archived study data + data card + checksums
data/reference/            experiments: sample, annotator prompts and raw answers, manifest
results/                   REPORT.md, summary.json, figures/, relabel/ (re-derived BART scores)
results/experiments/       experiments report, summary.json, figures, zero-shot grid scores
docs/                      methodology.md, audit.md
docs/experiments/          PLAN.md (pre-registered), CODEBOOK.md, DEVIATIONS.md
legacy/                    original scripts, unchanged (not maintained)
tests/                     pytest suite (runs without the deep-learning stack)
```

## Installation

Python ≥ 3.10.

```bash
pip install -e ".[dev]"          # analysis, tests, annotation app (no PyTorch)
make install-pipeline            # + PyTorch (CPU) and transformers, to run models
```

`requirements/lock-cpu.txt` pins the exact environment used to produce `results/`.

## Reproducing the results

```bash
zsd data verify                  # checksums; regenerates the 8 original CSVs byte-identically
zsd analyze                      # results/summary.json, REPORT.md, figures (~1 min)
pytest                           # unit tests + data invariants
pytest -m slow                   # committed results/ are reproducible from data/
```

`zsd analyze` uses the BART-MNLI scores stored in `results/relabel/`. To recompute
them (several hours on CPU):

```bash
zsd relabel --mode legacy                                   # all 28,490 strings, original rule
zsd relabel --mode segmented --coverage 0.95 --include-reviewed   # corrected rule, ablation
```

## Running the pipeline on videos

```bash
zsd run --config configs/default.yaml          # corrected pipeline, local videos/ directory
zsd run --config configs/original_study.yaml   # original behaviour (MinIO; set endpoint/buckets)
```

Each run writes `frames.jsonl` (objects, boxes, all zero-shot scores and the
decision source per frame), `videos.csv` (per-video counts, label, status) and
`manifest.json` (configuration fingerprint, model revisions, package versions,
git commit). Runs are resumable. MinIO credentials come from the
`MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` environment variables.

## Measuring accuracy: annotation protocol

```bash
zsd annotate sample --n-per-stratum 40 --stratum-size "nonmilitary|nonmilitary=200" --out annotation/
zsd review --sheet annotation/sheet.csv --answers annotation/answers.csv --videos /path/to/videos
zsd annotate agreement --answers annotation/answers.csv    # Cohen's / Fleiss' κ
zsd annotate estimate --answers annotation/answers.csv --key annotation/sample_key.csv
```

The estimator weights each stratum by its inclusion probability, so it gives
unbiased accuracy, precision, recall and true category prevalence for both
systems, with stratified-bootstrap intervals. See `docs/methodology.md` §6.

## Citation

See [`CITATION.cff`](CITATION.cff); please also cite the paper of the original study.

## License

GPL-3.0 (see [`LICENSE`](LICENSE)). `src/zeroshot_detector/resources/open_images_v7_boxable_classes.txt`
lists Open Images V7 class names (CC BY 4.0, Google LLC).
