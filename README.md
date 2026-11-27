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
  evaluation the original study lacked.

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
configs/                   default.yaml (corrected), original_study.yaml (exact reproduction)
data/                      archived study data + data card + checksums
results/                   REPORT.md, summary.json, figures/, relabel/ (re-derived BART scores)
docs/                      methodology.md, audit.md
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
