# Experiment plan: a text-level reference standard and what it reveals

*Written and committed before any reference label was collected. Deviations
from this plan are listed in `docs/experiments/REPORT.md` under "Deviations".*

## Motivation

The archive has no ground truth. Its LLM labels come from an undocumented
model and prompt, and its human labels cover only one kind of disagreement,
were made with both system labels visible, and contain obvious errors
(`docs/audit.md` D1–D6). Every archived label is a deterministic function of
the Florence-2 object string. A **reference label for the string**, collected
on a probability sample with a documented protocol, therefore supports valid,
design-weighted estimates of every string-based classifier's agreement with
that reference. This includes the original pipeline, the original LLM, and any
alternative.

The reference is **text-level**: "what do these detected objects depict?" It
cannot detect Florence-2 errors (a missed or hallucinated object). Image-level
evaluation remains future work (`zsd annotate` + `zsd review`).

## Research questions

* **RQ1 (construct).** How much do the study's conclusions depend on what the
  categories mean? Does *aircraft / maritime / ground vehicles* mean any such
  equipment, or military equipment only?
* **RQ2 (accuracy).** How accurate are the original pipeline, the original LLM
  labels and the corrected pipeline (separated labels, corrected keyword rule)
  against the reference, under each reading of the categories?
* **RQ3 (prevalence).** What share of sampled frames show military content
  according to the reference? How far are the pipeline and LLM estimates from it?
* **RQ4 (better classifiers).** Do alternative zero-shot set-ups (hypothesis
  templates, verbalizers that say "military", another NLI model, separated
  labels, threshold) or a small supervised model on detected-object labels
  agree better with the reference?
* **RQ5 (time).** Does the share of military videos change around
  24 February 2022 (full-scale invasion of Ukraine) and 7 October 2023? This is
  descriptive and exploratory.

## Design

**Population.** Frames of `data/frames.csv.gz`, one file per post
(1,027,433 frames).

**Sample.** Stratified simple random sample of frames, stratified by
`pipeline_label × llm_label` (21 non-empty strata), seed 20251127. Sizes:
40 per stratum, census for strata with ≤ 40 frames, 200 for
`nonmilitary|nonmilitary`, and 60 each for `aircraft|aircraft`,
`ground_vehicles|ground_vehicles` and `nonmilitary|ground_vehicles`. Design
weights are $N_h / n_h$. The unit that is labelled is the frame's string; frames
sharing a string share its label.

**Annotation.** Three independent annotator passes per string, each with the
same written codebook (`docs/experiments/CODEBOOK.md`) and an independently
shuffled order. Annotators are separate instances of a Claude model. They
receive the glued string and a suggested segmentation, and they do **not** see
the pipeline, LLM or human labels. Each pass records, per string:

* `categories`: every equipment type present (`weapons`, `aircraft`,
  `maritime`, `ground_vehicles`), possibly none;
* `primary`: the single most salient equipment type, or `none`;
* `military`: whether the objects indicate *military* use (`yes` / `no` / `ambiguous`);
* `confidence`: 1–3.

**Adjudication.** `primary` and `military` are decided by majority of the three
passes. Three-way ties go to a fourth, adjudicating pass that sees the three
answers and the codebook. Agreement among the three passes is reported as
Fleiss' κ for `primary` and for `military`.

**Reference readings** (decided before labelling):

| Reading | Reference label |
|---|---|
| *equipment* | `primary` (or `nonmilitary` when `none`) |
| *lenient military* | `primary` if `military ∈ {yes, ambiguous}`, else `nonmilitary` |
| *strict military* | `primary` if `military = yes`, else `nonmilitary` |

## Outcomes and analysis

* **Primary outcomes:** design-weighted (Horvitz–Thompson) accuracy, macro-F1
  and Cohen's κ against the reference, for the original pipeline and the
  original LLM, under each reading. 95% stratified-bootstrap intervals
  (2,000 replicates, seed 20251127).
* **Secondary:** per-class precision and recall; reference prevalence of each
  class; the same metrics for the corrected pipeline; lenient matching (system
  label ∈ `categories`).
* **RQ4:**
  * zero-shot grid: NLI model {BART-large-MNLI, DeBERTa-v3-large NLI} ×
    hypothesis template {original, "image shows"} × verbalizers {original
    identifiers, natural phrases with "military"} × input {glued, separated},
    with the threshold swept from 0 to 0.95;
  * supervised: L2-regularised logistic regression on bag-of-object-labels,
    either trained on the reference sample with stratified 5-fold
    cross-validation (design-weighted), or trained on the original LLM labels
    of all strings outside the sample (weak supervision) and evaluated on the
    sample.

  Every configuration is reported. Any "best" configuration is labelled
  as selected post hoc on the same sample.
* **RQ5:** monthly share of military videos (plurality rule, one file per post)
  for the pipeline and LLM labels, with Wilson intervals. Mean shares are
  compared in the 6 months before and after each date, and a segmented
  regression with a level and slope change is fitted. Interpreted descriptively:
  the selection of videos is undocumented and batches are chronological.

## Known limitations (stated in advance)

* The annotators are language models, not people; their errors may be
  correlated with each other and with the original LLM.
* A text-level reference cannot detect detection errors.
* The sample supports frame-weighted estimates over the population of
  frames. Video-level accuracy is not directly estimable from it.
