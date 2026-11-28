# Experiment plan v2: a text-level reference standard and what it reveals

*Version 2, committed before any reference label was collected. v1 (git history,
commit `9815c33`) was revised after a four-lens design review (statistics,
construct validity, LLM-annotation methodology, hostile reviewer); the changes
are listed in §9. Disclosures: the plan and codebook were written **after** the
system labels, the disagreement tables of `results/REPORT.md` and the archived
human labels had been inspected. A v1 frame sample (seed 20251127) was drawn and
its composition inspected during the review; it was never annotated. The v2
sample is drawn once, with the seed below, after this plan is committed.*

## 1. Motivation and scope

Every archived label (pipeline, OpenAI LLM, human) is a deterministic function
of the Florence-2 object string. A documented reference label for the
**string**, collected on a probability sample, therefore supports valid
design-weighted estimates of how well any string-based classifier agrees with
that reference.

The reference is **text-level** ("what do these detected objects depict, under
the codebook?") and is produced by **language-model annotators** applying a
researcher-written codebook. Outcomes are therefore called **TLR agreement**
(agreement with the text-level reference), never accuracy. Agreement between
passes of the same model family is called self-consistency, not inter-rater
reliability. No human audit is available: the panel's own error rate is not
estimated.

## 2. Research questions

* **RQ1 (construct).** How do the study's conclusions depend on what the
  categories mean? Answered **exactly** over the whole population by the
  deterministic *rule reference* (§4). The panel then measures how often careful
  application of the codebook departs from the mechanical rules.
* **RQ2.** TLR agreement, Cohen's κ, macro-F1, per-class precision/recall of the
  original pipeline, the original LLM labels and the corrected pipeline (§10 of
  REPORT), under each reading.
* **RQ3.** TLR prevalence of each class, compared with the pipeline and LLM shares.
* **RQ4.** Alternative zero-shot set-ups and a supervised model on detected-object labels.
* **RQ5 (exploratory).** Composition of the archive by batch around 24 Feb 2022 and 7 Oct 2023.

## 3. Readings (pre-registered)

Each vehicle category present gets a military score 0–3 (CODEBOOK §3). Weapons
are never gated.

| Reading | Reference label | Role |
|---|---|---|
| **lenient** | most salient category with score ≥ 2 (weapons always qualify), else nonmilitary | **primary** |
| equipment | the primary category at any score | bracketing |
| strict | most salient category with score = 3 (weapons always qualify) | bracketing |
| type | score ≥ 1 | secondary |

"Most salient" follows the ordered primary rules of CODEBOOK §3.2.

## 4. RQ1: rule reference (computed before annotation)

`experiments.labels.rule_reference` applies the label table and rules to all
28,490 segmented strings of the population (1,027,433 frames, one file per
post). Exact values (no sampling):

| Reading | Military share | Pipeline agreement (κ) | LLM agreement (κ) |
|---|---|---|---|
| equipment | 45.4% | 71.3% (0.453) | 74.7% (0.522) |
| type | 21.2% | 95.5% (0.867) | 98.9% (0.970) |
| **lenient** | 21.1% | 95.5% (0.865) | 98.8% (0.967) |
| strict | 6.4% | 88.1% (0.479) | 84.2% (0.373) |

The reference label changes between equipment and strict for 39.0% of frames,
and between type and lenient for only 0.1%. The detector vocabulary contains no
military-only aircraft or watercraft label, and almost no civilian context cues.

**Predictions (made before annotation).**

* P1: under lenient, the panel reference departs from the rule reference on less
  than 5% of frames (design-weighted).
* P2: under lenient, the LLM has higher TLR agreement than the pipeline; under
  strict, the pipeline has higher TLR agreement than the LLM.

## 5. Design (sample of strings)

Population: frames of one file per post. Seed: **20251128**.

* **C** (certainty): the 500 most frequent strings (each ≥ 159 frames; 83.5% of
  frames). Each enters with its exact frame count.
* **S** (screened census): every `nonmilitary|nonmilitary` string outside C whose
  segmented objects contain a weapon token (`weapon, rifle, handgun, shotgun,
  missile, rocket, cannon, sword, dagger, bow and arrow, knife`) or a watercraft
  token (`boat, canoe, kayak, jet ski, watercraft`).
* **T** (tail): the remaining frames. Simple random sample of frames within
  pipeline × LLM strata, n_h = 30 (census if N_h ≤ 30). `nonmilitary|nonmilitary`
  is split by a vehicle-token screen: vehicle-token n = 120, no-equipment-token
  n = 40. Weights N_h / n_h.
* `No object detected.` is assigned deterministically (no category) and never annotated.

Estimator: totals over records (C and S records weight M_s, fixed; tail records
weight N_h / n_h). Budget: about 1,110 distinct strings. Precision, simulated
with the rule reference as proxy truth (sampling only): TLR agreement ±0.1 pp and
LLM − pipeline difference ±0.1 pp. Annotation uncertainty therefore dominates.

## 6. Annotation protocol

* **Codebook v2** (`CODEBOOK.md`), frozen; its SHA-256 is recorded. Worked
  examples use strings absent from the archive.
* **Configurations:** A = primary model, objects in archived order; B = a
  smaller model tier, archived order; C = primary model, objects shuffled within
  each item. Every item gets one pass from each configuration. The 24 strings
  carrying more than 0.5% of frames get two extra passes: D = smaller tier with
  shuffled objects, E = primary model with reversed objects.
* **Batches:** 25 items, items in a fresh random order per configuration, prompts
  built mechanically (instruction = codebook; content = JSON of
  `item_id, raw, objects`). Annotators are instructed not to use tools. Every
  transcript is audited for tool calls, and a batch with any tool call or any
  mention of the systems' labels is discarded and re-run. Answers must echo
  `item_id` and `raw`, and are validated against a schema and the codebook's
  consistency constraints. A failed batch gets at most one retry, then
  single-item calls, flagged.
* **Embedded checks** (excluded from all estimates): (i) about 10% of items
  re-presented under fresh ids in the same pass (test-retest; half from the 24
  heavy strings); (ii) 12 synthetic strings absent from the archive whose answers
  follow from single rules (rule compliance; a configuration below 90% is
  flagged); (iii) 150 randomly chosen human-labelled strings outside the sample
  (consistency with the archived human labels); (iv) 100 random sample strings
  re-annotated with configuration A as single-item calls (batch-context effect).
* **Archive:** every call's configuration, item order, prompt and raw answers are
  stored under `data/reference/raw/` with SHA-256 sums. Model identities are
  described by tier, not by version string.

## 7. Adjudication and estimation

* For each pass, derive the reading labels from that pass's categories,
  primary and scores. The reference for a reading is the **majority of the
  per-pass reading labels**. Category scores are medians across passes; categories
  used for lenient matching are those recorded by at least 2 passes.
* A tie gets one fresh blind pass (configuration D) as a further vote. If it
  is still tied, the item is *unresolved*. Unresolved items are excluded from
  point estimates and reported, with bounds that score them all-correct and all-wrong.
* **Annotation-aware bootstrap** (2,000 replicates, seed 20251128): tail records
  are resampled within strata, and each string's passes are resampled with
  replacement and re-adjudicated. Sampling-only and total intervals are both reported.
* **Influence table:** the 25 heaviest strings with their votes, and the change in
  each primary outcome if a string's label flips.
* Agreement among configurations: Fleiss' κ on per-pass reading labels, on
  distinct strings (unweighted and design-weighted), and excluding the
  "decision-table" strings whose answer follows from the table alone (bare
  `helicopter, airplane, boat, truck, truckwheel, car tiretruck, tank, weapon,
  rifle, handgun, vehicle, land vehicle`).

**Confirmatory contrasts** (all others are secondary or exploratory):

1. RQ2: lenient TLR agreement, LLM − original pipeline.
2. RQ4: lenient TLR agreement, [DeBERTa-v3-large NLI, "military" verbalizers,
   original template, separated labels, corrected keyword rule, arg-max] −
   original pipeline.

**Sensitivity analyses:** (a) the alternative primary rule in which a weapon
stays primary even when it is armament; (b) bare `helicopter, airplane, boat,
truck` set to scores 1, 2 and 3; (c) excluding `screen_or_print` and
`toy_or_model` frames; (d) leave-one-configuration-out references.

## 8. RQ4 and RQ5

**RQ4 zero-shot grid:** {BART-large-MNLI, DeBERTa-v3-large NLI} × template {"This
example is {}.", "This image shows {}."} × verbalizers {identifiers, "military …"}
× input {glued, separated}. Each configuration is evaluated with and without the
keyword rule, at thresholds {arg-max, 0.05, 0.10, …, 0.95}. Scoring is done only on
reference strings. All configurations are reported as paired differences from the
original pipeline, labelled exploratory. The "best" configuration is estimated by
nested selection over 5 folds of strings (hash of item id). Baselines:
all-nonmilitary; the rule reference.

**RQ4 supervised:** L2 logistic regression on bag-of-object-labels, in two variants:

* trained on the reference strings, with GroupKFold(5) over strings, fitted
  unweighted, C ∈ {0.01, 0.1, 1, 10} chosen by inner grouped CV; out-of-fold
  predictions are evaluated with design weights;
* weakly supervised: cross-fitted over all 28,490 strings on the LLM labels
  (5 folds by item-id hash), evaluated on the reference sample.

**RQ5 (exploratory composition analysis):** the before/after change for the
batches that straddle each date, next to between-batch differences. A binomial
GLM on videos with batch fixed effects, a month trend and a post-event step. The
event estimate is ranked among placebo estimates for all months 2020-06 to
2024-04. A "change" is reported only if it exceeds the 95th placebo percentile and
has the same sign within and between batches. Window 18 months; event month excluded.

## 9. Changes from v1 (design review)

String-level design with certainty and screened census strata, replacing a
frame SRS (324 strings, about ±5 pp); researcher-decided label table and
per-category 0–3 scores, replacing a single yes/ambiguous/no flag (which made
lenient ≈ equipment and strict ≈ the pipeline's keyword); ordered primary rules;
exact population rule reference for RQ1 with predictions; examples from absent
strings; per-pass adjudication with blind tie-breaks and no silent imputation;
annotation-aware bootstrap; diversified configurations and embedded checks;
"TLR agreement" terminology; grouped CV and cross-fitting for supervised models;
one confirmatory contrast per RQ; RQ5 demoted to exploratory, because placebo
months showed that the v1 intervals had a 34–60% false-positive rate.

## 10. Known limitations

The annotators are language models applying the authors' codebook. Their
errors may be correlated with each other and with the original OpenAI labeller,
which would bias RQ2 towards the LLM. A text-level reference cannot detect
detector errors, and cannot tell video games, toys or museum pieces from real
equipment unless the objects say so. Intervals cover sampling and annotation
uncertainty for this archive only. Video-level agreement is not estimated.
