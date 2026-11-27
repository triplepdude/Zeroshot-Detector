# Methodology

This document describes the method as implemented, what the archived data can
and cannot support, the statistics used in [`results/REPORT.md`](../results/REPORT.md),
and the threats to validity. The study's paper
(<https://hdl.handle.net/20.500.14216/1718>) was not accessible when this
document was written, so the description is reconstructed from the code and data.

## 1. Task

The goal is to classify short-form videos (TikTok posts created between July
2018 and October 2024) into five categories, based on what appears in them:

| Label | Intended meaning |
|---|---|
| `weapons` | firearms, missiles, artillery, other weapons |
| `aircraft` | military aircraft |
| `maritime` | military ships and boats |
| `ground_vehicles` | military ground vehicles (tanks, armoured vehicles, military trucks) |
| `nonmilitary` | everything else (residual class) |

The unit of interest is the **video**. The system classifies sampled **frames**
and aggregates frame labels to a video label.

## 2. Pipeline

For a video with decoded frames $f_0, f_1, \dots$:

1. **Sampling.** Keep frames $t \in \{0, 30, 60, \dots\}$, about one per second
   at 30 fps. A frame is dropped if its L1 distance to the previously kept frame
   is ≤ 5000, which removes only near-identical frames.
2. **Detection.** Florence-2-large (Xiao et al., 2024) with the `<OD>` task
   prompt, beam search with 3 beams, at most 1024 new tokens. Output: a set of
   object labels $O_t$ with boxes. The label vocabulary largely follows Open
   Images (Kuznetsova et al., 2020).
3. **Text.** Original: $s_t$ = the labels concatenated *without a separator*
   (audit A1). Corrected: $s_t$ = the labels joined with `", "`.
4. **Keyword rule.** If any keyword in $K$ (18 military terms) matches $s_t$, the
   frame is labelled `ground_vehicles`. Original: case-sensitive substring test
   on the lower-cased string. Corrected: whole-word, case-insensitive, per object
   label, with exclusions (audit A2).
5. **Zero-shot classification.** BART-large fine-tuned on MNLI (Lewis et al.,
   2020; Williams et al., 2018) used as a zero-shot classifier (Yin et al.,
   2019). For each label $c$, the hypothesis is "This example is $c$." and the
   score is
   $p(c \mid s_t) = \mathrm{softmax}_c\, \ell_{\text{entail}}(s_t, h_c)$.
   The frame gets $\arg\max_c p(c \mid s_t)$ if $\max_c p(c \mid s_t) \ge \tau = 0.8$,
   and `nonmilitary` otherwise.
6. **Aggregation.** Original: arg-max of frame counts with label-order tie-breaking, plus an
   override that almost never fires (audit B1–B3). Corrected rules: `plurality`
   (ties go to `nonmilitary`, empty videos are undetermined) or `military_fraction`
   (military when at least a share $\phi$ of frames are military).

`zeroshot_detector.zero_shot.NLIZeroShotScorer` implements step 5 directly. It
was checked against the Hugging Face `zero-shot-classification` pipeline: the
maximum absolute score difference is 2.3 × 10⁻⁶.

Florence-2 can run through Microsoft's `trust_remote_code` implementation (as in
the study; it needs eager attention and `use_cache=False` with current
`transformers`) or through the native `transformers` implementation
(`florence-community/*` checkpoints). In smoke tests on a short video, the two
produced token-identical `<OD>` output, location tokens included: 3 frames with
the base checkpoints and 1 frame with the large checkpoints. This is a spot
check, not a proof of equivalence.

## 3. Data

See the data card, [`data/README.md`](../data/README.md). In brief:

* 1,099,951 sampled frames from 30,958 video files (30,942 with ≥ 1 frame);
* 28,676 distinct posts, because 2,266 posts were processed twice;
* 28,490 distinct Florence-2 strings;
* three labels per frame: **pipeline** (the system), **LLM** (an OpenAI model
  given the same string; model and prompt not recorded), and **human** (for 13,440
  frames only).

Each string always receives the same pipeline label and the same LLM label.
Both are therefore functions of the string, and most analyses can be done once
per distinct string and weighted by frame counts.

## 4. What the archive can and cannot support

| Question | Supported? | Section |
|---|---|---|
| Are the stored pipeline labels reproducible from the stored strings? | Yes | REPORT §2 |
| How often do the pipeline and LLM agree? | Yes: agreement, not accuracy | §4 |
| How were disagreements resolved in the original study? | Yes | §5 |
| Of the frames where the pipeline said military and the LLM said non-military, how often does a human side with each? | Yes, as a census of those cells; human judged the text, with both labels visible | §6 |
| Precision of the pipeline's military labels | Only under the discrepant-analysis assumption (optimistic), with bounds | §7 |
| How sensitive are video-level results to the aggregation rule, threshold, and separator bug? | Yes | §8–§10 |
| Accuracy, recall, or true prevalence of military content | **No.** There is no image-level ground truth and no random sample | use `zsd annotate` (§6 below) |
| Florence-2's own detection errors | **No.** No images are archived | — |

## 5. Statistical methods

**Weighting.** Frame-weighted statistics answer "what share of sampled frames …"
and are dominated by long videos and by a few very frequent strings: the 10
most frequent strings (led by `helicopter`, 66,199 frames) cover 36% of all
frames, and the top 100 cover 68%.
String-weighted statistics treat every distinct description once. Video-level
statistics answer the substantive question. All three are reported.

**Agreement.** The analysis reports observed agreement, Cohen's κ (Cohen, 1960)
and macro-F1, with the LLM label as the reference only for computing per-class
"precision/recall". These quantities measure agreement between two systems.

**Uncertainty.** Frames from the same video are strongly dependent; adjacent
samples usually produce the same string. Intervals for frame-level and
video-level statistics therefore come from a **cluster bootstrap**: posts are
resampled with replacement, and all files and frames of a post are kept
together (Davison & Hinkley, 1997, §3.8). The analysis uses 2,000 replicates,
95% percentile intervals and a fixed seed (2025). Statistics are computed from
pooled additive tables (confusion matrices, success/trial counts), so a
replicate is a multinomially weighted sum of per-post tables.
String-level and per-year proportions use Wilson score intervals (Wilson, 1927).
The test suite checks that the cluster bootstrap widens intervals for
correlated data.

**Discrepant analysis (REPORT §7).** For pipeline label $c$, let $A$ be the frames
where the LLM agrees, $R$ the adjudicated disagreements (with $H$ of them judged
$c$ by a human), and $U$ the unadjudicated disagreements. The point estimate
$(A + H)/(A + R)$ assumes that every agreement is correct. That assumption is
known to inflate accuracy estimates (Hadgu, 1996; Miller, 1998), so the report
also gives bounds that treat all of $U$ as wrong or as right. The bounds still
assume agreements are correct.

## 6. Gold-standard evaluation protocol (`zsd annotate`)

The package includes the design the archive lacks.

1. **Sampling.** `zsd annotate sample` draws a simple random sample of frames
   within each stratum *pipeline label × LLM label* (21 non-empty strata in the
   archive; default 40 frames per stratum). It removes duplicate files first and records $N_h$, $n_h$ and
   the design weight $N_h / n_h$. Rare but important cells, such as
   pipeline = weapons and LLM = nonmilitary, are guaranteed to be represented.
2. **Blind annotation.** The sheet given to annotators contains only
   `sample_id, video, frame_number`. `zsd review` shows the **frame image** with a
   short codebook. Annotators never see system labels, and see the Florence-2
   text only if `--show-text` is used. Each annotator gets a seeded random
   order. `unsure` and `unviewable` are explicit answers.
3. **Reliability.** At least two independent annotators per item.
   `zsd annotate agreement` reports pairwise Cohen's κ and Fleiss' κ
   (Fleiss, 1971). The gold label is the majority vote; ties are *unresolved*
   and reported.
4. **Estimation.** `zsd annotate estimate` computes Horvitz–Thompson-weighted
   (stratified) estimates of each system's accuracy, per-class precision,
   recall and F1, Cohen's κ, and the **true category prevalence** (Cochran,
   1977). Intervals come from a stratified bootstrap that resamples within
   strata and holds census strata fixed. A test checks that the weighted
   estimator is unbiased where the naive sample mean is not.

Sample-size guidance: overall accuracy is dominated by the largest stratum
(nonmilitary/nonmilitary, 78% of frames). Its precision is driven by
$n_h$ in that stratum, so allocate more items there, or use Neyman allocation
if pilot estimates are available. Per-class precision for a military label is
driven by the strata with that pipeline label.

## 7. Threats to validity

**Construct validity.**
* The taxonomy does not say whether *ground vehicles*, *aircraft* and
  *maritime* mean military only. The LLM and the original auto-correction rules
  treated civilian trucks and boats as military. That disagreement explains
  most of the pipeline/LLM disagreement (32,545 frames are `truck`/`car`
  strings).
* Classification is text-mediated. Both the pipeline and the LLM see a list of
  detected object names, never the image, so Florence-2's vocabulary and
  errors bound what either can detect. Toys, video games, airshows and museum
  pieces cannot be distinguished from operational equipment.
* `nonmilitary` is a residual class and absorbs low-confidence military frames
  (threshold 0.8).

**Internal validity.**
* There is no image-level ground truth (§4). The LLM is not an independent
  annotator (same input); its model and prompt are unknown.
* Human labels exist only for one kind of disagreement. They were made from
  text with both system labels shown, by an unknown number of annotators,
  and contain visible errors (audit D4–D6).

**External validity.**
* The collection criteria for the videos are not documented. Post dates
  are concentrated in 2022–2024 and come in chronological batches. Shares
  describe this archive, not the platform.

**Statistical conclusion validity.**
* Frames are clustered within videos, and 6.6% of frames are exact duplicates.
  Both are handled by clustering on posts, but frame-weighted figures remain
  dominated by long videos (up to 3,185 sampled frames).

**Reproducibility.**
* The LLM step cannot be re-run. Videos are not archived, so the
  Florence-2 step cannot be re-run on the original inputs. All downstream steps can,
  and were (REPORT §2).

## 8. Responsible use

The data contain public post ids and machine-generated object labels. They
contain no user names or media, but post ids can be resolved to public posts and
their authors. Do not publish per-post labels in a form that targets individuals,
and follow the platform's terms when re-collecting media. Detection of military
content is dual-use. The measured error modes (false positives on toys and
clothing, the civilian/military vehicle ambiguity, and low agreement on weapons)
make the system unsuitable for automated moderation or enforcement without human review.

## References

* Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley.
* Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
* Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
* Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5), 378–382.
* Hadgu, A. (1996). The discrepancy in discrepant analysis. *The Lancet*, 348(9027), 592–593.
* Kuznetsova, A., et al. (2020). The Open Images Dataset V4. *International Journal of Computer Vision*, 128, 1956–1981.
* Lewis, M., et al. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *ACL 2020*.
* Miller, W. C. (1998). Bias in discrepant analysis: When two wrongs don't make a right. *Journal of Clinical Epidemiology*, 51(3), 219–231.
* Williams, A., Nangia, N., & Bowman, S. (2018). A broad-coverage challenge corpus for sentence understanding through inference. *NAACL 2018*.
* Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212.
* Xiao, B., et al. (2024). Florence-2: Advancing a unified representation for a variety of vision tasks. *CVPR 2024*.
* Yin, W., Hay, J., & Roth, D. (2019). Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. *EMNLP-IJCNLP 2019*.
