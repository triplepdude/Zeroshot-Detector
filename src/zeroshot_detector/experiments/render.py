"""Markdown report and figures for the experiments (``results/experiments/``)."""

from __future__ import annotations

from pathlib import Path

from ..analysis.report import num, pct, table
from ..taxonomy import LABELS

READING_ORDER = ("lenient", "equipment", "strict", "type")
SYSTEM_NAMES = {
    "pipeline": "Original pipeline",
    "llm": "Original LLM labels",
    "corrected_pipeline": "Corrected pipeline (separated labels, corrected keyword rule, BART, τ=0.8)",
    "deberta_military_separated": "DeBERTa-v3 NLI, 'military' verbalizers, separated labels, keyword rule, arg-max",
    "rule_reference": "Rule reference (deterministic codebook)",
}


def ci(d: dict | None, fmt=pct) -> str:
    if not d or d.get("estimate") is None:
        return "–"
    return f"{fmt(d['estimate'])} ({fmt(d.get('ci_low'))}–{fmt(d.get('ci_high'))})"


def signed_pp(d: dict | None) -> str:
    if not d or d.get("estimate") is None:
        return "–"

    def f(x):
        return "–" if x is None else f"{100 * x:+.1f}"

    return f"{f(d['estimate'])} pp ({f(d.get('ci_low'))} to {f(d.get('ci_high'))})"


def _contrast(s: dict, reading: str, key: str) -> dict | None:
    return s.get("rq2", {}).get(reading, {}).get("total", {}).get("contrasts", {}).get(f"{key} - pipeline: agreement")


def _headline_rows(s: dict) -> list[list[str]]:
    """Confirmatory contrasts C1, C2 and predictions P1, P2 (PLAN v2 §4, §7)."""

    def excludes_zero(d: dict | None) -> str:
        if not d or d.get("ci_low") is None:
            return "–"
        if d["ci_low"] > 0:
            return "positive; interval excludes 0"
        if d["ci_high"] < 0:
            return "negative; interval excludes 0"
        return "interval includes 0"

    rows = []
    c1 = _contrast(s, "lenient", "llm")
    rows.append(
        ["C1: lenient TLR agreement, LLM − original pipeline", "confirmatory", signed_pp(c1), excludes_zero(c1)]
    )
    c2 = _contrast(s, "lenient", "deberta_military_separated")
    rows.append(
        [
            "C2: lenient TLR agreement, DeBERTa-v3 NLI ('military' verbalizers, separated labels, keyword rule, arg-max) − original pipeline",
            "confirmatory",
            signed_pp(c2),
            excludes_zero(c2) if c2 else "not computed (grid scores missing)",
        ]
    )
    pr = s.get("rq1_panel_vs_rule", {}).get("lenient")
    if pr and pr.get("estimate") is not None:
        dep = {"estimate": 1 - pr["estimate"], "ci_low": 1 - pr["ci_high"], "ci_high": 1 - pr["ci_low"]}
        rows.append(
            [
                "P1: panel departs from the rule reference on < 5% of frames (lenient)",
                "prediction",
                f"{ci(dep, lambda x: pct(x, 2))} of frames",
                "confirmed"
                if dep["ci_high"] < 0.05
                else ("not confirmed" if dep["estimate"] >= 0.05 else "inconclusive"),
            ]
        )
    strict = _contrast(s, "strict", "llm")
    if c1 and strict:
        ok = c1["ci_low"] > 0 and strict["ci_high"] < 0
        rows.append(
            [
                "P2: LLM ahead of the pipeline under lenient, behind it under strict",
                "prediction",
                f"lenient {signed_pp(c1)}; strict {signed_pp(strict)}",
                "confirmed" if ok else "not confirmed",
            ]
        )
    return rows


def figures(s: dict, fig_dir: Path) -> dict[str, Path]:
    try:
        import matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover
        return {}
    from ..analysis import figures as F

    plt = F._plt()
    paths = {}
    # forest plot: lenient/equipment/strict agreement per system (total intervals)
    systems = [k for k in SYSTEM_NAMES if k in s["rq2"]["lenient"]["total"]["systems"] and k != "rule_reference"]
    readings = [r for r in ("equipment", "lenient", "strict") if r in s["rq2"]]
    colors = [
        F.CATEGORY_COLORS["weapons"],
        F.CATEGORY_COLORS["aircraft"],
        F.CATEGORY_COLORS["maritime"],
        F.CATEGORY_COLORS["ground_vehicles"],
    ]
    fig, axes = plt.subplots(1, len(readings), figsize=(10.5, 0.5 * len(systems) + 1.8), sharey=True)
    for ax, reading in zip(axes, readings, strict=True):
        res = s["rq2"][reading]["total"]["systems"]
        for i, sysname in enumerate(systems):
            a = res[sysname]["agreement"]
            y = len(systems) - 1 - i
            ax.plot([a["ci_low"], a["ci_high"]], [y, y], color=colors[i % len(colors)], linewidth=2)
            ax.scatter([a["estimate"]], [y], color=colors[i % len(colors)], s=30, zorder=3, edgecolor="#ffffff")
            ax.text(a["estimate"], y + 0.25, f"{a['estimate']:.1%}", ha="center", fontsize=7, color=F.INK)
        ax.set_title(f"{reading} reading", loc="left", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", color=F.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(systems)), [SYSTEM_NAMES[k].split(" (")[0].split(",")[0] for k in reversed(systems)])
    fig.suptitle(
        "Agreement with the text-level reference (design-weighted, 95% intervals incl. annotation uncertainty)",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    paths["forest"] = F._save(fig, fig_dir / "tlr_agreement_by_reading.png")
    # RQ1: military share by reading (rule reference, exact) vs systems
    rq1 = s.get("rq1_rule_population")
    if rq1:
        rows = [
            (f"{r} reading", {lab: rq1["readings"][r]["prevalence"][lab] for lab in LABELS if lab != "nonmilitary"})
            for r in ("equipment", "type", "lenient", "strict")
        ]
        segs = [(lab, F.PRETTY[lab], F.CATEGORY_COLORS[lab]) for lab in LABELS if lab != "nonmilitary"]
        paths["rq1"] = F.stacked_share_figure(
            rows,
            segs,
            fig_dir / "rq1_military_share_by_reading.png",
            title="Share of frames that are military, by what 'military' means",
            subtitle="Rule reference applied to all 1,027,433 frames (exact). Same strings, different readings of the categories.",
            xlabel="Share of frames",
            xmax=0.55,
        )
    # RQ5 monthly series
    rq5 = s.get("rq5", {}).get("by_labelling", {})
    if rq5:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        for col, color, name in (
            ("pipeline_label", F.CATEGORY_COLORS["weapons"], "Pipeline"),
            ("llm_label", F.CATEGORY_COLORS["aircraft"], "LLM"),
        ):
            m = [r for r in rq5[col]["monthly"] if r["videos"] >= 50]
            xs = list(range(len(m)))
            ax.plot(xs, [r["share"] for r in m], color=color, linewidth=1.6, label=name)
            ax.fill_between(
                xs, [r["ci_low"] for r in m], [r["ci_high"] for r in m], color=color, alpha=0.12, linewidth=0
            )
            months = [r["month"] for r in m]
        for ev in ("2022-02", "2023-10"):
            if ev in months:
                ax.axvline(months.index(ev), color=F.INK_2, linestyle=(0, (3, 3)), linewidth=1)
                ax.text(months.index(ev), ax.get_ylim()[1], f" {ev}", va="top", fontsize=7, color=F.INK_2)
        ticks = [i for i, mo in enumerate(months) if mo.endswith("-01")]
        ax.set_xticks(ticks, [months[i][:4] for i in ticks])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", color=F.GRID, linewidth=0.6)
        ax.set_title("Monthly share of videos labelled military (months with ≥ 50 videos)", loc="left", pad=22)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, borderaxespad=0.2)
        fig.text(
            0.01,
            -0.04,
            "Descriptive. Collection batches are chronological; see the placebo analysis before reading any change.",
            color=F.INK_2,
            fontsize=7,
        )
        paths["rq5"] = F._save(fig, fig_dir / "rq5_monthly_share.png")
    return paths


def render(s: dict, figs: dict[str, Path], out_dir: Path) -> str:
    def fig(key, alt):
        return f"\n![{alt}]({figs[key].relative_to(out_dir).as_posix()})\n" if key in figs else ""

    L: list[str] = []
    w = L.append
    w("# Experiments: a text-level reference standard")
    w("")
    w(
        "> Generated by `zsd experiments report` — do not edit by hand. Plan: [`docs/experiments/PLAN.md`](../../docs/experiments/PLAN.md) "
        "(pre-registered); codebook: [`CODEBOOK.md`](../../docs/experiments/CODEBOOK.md); all numbers: [`summary.json`](summary.json)."
    )
    w("")
    w(
        "**Terminology.** *TLR agreement* = design-weighted agreement with the text-level reference produced by language-model "
        "annotators applying the authors' codebook. It is not accuracy against what is in the video. Intervals: 95%, stratified "
        "bootstrap that also resamples annotator passes (annotation-aware), unless marked *sampling only*."
    )
    w("\n## Pre-registered results\n")
    w(table(["Item", "Pre-registered as", "Result", "Outcome"], _headline_rows(s)))

    d, a = s["design"], s["annotation"]
    w("\n## 1. Design and annotation\n")
    w(
        table(
            ["Quantity", "Value"],
            [
                [
                    "Sample records (C / S / T strings)",
                    f"{d['records']:,} records, {d['strings']:,} strings ({', '.join(f'{k}: {v}' for k, v in d['parts'].items())})",
                ],
                [
                    "Annotator answers (by configuration)",
                    f"{a['answers']:,} ({', '.join(f'{k}: {v}' for k, v in sorted(a['by_config'].items()))})",
                ],
                ["Answers from single-item fallback", a["fallback_answers"]],
                ["Unresolved items per reading", ", ".join(f"{k}: {v}" for k, v in a["unresolved"].items())],
                ["Tool calls by annotators (other than the answer tool)", s.get("tool_audit_summary", "–")],
            ],
        )
    )
    sc = a["self_consistency_ABC"]
    sx = a["self_consistency_ABC_excluding_decision_table"]
    w("\nSelf-consistency across configurations A, B, C (Fleiss' κ on per-pass reading labels, distinct strings):\n")
    w(
        table(
            [
                "Reading",
                "κ (strings)",
                "κ (frame-weighted)",
                "unanimous",
                "κ (excl. decision-table strings)",
                "unanimous (excl.)",
            ],
            [
                [
                    r,
                    num(sc[r]["fleiss_kappa"], 4),
                    num(sc[r].get("fleiss_kappa_weighted"), 4),
                    pct(sc[r]["unanimous_share"], 2),
                    num(sx[r]["fleiss_kappa"], 4),
                    pct(sx[r]["unanimous_share"], 2),
                ]
                for r in READING_ORDER
                if r in sc
            ],
        )
    )
    ch = s["checks"]
    tr = ch["test_retest"]
    w(
        "\n**Embedded checks.** Test-retest (same configuration, item re-presented under a new id): "
        + ", ".join(f"{r} {pct(tr.get(f'agreement_{r}'))}" for r in READING_ORDER)
        + f" agreement over {tr['pairs']} pairs. "
        "Single-item vs. batched answers (configuration A): "
        + ", ".join(f"{r} {pct(ch['single_item_vs_batch'].get(f'agreement_{r}'))}" for r in READING_ORDER)
        + ". "
        "Rule compliance on synthetic strings (lenient label correct): "
        + ", ".join(
            f"{c} {pct(v['lenient'])} (exact primary {pct(v['primary'])}, scores {pct(v['scores'])})"
            for c, v in ch["synthetic_compliance"].items()
        )
        + ". Lenient disagreement with the reference by position in batch: "
        + ", ".join(f"{k}: {pct(v)}" for k, v in ch["position_effect_lenient_disagreement"].items())
        + "."
    )
    hc = ch["human_consistency"]
    w(
        f"\n**Consistency with the archived human labels** (unblinded, selective; {hc['strings']} strings): equipment reading "
        f"{pct(hc['agree_equipment'])}, lenient {pct(hc['agree_lenient'])}, strict {pct(hc['agree_strict'])}. By which system the "
        "human had sided with (lenient): " + ", ".join(f"{k} {pct(v)}" for k, v in hc["by_side_lenient"].items()) + ". "
        "This is consistency, not validation."
    )

    w("\n## 2. RQ1: what 'military' means\n")
    rq1 = s.get("rq1_rule_population")
    if rq1:
        w("Rule reference over the **whole population** (exact, computed and committed before annotation):\n")
        w(
            table(
                ["Reading", "Military share", "Pipeline agreement (κ)", "LLM agreement (κ)"],
                [
                    [
                        r,
                        pct(rq1["readings"][r]["military_share"]),
                        f"{pct(rq1['readings'][r]['pipeline']['agreement'])} ({num(rq1['readings'][r]['pipeline']['kappa'])})",
                        f"{pct(rq1['readings'][r]['llm']['agreement'])} ({num(rq1['readings'][r]['llm']['kappa'])})",
                    ]
                    for r in ("equipment", "type", "lenient", "strict")
                ],
            )
        )
        w(fig("rq1", "Military share by reading"))
    cg = s.get("rq1_contrast_by_group")
    if cg:
        rd = [r for r in ("lenient", "strict") if r in cg["readings"]]
        w(
            "**Where the LLM − pipeline difference comes from** (*post hoc*, added after seeing the results; exact, rule "
            "reference over the population). Strings are grouped by their equipment-reading category; contributions add up "
            "to the total difference in agreement.\n"
        )
        groups = sorted(cg["frame_share"], key=lambda g: -abs(cg["readings"][rd[0]]["by_group"][g]))
        w(
            table(
                ["Object group", "Frame share", *[f"{r} (pp)" for r in rd]],
                [
                    [
                        "no vehicle or weapon" if g == "nonmilitary" else g,
                        pct(cg["frame_share"][g]),
                        *[f"{100 * cg['readings'][r]['by_group'][g]:+.2f}" for r in rd],
                    ]
                    for g in groups
                ]
                + [["**total**", "100.0%", *[f"**{100 * cg['readings'][r]['total']:+.2f}**" for r in rd]]],
            )
        )
        w("")
    w("Panel reference vs. rule reference (design-weighted agreement; prediction P1: departure < 5% under lenient):\n")
    w(
        table(
            ["Reading", "Panel = rule reference"],
            [
                [r, ci(s["rq1_panel_vs_rule"][r], lambda x: pct(x, 2))]
                for r in READING_ORDER
                if r in s["rq1_panel_vs_rule"]
            ],
        )
    )

    w("\n## 3. RQ2 / RQ3: agreement with the reference\n")
    w(fig("forest", "TLR agreement by reading"))
    for reading in READING_ORDER:
        if reading not in s["rq2"]:
            continue
        tot, smp = s["rq2"][reading]["total"], s["rq2"][reading]["sampling_only"]
        w(f"\n### {reading.capitalize()} reading{' (primary)' if reading == 'lenient' else ''}\n")
        rows = []
        for k, v in tot["systems"].items():
            rows.append(
                [
                    SYSTEM_NAMES.get(k, k),
                    ci(v["agreement"]),
                    ci(smp["systems"][k]["agreement"]),
                    ci(v["kappa"], num),
                    ci(v["macro_f1"], num),
                    pct(v["military_share_system"]["estimate"]),
                ]
            )
        w(table(["System", "TLR agreement", "(sampling only)", "κ", "macro-F1", "military share (system)"], rows))
        ref_prev = next(iter(tot["systems"].values()))
        w(
            "\nReference prevalence (TLR, design-weighted): "
            + ", ".join(f"{lab} {ci(ref_prev[f'prevalence_{lab}'])}" for lab in LABELS)
            + ".\n"
        )
        if tot["contrasts"]:
            w(
                table(
                    ["Contrast (paired)", "Difference"],
                    [[k, signed_pp(v)] for k, v in tot["contrasts"].items() if k.endswith("agreement")],
                )
            )
        if "unresolved_bounds" in s["rq2"][reading]:
            b = s["rq2"][reading]["unresolved_bounds"]
            w(
                "\nBounds for unresolved items (all correct / all wrong): "
                + "; ".join(
                    f"{SYSTEM_NAMES.get(k, k).split(' (')[0]} {pct(v['all_correct'])}–{pct(v['all_wrong'])}"
                    for k, v in b.items()
                )
                + "."
            )
    w("\nPer-class precision / recall under the lenient reading:\n")
    lt = s["rq2"]["lenient"]["total"]["systems"]
    rows = []
    for k in ("pipeline", "llm", "corrected_pipeline", "deberta_military_separated"):
        if k in lt:
            for lab in LABELS:
                rows.append(
                    [
                        SYSTEM_NAMES[k].split(" (")[0].split(",")[0],
                        lab,
                        ci(lt[k].get(f"precision_{lab}")),
                        ci(lt[k].get(f"recall_{lab}")),
                    ]
                )
    w(table(["System", "Class", "Precision", "Recall"], rows))

    w("\n### Sensitivity (lenient reading; point estimate and sampling-only interval)\n")
    rows = []
    for name, res in s["sensitivity_lenient"].items():
        rows.append([name] + [ci(res["systems"][k]["agreement"]) for k in ("pipeline", "llm")])
    w(table(["Analysis", "Pipeline", "LLM"], rows))
    w("\n### Influence of the heaviest strings (lenient)\n")
    w(
        "Each string's lenient label is flipped (military ↔ non-military) and the outcomes recomputed. Both systems agree "
        "with the reference on most heavy strings, so a flip moves both agreements but rarely the LLM − pipeline contrast.\n"
    )
    w(
        table(
            ["String", "Frame share", "Reference", "Vote share", "Δ pipeline", "Δ LLM", "Δ (LLM − pipeline)"],
            [
                [
                    f"`{r['text']}`",
                    pct(r["frame_share"]),
                    r["lenient"],
                    pct(r["vote_share"]),
                    f"{100 * r['delta_pipeline']:+.1f} pp",
                    f"{100 * r['delta_llm']:+.1f} pp",
                    f"{100 * (r['delta_llm'] - r['delta_pipeline']):+.1f} pp",
                ]
                for r in s["influence_lenient"][:15]
            ],
        )
    )

    w("\n## 4. RQ4: better string classifiers? (lenient reading)\n")
    if "rq4_grid_lenient" in s:
        grid = sorted(s["rq4_grid_lenient"], key=lambda r: -(r["agreement"] or 0))
        w(
            "Top 15 of the exploratory grid (all configurations are in `summary.json`), as differences from the original pipeline:\n"
        )
        w(
            table(
                ["Configuration", "Keyword rule", "Threshold", "TLR agreement", "κ", "Δ vs pipeline"],
                [
                    [
                        r["config"],
                        "on" if r["keyword"] else "off",
                        r["threshold"],
                        pct(r["agreement"]),
                        num(r["kappa"]),
                        f"{100 * r['diff_vs_pipeline']:+.1f} pp",
                    ]
                    for r in grid[:15]
                ],
            )
        )
        ns = s["rq4_nested_selection"]
        w(
            f"\nNested selection (choose on 4 folds of strings, evaluate on the 5th): out-of-fold TLR agreement "
            f"{pct(ns['out_of_fold']['agreement'])}; chosen: "
            + "; ".join(f"{c['config']}/{'kw' if c['keyword'] else 'no kw'}/{c['threshold']}" for c in ns["chosen"])
            + "."
        )
    sup, weak, base = s["rq4_supervised_reference"], s["rq4_weak_supervision"], s["rq4_baselines"]
    w(
        f"\nSupervised bag-of-labels logistic regression trained on the reference strings (grouped 5-fold CV, {sup['n_training_strings']} strings): "
        f"out-of-fold TLR agreement {pct(sup['out_of_fold']['agreement'])} (κ {num(sup['out_of_fold']['kappa'])}). Weakly supervised on the original "
        f"LLM labels (cross-fitted over all strings): {pct(weak['agreement'])} (κ {num(weak['kappa'])}). Baselines: all non-military "
        f"{pct(base['all_nonmilitary']['agreement'])}; rule reference {pct(base['rule_reference']['agreement'])}; a single annotator "
        f"configuration (A) {pct(base['single_configuration_A']['agreement'])}."
    )

    w("\n## 5. RQ5 (exploratory): composition around two dates\n")
    rq5 = s["rq5"]["by_labelling"]
    rows = []
    for col, dct in rq5.items():
        for ev, e in dct["events"].items():
            rows.append(
                [
                    col.replace("_label", ""),
                    ev,
                    num(e["glm_step_logodds"]),
                    pct(e["placebo_percentile_of_abs_step"]),
                    f"{100 * e['within_batch_change']:+.1f} pp",
                    f"{100 * e['between_batch_change']:+.1f} pp",
                    ", ".join(e["straddling_batches"]),
                    "yes" if e["reported_as_change"] else "no",
                ]
            )
    w(
        table(
            [
                "Labels",
                "Date",
                "GLM step (log-odds)",
                "Placebo percentile of abs(step)",
                "Within straddling batches",
                "Between batches",
                "Straddling batches",
                "Reported as change",
            ],
            rows,
        )
    )
    w(fig("rq5", "Monthly military share"))
    w(
        "\nA change is reported only if the step exceeds the 95th percentile of placebo steps (all months 2020-06 to 2024-04) and has the "
        "same sign within and between batches (pre-registered)."
    )
    if s.get("deviations"):
        w("\n## Deviations from the plan\n")
        for dv in s["deviations"]:
            w(f"* {dv}")
    w("\n## Limitations\n")
    w(
        "* The reference is produced by language models applying the authors' codebook; its error rate against human judgement is not estimated.\n"
        "* It is text-level: detector errors, toys, video games and museum pieces are invisible unless the object list says so.\n"
        "* Intervals cover sampling and annotation uncertainty for this archive only."
    )
    return "\n".join(L) + "\n"
