"""``zsd`` command-line interface.

::

    zsd run       --config configs/default.yaml        # classify videos
    zsd relabel   --mode legacy|segmented              # re-score archived strings with BART-MNLI
    zsd analyze                                        # results/summary.json, REPORT.md, figures
    zsd data verify | export-legacy                    # data integrity / original CSV files
    zsd annotate  sample | agreement | estimate        # gold-standard evaluation protocol
    zsd review    --sheet ... --answers ...            # blind annotation web app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from . import __version__


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load_config
    from .pipeline import run

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg.output.dir) / f"run-{cfg.fingerprint()}"
    manifest = run(cfg, run_dir, limit=args.limit)
    print(json.dumps({k: manifest[k] for k in ("videos_processed_this_session", "errors_this_session")}, indent=2))
    print(f"outputs: {run_dir}")
    return 0


def _cmd_relabel(args: argparse.Namespace) -> int:
    import torch

    from .classify import FrameClassifier
    from .data import load_frames
    from .manifest import base_manifest, finish
    from .relabel import relabel_texts
    from .segment import build_vocabulary, segment_all
    from .zero_shot import NLIZeroShotScorer

    if args.threads:
        torch.set_num_threads(args.threads)
    frames = load_frames(args.data)
    counts = frames.florence_text.value_counts()  # most frequent first
    texts = counts.index.tolist()
    if args.coverage < 1.0:
        # the most frequent strings that together cover `coverage` of all frames
        share = counts.cumsum() / counts.sum()
        texts = counts.index[: int((share < args.coverage).sum()) + 1].tolist()
    if args.include_reviewed:
        reviewed = frames.loc[frames["human_label"].notna(), "florence_text"].unique().tolist()
        texts = list(dict.fromkeys(texts + reviewed))
    objects = None
    if args.mode == "segmented":
        objects = segment_all(texts, build_vocabulary(texts))
    scorer = NLIZeroShotScorer(model=args.model, revision=args.revision, batch_size=args.batch_size)
    clf = FrameClassifier(scorer, threshold=args.threshold, legacy=args.mode == "legacy", always_score=True)
    out = Path(args.out) / f"{args.mode}_{args.model.split('/')[-1]}.csv.gz"
    manifest = base_manifest(f"relabel_{args.mode}")
    relabel_texts(clf, texts, out, chunk_size=args.chunk_size, objects=objects)
    import transformers

    finish(
        manifest,
        model=args.model,
        revision=scorer.resolved_revision,
        threshold=args.threshold,
        mode=args.mode,
        coverage=args.coverage,
        include_reviewed=args.include_reviewed,
        n_texts=len(texts),
        transformers=transformers.__version__,
        torch=torch.__version__,
        output=str(out),
    )
    out.with_name(out.name.split(".")[0] + ".meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis.report import run_analysis

    run_analysis(args.data, args.out, args.relabel_dir, n_boot=args.n_boot, seed=args.seed, figures=not args.no_figures)
    print(f"wrote {Path(args.out) / 'REPORT.md'} and summary.json")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from .data import default_data_dir, export_legacy, sha256, verify_legacy_checksums

    data_dir = Path(args.data or default_data_dir())
    if args.data_command == "export-legacy":
        for name, path in export_legacy(data_dir, args.out).items():
            print(f"{name}\t{path}")
        return 0
    # verify
    ok = True
    sums = data_dir / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        good = sha256(data_dir / name.strip()) == digest
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {name.strip()}")
    with tempfile.TemporaryDirectory() as tmp:
        for name, good in verify_legacy_checksums(data_dir, tmp).items():
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} regenerated {name} is byte-identical to the original")
    return 0 if ok else 1


def _cmd_annotate(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import annotation as A
    from .analysis.stats import add_post_ids
    from .data import load_frames

    if args.annotate_command == "sample":
        frames = add_post_ids(load_frames(args.data))
        overrides = {}
        for item in args.stratum_size or []:
            name, _, n = item.rpartition("=")
            overrides[name] = int(n)
        sample = A.stratified_sample(frames, n_per_stratum=args.n_per_stratum, seed=args.seed, n_overrides=overrides)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out / "sample_key.csv", index=False)
        A.annotation_sheet(sample, seed=args.seed, include_text=args.include_text).to_csv(
            out / "sheet.csv", index=False
        )
        n_strata = sample["stratum"].nunique()
        print(f"{len(sample)} items in {n_strata} strata → {out}/sheet.csv (blind), sample_key.csv (keep private)")
        return 0
    answers = pd.read_csv(args.answers, dtype={"sample_id": str})
    if args.annotate_command == "agreement":
        print(json.dumps(A.inter_annotator_agreement(answers), indent=2))
        return 0
    key = pd.read_csv(args.key, dtype={"sample_id": str})
    gold = A.adjudicate(answers)
    result = A.estimate(key, gold, n_boot=args.n_boot, seed=args.seed)
    result["inter_annotator"] = A.inter_annotator_agreement(answers)
    text = json.dumps(result, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from .review.app import serve

    serve(args.sheet, args.answers, args.videos, args.show_text, args.host, args.port)
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    from .data import load_frames, load_videos

    ref_dir = Path(args.reference)
    if args.experiments_command == "materials":
        from .experiments.materials import build

        manifest = build(load_frames(args.data), args.codebook, ref_dir)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("sha256", "configs")}, indent=2))
        return 0
    if args.experiments_command == "score-grid":
        import pandas as pd

        from .experiments.grid import all_configs, score_config

        rec = pd.read_csv(ref_dir / "design_records.csv", keep_default_na=False)
        strings = pd.read_csv(ref_dir / "strings.csv.gz", keep_default_na=False)
        texts = list(dict.fromkeys(rec["text"]))
        seg = {t: tuple(o.split("; ")) for t, o in zip(strings["text"], strings["objects"], strict=True)}
        for config in all_configs():
            score_config(config, texts, seg, args.grid_scores)
        return 0
    # report
    from .experiments.reference import load_workflow_results
    from .experiments.render import figures, render
    from .experiments.report import clean, run
    from .experiments.rule_reference import contrast_by_group, rq1_summary, string_table

    passes = load_workflow_results(sorted((ref_dir / "raw").glob("workflow_*.json")))
    frames, videos = load_frames(args.data), load_videos(args.data)
    grid = Path(args.grid_scores) if args.grid_scores else None
    summary = run(ref_dir, passes, frames, videos, grid, n_boot=args.n_boot, seed=args.seed)
    table = string_table(frames)
    summary["rq1_rule_population"] = rq1_summary(table)
    summary["rq1_contrast_by_group"] = contrast_by_group(table)
    audit = ref_dir / "raw" / "tool_audit.json"
    if audit.exists():
        a = json.loads(audit.read_text())
        summary["tool_audit"] = a
        summary["tool_audit_summary"] = (
            f"{sum(v['other_tools'] for v in a.values())} in {sum(v['transcripts'] for v in a.values())} transcripts"
        )
    deviations = Path(args.deviations)
    if deviations.exists():
        summary["deviations"] = [
            line[2:].strip() for line in deviations.read_text().splitlines() if line.startswith("- ")
        ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = clean(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figs = figures(summary, out / "figures") if not args.no_figures else {}
    (out / "REPORT.md").write_text(render(summary, figs, out))
    print(f"wrote {out / 'REPORT.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zsd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="classify videos from a local directory or MinIO")
    r.add_argument("--config", required=True)
    r.add_argument("--run-dir")
    r.add_argument("--limit", type=int, help="process at most N new videos")
    r.set_defaults(func=_cmd_run)

    rl = sub.add_parser("relabel", help="re-score the archived Florence-2 strings")
    rl.add_argument("--mode", choices=["legacy", "segmented"], default="legacy")
    rl.add_argument("--data")
    rl.add_argument("--out", default="results/relabel")
    rl.add_argument("--model", default="facebook/bart-large-mnli")
    rl.add_argument("--revision")
    rl.add_argument("--threshold", type=float, default=0.8)
    rl.add_argument("--batch-size", type=int, default=64)
    rl.add_argument("--chunk-size", type=int, default=1024)
    rl.add_argument("--threads", type=int)
    rl.add_argument(
        "--coverage", type=float, default=1.0, help="score only the most frequent strings covering this share of frames"
    )
    rl.add_argument("--include-reviewed", action="store_true", help="always include strings with human labels")
    rl.set_defaults(func=_cmd_relabel)

    a = sub.add_parser("analyze", help="reproduce results/ from data/")
    a.add_argument("--data", default="data")
    a.add_argument("--out", default="results")
    a.add_argument("--relabel-dir")
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=2025)
    a.add_argument("--no-figures", action="store_true")
    a.set_defaults(func=_cmd_analyze)

    d = sub.add_parser("data", help="data integrity")
    dsub = d.add_subparsers(dest="data_command", required=True)
    dv = dsub.add_parser("verify", help="check checksums and byte-identical regeneration of the original CSVs")
    dv.add_argument("--data")
    de = dsub.add_parser("export-legacy", help="write the original CSV files")
    de.add_argument("--data")
    de.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_data)

    an = sub.add_parser("annotate", help="gold-standard evaluation protocol")
    asub = an.add_subparsers(dest="annotate_command", required=True)
    s = asub.add_parser("sample", help="draw a stratified sample and a blind annotation sheet")
    s.add_argument("--data")
    s.add_argument("--out", default="annotation")
    s.add_argument("--n-per-stratum", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--include-text", action="store_true", help="show the Florence-2 string to annotators")
    s.add_argument(
        "--stratum-size",
        action="append",
        metavar="PIPELINE|LLM=N",
        help="sample size for one stratum, e.g. 'nonmilitary|nonmilitary=200' (repeatable)",
    )
    ag = asub.add_parser("agreement", help="inter-annotator agreement")
    ag.add_argument("--answers", required=True)
    es = asub.add_parser("estimate", help="design-weighted accuracy against adjudicated labels")
    es.add_argument("--answers", required=True)
    es.add_argument("--key", required=True)
    es.add_argument("--n-boot", type=int, default=2000)
    es.add_argument("--seed", type=int, default=0)
    es.add_argument("--out")
    an.set_defaults(func=_cmd_annotate)

    rv = sub.add_parser("review", help="blind annotation web app")
    rv.add_argument("--sheet", required=True)
    rv.add_argument("--answers", required=True)
    rv.add_argument("--videos", help="directory containing the video files (searched recursively)")
    rv.add_argument("--show-text", action="store_true")
    rv.add_argument("--host", default="127.0.0.1")
    rv.add_argument("--port", type=int, default=5000)
    rv.set_defaults(func=_cmd_review)

    ex = sub.add_parser("experiments", help="text-level reference standard experiments (docs/experiments)")
    exs = ex.add_subparsers(dest="experiments_command", required=True)
    for name, helptext in (
        ("materials", "draw the sample and build annotation materials (run once)"),
        ("score-grid", "score the zero-shot grid on the reference strings"),
        ("report", "assemble results/experiments from the annotations"),
    ):
        e = exs.add_parser(name, help=helptext)
        e.add_argument("--data")
        e.add_argument("--reference", default="data/reference")
        e.add_argument("--codebook", default="docs/experiments/CODEBOOK.md")
        e.add_argument("--grid-scores", default="results/experiments/grid_scores")
        e.add_argument("--out", default="results/experiments")
        e.add_argument("--deviations", default="docs/experiments/DEVIATIONS.md")
        e.add_argument("--n-boot", type=int, default=2000)
        e.add_argument("--seed", type=int, default=20251128)
        e.add_argument("--no-figures", action="store_true")
    ex.set_defaults(func=_cmd_experiments)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
