"""Command line interface for the fire-rule evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from tools.fire_rule_eval.features import MOSAIC_REGIONS
from tools.fire_rule_eval.features import ExtractOptions
from tools.fire_rule_eval.features import extract_features
from tools.fire_rule_eval.features import iter_check_frames
from tools.fire_rule_eval.flicker import extract_flicker_features
from tools.fire_rule_eval.flicker import iter_frames_30fps
from tools.fire_rule_eval.protocol import evaluate_rule
from tools.fire_rule_eval.protocol import render
from tools.fire_rule_eval.protocol import render_continuous
from tools.fire_rule_eval.protocol import render_hard_detail
from tools.fire_rule_eval.protocol import render_per_day
from tools.fire_rule_eval.report import load_clips
from tools.fire_rule_eval.report import render_markdown
from tools.fire_rule_eval.rules import FLICKER_RULES
from tools.fire_rule_eval.rules import RULES
from tools.fire_rule_eval.synth import SynthOptions
from tools.fire_rule_eval.synth import composite_video

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv"}


def _cmd_scan(args: argparse.Namespace) -> int:
    """Appends every video in a directory to a manifest (creating it if needed)."""
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"clips": []}
    known = {entry["path"] for entry in manifest["clips"]}
    added = 0
    for path in sorted(Path(args.dir).rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES or str(path) in known:
            continue
        name = f"{args.group}__{path.stem}"
        manifest["clips"].append({"name": name, "path": str(path), "label": args.label, "group": args.group})
        added += 1
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"added {added} clips to {manifest_path}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """Extracts per-check features for every clip in the manifest."""
    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = ExtractOptions(args.interval, args.start, args.max_checks, args.scale)
    for entry in manifest["clips"]:
        if args.only and entry["name"] not in args.only:
            continue
        target = out_dir / f"{entry['name']}.npz"
        if args.skip_existing and target.exists():
            continue
        started = time.time()
        regions = MOSAIC_REGIONS if args.exclude_mosaic and entry.get("group", "").startswith(("goat_", "synth")) else ()
        features = extract_features(iter_check_frames(Path(entry["path"]), options), (args.blur, args.blur), regions)
        np.savez_compressed(target, **features)
        print(f"{entry['name']}: {len(features['t'])} checks in {time.time() - started:.0f}s", flush=True)
    return 0


def _extract_flicker_one(job: tuple[str, str, str, bool]) -> str:
    name, path, out, windowed = job
    cv2.setNumThreads(1)
    started = time.time()
    features = extract_flicker_features(iter_frames_30fps(Path(path)), windowed=windowed)
    if not len(features["t"]):
        raise ValueError(f"{name}: fewer than 31 frames decoded, nothing to store")
    target = Path(out) / f"{name}.npz"
    temporary = target.with_suffix(".tmp.npz")  # written aside and renamed: an interrupted run leaves no corrupt file
    np.savez_compressed(temporary, **features)
    temporary.replace(target)
    return f"{name}: {len(features['t'])} rows in {time.time() - started:.0f}s"


def _cmd_flicker(args: argparse.Namespace) -> int:
    """Extracts per-second flicker features (windowed reference and incremental) for manifest clips."""
    import multiprocessing

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (entry["name"], entry["path"], str(out_dir), not args.no_windowed)
        for entry in manifest["clips"]
        if (not args.only or entry["name"] in args.only)
        and not (args.skip_existing and (out_dir / f"{entry['name']}.npz").exists())
    ]
    # `fork`: the default start method on Python 3.14 (forkserver) hangs when run through `python -m`.
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        for line in pool.imap_unordered(_extract_flicker_one, jobs):
            print(line, flush=True)
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    """Writes a synthetic positive clip."""
    options = SynthOptions(args.fraction, args.anchor_x, args.anchor_y, args.seconds, args.fire_start, args.onset)
    written = composite_video(Path(args.background), Path(args.fire), Path(args.out), options)
    print(f"wrote {written} frames to {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Prints the Markdown report."""
    flicker_dir = Path(args.flicker) if args.flicker else None
    clips = load_clips(Path(args.manifest), Path(args.features), flicker_dir, args.warmup)
    if not clips:
        print("no clips with features found", file=sys.stderr)
        return 1
    print(render_markdown(clips, (*RULES, *FLICKER_RULES) if flicker_dir else RULES))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Prints the common #132 comparison table (best safe point per rule, warm-up, time to first alarm)."""
    from tools.fire_rule_eval.rules import RULES_BY_NAME

    clips = load_clips(Path(args.manifest), Path(args.features), None, args.warmup)
    if not clips:
        print("no clips with features found", file=sys.stderr)
        return 1
    clips = [c for c in clips if not (c.label == "neg" and any(token in c.name for token in args.exclude_neg))]
    from tools.fire_rule_eval.decision import build_candidates

    lookup = {**RULES_BY_NAME, **{cand.name: cand.rule for cand in build_candidates(clips)}}
    rules = [lookup[name] for name in args.rules]
    outcomes = [evaluate_rule(rule, clips) for rule in rules]
    print(render(outcomes, args.warmup, len(rules)))
    if args.continuous:
        grid_params = {o.rule: o.param for o in outcomes}
        print("\n" + render_continuous(rules, clips, grid_params))
        print("\n" + render_hard_detail(rules, clips, grid_params))
        print("\n" + render_per_day(rules, clips))
    return 0


def _cmd_decision(args: argparse.Namespace) -> int:
    """Prints the #137 decision-layer report (debounce vs CUSUM / SPRT, #130 table, conformal threshold)."""
    from tools.fire_rule_eval.decision_eval import run

    clips = load_clips(Path(args.manifest), Path(args.features), None, args.warmup)
    if not clips:
        print("no clips with features found", file=sys.stderr)
        return 1
    print(run(clips, args.label or args.features))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    """Measures the per-check cost of the P15 / P8 candidates on real frames."""
    if args.decision:
        from tools.fire_rule_eval.decision import bench_decision

        for name, microseconds in bench_decision().items():
            print(f"{name}: {microseconds:.3f} us/check")
        return 0
    from tools.fire_rule_eval import bench

    for path in args.video:
        for scale in args.scale:
            if args.local_contrast:
                r = bench.run_local_contrast(Path(path), args.checks, scale, args.repeats, args.threads)
                print(
                    f"{Path(path).name} {int(r['width'])}x{int(r['height'])} threads={args.threads} "
                    f"(warm-AND-moving share {r['joint_share']:.4f}): baseline mask {r['baseline_mask_ms']:.2f} ms; "
                    + "; ".join(
                        f"{k} {r[f'{k}_ms']:.2f} ms"
                        for k in ("p9_cfar", "p10_tophat", "p10_lcm", "p1_wavelet", "dense_p1", "all")
                    ),
                    flush=True,
                )
                continue
            if args.global_change:
                r = bench.run_global(Path(path), args.checks, scale, args.repeats, args.threads)
                print(
                    f"{Path(path).name} {int(r['width'])}x{int(r['height'])} threads={args.threads}: "
                    f"baseline mask {r['baseline_mask_ms']:.2f} ms; global features +{r['global_ms']:.2f} ms",
                    flush=True,
                )
                continue
            if args.invariance:
                r = bench.run_invariance(Path(path), args.checks, scale, args.repeats, args.threads)
                print(
                    f"{Path(path).name} {int(r['width'])}x{int(r['height'])} threads={args.threads}: "
                    f"baseline mask {r['baseline_mask_ms']:.2f} ms; "
                    + "; ".join(
                        f"{k} {r[f'{k}_ms']:.2f} ms" for k in ("p4s", "p4g", "p4t", "p4c_colour", "p11", "p12", "all_variants")
                    ),
                    flush=True,
                )
                continue
            r = bench.run(Path(path), args.checks, scale, args.repeats, args.threads)
            print(
                f"{Path(path).name} {int(r['width'])}x{int(r['height'])} threads={args.threads}: "
                f"baseline mask {r['baseline_mask_ms']:.2f} ms; P15 +{r['p15_ms']:.2f} ms "
                f"(tracker only {r['p15_tracker_only_ms']:.2f}); P8 +{r['p8_ms']:.2f} ms; P15+P8 +{r['p15_p8_ms']:.2f} ms",
                flush=True,
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(prog="python -m tools.fire_rule_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="add videos from a directory to a manifest")
    scan.add_argument("--dir", required=True)
    scan.add_argument("--label", required=True, choices=["pos", "neg", "neg_hard"])
    scan.add_argument("--group", required=True)
    scan.add_argument("--manifest", required=True)
    scan.set_defaults(func=_cmd_scan)

    extract = sub.add_parser("extract", help="compute per-check features for manifest clips")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--interval", type=float, default=1.0, help="seconds between checks (default 1.0)")
    extract.add_argument("--start", type=float, default=0.0)
    extract.add_argument("--max-checks", type=int, default=None)
    extract.add_argument("--scale", type=float, default=1.0, help="resize factor; 1.0 keeps production parity")
    extract.add_argument(
        "--blur", type=int, default=21, help="Gaussian blur kernel of the colour mask (production 21; 5 at --scale 0.25)"
    )
    extract.add_argument(
        "--exclude-mosaic",
        action="store_true",
        help="post hoc sensitivity run: ignore warm pixels in the privacy mosaic of the goat / synth clips",
    )
    extract.add_argument("--only", nargs="*")
    extract.add_argument("--skip-existing", action="store_true")
    extract.set_defaults(func=_cmd_extract)

    flicker = sub.add_parser("flicker", help="compute per-second flicker features (30 fps, 480x270) for manifest clips")
    flicker.add_argument("--manifest", required=True)
    flicker.add_argument("--out", required=True)
    flicker.add_argument("--jobs", type=int, default=4)
    flicker.add_argument("--no-windowed", action="store_true", help="skip the slow windowed reference")
    flicker.add_argument("--only", nargs="*")
    flicker.add_argument("--skip-existing", action="store_true")
    flicker.set_defaults(func=_cmd_flicker)

    synth = sub.add_parser("synth", help="composite real flames onto a fire-free background (synthetic positive)")
    synth.add_argument("--background", required=True)
    synth.add_argument("--fire", required=True)
    synth.add_argument("--out", required=True)
    synth.add_argument("--fraction", type=float, default=0.01, help="flame area as a fraction of the frame")
    synth.add_argument("--anchor-x", type=float, default=0.52)
    synth.add_argument("--anchor-y", type=float, default=0.76)
    synth.add_argument("--seconds", type=float, default=20.0)
    synth.add_argument("--fire-start", type=float, default=0.0)
    synth.add_argument("--onset", type=float, default=0.0)
    synth.set_defaults(func=_cmd_synth)

    report = sub.add_parser("report", help="print recall / false-positive tables")
    report.add_argument("--manifest", required=True)
    report.add_argument("--features", required=True)
    report.add_argument("--flicker", help="directory from the `flicker` command; adds the flicker rules")
    report.add_argument("--warmup", type=int, default=1, help="leading checks of every clip that do not count (default 1)")
    report.set_defaults(func=_cmd_report)

    bench = sub.add_parser("bench", help="per-check cost of the P15 / P8 candidates on real frames (decode excluded)")
    bench.add_argument("--video", nargs="*", default=[])
    bench.add_argument("--scale", nargs="+", type=float, default=[1.0, 0.25])
    bench.add_argument("--checks", type=int, default=21)
    bench.add_argument("--repeats", type=int, default=7)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--invariance", action="store_true", help="measure the #135 motion masks instead of P15 / P8")
    bench.add_argument("--local-contrast", action="store_true", help="measure the #136 local-contrast scores")
    bench.add_argument("--global-change", action="store_true", help="measure the #138 global-change features")
    bench.add_argument("--decision", action="store_true", help="measure the #137 decision layers (no video needed)")
    bench.set_defaults(func=_cmd_bench)

    compare = sub.add_parser("compare", help="common #132 comparison table at each rule's best safe point")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--features", required=True)
    compare.add_argument("--rules", nargs="+", required=True, help="rule names (see rules.RULES_BY_NAME)")
    compare.add_argument("--warmup", type=int, default=5, help="leading checks of every clip that do not count (default 5)")
    compare.add_argument(
        "--exclude-neg", nargs="*", default=[], help="drop negative clips whose name contains one of these (sensitivity runs)"
    )
    compare.add_argument(
        "--continuous", action="store_true", help="also print the continuous-margin table and the per-hard-clip breakdown"
    )
    compare.set_defaults(func=_cmd_compare)

    decision = sub.add_parser(
        "decision", help="#137: debounce vs CUSUM / SPRT on the gate score, #130 debounce table, conformal"
    )
    decision.add_argument("--manifest", required=True)
    decision.add_argument("--features", required=True)
    decision.add_argument("--warmup", type=int, default=5)
    decision.add_argument("--label", help="title of the feature set in the report")
    decision.set_defaults(func=_cmd_decision)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
