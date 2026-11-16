"""Tables of the #137 decision-layer experiment (see ``decision.py`` for the frozen design)."""

from __future__ import annotations

import numpy as np

from tools.fire_rule_eval.decision import DEBOUNCE_SECONDS
from tools.fire_rule_eval.decision import EXTINGUISH_DEFAULT
from tools.fire_rule_eval.decision import EXTINGUISH_SECONDS
from tools.fire_rule_eval.decision import FIXED_THRESHOLDS
from tools.fire_rule_eval.decision import N_PERM
from tools.fire_rule_eval.decision import Candidate
from tools.fire_rule_eval.decision import build_candidates
from tools.fire_rule_eval.decision import chain_stream
from tools.fire_rule_eval.decision import conformal_threshold
from tools.fire_rule_eval.decision import debounce_candidate
from tools.fire_rule_eval.decision import goat_chains
from tools.fire_rule_eval.decision import goat_day
from tools.fire_rule_eval.decision import operating_point
from tools.fire_rule_eval.decision import positive_results
from tools.fire_rule_eval.decision import saturated_response
from tools.fire_rule_eval.decision import summarise
from tools.fire_rule_eval.decision import verdict
from tools.fire_rule_eval.protocol import FIXED_CAMERA_EXCLUDED
from tools.fire_rule_eval.protocol import is_timelapse
from tools.fire_rule_eval.report import HARD_NEGATIVE
from tools.fire_rule_eval.report import Clip

CURVE_DEBOUNCE = tuple(range(0, 11))
CURVE_DELTAS = {"cusum": (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6), "sprt": (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)}
CONFORMAL_ALPHAS = (0.05, 0.02, 0.01, 0.005)


def _table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join(lines)


def _fmt(x: float | None, nd: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def hard_counts(cand: Candidate, p: float, clips: list[Clip]) -> tuple[int, int, int, int]:
    """Alarmed hard negatives at ``p``: (all, total, fixed-camera alarmed, fixed-camera total)."""
    hard = [c for c in clips if c.label == HARD_NEGATIVE and c.mask.any()]
    hits = [bool((cand.alarm(c.features, p) & c.mask).any()) for c in hard]
    fixed = [h for c, h in zip(hard, hits, strict=True) if not any(k in c.name for k in FIXED_CAMERA_EXCLUDED)]
    return sum(hits), len(hits), sum(fixed), len(fixed)


def hard_lapse(cand: Candidate, p: float, clips: list[Clip]) -> tuple[int, int]:
    """Alarmed time-lapse hard negatives at ``p``."""
    hard = [c for c in clips if c.label == HARD_NEGATIVE and c.mask.any() and is_timelapse(c.name)]
    return sum(bool((cand.alarm(c.features, p) & c.mask).any()) for c in hard), len(hard)


def goat_counts(cand: Candidate, p: float, negatives: list[Clip]) -> tuple[int, int, int]:
    """Alarmed counted checks, alarmed clips, counted checks of the goat negatives at ``p``."""
    alarms = [int((cand.alarm(c.features, p) & c.mask).sum()) for c in negatives if c.mask.any()]
    return sum(alarms), sum(a > 0 for a in alarms), sum(int(c.mask.sum()) for c in negatives)


def _row(cand: Candidate, p: float, clips: list[Clip], negatives: list[Clip], off_s: float = EXTINGUISH_DEFAULT) -> list[str]:
    res = positive_results(cand, p, off_s, clips)
    s = summarise(res)
    ga, gc, gn = goat_counts(cand, p, negatives)
    h, ht, hf, hft = hard_counts(cand, p, clips)
    lapse, lapse_n = hard_lapse(cand, p, clips)
    return [
        cand.name,
        f"{p:.4f}",
        f"{ga}/{gn} ({gc} clips)",
        f"{s['real_det']}/{s['real_n']}",
        f"{s['synth_det']}/{s['synth_n']}",
        f"{s['real_recall']:.2f}",
        f"{s['real_protocol_recall']:.2f}",
        f"{s['synth_recall']:.2f}",
        f"{_fmt(s['ttfa_real'])} / {_fmt(s['ttfa_synth'])}",
        str(s["censored"]),
        f"{h}/{ht} ({hf}/{hft}; {lapse}/{lapse_n})",
    ]


MAIN_HEAD = [
    "Rule",
    "Threshold t",
    "Goat alarms (counted checks)",
    "Real clips det.",
    "Synth det.",
    "Real recall (prod. state, cold)",
    "Real recall (W=5 protocol)",
    "Synth recall (prod. state)",
    "TTFA real / synth (s)",
    "Censored (of 13)",
    "Hard all (fixed; time-lapse)",
]


def run(clips: list[Clip], label: str) -> str:  # noqa: C901
    """Full report of one feature set."""
    negatives = [c for c in clips if c.label == "neg" and c.mask.any()]
    cands = build_candidates(clips)
    by_name = {c.name: c for c in cands}
    chains = goat_chains(negatives)
    points = {c.name: operating_point(c, negatives, chains) for c in cands}
    out = [f"# Feature set: {label}\n"]
    real_n = sum(1 for c in clips if c.is_positive and c.group.startswith("commons_fire"))
    out.append(
        f"{len(negatives)} goat clips ({sum(int(c.mask.sum()) for c in negatives)} counted checks, W=5), {real_n} real + "
        f"{sum(1 for c in clips if c.is_positive and c.group == 'synth_goat')} synthetic positives, "
        f"{sum(1 for c in clips if c.label == HARD_NEGATIVE)} hard negatives; {N_PERM} random chain orders.\n"
    )

    out.append("## Frozen parameters (goat-derived)\n")
    out.append(
        _table(
            [
                "Candidate",
                "Parameters",
                "t_clip",
                "t_chain (canonical)",
                "t_safe (clip + chain + perms)",
                f"orders alarming at t_clip (of {N_PERM})",
            ],
            [
                [
                    c.name,
                    str({k: round(v, 4) if isinstance(v, float) else v for k, v in c.params.items()}),
                    f"{points[c.name].t_clip:.4f}",
                    f"{points[c.name].t_chain:.4f}",
                    f"{points[c.name].t_safe:.4f}",
                    points[c.name].perm_alarm_orders_at_clip,
                ]
                for c in cands
            ],
        )
    )

    out.append(
        "\n## Main table: every rule at its own zero-goat-alarm point (alarm iff score > t_safe)\n"
        "Production replay E=5 s, cold start.\n"
    )
    rows = [_row(c, points[c.name].t_safe, clips, negatives) for c in cands]
    out.append(_table(MAIN_HEAD, rows))

    out.append(
        "\n## Verdicts of the #137 criterion: candidate (rows) vs debounce comparator\n"
        "(columns: Settings seconds, area_relative gate)\n"
    )
    comp = {
        d: positive_results(by_name[f"ar_deb{d}s"], points[f"ar_deb{d}s"].t_safe, EXTINGUISH_DEFAULT, clips)
        for d in DEBOUNCE_SECONDS
    }
    vrows = []
    for c in cands:
        if c.family == "debounce":
            continue
        res = positive_results(c, points[c.name].t_safe, EXTINGUISH_DEFAULT, clips)
        vrows.append([c.name, *[verdict(res, comp[d])[0] + " [" + verdict(res, comp[d])[1] + "]" for d in DEBOUNCE_SECONDS]])
    out.append(_table(["Candidate", *[f"vs D={d}s (K={d + 1})" for d in DEBOUNCE_SECONDS]], vrows))
    out.append(
        "\nDetail: `clips detected cand/base, real recall cand/base, "
        "paired median TTFA cand/base (n = clips detected by both)`.\n"
    )

    out.append(
        "\n## #130: effect of FIRE_DETECTED_DEBOUNCE_SECONDS (CHECKS_PER_SECOND=1; production replay, E=5 s, cold start)\n"
    )
    rows130 = []
    for mode, thresholds in FIXED_THRESHOLDS.items():
        for d in DEBOUNCE_SECONDS:
            cand = by_name[f"{mode}_deb{d}s"]
            for p in thresholds:
                rows130.append([f"{mode} D={d}s fixed p={p}", *_row(cand, p, clips, negatives)[1:]])
            rows130.append([f"{mode} D={d}s retuned", *_row(cand, points[cand.name].t_safe, clips, negatives)[1:]])
    out.append(_table(["Setting", *MAIN_HEAD[1:]], rows130))

    out.append(
        "\n## Recall vs time to alarm (all points have zero goat alarms)\n"
        "Debounce D=0..10 retuned; sequential tests: h = t_safe + delta.\n"
    )
    crows = []
    for d in CURVE_DEBOUNCE:
        cand = by_name.get(f"ar_deb{d}s") or debounce_candidate("ar", d)
        p = points[cand.name].t_safe if cand.name in points else operating_point(cand, negatives, chains).t_safe
        s = summarise(positive_results(cand, p, EXTINGUISH_DEFAULT, clips))
        crows.append(
            [
                "debounce",
                f"D={d}s",
                f"{p:.4f}",
                f"{s['real_det'] + s['synth_det']}/13",
                f"{s['real_recall']:.2f}",
                f"{s['synth_recall']:.2f}",
                _fmt(s["ttfa_all"]),
            ]
        )
    for c in cands:
        if c.family == "debounce":
            continue
        for delta in CURVE_DELTAS[c.family]:
            p = points[c.name].t_safe + delta
            s = summarise(positive_results(c, p, EXTINGUISH_DEFAULT, clips))
            crows.append(
                [
                    c.name,
                    f"h=t+{delta:g}",
                    f"{p:.4f}",
                    f"{s['real_det'] + s['synth_det']}/13",
                    f"{s['real_recall']:.2f}",
                    f"{s['synth_recall']:.2f}",
                    _fmt(s["ttfa_all"]),
                ]
            )
    out.append(
        _table(["Rule", "Point", "Threshold", "Clips detected", "Real recall", "Synth recall", "Median TTFA (s)"], crows)
    )

    out.append(
        "\n## Fire-state flicker on real positives (state exits / re-entries) for FIRE_EXTINGUISHED_DEBOUNCE_SECONDS E\n"
    )
    erows = []
    for name in [
        "ar_deb0s",
        "ar_deb2s",
        "ar_deb3s",
        "ar_deb10s",
        "ff_deb0s",
        "ff_deb3s",
        "cusum_half",
        "cusum_q99",
        "sprt_q99",
    ]:
        c = by_name[name]
        for e in EXTINGUISH_SECONDS:
            res = [r for r in positive_results(c, points[name].t_safe, float(e), clips) if r.group.startswith("commons_fire")]
            detected = [r for r in res if r.detected]
            erows.append(
                [
                    name,
                    f"E={e}s",
                    f"{len(detected)}/{len(res)}",
                    sum(r.entries for r in res),
                    sum(r.exits for r in res),
                    sum(max(r.entries - 1, 0) for r in res),
                    f"{np.mean([r.recall for r in res]):.2f}",
                ]
            )
    out.append(
        _table(["Rule", "E", "Real clips detected", "Entries", "Exits", "Re-entries (repeat alerts)", "Real recall"], erows)
    )

    out.append(_post_hoc(cands, points, comp, clips))
    out.append("\n## Simplified P6: split-conformal threshold, leave-one-day-out on the goat clips\n")
    out.append(_conformal(clips, negatives))
    return "\n".join(out)


def _conformal(clips: list[Clip], negatives: list[Clip]) -> str:
    days = sorted({goat_day(c.name) for c in negatives})
    rows = []
    for d_s in (0, 2, 3):
        cand = debounce_candidate("ar", d_s)
        score = cand.score
        per_clip = {c.name: score(c.features)[c.mask] for c in negatives}
        for alpha in CONFORMAL_ALPHAS:
            alarmed_days = held_clips = held_alarmed = held_checks = held_alarm_checks = 0
            for day in days:
                cal = np.concatenate([v for n, v in per_clip.items() if goat_day(n) != day])
                thr = conformal_threshold(cal, alpha)
                for n, v in per_clip.items():
                    if goat_day(n) == day:
                        held_clips += 1
                        held_checks += len(v)
                        held_alarm_checks += int((v > thr).sum())
                        held_alarmed += int((v > thr).any())
                alarmed_days += int(any((v > thr).any() for n, v in per_clip.items() if goat_day(n) == day))
            cal_all = np.concatenate(list(per_clip.values()))
            thr_all = conformal_threshold(cal_all, alpha)
            s = summarise(positive_results(cand, thr_all, EXTINGUISH_DEFAULT, clips)) if np.isfinite(thr_all) else None
            rows.append(
                [
                    f"D={d_s}s",
                    alpha,
                    f"{held_alarm_checks}/{held_checks} ({held_alarm_checks / max(held_checks, 1):.4f})",
                    f"{held_alarmed}/{held_clips}",
                    f"{alarmed_days}/{len(days)}",
                    f"{thr_all:.4f}",
                    "-" if s is None else f"{s['real_det'] + s['synth_det']}/13",
                    "-" if s is None else f"{s['real_recall']:.2f}",
                ]
            )
    head = [
        "Debounce",
        "alpha",
        "Held-out alarmed checks (rate)",
        "Held-out clips alarmed",
        "Days with an alarm",
        "Threshold (all goat)",
        "Clips det. at it",
        "Real recall at it",
    ]
    return _table(head, rows) + (
        "\n\nThreshold = ceil((n+1)(1-alpha))-th smallest calibration score (per-check scores of the other days; "
        "`inf` = no finite "
        "guarantee). Nominal per-check false-alarm level alpha vs the observed held-out rate shows whether "
        "day-to-day shift breaks exchangeability."
    )


def _post_hoc(cands: list[Candidate], points: dict, comp: dict, clips: list[Clip]) -> str:
    """Sections added after the first run (post hoc, not pre-registered): glitch protection, hold, equal-hold, per clip."""
    out = ["\n## POST HOC: glitch protection (N_min) and hold of every rule at its safe point\n"]
    out.append(
        "N_min = consecutive saturated checks (score 1) needed from a quiet start; hold = checks the rule's own alarm "
        "stays on after 5 saturated checks followed by silence.\n"
    )
    resp = {c.name: saturated_response(c, points[c.name].t_safe) for c in cands}
    out.append(
        _table(
            ["Rule", "N_min", "Hold after 5 hits (checks)"],
            [[c.name, *("-" if v is None else v for v in resp[c.name])] for c in cands],
        )
    )

    out.append("\n## POST HOC: verdict against the debounce with matched glitch protection (D = N_min - 1, capped at 10)\n")
    rows = []
    for c in cands:
        if c.family == "debounce":
            continue
        n_min = resp[c.name][0]
        d = 10 if n_min is None else min(max(n_min - 1, 0), 10)
        d = max(x for x in DEBOUNCE_SECONDS if x <= d)
        res = positive_results(c, points[c.name].t_safe, EXTINGUISH_DEFAULT, clips)
        label, detail = verdict(res, comp[d])
        rows.append([c.name, f"N_min={n_min}", f"D={d}s", label, detail])
    out.append(_table(["Candidate", "N_min", "Comparator", "Verdict (E=5 both)", "Detail"], rows))

    out.append("\n## POST HOC: leave-one-day-out safety of the zero-alarm threshold (threshold from the other 9 days)\n")
    out.append(_lodo(cands, negatives_of(clips)))

    out.append("\n## POST HOC: real-clip check recall vs the extinguish hold E (cold start; hold equalisation)\n")
    es = (0, 5, 10, 20, 40)
    erows = []
    for c in cands:
        if c.family == "debounce" and c.mode == "ff":
            continue
        cells = []
        for e in es:
            s = summarise(positive_results(c, points[c.name].t_safe, float(e), clips))
            cells.append(f"{s['real_recall']:.2f}")
        erows.append([c.name, *cells])
    out.append(_table(["Rule", *[f"E={e}s" for e in es]], erows))
    out.append("\nHold-equalised verdict: candidate at E=0 against each debounce comparator at E=10.\n")
    hrows = []
    for c in cands:
        if c.family == "debounce":
            continue
        res0 = positive_results(c, points[c.name].t_safe, 0.0, clips)
        cells = []
        for d in DEBOUNCE_SECONDS:
            base = positive_results(
                next(x for x in cands if x.name == f"ar_deb{d}s"), points[f"ar_deb{d}s"].t_safe, 10.0, clips
            )
            cells.append(verdict(res0, base)[0])
        hrows.append([c.name, *cells])
    out.append(_table(["Candidate (E=0)", *[f"vs D={d}s (E=10)" for d in DEBOUNCE_SECONDS]], hrows))

    out.append(
        "\n## POST HOC: time to alarm per positive clip (s from the first check with motion information; - = not detected)\n"
    )
    names = ["ar_deb0s", "ar_deb2s", "ar_deb3s", "cusum_q99", "cusum_half", "sprt_q99"]
    by = {c.name: c for c in cands}
    per = {n: positive_results(by[n], points[n].t_safe, EXTINGUISH_DEFAULT, clips) for n in names}
    prows = []
    for i, r in enumerate(per[names[0]]):
        prows.append(
            [
                r.name.split("__", 1)[-1][:42],
                f"{r.available:.0f}",
                *[("-" if per[n][i].ttfa is None else f"{per[n][i].ttfa:.0f}") for n in names],
            ]
        )
    out.append(_table(["Clip", "Max measurable (s)", *names], prows))
    return "\n".join(out)


def negatives_of(clips: list[Clip]) -> list[Clip]:
    """Goat clips with at least one counted check."""
    return [c for c in clips if c.label == "neg" and c.mask.any()]


def _lodo(cands: list[Candidate], negatives: list[Clip]) -> str:
    """Per rule: held-out days that alarm when parameters and threshold come from the other days only (clip + chain)."""
    days = sorted({goat_day(c.name) for c in negatives})
    folds = {}
    for day in days:
        rest = [x for x in negatives if goat_day(x.name) != day]
        folds[day] = (rest, [x for x in negatives if goat_day(x.name) == day], {f.name: f for f in build_candidates(rest)})
    rows = []
    for c in cands:
        if c.family == "debounce" and c.mode == "ff":
            continue
        alarmed, ratios = [], []
        for day, (rest, held, fold) in folds.items():
            fc = fold[c.name]  # goat-derived parameters (k, gate) recomputed from the other days only
            feats, mask = chain_stream(sorted(rest, key=lambda x: x.name))
            t_rest = max(max(float(fc.score(x.features)[x.mask].max()) for x in rest), float(fc.score(feats)[mask].max()))
            t_held = max(float(fc.score(x.features)[x.mask].max()) for x in held)
            if t_held > t_rest:
                alarmed.append(day[5:])
            ratios.append(t_held / t_rest if t_rest > 0 else float("inf") if t_held > 0 else 0.0)
        rows.append([c.name, f"{len(alarmed)}/{len(days)}", ", ".join(alarmed) or "-", f"{max(ratios):.2f}"])
    return _table(["Rule", "Days alarming", "Which", "Largest held-out max / threshold"], rows)
