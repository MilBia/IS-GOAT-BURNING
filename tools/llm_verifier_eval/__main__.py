"""Evaluate an OpenAI-compatible vision LLM as a fire *verifier* on a labelled image set.

Reads ``LITELLM_BASE_URL`` (e.g. http://HOST:PORT/v1) and ``LITELLM_API_KEY`` from the environment; nothing is stored.
Images are NOT part of the repository: pass a manifest (JSON list of {"path", "label": "yes"|"no", "cat"}).

    python -m tools.llm_verifier_eval frames --src clips/ --label yes --cat real_fire --out frames/ --manifest set.json
    python -m tools.llm_verifier_eval run --manifest set.json --model chat --out res.jsonl --width 640
    python -m tools.llm_verifier_eval report res.jsonl
"""

import argparse
import base64
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

import cv2

# The first prompt is the production GEMINI__PROMPT plus a JSON-output instruction.
JSON_TAIL = ' Answer ONLY with JSON: {"fire": true|false, "confidence": 0-1}.'
PROMPTS = {
    "config": "Analyze this image and determine if there is a visible fire. Ignore purely digital overlays." + JSON_TAIL,
    "strict": (
        "You verify alarms from a fixed outdoor webcam that watches a large straw goat (a Christmas figure with red ribbons). "
        "Answer true ONLY if you can see real flames or burning material (glowing fire, flickering flames, burning straw). "
        "Answer false for: sunsets and warm light, orange or red autumn leaves, Christmas lights, red ribbons, "
        "straw or hay, reflections, pixelated privacy-mask blocks, text overlays, smoke-free scenes." + JSON_TAIL
    ),
    "minimal": "Is there fire (visible flames) in this photo?" + JSON_TAIL,
    # Single-token answer; the score is P(yes) / (P(yes) + P(no)) from first-token logprobs, which allows a threshold sweep.
    "yesno": "Is there a visible fire (flames) in this image? Answer with exactly one word: yes or no.",
}


def encode(path: str, width: int) -> str:
    """Resize to `width` and return a JPEG data URI."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if width and img.shape[1] != width:
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def call(base: str, key: str, model: str, uri: str, text: str, logprobs: bool, timeout: float, max_tokens: int) -> dict:
    """One chat completion. The image goes FIRST so servers with prefix caching can reuse its encoding across prompts."""
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": text}]}
        ],
    }
    if logprobs:
        body.update(logprobs=True, top_logprobs=10)
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        return {"error": str(e)[:200], "latency": time.time() - t0}
    out = {"latency": time.time() - t0, "content": d["choices"][0]["message"].get("content") or "", "usage": d.get("usage", {})}
    lp = (d["choices"][0].get("logprobs") or {}).get("content")
    if lp:
        out["top"] = [(t["token"].strip().lower(), t["logprob"]) for t in lp[0].get("top_logprobs", [])]
    return out


def parse(content: str) -> tuple[bool | None, float | None]:
    """Extract (fire, confidence) from a JSON-ish answer; (None, None) when unparsable."""
    m = re.search(r"\{.*?\}", content, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            fire = j.get("fire")
            if isinstance(fire, str):
                fire = fire.strip().lower() == "true"
            conf = j.get("confidence")
            return (bool(fire) if fire is not None else None), (float(conf) if isinstance(conf, (int, float)) else None)
        except ValueError:
            pass
    w = content.strip().lower()
    if w.startswith("yes"):
        return True, None
    if w.startswith("no"):
        return False, None
    return None, None


def yes_score(top: list | None) -> float | None:
    """P(yes) / (P(yes) + P(no)) from first-token logprobs."""
    if not top:
        return None
    y = sum(math.exp(v) for k, v in top if k == "yes")
    n = sum(math.exp(v) for k, v in top if k == "no")
    return y / (y + n) if y + n > 0 else None


def run(a: argparse.Namespace) -> None:
    """Query every (image, prompt, repeat) and append JSON lines; resumable."""
    base, key = os.environ["LITELLM_BASE_URL"], os.environ.get("LITELLM_API_KEY", "")
    items = json.loads(Path(a.manifest).read_text())
    done = set()
    if os.path.exists(a.out):
        for line in Path(a.out).read_text().splitlines():
            r = json.loads(line)
            done.add((r["path"], r["prompt"], r["rep"], r["width"]))
    stab = {items[i]["path"] for i in range(0, len(items), max(1, len(items) // a.stability_n))} if a.repeats > 1 else set()
    with open(a.out, "a") as fo:
        for n, it in enumerate(items):
            uri = None
            for rep in range(a.repeats if it["path"] in stab else 1):
                for pn in a.prompts:
                    if (it["path"], pn, rep, a.width) in done:
                        continue
                    uri = uri or encode(it["path"], a.width)
                    r = call(base, key, a.model, uri, PROMPTS[pn], pn == "yesno", a.timeout, 4 if pn == "yesno" else 60)
                    r.update(
                        path=it["path"], label=it["label"], cat=it["cat"], prompt=pn, rep=rep, model=a.model, width=a.width
                    )
                    if "content" in r:
                        r["fire"], r["conf"] = parse(r["content"])
                        r["score"] = yes_score(r.get("top")) if pn == "yesno" else None
                    fo.write(json.dumps(r) + "\n")
                    fo.flush()
            print(f"{n + 1}/{len(items)}", file=sys.stderr, flush=True)


def frames(a: argparse.Namespace) -> None:
    """Extract evenly spaced frames from every video in a directory and append them to a manifest."""
    os.makedirs(a.out, exist_ok=True)
    rows = json.loads(Path(a.manifest).read_text()) if os.path.exists(a.manifest) else []
    for name in sorted(os.listdir(a.src)):
        cap = cv2.VideoCapture(os.path.join(a.src, name))
        fps, total = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first, last = int(a.start * fps), total - 1
        for i in range(a.per_clip):
            idx = first + int((last - first) * (i + 0.5) / a.per_clip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                continue
            img = cv2.resize(img, (a.width, round(img.shape[0] * a.width / img.shape[1])), interpolation=cv2.INTER_AREA)
            path = os.path.join(a.out, f"{os.path.splitext(name)[0][:24]}_{idx / fps:05.1f}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({"path": path, "label": a.label, "cat": a.cat})
    Path(a.manifest).write_text(json.dumps(rows, indent=1))


def wilson(k: int, n: int) -> tuple[float, float]:
    """95 % Wilson interval for k/n."""
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def fmt(k: int, n: int) -> str:
    """Format k/n as a percentage with its Wilson interval."""
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% ({k}/{n}; {100 * lo:.0f}-{100 * hi:.0f})" if n else "-"


def _alarms(rows: list[dict]) -> int:
    return sum(bool(r.get("fire")) for r in rows)


def _table(sel: list[dict]) -> None:
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in sel:
        cats[r["cat"]].append(r)
    print("| category | truth | alarm rate (95% CI) |\n|---|---|---|")
    for c, rs in sorted(cats.items()):
        print(f"| {c} | {'fire' if rs[0]['label'] == 'yes' else 'no fire'} | {fmt(_alarms(rs), len(rs))} |")
    pos = [r for r in sel if r["label"] == "yes" and not r["cat"].startswith("synth")]
    neg = [r for r in sel if r["label"] == "no"]
    goat = [r for r in neg if r["cat"].startswith("goat")]
    print(
        f"\nrecall on non-synthetic fire {fmt(_alarms(pos), len(pos))}; FPR all negatives {fmt(_alarms(neg), len(neg))}; "
        f"FPR goat frames {fmt(_alarms(goat), len(goat))}\n"
    )


def _sweep(sel: list[dict]) -> None:
    sc = [r for r in sel if r.get("score") is not None]
    pos = [r for r in sc if r["label"] == "yes" and not r["cat"].startswith("synth")]
    neg = [r for r in sc if r["label"] == "no"]
    goat = [r for r in neg if r["cat"].startswith("goat")]
    print("threshold sweep on P(yes):\n| thr | recall (non-synth fire) | FPR all neg | FPR goat |\n|---|---|---|---|")
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99):
        cols = [fmt(sum(r["score"] >= t for r in g), len(g)) for g in (pos, neg, goat)]
        print(f"| {t} | " + " | ".join(cols) + " |")
    print()


def report(a: argparse.Namespace) -> None:
    """Print Markdown tables: alarm rate per prompt and category, latency, parse failures, stability."""
    rows = [json.loads(line) for line in Path(a.results).read_text().splitlines() if line]
    ok = [r for r in rows if "error" not in r]
    print(f"{len(rows)} calls, {len(rows) - len(ok)} transport errors\n")
    for width, model, pn in sorted({(r["width"], r["model"], r["prompt"]) for r in ok}):
        sel = [r for r in ok if (r["width"], r["model"], r["prompt"]) == (width, model, pn) and r["rep"] == 0]
        lat = sorted(r["latency"] for r in sel)
        bad = sum(r.get("fire") is None for r in sel)
        print(
            f"### {model} | width {width} | prompt `{pn}` | median {statistics.median(lat):.1f}s, "
            f"p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s | unparsable {bad}/{len(sel)}"
        )
        _table(sel)
        if pn == "yesno":
            _sweep(sel)
    grp = defaultdict(list)
    for r in ok:
        grp[(r["model"], r["prompt"], r["path"], r["width"])].append(r.get("fire"))
    multi = [v for v in grp.values() if len(v) > 1]
    if multi:
        flips = sum(len(set(v)) > 1 for v in multi)
        print(f"stability: {len(multi)} (image, prompt) groups repeated, {flips} with differing answers")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(required=True)
    r = s.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--width", type=int, default=640, help="resize before sending (0 = original)")
    r.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    r.add_argument("--repeats", type=int, default=1, help="repeats on a stability subset")
    r.add_argument("--stability-n", type=int, default=20, help="size of the stability subset")
    r.add_argument("--timeout", type=float, default=300)
    r.set_defaults(f=run)
    f = s.add_parser("frames")
    f.add_argument("--src", required=True, help="directory with videos")
    f.add_argument("--label", required=True, choices=["yes", "no"])
    f.add_argument("--cat", required=True, help="category name used in the report, e.g. real_fire, goat_neg, hard_neg")
    f.add_argument("--per-clip", type=int, default=3)
    f.add_argument("--start", type=float, default=1.0, help="skip the first N seconds")
    f.add_argument("--width", type=int, default=1280)
    f.add_argument("--out", required=True)
    f.add_argument("--manifest", required=True, help="created or appended to")
    f.set_defaults(f=frames)
    q = s.add_parser("report")
    q.add_argument("results")
    q.set_defaults(f=report)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
