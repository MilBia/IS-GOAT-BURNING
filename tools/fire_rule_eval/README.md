# fire_rule_eval

Offline harness for measuring the classical (colour + motion) fire rule on recorded video, built for
issue #104. It reuses the HSV range and blur settings of the production detector, so the numbers
describe what `CPUFireDetector` would do at the production check rate.

Videos are **not** kept in the repository. Pass directories on the command line.

## Workflow

```bash
export PYTHONPATH=.

# 1. Describe the data. Labels: pos (fire), neg (no fire), neg_hard (warm / flickering non-fire scenes).
python -m tools.fire_rule_eval scan --dir /data/events   --label neg      --group goat   --manifest manifest.json
python -m tools.fire_rule_eval scan --dir /data/fire     --label pos      --group fire   --manifest manifest.json
python -m tools.fire_rule_eval scan --dir /data/sunsets  --label neg_hard --group sunset --manifest manifest.json

# 2. (optional) synthetic positives: real flames composited onto a fire-free background.
python -m tools.fire_rule_eval synth --background bg.mp4 --fire flames.mp4 --fraction 0.01 --onset 5 --out synth_1pct.mp4

# 3. Heavy step, once per video: decode at the check rate and store per-check features (.npz).
python -m tools.fire_rule_eval extract --manifest manifest.json --out features/ --skip-existing

# 4. Cheap step, rerun freely: sweep rule parameters and print Markdown tables.
python -m tools.fire_rule_eval report --manifest manifest.json --features features/
```

`extract` runs at native resolution (`--scale 1.0`) for parity with production; use `--scale` only for quick looks.
Edit the manifest by hand to add `fire_from_s` / `fire_to_s` for positives where the fire is not visible for the whole clip.

## Rules

| Rule | Idea |
|---|---|
| `deployed` | Production before the first-check fix: warm AND motion(25) pixels > p x frame area. The first check after (re)start has no previous frame and counted colour only. |
| `skip_first` | As `deployed`, but never alarms on the first check. Equals current production. |
| `motion15`, `motion40` | `skip_first` with a different motion threshold. |
| `avg_motion` | Motion against the mean of the last 3 checks. |
| `area_relative` | Share of the warm region that moves, instead of share of the whole frame. |
| `tiled` | Maximum over an 8x4 tile grid of the warm-and-moving tile fraction. |
| `component`, `component_closed` | Largest connected warm-and-moving component. |
| `white_core` | Adds bright, desaturated moving pixels next to warm regions (white-hot flame cores). |

## Interpreting the report

* **Best operating point** = the most sensitive parameter with **zero** alarms on all negative checks.
  Recall is the mean over positive clips of the fraction of checks alarmed (steady state, first check dropped).
* The production default has `FIRE_DETECTED_DEBOUNCE_SECONDS=0`, so a single alarmed check triggers; hence "zero negative alarms".
* Synthetic positives are proxies. Report them separately from real footage.
* Small negative sets give optimistic false-positive estimates. Grow the negative pool before changing defaults.

## Tests

```bash
PYTHONPATH=. pytest tests/tools
```

`test_features_match_production_detector` guards parity with `CPUFireDetector`.
