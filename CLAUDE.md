# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Monitors the Gävle Goat YouTube webcam feed for fire and sends email/Discord alerts. Pipeline: resolve a YouTube live URL with `yt-dlp` → grab frames with OpenCV → analyze each (throttled) frame with a fire detector → debounce → fire an idempotent notification action. Optionally archives video around a fire event.

## Commands

```bash
# Run the app (reads .env; see .env.example)
python burning_goat_detection.py

# Tests — PYTHONPATH=. is REQUIRED so the is_goat_burning package resolves
PYTHONPATH=. pytest
PYTHONPATH=. pytest tests/fire_detection/test_detectors.py          # single file
PYTHONPATH=. pytest tests/fire_detection/test_detectors.py::test_x  # single test
PYTHONPATH=. pytest --cov=is_goat_burning --cov-report=term-missing # with coverage

# Lint / format (must pass before commit; CI runs both)
ruff check .
ruff format --check .        # add --fix / drop --check to apply

# Install dev env
pip install -e .[dev,cpu] && pre-commit install

# GitHub operations — use the `gh` CLI (issues, PRs, CI status)
gh issue view <n>            # read an issue before starting work
gh pr create --fill         # open a PR for the current branch
gh pr diff <n>              # inspect a PR's diff
gh pr checks <n>            # check CI status
gh run view <run-id> --log  # inspect a failed CI run
```

Tests auto-load `.env.tests` first (then `.env`) via `SettingsConfigDict(env_file=(".env.tests", ".env"))`, so the suite runs against mock config with all external I/O disabled. Async tests use explicit `@pytest.mark.asyncio` markers (no global asyncio mode configured).

Docker builds are multi-stage — select with `--target`: `cpu`, `gpu` (needs `--build-arg CUDA_ARCH=<compute_capability>`), `opencl`, and matching `*-test` stages. See README for the full matrix. `gpu`/`opencl` build OpenCV from source: keep `OPENCV_VERSION`/`OPENCV_CONTRIB_VERSION` (Dockerfile, `scripts/build_opencv.sh`) at the `opencv-python` major.minor — the `docker-cpu` CI job checks this and runs `cpu-test` with `--network none` (#214).

## Architecture

All I/O is `asyncio`-based; every new networking/file operation must be non-blocking (blocking OpenCV/`yt-dlp` calls are offloaded via `asyncio.to_thread` / `run_in_executor`). Major concerns are isolated by directory — respect that separation.

**Control flow** (`burning_goat_detection.py` → `is_goat_burning/`):
- `app.py` `Application` is the orchestrator: builds notification actions from settings, runs `StreamFireDetector` in a reconnect loop (`asyncio.TaskGroup` + `except*`), and drains a fire-event queue through an action manager. Keep business logic out of here.
- `fire_detection/detector_core.py` `StreamFireDetector` owns the detect loop: throttles frames to `checks_per_second`, calls the detector, and applies **debounce** state (`fire_detected_debounce_seconds` / `fire_extinguished_debounce_seconds`) before signaling.
- `fire_detection/detectors.py` — `create_fire_detector()` factory picks the implementation from `settings.detection_strategy`:
  - `classic` → local CV detector (`CPUFireDetector`, or `CUDAFireDetector`/`OpenCLFireDetector` per `settings.cuda`/`settings.open_cl`). HSV color mask **AND** temporal motion mask (frame differencing) to reject static warm-colored objects.
  - `gemini` → `GeminiFireDetector` (google-genai, structured JSON output; call timeout `GEMINI__TIMEOUT_SECONDS`, default 10 s).
  - `hybrid_cloud` → local CV as a cheap gatekeeper; only escalates to Gemini when CV fires, filtering false positives while minimizing API calls. The legacy value `hybrid` is auto-migrated to `hybrid_cloud` by a validator in `Settings`.
  - `gemini_decision` / `hybrid_decision` → `DecisionFireDetector` (`decision_detector.py`): Gemini Decisions API (`gemini-decision-1`, `v1beta`, needs google-genai ≥ 3.2) asks a `noul` yes/no question and thresholds the returned `P(true)` (`DECISION__*` settings, key shared with `GEMINI__API_KEY`). API errors are logged at ERROR and escalate to CRITICAL after 5 in a row (same for `GeminiFireDetector`). `hybrid_decision` reuses `HybridFireDetector` with CV as gatekeeper.
  - `hybrid_edge` → `HybridFireDetector` with local CV as gatekeeper and `EdgeInferenceDetector` (`edge_detector.py`, offline fire/no-fire image classifier via cv2.dnn, onnxruntime or OpenVINO — `EDGE__ACCELERATOR=openvino`/`openvino_gpu`, optional `openvino` extra, never picked by `auto`; `OpenVinoBackend` blocks `openvino_telemetry` via `sys.modules` before importing openvino so no telemetry is sent; likewise onnxruntime is only imported via `_import_onnxruntime()`, which sets `ORT_DISABLE_TELEMETRY=1` before the import) as verifier, configured via the `EDGE__*` settings (`EDGE__MODEL_PATH` is required, enforced by a `Settings` model validator). A missing/unset `EDGE__MODEL_PATH` is a hard startup error (fail fast). If the model cannot be LOADED (corrupt/invalid model, input-size mismatch, unavailable accelerator, missing onnxruntime/openvino, any other error), the factory logs CRITICAL and **falls back to the local detector (classic rule; honours CUDA/OpenCL settings)** wrapped in `UnverifiedLocalDetector`, which applies `VERIFIER_FAILURE_POLICY` to every check (`alert_unverified`: fires reported with `last_result_unverified=True`; `suppress`: no fire) instead of failing startup. The fallback publishes `VerifierStatus(available=False, verifier="EdgeInferenceDetector", policy=<VERIFIER_FAILURE_POLICY>)` to `verifier_health` (and a later successful build publishes the recovery), so the service channel never reports a healthy verifier during a fallback. The built verifier is cached across factory calls (stream reconnects) keyed by model path+mtime+size and the EDGE settings (`clear_edge_verifier_cache()` resets it); failures are never cached, and a verifier whose last inference failed (`needs_rebuild`) is rebuilt.
  All detectors conform to the `FireDetector` protocol: `async detect(frame) -> tuple[bool, annotated_frame]`.
  **Escalation log (#175):** `fire_detection/escalation_log.py` (`ESCALATION_LOG__*`, off by default). `HybridFireDetector` hands each actual verifier call (not cooldown reuses) to `EscalationLog.record()`: JPEG + `escalations.jsonl` line (verdict `fire`/`no_fire`/`error`, `last_probability` of the Decision/Edge verifiers, latency, error class name only, policy); hourly-cap skips are index-only `capped` lines. Writes run in `asyncio.to_thread` with at most 4 in flight (extra records dropped), never raise into the detect loop, failures are logged rate-limited; oldest JPEGs are pruned from a running total (scanned once). `Application.shutdown` drains pending writes via `close_escalation_log()`.

  **Verifier errors (#169):** the verifiers (`GeminiFireDetector`, `DecisionFireDetector`, `EdgeInferenceDetector`) implement the `FireVerifier` protocol `async verify(frame) -> bool`, which raises `VerifierError` (`verifier_health.py`) when no verdict could be obtained (API error, invalid key, quota, timeout, unparseable answer, failed inference). Their `detect()` wraps `verify()` and never raises (standalone `gemini`/`gemini_decision`: an error is "no fire" + logs; after 5 consecutive errors `detect()` also publishes `VerifierStatus(available=False, policy="no_fire")` and the next verdict the recovery). `HybridFireDetector` calls `verify()`: a positive verdict is reused for `VERIFIER_COOLDOWN_SECONDS`; a negative one is not reused while the last positive is at most cooldown + `FIRE_EXTINGUISHED_DEBOUNCE_SECONDS` old (#188) nor for `FIRE_EXTINGUISHED_DEBOUNCE_SECONDS` after the first negative following a reported fire (verified/reused positive or `alert_unverified` policy result; later negatives don't extend it, a positive clears it — #191), otherwise for the cooldown; an error is never cached for `VERIFIER_COOLDOWN_SECONDS` (re-asked on the next check) and, below the threshold, is inconclusive (`last_result_inconclusive=True`; `StreamFireDetector` then neither starts nor resets either debounce); after `VERIFIER_ERRORS_BEFORE_POLICY` consecutive errors, or at `VERIFIER_MAX_CALLS_PER_HOUR`, `VERIFIER_FAILURE_POLICY` applies — `alert_unverified` (default) reports the CV fire with `last_result_unverified=True`; `StreamFireDetector` calls `on_fire_action(unverified=True)` only if no fire check of the debounce window was verified → `FIRE_UNVERIFIED` queue event → `OnceAction(unverified=True)` → actions append an `[UNVERIFIED]` note (text: `UNVERIFIED_ALERT_NOTE`); `suppress` reports no fire (so an ongoing fire ends after the extinguish debounce). Entering/leaving that state logs CRITICAL/INFO once and calls `publish_verifier_status()`; subscribe with `add_verifier_status_listener()` (sync, must not block). Other exceptions from `verify()` still propagate.
- `stream.py` — `YouTubeStream` resolves the stream URL (preferred format with fallback to `best`; yt-dlp gets a `logger` adapter so its own WARNING/ERROR lines go to the app logger at DEBUG with URLs masked, #193); `VideoStreamer` opens the capture, chooses backend (cpu/cuda/opencl), yields frames, and feeds each raw frame to the video saver. Every blocking call on the capture after `create()` (read/grab/retrieve/release) runs on the streamer's own single-thread executor, so `stop()` (idempotent) releases only after an in-flight read returns (#201: release under a read segfaulted on shutdown); it waits `_RELEASE_TIMEOUT_SECONDS` (5 s), then logs a WARNING, leaves the release queued on the worker and still stops the saver.
- `on_fire_actions/` — each notification is a class with `async def __call__(self, *, unverified: bool = False)`, configured entirely via `__init__` (never reads env itself); `OnceAction` passes `unverified=True` only for alerts sent while the verifier is unavailable, so every alert action must accept that keyword. All actions are wrapped by `OnceAction` (`action_containers.py`), which sends one alert per fire event: it re-arms on the detector's EXTINGUISHED event (queued to the action manager; also queued on reconnect if the stream dropped mid-fire) and keeps `ALERT_COOLDOWN_SECONDS` between alerts (a fire re-detected inside the window is deferred to its end, cancelled if extinguished first). An action reports failure by raising or returning `False`; the alert counts as sent only if at least one action succeeded. If all fail it stays armed, but since the detector only signals on a no-fire→fire transition, a retry happens only after re-detection (extinguish → fire) or a stream reconnect. `stop()` (shutdown) cancels a waiting deferral and gives an in-flight send up to 5 s before cancelling it.
- `stream_recording/` — `AsyncVideoChunkSaver` + a Strategy pattern (`strategies.py`): `MemoryBufferStrategy` (default; keeps compressed pre-fire frames in RAM, flushes on fire) vs `DiskBufferStrategy` (continuous chunk writes). Don't alter the saver's state machine unless asked.
- `service_notifier.py` — `ServiceNotifier` ("sign of life", #179): service messages to the separate `DISCORD__SERVICE_HOOKS` list (empty → disabled no-op; never the alert hooks): startup, heartbeat every `DISCORD__SERVICE_HEARTBEAT_INTERVAL_SECONDS`, one warning per stream outage longer than `DISCORD__SERVICE_STREAM_DOWN_WARNING_SECONDS` plus its recovery, verifier unavailable/recovered (a `verifier_health` listener), best-effort bounded shutdown. Event methods are sync and non-blocking (bounded queue, own task, failures only logged); `app.py` only wires them — the reconnect loop reports `stream_connected()` / `stream_failed(<exception type>)`.
- `signal_handler.py` — `SignalHandler` is a singleton (SIGINT/SIGTERM graceful exit) that also carries the fire-detected/-extinguished `asyncio.Event`s coordinating detector and video saver.

## Conventions

- **Config is centralized in `is_goat_burning/config.py`** (Pydantic `pydantic-settings`), exposed as the module-level `settings` singleton. This is the single source of truth — do **not** call `os.environ` / read env vars elsewhere. Sole exception: setting a third-party library's telemetry opt-out switch right before importing it (e.g. `ORT_DISABLE_TELEMETRY` in `edge_detector._import_onnxruntime()`). Env vars use `__` as the nested delimiter, e.g. `EMAIL__SENDER`, `VIDEO__BUFFER_MODE`, `GEMINI__API_KEY`, `EDGE__MODEL_PATH`. Cross-field rules live in `Settings` validators (`field_validator` / `model_validator`) — put new ones there rather than in consumers. Config tests must be isolated from a developer's local `.env` (see `tests/` for the pattern from #74).
- **Adding a config option:** add it with a default to `config.py` (in the right nested model), document it in `.env.example`, add a value to `.env.tests`, then import from `settings`.
- **Dependencies:** `pyproject.toml` is the single source of truth. `requirements*.txt` are generated by `pip-tools` — never edit them by hand. Regenerate:
  ```bash
  pip-compile --output-file=requirements.txt pyproject.toml
  pip-compile --extra=cpu --output-file=requirements-cpu.txt pyproject.toml
  pip-compile --extra=dev --extra=cpu --output-file=requirements-dev.txt pyproject.toml
  pip-compile --extra=edge-gpu --output-file=requirements-edge-gpu.txt pyproject.toml   # optional: onnxruntime-gpu for hybrid_edge on CUDA, bare-metal only (needs CUDA 13 libs; not usable in the Docker `gpu` image)
  pip-compile --extra=openvino --unsafe-package=openvino-telemetry --output-file=requirements-openvino.txt pyproject.toml   # optional: OpenVINO for hybrid_edge (EDGE__ACCELERATOR=openvino/openvino_gpu), bare-metal only; leaves openvino-telemetry out (install with `pip install --no-deps -r requirements-openvino.txt`)
  ```
  Every new or updated dependency must pass `PYTHONPATH=. pytest tests/security` (#210: local inference is network-blocked with no Python-level connection attempts, no device-id files in a temp `HOME`/`XDG_CACHE_HOME` and no leftover processes; the `no-outbound` CI job installs the optional OpenVINO/onnxruntime backends for it — see CONTRIBUTING).
  Dependabot (`.github/dependabot.yml`: pip, github-actions, pre-commit) bumps `pyproject.toml` and also transitive pins in `requirements*.txt` (#209) — regenerate all of them on its PRs (procedure in CONTRIBUTING); the `requirements-in-sync` CI job fails on drift, and transitive packages capped by a dependent get a Dependabot `ignore` (e.g. `tenacity`, #216). The docker ecosystem is deliberately disabled (CUDA/apt-coupled base images).
- **Style:** ruff (line length 128, double quotes, isort with `force-single-line` + `force-sort-within-sections`), Google-convention docstrings (enforced by pydocstyle in pre-commit).
- **Commits:** Conventional Commits — `<type>(<scope>): <description> (#<issue>)`, e.g. `feat(notifications): add Slack service (#42)`.
- **Tests** mirror the source tree under `tests/`; mock external services (SMTP, Discord, Gemini) with `unittest.mock` / `pytest-mock`.

`GEMINI.md` holds the fuller developer playbook (issue/PR workflow, security mandates).
