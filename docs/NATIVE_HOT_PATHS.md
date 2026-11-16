# Rewriting hot paths in a compiled language? (research, #133)

**Verdict: do not rewrite anything in Rust/C++/Cython.** Almost all the CPU is spent in code that is already native
(FFmpeg H.264 decode, libjpeg-turbo, OpenCV). The detector's own cost is a few percent of a core. The measurements
instead found three much bigger, Python-level problems (section 4), the worst of which makes a Raspberry Pi 4 run out
of memory.

Everything here was measured in containers with Python 3.14 (OpenCV 5.0.0, numpy 2.5) on x86_64 (8 cores) and a
Raspberry Pi 4 (4 cores, 4 GB, `ondemand` governor). Input is a 60 s 1080p/30 fps H.264 clip of the public live stream
`Shj6eQEMR_E` (forest; the Gävle stream was offline), `CHECKS_PER_SECOND=1`, classic detector. Tools: `tools/native_bench/`.

## 1. End-to-end cost (real `StreamFireDetector`, paced to real time)

| Mode | x86_64 CPU / peak RSS | RPi4 CPU / peak RSS | RPi4 frames processed |
|---|---|---|---|
| `classic`, no archiving | 0.50 core / 250 MB | 1.01 core / 202 MB | 2700/2700 |
| + `VIDEO__BUFFER_MODE=memory` (60 s) | 1.07 core / 1.7 GB | 1.46 core / 1.6 GB | **2054/3000 (68 %)** |
| memory, fire flush at 70 s | 1.33 core / 2.4 GB | 1.38 core / 1.7 GB | 1931/3000 |
| + `VIDEO__BUFFER_MODE=disk` | 1.21 core / 335 MB | 1.61 core / **3.4 GB** | **1580/2700 (59 %)** |

Event-loop lag with memory archiving: mean 7 ms / max 31 ms (x86), mean 30 ms / max 213 ms (RPi4), versus under 1 ms
without it. Not measured: `hybrid_*` end to end. The verifier only runs when the colour+motion rule fires, so it
adds nothing in the steady state; its inference latency is in #119/#101.

## 2. Where the CPU goes

* **Decode dominates.** Of the 0.50 core (x86) roughly 0.4 is FFmpeg worker threads; the detector thread is about
  0.04 core. On the RPi, `cap.read()` costs 29.5 ms CPU per 1080p frame, 0.88 core at 30 fps.
* **Detector** (per check, ms; RPi4 4 threads / 1 thread, x86 8 threads / 1 thread): whole `_detect_logic` 43 / 102
  and 19 / 52. Gaussian blur 21×21 is 42-65 % of it. At 1 check/s that is about 4 % of a core on the RPi.
  `bitwise_and(frame, frame, mask)` builds an annotated frame nobody reads unless `VIDEO_OUTPUT` is on: 7 ms/check on the RPi.
  The Python overhead around it (`asyncio.to_thread` hop) is 0.1-0.2 ms.
* **JPEG for the memory buffer** is paid on *every* stream frame, synchronously on the event-loop thread:
  18 ms (x86) / 37 ms (RPi4) per 1080p frame at the default quality 95 (about 820 KB each for this clip). Lowering the
  quality hardly helps (q70: 13 / 30 ms), so the 33 ms frame budget is exceeded on the RPi.

## 3. Native prototypes of the fused colour+motion kernel

One pass doing OpenCV-exact BGR2HSV, `inRange`, BGR2GRAY, abs-diff threshold and both counts (the part after the blur).
Verified bit-exact against OpenCV on real and adversarial frames (note: OpenCV 5's gray is
`(b*3735 + g*19235 + r*9798 + 16384) >> 15`).

| Post-blur stage, 1080p, ms | x86_64 | RPi4 |
|---|---|---|
| OpenCV, 1 thread | 26-30 | 44 |
| OpenCV, all threads | 15-18 | 30.7 |
| C++/pybind11, scalar `-O3` | ≈ OpenCV 1-thread (1.08×) | not run |
| Rust/pyo3, 1 thread | 26 (1.12×) | 34.9 (1.28×) |
| Rust/pyo3, multi-threaded | 9.5-10.4 (1.7× vs OpenCV all threads) | 12.7 (2.4×) |

* Single-threaded, a native kernel is only 10-30 % faster; the gain comes from using several cores.
* Best case saving on the RPi: about 18 ms per check, i.e. under 2 % of a core at 1 check/s. The blur (the biggest
  stage) stays in OpenCV, and the kernel needs a host `np.ndarray`, so it does not apply to the OpenCL (`UMat`) or
  CUDA (`GpuMat`) paths without a download.
* Timing is data dependent (early exit on S/V); fire-heavy frames were not measured.
* Packaging: Rust wheel is abi3 (`cp311-abi3`, works on 3.14), about 239 KB per architecture, `manylinux_2_34`.
  Build 27 s cold on x86, 74 s on the RPi; toolchain adds about 650 MB to a build stage. pyo3 0.30 supports Python 3.14.
  Free-threaded builds, `manylinux_2_28` and CI integration were not tried.

**Amdahl:** removing the *entire* detector would save about 4 % of the RPi's 1.0 core. A native rewrite of the
glue/kernel can win at most 1-2 %. Not worth a second toolchain, per-architecture wheels and a build stage in the
`cpu`/`gpu`/`opencl` images.

## 4. What is worth doing instead (suggested follow-up issues)

1. **`grab()` instead of `read()` for frames that are not analysed** (when archiving is off). Decode CPU per frame,
   `read` → `grab` + `retrieve` every 30th: RPi4 1080p 29.5 → 16.4 ms (**-44 %**, 0.88 → 0.49 core), 720p
   -44 %, 480p -54 %; x86 -31 %. Pure Python change in `VideoStreamer.frames()`.
2. **Lower stream resolution** (`YTDLP_FORMAT`, config only), RPi4 decode core at 30 fps with `read`: 1080p 0.88,
   720p 0.38, 480p 0.22, 360p 0.11. The cost is sensitivity to small fires (#108), so decide after #109.
3. **Archiving does not fit the RPi4 at 1080p/30:** per-frame JPEG on the event loop (memory mode), and in disk mode an
   unbounded `frame_queue` of raw 6 MB frames that reached 3.4 GB of 3.8 GB. The first disk-mode run made the test Pi
   reboot (no persistent journal, so the cause is inferred from the second run's memory curve, not proven). Options:
   bound the queue, encode off the loop, downscale or drop the archive frame rate, or keep the original H.264 packets
   (about 0.24 MB/s instead of about 25 MB/s of JPEG) and mux them with `ffmpeg -c copy` on flush (feasibility not tested).
4. Skip building the annotated frame unless `VIDEO_OUTPUT` or a verifier needs it (7 ms/check on the RPi).

## 5. Criteria for revisiting a native module

Reconsider only if a profile shows **more than 20 % of process CPU in code we own** (not FFmpeg/OpenCV), or if the
flicker feature (#105) is adopted and its measured 3-6 ms/frame must run at the full frame rate. Then prefer a
multi-threaded Rust kernel (abi3 wheel per architecture); do not expect more than the numbers above.

## Limits of this study

The forest clip is a proxy for the goat stream (JPEG size and decode cost depend on content); one 30 fps clip was used
for the end-to-end matrix. `py-spy`, `cProfile` and `tracemalloc` were not used: attribution comes from per-thread CPU
time, RSS series and per-stage timings. CUDA/OpenCL were not measured.

Reproduce: `docker build -f tools/native_bench/Dockerfile -t goat-native-bench .`, then
`python -m tools.native_bench.{e2e,microbench,decode_bench,proto_bench}` inside it (see each module's docstring).
