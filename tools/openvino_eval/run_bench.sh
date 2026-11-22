#!/usr/bin/env bash
# Run the #152 OpenVINO bench matrix (model x variant x device) with the ov-bench-152 image, one container per run.
# Usage: run_bench.sh <work dir with ov/<model>/<variant>.xml and jpgs/> <results dir> <label, e.g. prodoff>
# Env: RENDER_GID (default: gid of the host's render group), RUNS (50), MODELS, VARIANTS, DEVICES.
# Prints `uptime` before each model batch; containers are started with --rm, nothing else is touched.
set -euo pipefail
WORK=$(realpath "$1")
OUT=$(realpath "$2")
LABEL=$3
RUNS=${RUNS:-50}
RENDER_GID=${RENDER_GID:-$(getent group render | cut -d: -f3)}
MODELS=${MODELS:-"p16_224_minimal p32_256_minimal"}
VARIANTS=${VARIANTS:-"fp32 fp16 int8w int8q"}
DEVICES=${DEVICES:-"CPU GPU"}
mkdir -p "$OUT"
for m in $MODELS; do
    echo "== $m $(uptime)" | tee -a "$OUT/uptime_$LABEL.log"
    for v in $VARIANTS; do
        for d in $DEVICES; do
            f="$OUT/${LABEL}_${m}_${v}_${d}.json"
            if docker run --rm --cpus 4 --device /dev/dri --group-add "$RENDER_GID" -v "$WORK:/work:ro" ov-bench-152 \
                --model "/work/ov/$m/$v.xml" --images /work/jpgs --device "$d" --threads 4 --runs "$RUNS" \
                >"$f" 2>"${f%.json}.err"; then
                python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], d['median_ms'], d['p95_ms'], d['max_ms'], d['peak_rss_mib'], d['load_s'], d['precision_hint'])" "$f" "$m $v $d"
            else
                echo "$m $v $d FAILED (see ${f%.json}.err)"
            fi
        done
    done
done
echo "== end $(uptime)" | tee -a "$OUT/uptime_$LABEL.log"
