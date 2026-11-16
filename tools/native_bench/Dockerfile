# Benchmark image for the native-hot-paths research (#133): Python 3.14 + the CPU requirements + profilers
# + compilers for the micro-prototypes. Build from the repository root (x86_64 and aarch64 alike):
#   docker build -f tools/native_bench/Dockerfile -t goat-native-bench .
FROM python:3.14-slim-trixie
COPY --from=denoland/deno:bin-2.5.6 /deno /usr/local/bin/deno
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg gcc g++ make curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-cpu.txt /tmp/requirements-cpu.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cpu.txt py-spy pytest pytest-asyncio pytest-mock \
    pybind11 cython setuptools wheel numba
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
