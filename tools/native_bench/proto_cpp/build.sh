#!/bin/bash
# Build the pybind11 extension in place (run inside the goat-native-bench image). Usage: build.sh <outdir> [extra g++ flags]
set -e
out=$1; shift
mkdir -p "$out"
g++ -O3 -shared -std=c++17 -fPIC "$@" $(python -m pybind11 --includes) cpp_fused.cpp -o "$out/cpp_fused$(python3-config --extension-suffix)"
