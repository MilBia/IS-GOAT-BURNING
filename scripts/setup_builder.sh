#!/bin/bash
set -e

# This script sets up the builder environment for GPU or OpenCL.
# It uses functions from common.sh for the base Python setup and then installs
# additional build dependencies based on the build type.

# Default values
BUILD_TYPE=""
SETUPTOOLS_VERSION=""
NUMPY_VERSION=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --type requires a value (gpu or opencl)." >&2
                exit 1
            fi
            BUILD_TYPE="$2"
            shift 2
            ;;
        *)
            # Assume positional arguments for versions if not flagged
            if [[ -z "$SETUPTOOLS_VERSION" ]]; then
                SETUPTOOLS_VERSION="$1"
            elif [[ -z "$NUMPY_VERSION" ]]; then
                NUMPY_VERSION="$1"
            else
                echo "Unknown parameter passed: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$BUILD_TYPE" ]]; then
    echo "Error: --type argument is required." >&2
    exit 1
fi

if [[ "$BUILD_TYPE" != "gpu" && "$BUILD_TYPE" != "opencl" ]]; then
    echo "Error: Invalid build type '$BUILD_TYPE'. Must be 'gpu' or 'opencl'." >&2
    exit 1
fi

if [[ -z "$SETUPTOOLS_VERSION" ]] || [[ -z "$NUMPY_VERSION" ]]; then
    echo "Error: SETUPTOOLS_VERSION and NUMPY_VERSION arguments are required." >&2
    exit 1
fi

# Source the shared functions
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/common.sh"

# Install PPA prerequisites
install_base_dependencies_and_ppa

# Install Python variants required for building
echo "Installing Python build variants..."
apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-full \
    python3.13-dev \
    python3.13-venv

# Finalize the Python installation
finalize_python_setup "python3.13"

# Install common build dependencies
echo "Installing common build dependencies..."
COMMON_DEPS=(
    build-essential cmake git pkg-config
    libjpeg-dev libpng-dev libtiff-dev
    libavcodec-dev libavformat-dev libswscale-dev
    libv4l-dev libxvidcore-dev libx264-dev
    libgtk-3-dev wget unzip
)

# Add curl for GPU build (it was present in setup_gpu_builder.sh)
if [[ "$BUILD_TYPE" == "gpu" ]]; then
    COMMON_DEPS+=(curl)
fi

apt-get install -y --no-install-recommends "${COMMON_DEPS[@]}"

# Install type-specific dependencies
if [[ "$BUILD_TYPE" == "opencl" ]]; then
    echo "Installing OpenCL build dependencies..."
    apt-get install -y --no-install-recommends \
        ocl-icd-opencl-dev opencl-headers clinfo
fi

# Install pinned Python build dependencies
echo "Installing pinned Python build dependencies..."
python3 -m pip install --no-cache-dir \
    setuptools=="${SETUPTOOLS_VERSION}" \
    numpy=="${NUMPY_VERSION}"
