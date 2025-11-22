#!/bin/bash
set -e

# This script sets up the GPU builder environment. It uses functions from common.sh
# for the base Python setup and then installs additional build dependencies.

# 1. Validate version arguments for build tools.
if [ "$#" -ne 2 ]; then
    echo "Error: This script requires exactly two arguments: SETUPTOOLS_VERSION and NUMPY_VERSION." >&2
    exit 1
fi
SETUPTOOLS_VERSION=$1
NUMPY_VERSION=$2

# 2. Source the shared functions using a robust method to find the script's directory.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/common.sh"

# 3. Install PPA prerequisites.
install_base_dependencies_and_ppa

# 4. Install Python variants required for building.
echo "Installing Python build variants..."
apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-full \
    python3.13-dev \
    python3.13-venv

# 5. Finalize the Python installation (set default, bootstrap pip).
finalize_python_setup "python3.13"

# 6. Install additional build dependencies for OpenCV.
echo "Installing OpenCV build dependencies..."
apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev wget unzip curl

# 7. Install pinned Python build dependencies.
echo "Installing pinned Python build dependencies..."
python3 -m pip install --no-cache-dir \
    setuptools=="${SETUPTOOLS_VERSION}" \
    numpy=="${NUMPY_VERSION}"
