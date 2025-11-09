#!/bin/bash
set -e

# This script centralizes the logic for setting up the GPU builder environment.
# It installs system dependencies, adds the Python PPA, and bootstraps pip
# with pinned versions for critical build dependencies.

# 1. Validate that version arguments for pip packages are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: SETUPTOOLS_VERSION and NUMPY_VERSION must be provided as arguments." >&2
    exit 1
fi
SETUPTOOLS_VERSION=$1
NUMPY_VERSION=$2

# Source the common functions library
source /tmp/scripts/common.sh

# 2. Setup PPA prerequisites
setup_ppa_prerequisites

# 3. Install Python development packages and build tools
apt-get install -y --no-install-recommends \
    python3.13-full python3.13-dev \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev wget unzip curl

# 4. Clean up build-time dependencies and caches (build tools like wget/curl
#    are cleaned up later in the Dockerfile after they are used).
apt-get purge -y --auto-remove software-properties-common gnupg
apt-get clean
rm -rf /var/lib/apt/lists/*

# 5. Set python3.13 as the default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# 6. Bootstrap pip and install pinned Python build dependencies
python3 -m ensurepip --upgrade
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir \
    setuptools=="${SETUPTOOLS_VERSION}" \
    numpy=="${NUMPY_VERSION}"
