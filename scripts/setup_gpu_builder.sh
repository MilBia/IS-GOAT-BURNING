#!/bin/bash
set -e

# This script centralizes the logic for setting up the GPU builder environment.
# It installs system dependencies, adds the Python PPA, and bootstraps pip
# with pinned versions for critical build dependencies.

# Ensure version arguments are passed, with sensible defaults
SETUPTOOLS_VERSION=${1:-"75.8.0"}
NUMPY_VERSION=${2:-"2.3.3"}

# 1. Install system dependencies for adding PPA and building OpenCV
apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates

# 2. Add the deadsnakes PPA and install Python + build tools
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
    python3.13-full python3.13-dev \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev wget unzip curl

# 3. Clean up build-time dependencies and caches
apt-get purge -y --auto-remove software-properties-common gnupg
apt-get clean
rm -rf /var/lib/apt/lists/*

# 4. Set python3.13 as the default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# 5. Bootstrap pip and install pinned Python build dependencies
python3 -m ensurepip --upgrade
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir \
    setuptools=="${SETUPTOOLS_VERSION}" \
    numpy=="${NUMPY_VERSION}"
