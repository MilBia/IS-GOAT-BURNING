#!/bin/bash
set -e

# This script centralizes the logic for installing Python from the deadsnakes PPA.
# It takes a list of packages to install as arguments.

# 1. Install prerequisites for adding a PPA
apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates

# 2. Add the deadsnakes PPA
add-apt-repository -y ppa:deadsnakes/ppa

# 3. Install Python and any other specified packages
apt-get update
apt-get install -y --no-install-recommends "$@"

# 4. Clean up build-time dependencies and caches
apt-get purge -y --auto-remove software-properties-common gnupg
apt-get clean
rm -rf /var/lib/apt/lists/*

# 5. Set python3.13 as the default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# 6. Bootstrap a modern, compatible version of pip
python3 -m ensurepip --upgrade
python3 -m pip install --no-cache-dir --upgrade pip
