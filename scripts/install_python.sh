#!/bin/bash
set -e

# This script centralizes the logic for installing Python from the deadsnakes PPA.
# It sources common functions and installs a specific set of Python-related packages.

# Source the common functions library
source /tmp/scripts/common.sh

# 1. Setup PPA prerequisites
setup_ppa_prerequisites

# 2. Install Python and any other specified packages passed as arguments
apt-get install -y --no-install-recommends "$@"

# 3. Clean up build-time dependencies and caches
apt-get purge -y --auto-remove software-properties-common gnupg
apt-get clean
rm -rf /var/lib/apt/lists/*

# 4. Set python3.13 as the default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# 5. Bootstrap a modern, compatible version of pip
python3 -m ensurepip --upgrade
python3 -m pip install --no-cache-dir --upgrade pip
