#!/bin/bash
set -e

# This script installs Python 3.13 and other specified packages by leveraging
# the shared functions in common.sh.

# 1. Source the shared functions.
SCRIPT_DIR=$(dirname "$0")
source "${SCRIPT_DIR}/common.sh"

# 2. Install PPA prerequisites.
install_base_dependencies_and_ppa

# 3. Install Python and all other packages passed as arguments.
echo "Installing Python and specified packages: $@"
apt-get install -y --no-install-recommends "$@"

# 4. Finalize the Python installation.
finalize_python_setup

# 5. Clean up apt caches. Build-time dependencies like software-properties-common
#    are intentionally kept in runtime images for simplicity and stability.
echo "Cleaning up apt caches..."
apt-get clean
rm -rf /var/lib/apt/lists/*
