#!/bin/bash
set -e

# This script provides shared, reusable functions for setting up the Python
# environment. It is intended to be sourced by other scripts, not executed directly.

# Installs prerequisite packages for adding apt PPAs and then adds the
# deadsnakes PPA for recent Python versions.
install_base_dependencies_and_ppa() {
    echo "Updating package list and installing PPA prerequisites..."
    apt-get update
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        ca-certificates

    echo "Adding deadsnakes PPA..."
    add-apt-repository -y ppa:deadsnakes/ppa
}

# Finalizes the Python 3.13 installation by setting it as the default python3
# and bootstrapping a modern version of pip.
finalize_python_setup() {
    echo "Setting python3.13 as the default python3 alternative..."
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

    echo "Bootstrapping pip for python3.13..."
    python3.13 -m ensurepip --upgrade
    python3.13 -m pip install --no-cache-dir --upgrade pip
}
