#!/bin/bash
set -e

# This script provides shared, reusable functions for setting up the Python
# environment. It is intended to be sourced by other scripts, not executed directly.

# Installs prerequisite packages for adding apt PPAs and then adds the
# deadsnakes PPA for recent Python versions. On releases that package
# python3.14 natively (Ubuntu 26.04, used by the opencl stages, #223) the PPA
# is skipped (such stages set PIP_BREAK_SYSTEM_PACKAGES=1 for PEP 668).
install_base_dependencies_and_ppa() {
    echo "Updating package list..."
    apt-get update

    # apt-cache treats the name as a regex (python3.14 also matches postgresql-plpython3-14), so filter exactly.
    if apt-cache madison python3.14 | awk '$1 == "python3.14"' | grep -q .; then
        echo "python3.14 is packaged natively; skipping the deadsnakes PPA."
        PYTHON_FROM_DISTRO=1
        apt-get install -y --no-install-recommends ca-certificates
        return
    fi

    echo "Installing PPA prerequisites..."
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        ca-certificates

    echo "Adding deadsnakes PPA..."
    add-apt-repository -y ppa:deadsnakes/ppa
}

# Finalizes a Python installation by setting it as the default python3
# and bootstrapping a modern version of pip.
#
# Arguments:
#   $1: The Python executable name (e.g., "python3.14").
finalize_python_setup() {
    if [[ -z "$1" ]]; then
        echo "Error: Python executable name must be provided to finalize_python_setup." >&2
        exit 1
    fi
    local python_executable="$1"

    echo "Setting ${python_executable} as the default python3 alternative..."
    # Use a high priority to ensure this version becomes the default.
    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/${python_executable}" 100

    echo "Bootstrapping pip for ${python_executable}..."
    if [[ "${PYTHON_FROM_DISTRO:-}" == "1" ]]; then
        # Debian/Ubuntu disable ensurepip for the system Python; bootstrap from the pip wheel that
        # pythonX.Y-venv pulls in (python3-pip-whl). Unlike the python3-pip package, this pip has a
        # RECORD file, so pinned pip/packaging/wheel versions in requirements*.txt can replace it.
        local pip_wheel
        pip_wheel=$(ls /usr/share/python-wheels/pip-*.whl)
        "${python_executable}" "${pip_wheel}/pip" install --no-cache-dir "${pip_wheel}"
    else
        "${python_executable}" -m ensurepip --upgrade
    fi
    "${python_executable}" -m pip install --no-cache-dir --upgrade pip
}
