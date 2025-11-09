#!/bin/bash

# This script is not meant to be executed directly.
# It provides a library of common shell functions to be sourced by other scripts.

# Sets up the basic prerequisites for installing packages from a PPA.
# Specifically, it installs tools for adding repositories and then adds the
# deadsnakes PPA for newer Python versions.
setup_ppa_prerequisites() {
    apt-get update
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        ca-certificates

    # add-apt-repository automatically runs apt-get update
    add-apt-repository -y ppa:deadsnakes/ppa
}
