#!/bin/sh

# Ensure the recordings directory exists and has the correct permissions.
VIDEO_DIR="${VIDEO_OUTPUT_DIRECTORY:-/app/recordings}"
mkdir -p "$VIDEO_DIR"
chown nobody:nogroup "$VIDEO_DIR"

# Execute the command passed to this script (the Dockerfile's CMD)
# as the nobody user. `exec` replaces the shell process with the new process,
# ensuring that signals are passed correctly.
#
# We use `gosu` here, which is a lightweight `sudo` alternative perfect for containers.
exec gosu nobody:nogroup "$@"
