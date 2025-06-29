#!/bin/sh

# Ensure the recordings directory exists and has the correct permissions.
VIDEO_DIR="${VIDEO_OUTPUT_DIRECTORY:-/app/recordings}"

if ! echo "$VIDEO_DIR" | grep -q '^/app/' || echo "$VIDEO_DIR" | grep -q '\.\.'; then
    echo "Error: VIDEO_OUTPUT_DIRECTORY must be an absolute path under /app and cannot contain '..'." >&2
    exit 1
fi

mkdir -p "$VIDEO_DIR"
chown nobody:nogroup "$VIDEO_DIR"

# Execute the command passed to this script (the Dockerfile's CMD)
# as the nobody user. `exec` replaces the shell process with the new process,
# ensuring that signals are passed correctly.
#
# We use `gosu` here, which is a lightweight `sudo` alternative perfect for containers.
exec gosu nobody:nogroup "$@"
