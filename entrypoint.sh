#!/bin/sh

# Ensure the recordings directory exists and has the correct permissions.
VIDEO_DIR="${VIDEO_OUTPUT_DIRECTORY:-/app/recordings}"

case "$VIDEO_DIR" in
  # 1. Check for the path traversal.
  *..*)
    # If this matches, give a very specific error and exit.
    echo "Error: VIDEO_OUTPUT_DIRECTORY cannot contain '..'." >&2
    exit 1
    ;;

  # 2. If the first check passed, check if the path is a valid subdir.
  # The '/app/?*' pattern means:
  # - Must start with /app/
  # - Must have at least one character after the slash (`?`)
  # - Can have anything after that (`*`)
  /app/?*)
    # This is the "success" case. Do nothing and continue the script.
    ;;

  # 3. If neither of the above patterns matched, it's an invalid format.
  *)
    # Give a clear error explaining the required format and exit.
    echo "Error: VIDEO_OUTPUT_DIRECTORY must be a subdirectory of /app (e.g., /app/recordings)." >&2
    exit 1
    ;;
esac

mkdir -p "$VIDEO_DIR" || { echo "Error: Failed to create directory '$VIDEO_DIR'. It might be a file or you may not have permissions." >&2; exit 1; }
chown nobody:nogroup "$VIDEO_DIR"

# Execute the command passed to this script (the Dockerfile's CMD)
# as the nobody user. `exec` replaces the shell process with the new process,
# ensuring that signals are passed correctly.
#
# We use `gosu` here, which is a lightweight `sudo` alternative perfect for containers.
exec gosu nobody:nogroup "$@"
