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

# Dynamic permission fix for OpenCL/GPU access:
# The /dev/dri/renderD128 device is mounted from the host, so it retains the host's Group ID (GID).
# This GID might not match the 'render' group GID inside the container.
# We must detect the device's GID at runtime and ensure the 'nobody' user is part of that group.
# The device path can be overridden by the RENDER_DEVICE environment variable.
RENDER_DEVICE="${RENDER_DEVICE:-/dev/dri/renderD128}"
if [ -c "$RENDER_DEVICE" ]; then
    RENDER_GID=$(stat -c '%g' "$RENDER_DEVICE")
    echo "Detected render device $RENDER_DEVICE with GID $RENDER_GID"
    
    # Check if a group with this GID already exists
    EXISTING_GROUP_ENTRY=$(getent group "$RENDER_GID")
    if [ -z "$EXISTING_GROUP_ENTRY" ]; then
        echo "Creating group 'render_host' with GID $RENDER_GID"
        groupadd -g "$RENDER_GID" render_host || { echo "Error: Failed to create group 'render_host' with GID $RENDER_GID" >&2; exit 1; }
        RENDER_GROUP="render_host"
    else
        # Extract the group name from the getent output
        RENDER_GROUP=$(echo "$EXISTING_GROUP_ENTRY" | cut -d: -f1)
    fi
    
    echo "Adding 'nobody' to group '$RENDER_GROUP' ($RENDER_GID)"
    usermod -a -G "$RENDER_GROUP" nobody || { echo "Error: Failed to add 'nobody' to group '$RENDER_GROUP'" >&2; exit 1; }
fi

# Execute the command passed to this script (the Dockerfile's CMD)
# as the nobody user. `exec` replaces the shell process with the new process,
# ensuring that signals are passed correctly.
#
# We use `gosu` here, which is a lightweight `sudo` alternative perfect for containers.
# We use 'nobody' instead of 'nobody:nogroup' to ensure that supplementary groups
# (like 'video' and 'render', added in setup_runtime.sh) are correctly applied.
exec gosu nobody "$@"
