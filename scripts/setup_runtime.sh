#!/bin/bash
set -e

# This script centralizes the setup of the runtime environment for the application.
# It handles copying the entrypoint, setting permissions, and creating necessary
# application directories. It is designed to be run from within a Docker build.

echo "Setting up runtime environment..."

# 1. Copy the entrypoint script from the build context's scripts directory
#    and make it executable. The script is expected to be in /tmp/scripts.
echo "Copying entrypoint and setting permissions..."
cp /tmp/scripts/entrypoint.sh /usr/local/bin/
chmod +x /usr/local/bin/entrypoint.sh

# 2. Create the application directories for recordings and cache.
echo "Creating application directories..."
mkdir -p /app/recordings
mkdir -p /app/.cache

# 3. Set ownership of the cache directory to the non-root 'nobody' user.
#    This allows the application to write to the cache without root privileges.
echo "Setting ownership for cache directory..."
chown -R nobody:nogroup /app/.cache

# 4. Add the 'nobody' user to the 'video' and 'render' groups.
#    This is required for OpenCL and GPU access when running as a non-root user.
#    We explicitly check for group existence to avoid errors if the groups are missing.
echo "Adding nobody to video and render groups..."
if getent group video > /dev/null 2>&1; then
    usermod -a -G video nobody
else
    echo "Warning: Group 'video' does not exist. Skipping addition of 'nobody' to 'video'." >&2
fi

if getent group render > /dev/null 2>&1; then
    usermod -a -G render nobody
else
    echo "Warning: Group 'render' does not exist. Skipping addition of 'nobody' to 'render'." >&2
fi

echo "Runtime environment setup complete."
