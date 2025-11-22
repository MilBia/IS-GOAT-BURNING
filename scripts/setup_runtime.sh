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

echo "Runtime environment setup complete."
