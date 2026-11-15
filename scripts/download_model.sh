#!/usr/bin/env bash
# Download a model file over HTTPS, verify its SHA-256 and install it atomically.
#
# Usage: scripts/download_model.sh [--force] <url> <sha256> [dest]
#   url     https:// URL of the model (no default: no model is published yet, see #73).
#   sha256  expected SHA-256 hex digest of the file.
#   dest    destination path (default: ./models/<basename of the URL path>).
#   --force overwrite an existing destination file.
set -euo pipefail

usage() {
    echo "Usage: $0 [--force] <https-url> <sha256> [dest]" >&2
    exit 2
}

force=0
args=()
for arg in "$@"; do
    case "$arg" in
        --force) force=1 ;;
        -h | --help) usage ;;
        *) args+=("$arg") ;;
    esac
done
if [ "${#args[@]}" -lt 2 ] || [ "${#args[@]}" -gt 3 ]; then
    usage
fi

url="${args[0]}"
expected="${args[1],,}"
case "$url" in
    https://*) ;;
    *)
        echo "error: only https:// URLs are accepted." >&2
        exit 2
        ;;
esac
if ! [[ "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: sha256 must be 64 hexadecimal characters." >&2
    exit 2
fi

if [ "${#args[@]}" -eq 3 ]; then
    dest="${args[2]}"
else
    name="${url%%[?#]*}"
    name="${name##*/}"
    if [ -z "$name" ]; then
        echo "error: cannot derive a file name from the URL; pass dest." >&2
        exit 2
    fi
    dest="./models/${name}"
fi

if [ -e "$dest" ] && [ "$force" -ne 1 ]; then
    echo "error: $dest already exists; use --force to overwrite." >&2
    exit 1
fi

dest_dir="$(dirname "$dest")"
mkdir -p "$dest_dir"
tmp="$(mktemp "${dest_dir}/.download.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --retry 3 --connect-timeout 20 --output "$tmp" "$url"

actual="$(sha256sum "$tmp" | cut -d' ' -f1)"
if [ "$actual" != "$expected" ]; then
    echo "error: SHA-256 mismatch (expected $expected, got $actual); nothing written." >&2
    exit 1
fi

chmod 644 "$tmp"
mv -f "$tmp" "$dest"
trap - EXIT
echo "Model saved to $dest (sha256 verified)."
