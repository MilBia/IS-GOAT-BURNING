#!/usr/bin/env bash
# Download a model file over HTTPS, verify its SHA-256 and install it atomically.
#
# Usage: scripts/download_model.sh [--force] <url> <sha256> [dest]
#   url     https:// URL of the model (no default: no model is published yet, see #73).
#   sha256  expected SHA-256 hex digest of the file.
#   dest    destination path (default: ./models/<basename of the URL path>).
#   --force overwrite an existing destination file.
#   --      end of options.
# Environment: MODEL_DOWNLOAD_MAX_TIME (total seconds, default 600).
set -euo pipefail

usage() {
    echo "Usage: $0 [--force] <https-url> <sha256> [dest]" >&2
    exit 2
}

force=0
args=()
parse_opts=1
for arg in "$@"; do
    if [ "$parse_opts" -eq 1 ]; then
        case "$arg" in
            --) parse_opts=0; continue ;;
            --force) force=1; continue ;;
            -h | --help) usage ;;
        esac
    fi
    args+=("$arg")
done
if [ "${#args[@]}" -lt 2 ] || [ "${#args[@]}" -gt 3 ]; then
    usage
fi

url="${args[0]}"
expected="$(printf '%s' "${args[1]}" | tr '[:upper:]' '[:lower:]')"
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

case "$(basename "$dest")" in
    . | ..)
        echo "error: invalid destination file name '$dest'." >&2
        exit 2
        ;;
esac
if [ -d "$dest" ]; then
    echo "error: $dest is a directory." >&2
    exit 2
fi
if [ -e "$dest" ] && [ "$force" -ne 1 ]; then
    echo "error: $dest already exists; use --force to overwrite." >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    hash_cmd=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    hash_cmd=(shasum -a 256)
else
    echo "error: neither sha256sum nor shasum is available." >&2
    exit 2
fi

tmp="$(mktemp "${TMPDIR:-/tmp}/download_model.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --retry 3 --connect-timeout 20 --max-time "${MODEL_DOWNLOAD_MAX_TIME:-600}" \
    --speed-limit 1024 --speed-time 60 --output "$tmp" -- "$url"

actual="$("${hash_cmd[@]}" "$tmp" | cut -d' ' -f1)"
if [ "$actual" != "$expected" ]; then
    echo "error: SHA-256 mismatch (expected $expected, got $actual); nothing written." >&2
    exit 1
fi

dest_dir="$(dirname "$dest")"
mkdir -p "$dest_dir"
# Stage next to the destination so the final mv is an atomic rename on the same filesystem.
staged="$(mktemp "${dest_dir}/.download.XXXXXX")"
trap 'rm -f "$tmp" "$staged"' EXIT
cp "$tmp" "$staged"
chmod 644 "$staged"
mv -f "$staged" "$dest"
trap 'rm -f "$tmp"' EXIT
echo "Model saved to $dest (sha256 verified)."
