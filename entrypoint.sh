#!/bin/sh
# Seed /app/data/ from embedded image copies baked into the Dockerfile.
# Uses `cp -rn` (no-clobber): copies any file present in the image but missing
# from the volume, without overwriting existing files. This way:
#   - first boot: full seed from image
#   - subsequent boots: only NEW files in the image get copied (e.g. when we
#     add enriched_kg.json), and any in-place edits on the volume (e.g. KG
#     enrichments writing to enriched_kg.json) are preserved.
echo "[entrypoint] Syncing data/documents from image (no-clobber)..."
mkdir -p /app/data/documents
cp -rn /app/data_init/documents/. /app/data/documents/ 2>/dev/null || true
echo "[entrypoint] documents — $(find /app/data/documents -type f | wc -l) files present."

echo "[entrypoint] Syncing data/canonical from image (no-clobber)..."
mkdir -p /app/data/canonical
cp -rn /app/data_init/canonical/. /app/data/canonical/ 2>/dev/null || true
echo "[entrypoint] canonical — $(find /app/data/canonical -type f | wc -l) files present."

exec "$@"
