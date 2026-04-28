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

# Versioned forward migration. The image's enriched_kg.json embeds a
# _data_version marker in its metadata. If the volume's copy lacks the same
# marker, replace it with the image copy. This is bounded: once the volume
# has the current version, the check skips and subsequent /api/kg/enrich
# writes are preserved (they don't change _data_version, so the next image
# bump triggers a re-migration).
if [ -f /app/data/canonical/enriched_kg.json ] && [ -f /app/data_init/canonical/enriched_kg.json ]; then
    IMG_VER=$(grep -o '"_data_version"[^,]*' /app/data_init/canonical/enriched_kg.json | head -1)
    VOL_VER=$(grep -o '"_data_version"[^,]*' /app/data/canonical/enriched_kg.json | head -1)
    if [ "$IMG_VER" != "$VOL_VER" ]; then
        echo "[entrypoint] enriched_kg.json migration: $VOL_VER -> $IMG_VER"
        cp /app/data_init/canonical/enriched_kg.json /app/data/canonical/enriched_kg.json
    fi
fi

exec "$@"
