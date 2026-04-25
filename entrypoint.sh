#!/bin/sh
# On first start (volume is empty), seed /app/data/ from embedded copies baked into the image.
if [ ! -f "/app/data/documents/index.json" ]; then
    echo "[entrypoint] Seeding data/documents from embedded image copy..."
    mkdir -p /app/data/documents
    cp -r /app/data_init/documents/. /app/data/documents/
    echo "[entrypoint] Done — $(find /app/data/documents -type f | wc -l) files copied."
fi
if [ ! -d "/app/data/canonical" ]; then
    echo "[entrypoint] Seeding data/canonical from embedded image copy..."
    mkdir -p /app/data/canonical
    cp -r /app/data_init/canonical/. /app/data/canonical/
    echo "[entrypoint] Canonical — $(find /app/data/canonical -type f | wc -l) files copied."
fi
exec "$@"
