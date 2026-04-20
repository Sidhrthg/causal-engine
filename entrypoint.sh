#!/bin/sh
# On first start (volume is empty), seed /app/data/documents/ from the
# embedded copy baked into the image at /app/data_init/.
if [ ! -f "/app/data/documents/index.json" ]; then
    echo "[entrypoint] Seeding data/documents from embedded image copy..."
    mkdir -p /app/data/documents
    cp -r /app/data_init/documents/. /app/data/documents/
    echo "[entrypoint] Done — $(find /app/data/documents -type f | wc -l) files copied."
fi
exec "$@"
