#!/usr/bin/env bash
# rsync_data.sh — Upload large data files from your local machine to the VM.
# Run this from your LOCAL machine (not on the VM).
#
# Usage:
#   bash deploy/rsync_data.sh causal@YOUR_VM_IP
#   bash deploy/rsync_data.sh causal@api.yourdomain.com
#
# Optional: pass SSH key with -i flag:
#   bash deploy/rsync_data.sh causal@YOUR_VM_IP ~/.ssh/id_ed25519

set -euo pipefail

TARGET="${1:?Usage: $0 USER@HOST [SSH_KEY]}"
SSH_KEY="${2:-}"

# Root of this repo (script lives in deploy/, so go up one level)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/srv/causal-engine"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
if [ -n "${SSH_KEY}" ]; then
    SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY}"
fi

RSYNC_OPTS=(
    -avz
    --progress
    --human-readable
    --exclude="*.pyc"
    --exclude="__pycache__"
    -e "ssh ${SSH_OPTS}"
)

echo "==> Uploading data files to ${TARGET}:${REMOTE_DIR}"
echo "    Source: ${REPO_ROOT}"
echo ""

# ── BACI canonical CSVs ────────────────────────────────────────────────────────
if [ -d "${REPO_ROOT}/data" ]; then
    echo "--- Syncing data/ (CEPII BACI canonical CSVs)..."
    rsync "${RSYNC_OPTS[@]}" \
        "${REPO_ROOT}/data/" \
        "${TARGET}:${REMOTE_DIR}/data/"
else
    echo "    [skip] data/ not found locally"
fi

# ── HippoRAG index ────────────────────────────────────────────────────────────
if [ -d "${REPO_ROOT}/hipporag_index" ]; then
    echo "--- Syncing hipporag_index/ ..."
    rsync "${RSYNC_OPTS[@]}" \
        "${REPO_ROOT}/hipporag_index/" \
        "${TARGET}:${REMOTE_DIR}/hipporag_index/"
else
    echo "    [skip] hipporag_index/ not found locally"
fi

# ── Scenarios (YAMLs + calibrated) ────────────────────────────────────────────
if [ -d "${REPO_ROOT}/scenarios" ]; then
    echo "--- Syncing scenarios/ ..."
    rsync "${RSYNC_OPTS[@]}" \
        "${REPO_ROOT}/scenarios/" \
        "${TARGET}:${REMOTE_DIR}/scenarios/"
else
    echo "    [skip] scenarios/ not found locally"
fi

# ── Outputs / cached results (optional) ───────────────────────────────────────
if [ -d "${REPO_ROOT}/outputs" ]; then
    echo "--- Syncing outputs/ ..."
    rsync "${RSYNC_OPTS[@]}" \
        "${REPO_ROOT}/outputs/" \
        "${TARGET}:${REMOTE_DIR}/outputs/"
fi

# ── Fix ownership on the VM ────────────────────────────────────────────────────
echo ""
echo "--- Fixing ownership on VM..."
ssh ${SSH_OPTS} "${TARGET}" "sudo chown -R causal:causal ${REMOTE_DIR}/data ${REMOTE_DIR}/hipporag_index ${REMOTE_DIR}/scenarios 2>/dev/null || true"

echo ""
echo "==> Upload complete."
echo "    Restart the service to pick up new data:"
echo "    ssh ${TARGET} 'sudo systemctl restart causal-engine'"
