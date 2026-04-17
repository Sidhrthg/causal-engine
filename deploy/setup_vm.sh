#!/usr/bin/env bash
# setup_vm.sh — Run this once on a fresh Ubuntu 22.04/24.04 VM as root (or with sudo).
# Usage: sudo bash setup_vm.sh
#
# What it does:
#   1. Installs Python 3.12, nginx, certbot, git
#   2. Creates a dedicated 'causal' system user
#   3. Clones the repo and installs Python deps in a venv
#   4. Sets up log directory
#   5. Installs and enables the systemd service
#   6. Installs the nginx site config (disabled until you fill in your domain)
#
# After running this script you still need to:
#   a. Upload your data files:  bash rsync_data.sh USER@YOUR_VM_IP
#   b. Edit /etc/nginx/sites-available/causal-engine — replace YOUR_DOMAIN
#   c. sudo certbot --nginx -d YOUR_DOMAIN
#   d. sudo systemctl reload nginx

set -euo pipefail

###############################################################################
# CONFIG — edit these before running
###############################################################################
REPO_URL="https://github.com/YOUR_ORG/Causal-engine.git"   # your repo URL
APP_DIR="/srv/causal-engine"
APP_USER="causal"
PYTHON="python3.12"
###############################################################################

echo "==> [1/7] Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    nginx \
    certbot \
    python3-certbot-nginx \
    curl \
    rsync

echo "==> [2/7] Creating system user '${APP_USER}'..."
if ! id "${APP_USER}" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "${APP_USER}"
fi

echo "==> [3/7] Cloning repo to ${APP_DIR}..."
if [ -d "${APP_DIR}/.git" ]; then
    echo "    Repo already exists — pulling latest."
    git -C "${APP_DIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${APP_DIR}"
fi
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

echo "==> [4/7] Creating Python venv and installing deps..."
sudo -u "${APP_USER}" bash -c "
    ${PYTHON} -m venv ${APP_DIR}/.venv
    ${APP_DIR}/.venv/bin/pip install --upgrade pip wheel
    ${APP_DIR}/.venv/bin/pip install gunicorn
    ${APP_DIR}/.venv/bin/pip install -e '${APP_DIR}[rag,hipporag]'
"

echo "==> [5/7] Setting up log directory..."
mkdir -p /var/log/causal-engine
chown "${APP_USER}:${APP_USER}" /var/log/causal-engine

echo "==> [6/7] Installing systemd service..."
cp "${APP_DIR}/deploy/causal-engine.service" /etc/systemd/system/causal-engine.service
systemctl daemon-reload
systemctl enable causal-engine
systemctl restart causal-engine
echo "    Service status:"
systemctl status causal-engine --no-pager || true

echo "==> [7/7] Installing nginx config (disabled until domain is set)..."
cp "${APP_DIR}/deploy/nginx.conf" /etc/nginx/sites-available/causal-engine
# Do NOT symlink yet — user must fill in domain first
echo ""
echo "========================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Upload data files from your local machine:"
echo "     bash ${APP_DIR}/deploy/rsync_data.sh ${APP_USER}@YOUR_VM_IP"
echo ""
echo "  2. Edit nginx config — replace YOUR_DOMAIN:"
echo "     nano /etc/nginx/sites-available/causal-engine"
echo ""
echo "  3. Enable nginx site and get TLS cert:"
echo "     ln -s /etc/nginx/sites-available/causal-engine /etc/nginx/sites-enabled/"
echo "     nginx -t && systemctl reload nginx"
echo "     certbot --nginx -d YOUR_DOMAIN"
echo ""
echo "  4. Check service logs:"
echo "     journalctl -u causal-engine -f"
echo "========================================================"
