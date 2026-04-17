# Deployment Guide

Two-part deployment: FastAPI backend on a VM, Next.js frontend on Vercel.

```
[ Vercel (Next.js) ] ──API_URL──▶ [ Your VM (FastAPI + data) ]
```

---

## 1. Provision a VM

**Recommended specs:**
- Ubuntu 22.04 or 24.04
- 2 vCPU, 4 GB RAM (igraph and sentence-transformers are memory-hungry)
- 40 GB disk (BACI data + hipporag index)

Good options: DigitalOcean $24/mo droplet, AWS t3.medium, Hetzner CX22.

Open firewall ports: **22** (SSH), **80** (HTTP), **443** (HTTPS).

---

## 2. Set up the VM

SSH in as root, then:

```bash
# 1. Edit REPO_URL in the script first
nano deploy/setup_vm.sh   # set your GitHub repo URL

# 2. Copy and run the setup script
scp deploy/setup_vm.sh root@YOUR_VM_IP:/tmp/
ssh root@YOUR_VM_IP "bash /tmp/setup_vm.sh"
```

This installs Python 3.12, nginx, certbot, clones the repo, creates a venv, installs all deps, and starts the systemd service.

---

## 3. Upload data files

Run this from your **local machine**:

```bash
# Basic (uses your default SSH key)
bash deploy/rsync_data.sh causal@YOUR_VM_IP

# With a specific SSH key
bash deploy/rsync_data.sh causal@YOUR_VM_IP ~/.ssh/id_ed25519
```

This uploads `data/`, `hipporag_index/`, and `scenarios/` via rsync (resumable, shows progress).

After uploading, restart the service:
```bash
ssh causal@YOUR_VM_IP "sudo systemctl restart causal-engine"
```

---

## 4. Configure nginx + HTTPS

```bash
# SSH into the VM
ssh causal@YOUR_VM_IP

# Replace YOUR_DOMAIN in the nginx config
sudo nano /etc/nginx/sites-available/causal-engine

# Enable the site
sudo ln -s /etc/nginx/sites-available/causal-engine /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Get a free TLS cert from Let's Encrypt
sudo certbot --nginx -d YOUR_DOMAIN
```

If you don't have a domain yet, use the IP-only fallback block at the bottom of `nginx.conf` (comment out the domain blocks, uncomment the IP block).

---

## 5. Verify the backend

```bash
curl https://YOUR_DOMAIN/health
# → {"status":"healthy","version":"0.1.0"}

curl https://YOUR_DOMAIN/minerals/scenarios
# → {"scenarios":[...]}
```

---

## 6. Deploy frontend to Vercel

```bash
# Install Vercel CLI if you don't have it
npm i -g vercel

# From the frontend/ directory
cd frontend
vercel

# Follow the prompts:
#   Set up and deploy? → Yes
#   Which scope? → your account
#   Link to existing project? → No (first time)
#   Directory? → ./  (you're already in frontend/)
```

Then in the **Vercel dashboard → Project → Settings → Environment Variables**, add:

| Name | Value | Environments |
|------|-------|--------------|
| `api-url` | `https://YOUR_DOMAIN` | Production, Preview, Development |

This populates `API_URL` via the `@api-url` secret reference in `vercel.json`.

Redeploy after setting the env var:
```bash
vercel --prod
```

---

## Ongoing operations

**View backend logs:**
```bash
journalctl -u causal-engine -f
tail -f /var/log/causal-engine/access.log
```

**Restart backend:**
```bash
sudo systemctl restart causal-engine
```

**Update backend code (pull + restart):**
```bash
ssh causal@YOUR_VM_IP
cd /srv/causal-engine
sudo -u causal git pull
sudo systemctl restart causal-engine
```

**Renew TLS cert** (auto-renews via cron, but to force):
```bash
sudo certbot renew --dry-run
```

**Re-upload data after local changes:**
```bash
bash deploy/rsync_data.sh causal@YOUR_VM_IP
```

---

## Docker alternative

If you prefer Docker over bare-metal:

```bash
# Build
docker build -t causal-engine .

# Run (mount data dirs as volumes)
docker run -d \
    --name causal-engine \
    --restart unless-stopped \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/hipporag_index:/app/hipporag_index \
    -v $(pwd)/scenarios:/app/scenarios \
    -e WORKERS=2 \
    causal-engine
```

Then point nginx at `127.0.0.1:8000` the same way.

---

## File summary

| File | Purpose |
|------|---------|
| `deploy/setup_vm.sh` | One-shot VM provisioning script |
| `deploy/rsync_data.sh` | Upload data files from local → VM |
| `deploy/causal-engine.service` | systemd unit for FastAPI |
| `deploy/nginx.conf` | nginx reverse proxy + TLS config |
| `Dockerfile` | Docker image for the backend |
| `.dockerignore` | Excludes data/venv from Docker build context |
| `frontend/vercel.json` | Vercel project config |
| `frontend/.env.local.example` | Local dev env template |
