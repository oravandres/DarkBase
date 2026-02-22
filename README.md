# DarkBase GPU Node, MinIO Backup Server & LLM Platform

This repository contains the configuration for **DarkBase**, acting as a GPU worker node for the MiMi cluster, a MinIO backup server, and a local LLM serving platform.

## Features

- **MinIO Backup Server**: Provides S3-compatible storage for Velero backups.
- **GPU Worker Node**: Adds GPU compute capacity to the MiMi cluster.
- **LLM Serving (Ollama)**: Runs local LLMs with GPU acceleration, exposing an OpenAI-compatible API.
- **Local Deployment**: Runs directly on the host machine via Ansible.

## Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                       DarkBase Server                                 │
│              i7-14700K · 128GB RAM · RTX 4080 16GB                    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                  Ollama LLM Server (:11434)                      │ │
│  │                                                                  │ │
│  │  Models:                                                         │ │
│  │    ├── deepseek-r1:32b      (Reasoning, math, code)             │ │
│  │    ├── qwen3:32b            (General purpose, multilingual)      │ │
│  │    ├── qwen3:14b            (Fast general purpose)              │ │
│  │    └── qwen2.5-coder:32b   (Code generation)                   │ │
│  │                                                                  │ │
│  │  API: http://192.168.1.239:11434 (OpenAI-compatible)             │ │
│  │  Storage: /media/andres/data/ollama (1.8TB NVMe)                 │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                  MinIO Object Storage                            │ │
│  │                                                                  │ │
│  │  API:     https://s3.mimi.local                                  │ │
│  │  Console: https://minio.mimi.local                               │ │
│  │                                                                  │ │
│  │  Buckets:                                                        │ │
│  │    ├── etcd-backups     (K3s etcd snapshots)                     │ │
│  │    ├── velero-backups   (Cluster resources & PVs)                │ │
│  │    ├── loki-chunks      (Log storage)                            │ │
│  │    └── grafana-backups  (Dashboard exports)                      │ │
│  │                                                                  │ │
│  │  Storage: /media/andres/data/minio (1.8TB NVMe)                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    MiMi K3s Cluster (6 nodes)                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                         │
│  │   pi-c1    │ │   pi-c2    │ │   pi-c3    │  Control Plane          │
│  └────────────┘ └────────────┘ └────────────┘                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                         │
│  │   pi-n1    │ │   pi-n2    │ │   pi-n3    │  Workers                │
│  └────────────┘ └────────────┘ └────────────┘                         │
└───────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Deploy MinIO (requires sudo password)
ansible-playbook playbooks/setup-minio.yml --ask-become-pass

# Deploy Ollama + download models (requires sudo password)
ansible-playbook playbooks/setup-ollama.yml --ask-become-pass
```

## LLM API Access

| Service | URL | Protocol |
|---------|-----|----------|
| Ollama API | http://192.168.1.239:11434 | OpenAI-compatible |
| Model list | http://localhost:11434/api/tags | REST |

### Usage Examples

```bash
# List available models
curl http://localhost:11434/api/tags | jq '.models[].name'

# Generate text (streaming)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "prompt": "Explain K3s in one paragraph"
}'

# Generate text (non-streaming)
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1:32b",
  "prompt": "Write a Python function to sort a linked list",
  "stream": false
}'

# OpenAI-compatible chat endpoint (works with any OpenAI SDK)
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "qwen3:32b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# Pull additional models
ollama pull deepseek-r1:70b
```

### Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.239:11434/v1",
    api_key="unused"  # Ollama doesn't require a key
)

response = client.chat.completions.create(
    model="qwen3:32b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## MinIO Access

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | https://minio.mimi.local | See MiMi-Secrets repo |
| MinIO S3 API | https://s3.mimi.local | S3-compatible endpoint |

> **Note:** These URLs route through the MiMi cluster's Traefik ingress.
> Direct access: `http://192.168.1.239:9001` (console) / `:9000` (API)

## Configuration

Credentials and settings are stored in `inventory/group_vars/` (gitignored).

### Storage Locations

| Service | Path | Purpose |
|---------|------|---------|
| MinIO | `/media/andres/data/minio` | Backup object storage |
| Ollama | `/media/andres/data/ollama` | LLM model weights |

All data lives on a 1.8TB NVMe drive mounted at `/media/andres/data`.

## Troubleshooting

### Ollama

```bash
# Check Ollama service status
sudo systemctl status ollama

# View Ollama logs
sudo journalctl -u ollama -f

# Test API health
curl http://localhost:11434/api/tags

# List models
ollama list

# Check GPU usage during inference
nvidia-smi

# Check model storage usage
du -sh /media/andres/data/ollama/
```

### MinIO

```bash
# Check MinIO service status
sudo systemctl status minio

# View MinIO logs
sudo journalctl -u minio -f

# Test API health
curl http://localhost:9000/minio/health/live

# Check disk usage
df -h /media/andres/data
```

## Using MinIO Client (mc)

```bash
# List buckets
sudo mc ls local/

# Upload a file
sudo mc cp myfile.tar.gz local/etcd-backups/

# Download a file
sudo mc cp local/etcd-backups/myfile.tar.gz ./

# Check bucket disk usage
sudo mc du local/
```

## Project Structure

```
DarkBase/
├── inventory/
│   ├── hosts.yml           # Target hosts (localhost)
│   └── group_vars/         # Credentials (gitignored)
├── playbooks/
│   ├── setup-minio.yml     # MinIO deployment
│   ├── setup-ollama.yml    # Ollama + models deployment
│   └── join-k3s.yml        # K3s agent join
└── roles/
    ├── minio/              # MinIO object storage
    │   ├── defaults/       # Default variables
    │   ├── handlers/       # Service restart handlers
    │   ├── tasks/          # Installation tasks
    │   └── templates/      # Systemd & env templates
    └── ollama/             # Ollama LLM server
        ├── defaults/       # Default variables (models list, paths)
        ├── handlers/       # Service restart handlers
        ├── tasks/          # Install, configure, pull models
        └── templates/      # Environment config template
```

## License

MIT
