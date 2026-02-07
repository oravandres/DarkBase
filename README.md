# DarkBase - MinIO Backup Server

Ansible project to deploy MinIO as an S3-compatible backup server for the MiMi K3s cluster.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DarkBase Server                                 │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    MinIO Object Storage                        │ │
│  │                                                                │ │
│  │  API:     https://s3.mimi.local                                │ │
│  │  Console: https://minio.mimi.local                             │ │
│  │                                                                │ │
│  │  Buckets:                                                      │ │
│  │    ├── etcd-backups     (K3s etcd snapshots)                   │ │
│  │    ├── velero-backups   (Cluster resources & PVs)              │ │
│  │    ├── loki-chunks      (Log storage)                          │ │
│  │    └── grafana-backups  (Dashboard exports)                    │ │
│  │                                                                │ │
│  │  Storage: /media/andres/data/minio (1.8TB External HDD)        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MiMi K3s Cluster (6 nodes)                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │   pi-c1    │ │   pi-c2    │ │   pi-c3    │  Control Plane        │
│  └────────────┘ └────────────┘ └────────────┘                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │   pi-n1    │ │   pi-n2    │ │   pi-n3    │  Workers              │
│  └────────────┘ └────────────┘ └────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Deploy MinIO (requires sudo password)
ansible-playbook playbooks/setup-minio.yml --ask-become-pass

# Verify installation
curl http://localhost:9000/minio/health/live
```

## Access

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | https://minio.mimi.local | See MiMi-Secrets repo |
| MinIO S3 API | https://s3.mimi.local | S3-compatible endpoint |

> **Note:** These URLs route through the MiMi cluster's Traefik ingress.
> Direct access: `http://192.168.1.239:9001` (console) / `:9000` (API)

## Configuration

Credentials and settings are stored in `inventory/group_vars/minio.yml` (gitignored).

### Storage Location

MinIO stores data on the external HDD:
- Mount point: `/media/andres/data`
- Data directory: `/media/andres/data/minio`
- Capacity: ~1.8TB

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

## Integration with MiMi Cluster

### etcd Backup (Recommended First Step)

Create a CronJob on the cluster to backup etcd snapshots:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: etcd-backup
            image: bitnami/kubectl:latest
            command: ["/bin/sh", "-c"]
            args:
              - |
                # Snapshot etcd and upload to MinIO
                k3s etcd-snapshot save
                # Upload logic here
          restartPolicy: OnFailure
```

### Velero (Full Cluster Backup)

```bash
# Install Velero with MinIO backend
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket velero-backups \
  --secret-file ./credentials-velero \
  --backup-location-config region=minio,s3ForcePathStyle="true",s3Url=https://s3.mimi.local \
  --use-volume-snapshots=false
```

## Troubleshooting

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

## Project Structure

```
DarkBase/
├── inventory/
│   ├── hosts.yml           # Target hosts (localhost)
│   └── group_vars/
│       └── minio.yml       # Credentials (gitignored)
├── playbooks/
│   └── setup-minio.yml     # Main deployment playbook
└── roles/
    └── minio/
        ├── defaults/       # Default variables
        ├── handlers/       # Service restart handlers
        ├── tasks/          # Installation tasks
        └── templates/      # Systemd & env templates
```

## License

MIT
