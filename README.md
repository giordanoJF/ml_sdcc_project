# P2P Federated Learning — FEMNIST

## Requirements

- Docker + Docker Compose
- Python 3.11+
- git
- Terraform (for AWS deployments)

```bash
pip install -r requirements.debug.txt
```

## Deployment Modes

| Mode | EC2 instances | Containers | When to use |
|---|:---:|:---:|---|
| **Local** | 0 | `num_workers` + 1 | development, accuracy experiments |
| **Single EC2** | 1 | `num_workers` + 1 | same as local, on AWS |
| **Multi-instance EC2** | `num_workers` + 1 | 1 per EC2 | convergence-time experiments (real TCP/IP) |

See `docs/report.md` for architecture, experiments, and AWS constraints.

---

## Setup

```bash
# 1. Edit config.yaml (num_workers, gossip_fanout, learning_rate, use_gpu, …)

# 2. Download FEMNIST — re-run only when local_test_set changes
python scripts/download_femnist.py          # full dataset
# python scripts/download_femnist.py --sf 0.05   # 5% subset for quick install checks

# 3. Partition data and generate docker-compose.yml
#    Re-run when num_workers, local_test_set, or global_test_set changes
python scripts/split_dataset.py
python scripts/generate_compose.py
```

### Test sets

| Flag | Split | Purpose |
|---|---|---|
| `local_test_set: false` (default) | 90/10 train/val | val used for early stopping and as comparison metric |
| `local_test_set: true` | 80/10/10 train/val/local_test | independent estimate on the worker's own writers |
| `global_test_set: true` | reserves `global_test_fraction` of writers globally | fully unbiased convergence metric across all workers |

**Comparison metric:** `mean_best_val_accuracy` — printed by `aggregate_metrics.py`.

---

## Local

```bash
docker compose up --build
```

**Cycle between runs (same `num_workers`):**

```bash
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>   # run BEFORE docker compose down — logs are lost after
docker compose down
# edit config.yaml
docker compose up --build
```

**When `num_workers` changes** — re-partition and regenerate:

```bash
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
docker compose down
# edit num_workers in config.yaml
python scripts/split_dataset.py
python scripts/generate_compose.py
docker compose up --build
```

### GPU (local only)

Set `federated_learning.use_gpu: true` in `config.yaml`, then:

```bash
python scripts/generate_compose.py
docker compose up --build
```

Requires NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Single EC2

```bash
# Export credentials from Learner Lab panel (AWS Academy → Start Lab → AWS Details → Show)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

python scripts/aws_deploy.py provision_single

# Follow printed instructions, then on the EC2 host:
scp -r . ubuntu@<ip>:~/project
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip>
  cd ~/project
  pip install -r requirements.debug.txt   # for analysis scripts (host, not container)
  docker compose up --build
  python scripts/aggregate_metrics.py --plot
  python scripts/save_experiment.py <name>

python scripts/aws_deploy.py destroy_single   # IMPORTANT: stop billing

# If the lab session restarted (new public IP):
python scripts/aws_deploy.py resume_single
```

---

## Multi-Instance EC2

```bash
# Export credentials (same as above)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

python scripts/aws_deploy.py provision   # create EC2 instances
python scripts/aws_deploy.py deploy      # build images, upload data, start containers
python scripts/aws_deploy.py status      # check container status
python scripts/aws_deploy.py logs 0      # tail worker_0 logs (Ctrl+C to exit)
python scripts/aws_deploy.py collect     # download metrics when training ends
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
python scripts/aws_deploy.py destroy     # IMPORTANT: stop billing

# If the lab session restarted (instances get new public IPs):
python scripts/aws_deploy.py resume
```
