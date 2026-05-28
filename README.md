# P2P Federated Learning — FEMNIST

## Requirements

- Docker + Docker Compose
- Python 3.11+
- Terraform (for both AWS single-EC2 and multi-instance deploy)

```bash
pip install -r requirements.debug.txt
```

## Deployment Modes

| Mode | EC2 instances | Containers | When to use |
|---|:---:|:---:|---|
| **Local** | 0 | `num_workers` + 1 | development, accuracy experiments |
| **Single EC2** | 1 | `num_workers` + 1 | same as local, but running on AWS |
| **Multi-instance EC2** | `num_workers` + 1 | 1 per EC2 | convergence-time experiments (real TCP/IP) |

`num_workers` in `config.yaml` sets the number of worker containers in all modes.
In multi-instance mode each worker gets its own EC2 instance; the registry runs on a separate one.
See `docs/report.md` for full documentation on architecture, experiments, and AWS constraints.

---

## Setup

```bash
# 1. Edit config.yaml — set num_workers, gossip_fanout, learning_rate, etc.

# 2. Download FEMNIST dataset
#    Re-run only when use_test_set changes (different --tf flag to LEAF)
python scripts/download_femnist.py             # full dataset (default)
# python scripts/download_femnist.py --sf 0.05  # 5% subset for quick install checks only

# 3. Partition dataset and generate Docker Compose files
#    Re-run when num_workers OR use_test_set changes
python scripts/split_dataset.py
python scripts/generate_compose.py
```

---

## Local (and Single EC2)

```bash
docker compose up --build

# Analyze results
python scripts/aggregate_metrics.py
# Prints: per-round accuracy table, per-worker timing breakdown (phase A/B/C),
#         system convergence verdict (converged? at which round? wall-clock time),
#         weight divergence between workers.
python scripts/save_experiment.py <name>   # e.g. fanout_3, lr_1e-3, baseline
# Archives config + metrics to results/<timestamp>_<name>/  and cleans working dir.
```

For **single EC2** — Terraform handles instance creation and Docker install automatically:

```bash
# Each session: export credentials from Learner Lab panel
# (AWS Academy → Start Lab → AWS Details → Show → copy the three values below)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

# Edit config.yaml: aws.key_name, aws.key_path, aws.instance_type_single (default t3.large)
# One-time: install Terraform — https://developer.hashicorp.com/terraform/install

python scripts/aws_deploy.py provision_single   # create EC2 instance, wait for Docker ready

# Follow the printed instructions (scp + ssh), then on the EC2 host:
scp -r . ubuntu@<ip>:~/project
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip>
  cd ~/project
  pip install -r requirements.debug.txt          # for analysis scripts (run on host, not in container)
  docker compose up --build
  python scripts/aggregate_metrics.py
  python scripts/save_experiment.py <name>

python scripts/aws_deploy.py destroy_single     # IMPORTANT: stop billing

# If the lab session restarted (instance has new public IP):
python scripts/aws_deploy.py resume_single
```

**Experiment workflow:**
- *Phase 1 — hyperparameter search*: fix `num_workers` (e.g. 3), vary one parameter at a time, repeat the two commands above.
- *Phase 2 — scalability*: fix the best config, vary `num_workers` (3 → 5 → 8); re-run setup steps 1–3 before each run.
- *Phase 3 — final test evaluation*: set `use_test_set: true`, re-run all three setup steps, then train once; `aggregate_metrics.py` will also print unbiased test accuracy.

See `docs/report.md` section 11.1.1 for detailed per-step tables (who does what, on which machine).

---

## AWS Multi-Instance

```bash
# One-time prerequisites
# - Install Terraform: https://developer.hashicorp.com/terraform/install
# - Key pair: us-east-1 → use "vockey" (AWS Details → Download PEM → ~/Downloads/labsuser.pem)
#             us-west-2 → create a new key pair in EC2 Console
# - Edit config.yaml: aws.key_name, aws.key_path

# Each session: export credentials from Learner Lab panel
# (AWS Academy → Start Lab → AWS Details → Show → copy the three values below)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

python scripts/aws_deploy.py provision   # create EC2 instances via Terraform
python scripts/aws_deploy.py deploy      # build images, upload data, start containers
python scripts/aws_deploy.py status      # check container status
python scripts/aws_deploy.py logs 0      # tail worker_0 logs (Ctrl+C to exit)
python scripts/aws_deploy.py collect     # download metrics once training ends
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <name>
python scripts/aws_deploy.py destroy     # IMPORTANT: stop billing

# If the lab session restarted (instances get new public IPs):
python scripts/aws_deploy.py resume
```
