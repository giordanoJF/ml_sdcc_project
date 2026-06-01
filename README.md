# P2P Federated Learning — FEMNIST

## Requirements

- Docker + Docker Compose
- Python 3.11+
- git (used by `scripts/download_femnist.py` to clone the LEAF repository)
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
```

**Cycle between runs (same `num_workers`, different config parameter):**

```bash
python scripts/aggregate_metrics.py --plot  # --plot generates accuracy/loss/timing PNG charts
python scripts/save_experiment.py <name>    # archives config + metrics + Docker logs + plots
# MUST run before docker compose down — logs are lost when containers are removed
docker compose down                         # stops + removes containers and networks;
                                            # does NOT remove images or data/femnist/ files
# edit config.yaml
docker compose up --build                   # --build always required when config.yaml changes
                                            # (config is baked into the image, not volume-mounted)
```

**When `num_workers` changes** — dataset must be re-partitioned and compose regenerated:

```bash
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <name>
docker compose down
# edit num_workers in config.yaml
python scripts/split_dataset.py            # re-partition data for new worker count
python scripts/generate_compose.py         # regenerate docker-compose.yml with N services
docker compose up --build
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

**Experiment workflow — 13 runs total:**
- *Esp. 1 — no-FL baseline*: set `gossip_fanout: 0`, `num_workers=5`. One run.
- *Esp. 2 — FL default*: set `gossip_fanout: 3`, `num_workers=5`, `lr=1e-3`, `H=500`. One run.
- *Esp. 3 — hyperparameter grid*: `num_workers=5`, `fanout=3`, `batch_size=32` fixed. Run all 9 combinations of `learning_rate` ∈ {1e-4, 1e-3, 5e-3} × `inner_steps_H` ∈ {100, 500, 1000}. Pick best `(lr, H)` by `mean_accuracy`.
- *Esp. 4 — scalability*: apply best `(lr, H)` from Esp. 3 with `(num_workers=3, fanout=1)` then `(num_workers=8, fanout=5)`. Re-run setup steps 1–3 before each. Pick overall best config.
- *Esp. 5 — honest test evaluation*: set `use_test_set: true` with best config → re-download dataset (required, `--tf` changes), re-run steps 1–3, train once. `aggregate_metrics.py` prints unbiased `test_accuracy`.
- *Esp. 6 — fault injection*: best config, low `drop_probability` and `crash_probability` (e.g. 0.10 / 0.03). One run to document graceful degradation.

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
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
python scripts/aws_deploy.py destroy     # IMPORTANT: stop billing

# If the lab session restarted (instances get new public IPs):
python scripts/aws_deploy.py resume
```
