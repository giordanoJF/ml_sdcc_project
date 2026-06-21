# P2P Federated Learning — FEMNIST

## Host Requirements

- Docker + Docker Compose (runs containers)
- Python 3.11+ (host-side scripts)
- git (to clone the LEAF dataset repository)
- Terraform ≥ 1.2 (AWS provisioning)
- Python libraries (host-side scripts):
  ```bash
  pip install -r requirements.debug.txt
  ```

## Deployment Modes

**Local** — all containers run on the host machine via Docker Compose: one registry + `num_workers` workers.

**Single EC2** — same Docker Compose setup, deployed to one AWS instance. Useful to run experiments on cloud hardware without changing the architecture.

**Multi-instance EC2** — one EC2 instance per worker, plus one for the registry. Workers communicate over real TCP/IP across separate machines.

---

## Setup

**1. Edit `config.yaml`** .

**2. Download FEMNIST** — run once; re-run only if `local_test_set` changes in `config.yaml`. Use `--sf 0.05` for download just 5%

```bash
python scripts/download_femnist.py
```

**3. Partition data across workers** — re-run when `num_workers`, `local_test_set`, or `global_test_set` changes.

```bash
python scripts/split_dataset.py
```

**4. Generate `docker-compose.yml`** — re-run when `num_workers`, `use_gpu`, or `global_test_set` changes.

```bash
python scripts/generate_compose.py
```

### Test sets

Two independent flags in `config.yaml` control how data is split for evaluation:

**`local_test_set`** (default: `false`):
- `false` — 90/10 train/val per worker. Val is used for early stopping and as the final accuracy metric.
- `true` — 80/10/10 train/val/local_test per worker. Val is used only for early stopping; `local_test` contains the same writers as the training set, but a held-out portion of their samples never used for gradient updates.

**`global_test_set`** (default: `true`): reserves `global_test_fraction` of writers before any per-worker split. All workers evaluate on this shared set every round — a fully unbiased convergence metric across the whole federation.

**Comparison metric:** `mean_best_val_accuracy` — average across all workers of each worker's peak validation accuracy over all rounds. Used to compare different configurations across runs. Saved to `data/femnist/summary.txt` by `aggregate_metrics.py`.

---

## Local

**Automatic — `run_grid.py` manages the full experimental plan:**

```bash
python scripts/run_grid.py
```

`run_grid.py` handles config updates, dataset re-partitioning when N changes, and result archiving automatically.

---

**Manual — start and cycle between runs:**

```bash
docker compose up --build   # start training
```

When training finishes, save results and start the next run:

```bash
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>   # run BEFORE docker compose down — logs are lost after
docker compose down
# edit config.yaml
docker compose up --build
```

Depending on what changed in config.yaml, insert these steps before `docker compose up --build`:

**`use_gpu` changed** (requires NVIDIA GPU + NVIDIA Container Toolkit):
```bash
python scripts/generate_compose.py
```

**`num_workers` or `global_test_set` changed:**
```bash
python scripts/split_dataset.py
python scripts/generate_compose.py
```

**`local_test_set` changed:**
```bash
python scripts/download_femnist.py
python scripts/split_dataset.py
python scripts/generate_compose.py
```

---

## Single EC2

The **local machine** runs `download_femnist.py`, `split_dataset.py`, and `generate_compose.py` before the `scp` — the EC2 instance receives data already partitioned and runs only `docker compose up`. For subsequent runs, `split_dataset.py` and `generate_compose.py` can be re-run directly on the EC2 host (raw data is already there), except when `local_test_set` changes — that requires re-downloading LEAF locally (needs git) and re-uploading.

Export credentials (Learner Lab panel → Start Lab → AWS Details → Show):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

Create the EC2 instance via Terraform, wait until Docker is ready, then print the public IP assigned by AWS:

```bash
python scripts/aws_deploy.py provision_single
```

Copy the project to the instance (`<ip>` from the previous step), then connect:

```bash
scp -r . ubuntu@<ip>:~/project
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip>
```

`scp -r` copies the project recursively to `~/project` on the instance. `ssh -i` authenticates with the private key downloaded from Learner Lab (`aws.key_path` in `config.yaml`). `ubuntu` is the default user on EC2 Ubuntu instances.

<br>

**On the EC2 host — setup:**

```bash
cd ~/project
pip install -r requirements.debug.txt
```

<br>

**On the EC2 host — Automatic (`run_grid.py` manages the full experimental plan):**

```bash
python scripts/run_grid.py
```

<br>


**On the EC2 host — Manual (start and cycle between runs):**

```bash
docker compose up --build
```

When training finishes, save results and start the next run:

```bash
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
docker compose down
# edit config.yaml
docker compose up --build
```

Depending on what changed in config.yaml, insert these steps before `docker compose up --build`:

**`use_gpu` changed** (requires NVIDIA GPU + NVIDIA Container Toolkit):
```bash
python scripts/generate_compose.py
```

**`num_workers` or `global_test_set` changed:**
```bash
python scripts/split_dataset.py
python scripts/generate_compose.py
```

**`local_test_set` changed** — `download_femnist.py` requires git, which is not installed on the EC2 instance. Run these **on the local machine**, delete the old data on EC2, then re-upload:
```bash
# on the local machine:
python scripts/download_femnist.py
python scripts/split_dataset.py
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip> "rm -rf ~/project/data/femnist"
scp -r data/femnist ubuntu@<ip>:~/project/data/
```

When all runs are done:

```bash
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
```

<br>


**Back on the local machine — download results, then stop billing:**

```bash
scp -r ubuntu@<ip>:~/project/results ./
python scripts/aws_deploy.py destroy_single   # IMPORTANT: stop billing
```

**If the Learner Lab session restarted** (instance gets a new public IP):

```bash
python scripts/aws_deploy.py resume_single   # prints the new IP
```

---

## Multi-Instance EC2

All orchestration runs locally via `aws_deploy.py` — no manual SSH needed. Training runs on the EC2 instances. `run_grid.py` is not supported in this mode: runs must be managed one at a time.

Export credentials (same as Single EC2):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

Create all EC2 instances (one per worker + one for the registry), then build images, upload the dataset partitions, and start containers:

```bash
python scripts/aws_deploy.py provision
python scripts/aws_deploy.py deploy
```

Monitor training — poll `status` until all worker containers show "Exited":

```bash
python scripts/aws_deploy.py status      # container status on every instance
python scripts/aws_deploy.py logs 0      # tail worker_0 logs (Ctrl+C to exit)
```

When all containers have exited, `collect` downloads `metrics.csv` and `local_test_result.json` from each worker to `data/femnist/worker_i/` locally. Run it once — the EC2 instances stay up (EBS data survives) until you call `destroy`:

```bash
python scripts/aws_deploy.py collect
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
python scripts/aws_deploy.py destroy     # IMPORTANT: stop billing
```

**If config changes between runs** — update config locally, re-run the relevant scripts, then call `deploy` again (it stops old containers, re-uploads data, and restarts):

**`num_workers` or `global_test_set` changed:**
```bash
python scripts/split_dataset.py
python scripts/aws_deploy.py deploy
```

**`local_test_set` changed** — requires re-download locally (git not on EC2):
```bash
python scripts/download_femnist.py
python scripts/split_dataset.py
python scripts/aws_deploy.py deploy
```

No need to `destroy` and re-`provision` between runs — instances stay up, `deploy` handles the rest. `deploy` always deletes the old data partition on each worker before re-uploading, so no stale files survive between runs.

**Inter-worker communication:** workers register their private VPC IP with the registry at startup. Gossip gRPC calls go directly between workers via private IPs — the registry is used only for initial peer discovery. All instances are in the same availability zone, so intra-cluster traffic is free.

**If the Learner Lab session restarted** (instances get new public IPs):

```bash
python scripts/aws_deploy.py resume
```
