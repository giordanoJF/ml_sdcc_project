# P2P Federated Learning — FEMNIST

## Host Requirements

- Docker + Docker Compose (runs containers)
- Python 3.11+ (host-side scripts)
- git (to clone the LEAF dataset repository)
- Terraform ≥ 1.2 (AWS provisioning — Single EC2 and Multi-instance only)
- Python libraries (host-side scripts):
  ```bash
  pip install -r requirements.debug.txt
  ```

<br>

## Deployment Modes

**Local** — all containers run on the host machine via Docker Compose: one registry + `num_workers` workers.

**Single EC2** — same Docker Compose setup, deployed to one AWS instance. Useful to run experiments on cloud hardware without changing the architecture.

**Multi-instance EC2** — one EC2 instance per worker, plus one for the registry. Workers communicate over real TCP/IP across separate machines.

---

## Setup

Steps 1–4 are common to all deployment modes and always run on the **local machine**.

<br>

**1. Edit `config.yaml`**

**2. Download FEMNIST** — run once; re-run only if `local_test_set` changes in `config.yaml`.

```bash
python scripts/download_femnist.py
```

> Use `--sf 0.05` to download only 5% of the data for quick testing.

<br>

**3. Partition data across workers** — re-run when `num_workers`, `local_test_set`, or `global_test_set` changes.

```bash
python scripts/split_dataset.py
```

**4. Generate `docker-compose.yml`** — re-run when `num_workers`, `use_gpu`, or `global_test_set` changes.

```bash
python scripts/generate_compose.py
```

<br>

### Test sets

Two independent flags in `config.yaml` control how data is split for evaluation:

**`local_test_set`** (default: `false`):
- `false` — 90/10 train/val per worker. Val is used for early stopping and as the final accuracy metric.
- `true` — 80/10/10 train/val/local_test per worker. Val is used only for early stopping; `local_test` holds a per-worker held-out split never used for gradient updates.

<br>

**`global_test_set`** (default: `true`): reserves `global_test_fraction` of writers before any per-worker split. All workers evaluate on this shared set every round — a fully unbiased convergence metric across the whole federation.

<br>

**Comparison metric:** `mean_best_val_accuracy` — average across all workers of each worker's peak validation accuracy over all rounds. Used to compare configurations across runs. Saved to `data/femnist/summary.txt` by `aggregate_metrics.py`.

---

## Local

**Automatic — `run_grid.py` manages the full experimental plan:**

```bash
python scripts/run_grid.py
```

`run_grid.py` handles config updates, dataset re-partitioning when `num_workers` changes, and result archiving automatically.

<br>

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
# edit config.yaml, then start the next run
docker compose up --build
```

> **If training was interrupted** before completion, `metrics.csv` contains all rounds completed so far and must be deleted before re-running — `MetricsWriter` opens it in append mode, so a new run would accumulate on top of the old data, corrupting results. Delete manually or run `save_experiment.py` (which cleans the working directory automatically):
> ```bash
> rm -f data/femnist/worker_*/metrics.csv \
>        data/femnist/worker_*/model_best.pt
> ```

<br>

Depending on what changed in `config.yaml`, insert these steps before `docker compose up --build`:

<br>

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

All setup and deployment is automated via `aws_deploy.py` — no manual SCP or SSH needed to start training. Analysis runs on the EC2 host via SSH after training. `run_grid.py` is also supported (SSH in after `deploy_single`).

<br>

Export credentials (Learner Lab panel → Start Lab → AWS Details → Show):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

<br>

Create the EC2 instance via Terraform (installs Docker automatically, waits until ready):

```bash
python scripts/aws_deploy.py provision_single
```

<br>

Upload code and dataset, install analysis dependencies, and start training — all in one step:

```bash
python scripts/aws_deploy.py deploy_single
```

`deploy_single` handles everything: SCP of sources and `docker-compose.yml`, `pip install -r requirements.debug.txt` on the EC2 host, clean re-upload of `data/femnist/` (old data deleted first), and `docker compose up --build`. Re-running `deploy_single` between runs stops the old containers and reloads everything from scratch.

<br>

**Monitor training (optional):**

```bash
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip> 'cd ~/project && docker compose logs -f'
```

<br>

**When training finishes — analyze on the EC2 host:**

```bash
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip>
cd ~/project
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
```

<br>

**Download results to the local machine, then stop billing:**

```bash
scp -r ubuntu@<ip>:~/project/results ./
python scripts/aws_deploy.py destroy_single   # IMPORTANT: stop billing
```

<br>

---

**If config changes between runs** — update `config.yaml` locally, re-run the relevant scripts, then call `deploy_single` again. It stops old containers, re-uploads everything, and restarts training cleanly.

<br>

**`num_workers` or `global_test_set` changed:**
```bash
python scripts/split_dataset.py
python scripts/generate_compose.py
python scripts/aws_deploy.py deploy_single
```

**`local_test_set` changed** — `download_femnist.py` requires git, which is not installed on the EC2 instance. Run the download locally, then re-deploy:
```bash
python scripts/download_femnist.py
python scripts/split_dataset.py
python scripts/generate_compose.py
python scripts/aws_deploy.py deploy_single
```

<br>

**If the Learner Lab session expired** — AWS stops the instance (not terminates it): disk data on EBS survives, but RAM is wiped and all processes — including the Docker daemon — are lost. Containers do not restart automatically. The instance comes back with a new public IP:

```bash
python scripts/aws_deploy.py resume_single   # updates Terraform state, prints new IP
# Case A — training had already finished: SSH in to analyze, then destroy
# Case B — training was still running: partial metrics.csv and model_best.pt are on EBS.
#           WARNING: deploy_single wipes data/femnist before re-uploading — collect first
#           if you want to keep the partial data, otherwise it is permanently lost.
python scripts/aws_deploy.py collect         # optional: save partial data locally
python scripts/aws_deploy.py deploy_single   # wipes EBS data, re-uploads, restarts
```

---

## Multi-Instance EC2

All orchestration runs locally via `aws_deploy.py` — no manual SSH needed at any step. Training runs on the EC2 instances. `run_grid.py` is not supported in this mode: runs must be managed one at a time.

<br>

Export credentials (same as Single EC2):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

<br>

Create all EC2 instances (one per worker + one for the registry):

```bash
python scripts/aws_deploy.py provision
```

Build images, upload dataset partitions, and start containers — all in one step:

```bash
python scripts/aws_deploy.py deploy
```

<br>

Monitor training — poll `status` until all worker containers show "Exited":

```bash
python scripts/aws_deploy.py status      # container status on every instance
python scripts/aws_deploy.py logs 0      # tail worker_0 logs (Ctrl+C to exit)
python scripts/aws_deploy.py logs registry
```

<br>

When all containers have exited, `collect` downloads `metrics.csv`, `local_test_result.json`, and `model_best.pt` from each worker to `data/femnist/worker_i/` locally. `model_best.pt` is required by `aggregate_metrics.py` for weight divergence analysis. Run `collect` **before** `destroy` — data on EBS is permanently lost after termination:

```bash
python scripts/aws_deploy.py collect
python scripts/aggregate_metrics.py --plot
python scripts/save_experiment.py <name>
python scripts/aws_deploy.py destroy     # IMPORTANT: stop billing
```

<br>

---

**If config changes between runs** — update `config.yaml` locally, re-run the relevant scripts, then call `deploy` again. It stops old containers, re-uploads data, and restarts. No need to `destroy` and re-`provision` between runs.

<br>

**`num_workers` or `global_test_set` changed:**
```bash
python scripts/split_dataset.py
python scripts/aws_deploy.py deploy
```

**`local_test_set` changed** — requires re-download locally (git not installed on EC2):
```bash
python scripts/download_femnist.py
python scripts/split_dataset.py
python scripts/aws_deploy.py deploy
```

<br>

**Inter-worker communication:** workers register their private VPC IP with the registry at startup. Gossip gRPC calls go directly between workers via private IPs — the registry is used only for initial peer discovery. All instances are pinned to the same availability zone (`aws.availability_zone` in `config.yaml`), so intra-cluster traffic is free.

<br>

**If the Learner Lab session expired** — AWS stops all instances (not terminates them): disk data on EBS survives, but RAM is wiped and all processes — including the Docker daemon on each instance — are lost. Containers do not restart automatically. Instances come back with new public IPs:

```bash
python scripts/aws_deploy.py resume     # updates Terraform state, prints new IPs
# Case A — training had already finished: collect, analyze, destroy as normal
# Case B — training was still running: partial metrics.csv and model_best.pt are on EBS.
#           WARNING: deploy wipes data/femnist on every worker before re-uploading — collect
#           first if you want to keep the partial data, otherwise it is permanently lost.
python scripts/aws_deploy.py collect         # optional: save partial data locally
python scripts/aws_deploy.py deploy          # wipes EBS data on all workers, re-uploads, restarts
```
