#!/usr/bin/env python3
"""
AWS deployment orchestrator for P2P Federated Learning.

Supports two deployment modes:
  - Single EC2:      all containers (N workers + registry) on one instance.
                     Uses docker-compose.yml, just like local mode.
  - Multi-instance:  one container per EC2 instance, workers communicate
                     over real TCP/IP between separate machines.

--- SINGLE EC2 ---

  # 0. Set Learner Lab credentials (copy from the AWS Academy panel each session)
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_SESSION_TOKEN=...

  # 1. Edit config.yaml (num_workers, aws.key_name, aws.key_path, aws.instance_type_single)
  # 2. Download and split dataset (once, or when local_test_set / num_workers changes)
  python scripts/download_femnist.py
  python scripts/split_dataset.py
  python scripts/generate_compose.py

  # 3. Provision the instance via Terraform (creates EC2 + installs Docker automatically)
  python scripts/aws_deploy.py provision_single

  # 4. Upload project, install dependencies, start training
  scp -r . ubuntu@<ip>:~/project
  ssh -i ~/Downloads/labsuser.pem ubuntu@<ip>
    cd ~/project
    pip install -r requirements.debug.txt
    docker compose up --build

  # 5. Analyze results (still on EC2 host)
  python scripts/aggregate_metrics.py
  python scripts/save_experiment.py <name>

  # 6. Destroy the instance to stop billing
  python scripts/aws_deploy.py destroy_single

  # --- If the Learner Lab session expired while the instance was still running ---
  #
  # The Learner Lab has a ~4h session limit. If it expires before you run
  # destroy_single, AWS stops the instance automatically. Data on disk (metrics.csv,
  # dataset) survives. But the Docker containers are stopped.
  #
  # When you start a new session, the instance restarts with a NEW public IP.
  # resume_single updates the Terraform state so the new IP is known.
  #
  # Then, depending on what happened:
  #
  #   Case A — training had already finished:
  #     python scripts/aws_deploy.py resume_single   # prints new IP
  #     ssh ubuntu@<new_ip>
  #       python scripts/aggregate_metrics.py
  #       python scripts/save_experiment.py <name>
  #     python scripts/aws_deploy.py destroy_single
  #
  #   Case B — training was still running when session expired (model state lost):
  #     python scripts/aws_deploy.py resume_single   # prints new IP
  #     ssh ubuntu@<new_ip>
  #       cd ~/project && docker compose up          # restart from round 1
  #     python scripts/aws_deploy.py destroy_single

--- MULTI-INSTANCE ---

  # 0. Set Learner Lab credentials (same as above)
  # 1. Edit config.yaml (num_workers, aws.key_name, aws.key_path, aws.instance_type_worker)
  # 2. Download and split dataset
  python scripts/download_femnist.py
  python scripts/split_dataset.py

  # 3. Provision EC2 instances via Terraform
  python scripts/aws_deploy.py provision

  # 4. Build images, upload partitions, start containers
  python scripts/aws_deploy.py deploy

  # 5. Monitor training
  python scripts/aws_deploy.py status
  python scripts/aws_deploy.py logs 0        # tail worker_0
  python scripts/aws_deploy.py logs registry

  # 6. Collect metrics once training finishes
  python scripts/aws_deploy.py collect
  python scripts/aggregate_metrics.py
  python scripts/save_experiment.py <name>

  # 7. Destroy all instances to stop billing (IMPORTANT: do this after every session)
  python scripts/aws_deploy.py destroy

  # --- If the Learner Lab session expired while instances were still running ---
  #
  # The Learner Lab has a ~4h session limit. If it expires before you run destroy,
  # AWS stops all instances automatically. Data on disk (metrics.csv, dataset
  # partitions) survives on EBS. But Docker containers are stopped and do not
  # restart automatically (workers have no restart policy).
  #
  # When you start a new session, all instances restart with NEW public IPs.
  # resume updates the Terraform state so the new IPs are known.
  #
  # Then, depending on what happened:
  #
  #   Case A — training had already finished before the session expired:
  #     python scripts/aws_deploy.py resume    # prints new IPs for all instances
  #     python scripts/aws_deploy.py collect   # SCP metrics.csv from each worker
  #     python scripts/aggregate_metrics.py
  #     python scripts/save_experiment.py <name>
  #     python scripts/aws_deploy.py destroy
  #
  #   Case B — training was still running when the session expired (model state lost,
  #            no checkpointing — training must restart from round 1):
  #     python scripts/aws_deploy.py resume    # prints new IPs
  #     python scripts/aws_deploy.py deploy    # rebuild images, restart containers
  #     # ... wait for training to finish ...
  #     python scripts/aws_deploy.py collect
  #     python scripts/aws_deploy.py destroy

Usage
-----
  python scripts/aws_deploy.py <command> [args]

Commands — single EC2
---------------------
  provision_single   Create one EC2 instance via Terraform (docker-compose mode)
  destroy_single     Terminate the single EC2 instance
  resume_single      Refresh Terraform state after a session restart (IP changed)

Commands — multi-instance
-------------------------
  provision   Create EC2 instances and security group via Terraform
  deploy      Build Docker images, upload dataset partitions, start containers
  collect     Download metrics.csv (and local_test_result.json) from each worker
  status      Show Docker container status on every instance
  logs [id]   Tail logs from worker <id> (default 0) or 'registry'
  resume      Refresh Terraform state after a Learner Lab session restart (IPs change)
  destroy     Terminate all EC2 instances and remove security group
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

PROJECT_ROOT         = Path(__file__).parent.parent
TERRAFORM_DIR        = PROJECT_ROOT / "terraform"
TERRAFORM_SINGLE_DIR = PROJECT_ROOT / "terraform" / "single"
DATA_ROOT            = PROJECT_ROOT / "data" / "femnist"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
]


# ---------------------------------------------------------------------------
# Helpers: SSH / SCP
# ---------------------------------------------------------------------------

def _ssh(ip: str, key_path: str, cmd: str, *, check: bool = True, capture: bool = False):
    full = ["ssh", *SSH_OPTS, "-i", key_path, f"ubuntu@{ip}", cmd]
    if capture:
        return subprocess.run(full, capture_output=True, text=True)
    return subprocess.run(full, check=check)


def _scp_to(ip: str, key_path: str, local: Path, remote: str, *, recursive: bool = False):
    flags = ["-r"] if recursive else []
    subprocess.run(
        ["scp", *flags, *SSH_OPTS, "-i", key_path, str(local), f"ubuntu@{ip}:{remote}"],
        check=True,
    )


def _scp_from(ip: str, key_path: str, remote: str, local: Path, *, recursive: bool = False):
    local.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-r"] if recursive else []
    subprocess.run(
        ["scp", *flags, *SSH_OPTS, "-i", key_path, f"ubuntu@{ip}:{remote}", str(local)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Helpers: Terraform (dir-parametric so multi and single can share them)
# ---------------------------------------------------------------------------

def _terraform(terraform_dir: Path, *args):
    subprocess.run(["terraform", f"-chdir={terraform_dir}", *args], cwd=PROJECT_ROOT, check=True)


def _terraform_output(terraform_dir: Path) -> dict:
    result = subprocess.run(
        ["terraform", f"-chdir={terraform_dir}", "output", "-json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    raw = json.loads(result.stdout)
    return {k: v["value"] for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Helpers: instance readiness
# ---------------------------------------------------------------------------

def _wait_for_docker(ip: str, key_path: str, timeout: int = 300) -> bool:
    """Poll until SSH is reachable and Docker daemon is running."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _ssh(ip, key_path, "docker info", check=False, capture=True)
        if r.returncode == 0:
            return True
        time.sleep(10)
    return False


# ---------------------------------------------------------------------------
# Image build helpers
# ---------------------------------------------------------------------------

def _create_source_archive() -> str:
    """Pack source files needed for docker build into a temp .tar.gz."""
    src_files = [
        "docker/Dockerfile.worker", "docker/Dockerfile.registry",
        "requirements.worker.txt", "requirements.registry.txt",
        "proto/gossip.proto", "main_worker.py", "registry_server.py", "config.yaml",
    ]
    src_dirs = ["core", "network"]
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()
    with tarfile.open(tmp.name, "w:gz") as tar:
        for f in src_files:
            p = PROJECT_ROOT / f
            if p.exists():
                tar.add(p, arcname=f)
        for d in src_dirs:
            p = PROJECT_ROOT / d
            if p.exists():
                tar.add(p, arcname=d)
    return tmp.name


def _build_on_instance(ip: str, key_path: str, archive: str, role: str):
    """Upload source archive and docker build the appropriate image."""
    _scp_to(ip, key_path, Path(archive), "/home/ubuntu/src.tar.gz")
    _ssh(ip, key_path,
         "mkdir -p /home/ubuntu/build && "
         "tar -xzf /home/ubuntu/src.tar.gz -C /home/ubuntu/build")
    if role == "registry":
        _ssh(ip, key_path,
             "docker build -t fl-registry:latest "
             "-f /home/ubuntu/build/docker/Dockerfile.registry /home/ubuntu/build")
    else:
        _ssh(ip, key_path,
             "docker build -t fl-worker:latest "
             "-f /home/ubuntu/build/docker/Dockerfile.worker /home/ubuntu/build")


# ---------------------------------------------------------------------------
# Commands — single EC2
# ---------------------------------------------------------------------------

def cmd_provision_single(cfg: dict):
    """Create one EC2 instance via Terraform (single-EC2 / docker-compose mode).

    Docker is installed automatically via user_data; this command waits until
    the instance is SSH-reachable and Docker is ready before returning.
    """
    aws_cfg = cfg.get("aws", {})
    key_path = os.path.expanduser(aws_cfg.get("key_path", "~/Downloads/labsuser.pem"))

    tfvars = {
        "key_name":           aws_cfg.get("key_name", "vockey"),
        "region":             aws_cfg.get("region", "us-east-1"),
        "availability_zone":  aws_cfg.get("availability_zone", "us-east-1a"),
        "instance_type":      aws_cfg.get("instance_type_single", "t3.large"),
        "volume_size":        aws_cfg.get("volume_size_single", 20),
    }
    tfvars_path = TERRAFORM_SINGLE_DIR / "terraform.tfvars"
    with open(tfvars_path, "w") as f:
        for k, v in tfvars.items():
            f.write(f'{k} = "{v}"\n' if isinstance(v, str) else f"{k} = {v}\n")
    print(f"Written {tfvars_path}")

    _terraform(TERRAFORM_SINGLE_DIR, "init")
    _terraform(TERRAFORM_SINGLE_DIR, "apply", "-auto-approve")

    out = _terraform_output(TERRAFORM_SINGLE_DIR)
    ip  = out["public_ip"]
    print(f"\nInstance: {ip}  (private {out['private_ip']})")

    print("Waiting for SSH + Docker to be ready...")
    if _wait_for_docker(ip, key_path):
        print(f"  {ip}: ready")
    else:
        print(f"  WARNING: instance may not be fully ready yet — wait a moment before connecting.")

    print(f"\nNext steps:")
    print(f"  scp -r . ubuntu@{ip}:~/project")
    print(f"  ssh -i {key_path} ubuntu@{ip}")
    print(f"    cd ~/project")
    print(f"    pip install -r requirements.debug.txt")
    print(f"    docker compose up --build")


def cmd_destroy_single():
    """Terminate the single EC2 instance and remove its security group."""
    _terraform(TERRAFORM_SINGLE_DIR, "destroy", "-auto-approve")


def cmd_resume_single():
    """Refresh Terraform state after a Learner Lab session restart (single EC2).

    The Learner Lab has a ~4h session limit. To avoid expiry during a long run,
    click "Start Lab" again before the timer reaches 0:00 to renew the session —
    this makes resume_single unnecessary.

    If the session expires before destroy_single is run, AWS stops the instance
    automatically — data on EBS (metrics.csv, dataset) survives, but Docker
    containers are stopped.

    When a new session starts and the instance restarts, it gets a NEW public IP.
    This command syncs the Terraform state file so the new IP is known.

    After this command, check whether training had finished (Case A) or not (Case B):

      Case A — training finished:
        ssh ubuntu@<new_ip> "cd ~/project && python scripts/aggregate_metrics.py"
        python scripts/aws_deploy.py destroy_single

      Case B — training was mid-run (model state lost, no checkpointing):
        ssh ubuntu@<new_ip> "cd ~/project && docker compose up"  # restart from round 1
        python scripts/aws_deploy.py destroy_single
    """
    print("Refreshing Terraform state for single EC2 instance...")
    _terraform(TERRAFORM_SINGLE_DIR, "apply", "-refresh-only", "-auto-approve")
    out = _terraform_output(TERRAFORM_SINGLE_DIR)
    print(f"\nInstance: {out['public_ip']}  (private {out['private_ip']})")


# ---------------------------------------------------------------------------
# Commands — multi-instance
# ---------------------------------------------------------------------------

def cmd_resume():
    """Refresh Terraform state after a Learner Lab session restart (multi-instance).

    The Learner Lab has a ~4h session limit. To avoid expiry during a long run,
    click "Start Lab" again before the timer reaches 0:00 to renew the session —
    this makes resume unnecessary.

    If the session expires before destroy is run, AWS stops all instances
    automatically — data on EBS (metrics.csv, dataset partitions) survives,
    but Docker containers are stopped (workers have no restart policy).

    When a new session starts and instances restart, they get NEW public IPs.
    This command syncs the Terraform state file so the new IPs are known.
    It does not change anything on AWS and does not restart containers.

    After this command, check whether training had finished (Case A) or not (Case B):

      Case A — training finished before the session expired:
        python scripts/aws_deploy.py collect   # metrics.csv is on disk, SCP works
        python scripts/aggregate_metrics.py
        python scripts/aws_deploy.py destroy

      Case B — training was mid-run (model state in RAM was lost, no checkpointing):
        python scripts/aws_deploy.py deploy    # re-upload source, restart containers
        # training restarts from round 1
        python scripts/aws_deploy.py collect
        python scripts/aws_deploy.py destroy
    """
    print("Refreshing Terraform state (public IPs may have changed after session restart)...")
    _terraform(TERRAFORM_DIR, "apply", "-refresh-only", "-auto-approve")
    out = _terraform_output(TERRAFORM_DIR)
    print(f"\nRegistry : {out['registry_public_ip']}  (private {out['registry_private_ip']})")
    for i, (pub, priv) in enumerate(zip(out["worker_public_ips"], out["worker_private_ips"])):
        print(f"Worker {i:2d}: {pub}  (private {priv})")


def cmd_provision(cfg: dict):
    """Generate terraform.tfvars from config.yaml and run terraform apply."""
    aws_cfg = cfg.get("aws", {})
    net_cfg = cfg["network"]

    num_workers = net_cfg["num_workers"]

    # Learner Lab hard limit: maximum 9 concurrently running EC2 instances.
    # This deployment needs num_workers + 1 (registry) instances.
    total_instances = num_workers + 1
    if total_instances > 9:
        sys.exit(
            f"ERROR: num_workers={num_workers} would require {total_instances} EC2 instances, "
            f"but AWS Learner Lab allows a maximum of 9. "
            f"Set num_workers <= 8 in config.yaml before provisioning."
        )

    # vCPU check: t3.small/medium/large all use 2 vCPUs; limit is 32.
    # 9 × t3.large (2 vCPU) = 18 vCPUs — always within limits for supported types.

    tfvars = {
        "num_workers":            num_workers,
        "instance_type_worker":   aws_cfg.get("instance_type_worker", "t3.small"),
        "instance_type_registry": aws_cfg.get("instance_type_registry", "t3.micro"),
        "volume_size_worker":     aws_cfg.get("volume_size_worker", 20),
        "volume_size_registry":   aws_cfg.get("volume_size_registry", 8),
        "availability_zone":      aws_cfg.get("availability_zone", "us-east-1a"),
        "key_name":               aws_cfg.get("key_name", "vockey"),
        "region":                 aws_cfg.get("region", "us-east-1"),
        "registry_port":          net_cfg["registry_port"],
        "grpc_port":              net_cfg["grpc_port"],
    }

    tfvars_path = TERRAFORM_DIR / "terraform.tfvars"
    with open(tfvars_path, "w") as f:
        for k, v in tfvars.items():
            f.write(f'{k} = "{v}"\n' if isinstance(v, str) else f"{k} = {v}\n")
    print(f"Written {tfvars_path}")

    _terraform(TERRAFORM_DIR, "init")
    _terraform(TERRAFORM_DIR, "apply", "-auto-approve")

    out = _terraform_output(TERRAFORM_DIR)
    print(f"\nRegistry : {out['registry_public_ip']}  (private {out['registry_private_ip']})")
    for i, (pub, priv) in enumerate(zip(out["worker_public_ips"], out["worker_private_ips"])):
        print(f"Worker {i:2d}: {pub}  (private {priv})")


def cmd_deploy(cfg: dict):
    """Build images on all instances, upload partitions, start containers."""
    aws_cfg  = cfg.get("aws", {})
    net_cfg  = cfg["network"]
    ml_cfg   = cfg.get("machine_learning", {})

    key_path        = os.path.expanduser(aws_cfg["key_path"])
    num_workers     = net_cfg["num_workers"]
    grpc_port       = net_cfg["grpc_port"]
    reg_port        = net_cfg["registry_port"]
    img_source      = aws_cfg.get("image_source", "build")
    global_test_set = ml_cfg.get("global_test_set", False)

    out          = _terraform_output(TERRAFORM_DIR)
    reg_pub      = out["registry_public_ip"]
    reg_priv     = out["registry_private_ip"]
    worker_pubs  = out["worker_public_ips"]
    worker_privs = out["worker_private_ips"]
    all_ips      = [reg_pub] + worker_pubs

    # ------------------------------------------------------------------
    # 1. Wait for every instance to have Docker ready
    # ------------------------------------------------------------------
    print("\n[1/5] Waiting for instances (SSH + Docker)...")
    with ThreadPoolExecutor(max_workers=len(all_ips)) as ex:
        futures = {ex.submit(_wait_for_docker, ip, key_path): ip for ip in all_ips}
        for fut in as_completed(futures):
            ip = futures[fut]
            if not fut.result():
                sys.exit(f"ERROR: {ip} never became ready (timeout)")
            print(f"  {ip}: ready")

    # ------------------------------------------------------------------
    # 2. Build (or pull) Docker images on all instances in parallel
    # ------------------------------------------------------------------
    print(f"\n[2/5] Getting Docker images (image_source={img_source})...")
    if img_source == "build":
        archive = _create_source_archive()
        try:
            roles = ["registry"] + ["worker"] * num_workers
            def build(args):
                ip, role = args
                _build_on_instance(ip, key_path, archive, role)
                print(f"  {ip} ({role}): build OK")

            with ThreadPoolExecutor(max_workers=len(all_ips)) as ex:
                futures = {ex.submit(build, (ip, role)): ip
                           for ip, role in zip(all_ips, roles)}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        sys.exit(f"ERROR: build failed on {futures[fut]}: {e}")
        finally:
            os.unlink(archive)
    else:
        image = aws_cfg.get("dockerhub_image", "")
        if not image:
            sys.exit("ERROR: dockerhub_image must be set when image_source=dockerhub")
        for ip in all_ips:
            _ssh(ip, key_path, f"docker pull {image}")

    # ------------------------------------------------------------------
    # 3. Upload dataset partitions to worker instances
    # ------------------------------------------------------------------
    print("\n[3/5] Uploading dataset partitions...")
    def upload(args):
        i, ip = args
        local = DATA_ROOT / f"worker_{i}"
        if not local.exists():
            sys.exit(f"ERROR: {local} not found — run split_dataset.py first")
        _ssh(ip, key_path,
             "rm -rf /home/ubuntu/data/femnist && mkdir -p /home/ubuntu/data/femnist")
        _scp_to(ip, key_path, local, "/home/ubuntu/data/femnist/", recursive=True)
        print(f"  worker_{i} → {ip}: done")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(upload, enumerate(worker_pubs)))

    if global_test_set:
        global_test_local = DATA_ROOT / "global_test"
        if not global_test_local.exists():
            sys.exit(
                "ERROR: data/femnist/global_test/ not found — "
                "run split_dataset.py with global_test_set: true first"
            )
        print("  Uploading global_test to all workers...")
        def upload_global_test(ip):
            _scp_to(ip, key_path, global_test_local,
                    "/home/ubuntu/data/femnist/", recursive=True)
            print(f"  global_test → {ip}: done")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(upload_global_test, worker_pubs))

    # ------------------------------------------------------------------
    # 4. Start registry container
    # ------------------------------------------------------------------
    print("\n[4/5] Starting registry...")
    _ssh(reg_pub, key_path,
         f"docker rm -f fl-registry 2>/dev/null || true && "
         f"docker run -d --name fl-registry --restart unless-stopped "
         f"-p {reg_port}:{reg_port} "
         f"-e REGISTRY_PORT={reg_port} "
         f"-e TOTAL_WORKERS={num_workers} "
         f"fl-registry:latest")

    print(f"  Waiting for registry health...", end="", flush=True)
    health_url = f"http://{reg_pub}:{reg_port}/peers"
    for _ in range(30):
        try:
            urllib.request.urlopen(health_url, timeout=3)
            print(" OK")
            break
        except Exception:
            time.sleep(3)
            print(".", end="", flush=True)
    else:
        sys.exit("ERROR: registry never became healthy")

    # ------------------------------------------------------------------
    # 5. Start worker containers
    # ------------------------------------------------------------------
    print("\n[5/5] Starting workers...")
    def start_worker(args):
        i, pub, priv = args
        # Mount worker_{i}/ directly as /app/data/femnist so the container
        # sees train/ and val/ at the top level, matching load_partition(data_dir).
        global_test_mount = (
            f"-v /home/ubuntu/data/femnist/global_test:/app/data/femnist/global_test:ro "
            if global_test_set else ""
        )
        _ssh(pub, key_path,
             f"docker rm -f fl-worker 2>/dev/null || true && "
             f"docker run -d --name fl-worker "
             f"-p {grpc_port}:{grpc_port} "
             f"-v /home/ubuntu/data/femnist/worker_{i}:/app/data/femnist "
             f"{global_test_mount}"
             f"-e WORKER_ID={i} "
             f"-e TOTAL_WORKERS={num_workers} "
             f"-e MY_HOST={priv} "
             f"-e REGISTRY_URL=http://{reg_priv}:{reg_port} "
             f"fl-worker:latest")
        print(f"  worker_{i} started on {pub}")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(start_worker,
                    [(i, worker_pubs[i], worker_privs[i]) for i in range(num_workers)]))

    print("\nDeploy complete.")
    print(f"  Monitor : python scripts/aws_deploy.py status")
    print(f"  Logs    : python scripts/aws_deploy.py logs 0")
    print(f"  Collect : python scripts/aws_deploy.py collect  (after training ends)")


def cmd_collect(cfg: dict):
    """Download per-worker output files from every worker instance."""
    aws_cfg  = cfg.get("aws", {})
    net_cfg  = cfg["network"]
    key_path = os.path.expanduser(aws_cfg["key_path"])
    num_workers = net_cfg["num_workers"]

    out         = _terraform_output(TERRAFORM_DIR)
    worker_pubs = out["worker_public_ips"]

    print("Collecting metrics...")
    for i, ip in enumerate(worker_pubs):
        local_dir = DATA_ROOT / f"worker_{i}"
        local_dir.mkdir(parents=True, exist_ok=True)
        # Files are written to /app/data/femnist/ inside the container, which
        # maps to /home/ubuntu/data/femnist/worker_{i}/ on the host (the mount point).
        remote_dir = f"/home/ubuntu/data/femnist/worker_{i}"
        for fname in ("metrics.csv", "local_test_result.json"):
            probe = _ssh(ip, key_path,
                         f"test -f {remote_dir}/{fname} && echo yes || echo no",
                         capture=True)
            if probe.stdout.strip() == "yes":
                _scp_from(ip, key_path,
                          f"{remote_dir}/{fname}",
                          local_dir / fname)
                print(f"  worker_{i}: {fname} ✓")
            else:
                print(f"  worker_{i}: {fname} — not found (training still running?)")


def cmd_status(cfg: dict):
    """Print Docker container status on every instance."""
    aws_cfg  = cfg.get("aws", {})
    key_path = os.path.expanduser(aws_cfg["key_path"])

    out      = _terraform_output(TERRAFORM_DIR)
    all_ips  = [out["registry_public_ip"]] + out["worker_public_ips"]
    labels   = ["registry"] + [f"worker_{i}" for i in range(len(out["worker_public_ips"]))]

    for label, ip in zip(labels, all_ips):
        r = _ssh(ip, key_path,
                 "docker ps --format '{{.Names}}\t{{.Status}}'",
                 capture=True)
        status = r.stdout.strip() or "(no running containers)"
        print(f"{label:12s} {ip}  {status}")


def cmd_logs(cfg: dict, worker_id: str):
    """Tail container logs — replaces this process with SSH (Ctrl+C to exit)."""
    aws_cfg  = cfg.get("aws", {})
    key_path = os.path.expanduser(aws_cfg["key_path"])

    out = _terraform_output(TERRAFORM_DIR)
    if worker_id == "registry":
        ip        = out["registry_public_ip"]
        container = "fl-registry"
    else:
        idx       = int(worker_id)
        ip        = out["worker_public_ips"][idx]
        container = "fl-worker"

    # Replace current process with interactive SSH — Ctrl+C works cleanly
    os.execlp("ssh", "ssh", *SSH_OPTS, "-i", key_path,
              f"ubuntu@{ip}", f"docker logs -f {container}")


def cmd_destroy():
    """Terminate all EC2 instances and remove the security group."""
    _terraform(TERRAFORM_DIR, "destroy", "-auto-approve")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _check_terraform():
    if subprocess.run(["which", "terraform"], capture_output=True).returncode != 0:
        sys.exit("ERROR: terraform not found in PATH. Install from https://developer.hashicorp.com/terraform/install")


def main():
    parser = argparse.ArgumentParser(
        description="AWS FL deployment orchestrator (single EC2 and multi-instance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Single EC2 commands
    sub.add_parser("provision_single", help="Create one EC2 instance via Terraform (docker-compose mode)")
    sub.add_parser("destroy_single",   help="Terminate the single EC2 instance")
    sub.add_parser("resume_single",    help="Refresh state after a session restart (single EC2)")

    # Multi-instance commands
    sub.add_parser("provision", help="Create EC2 instances via Terraform (multi-instance mode)")
    sub.add_parser("deploy",    help="Build images, upload data, start containers")
    sub.add_parser("collect",   help="Download metrics from all workers")
    sub.add_parser("status",    help="Show container status on all instances")
    sub.add_parser("destroy",   help="Terminate all EC2 instances")
    sub.add_parser("resume",    help="Refresh state after a Learner Lab session restart (multi-instance)")
    p = sub.add_parser("logs",  help="Tail logs from a worker or registry")
    p.add_argument("id", nargs="?", default="0", help="Worker ID or 'registry' (default: 0)")

    args = parser.parse_args()
    _check_terraform()
    cfg = _load_config()

    if   args.command == "provision_single": cmd_provision_single(cfg)
    elif args.command == "destroy_single":   cmd_destroy_single()
    elif args.command == "resume_single":    cmd_resume_single()
    elif args.command == "provision":        cmd_provision(cfg)
    elif args.command == "deploy":           cmd_deploy(cfg)
    elif args.command == "collect":          cmd_collect(cfg)
    elif args.command == "status":           cmd_status(cfg)
    elif args.command == "destroy":          cmd_destroy()
    elif args.command == "resume":           cmd_resume()
    elif args.command == "logs":             cmd_logs(cfg, args.id)


if __name__ == "__main__":
    main()
