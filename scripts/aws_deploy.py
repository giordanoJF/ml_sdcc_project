#!/usr/bin/env python3
"""
AWS multi-instance deployment orchestrator for P2P Federated Learning.

Each EC2 instance runs exactly one Docker container (one worker or the registry).
Workers communicate over real TCP/IP via private IPs within the same VPC,
producing genuine network latency instead of loopback communication.

Workflow
--------
  # 0. Set Learner Lab credentials (copy from the AWS Academy panel each session)
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_SESSION_TOKEN=...

  # 1. Edit config.yaml (num_workers, aws.key_name, aws.key_path, ...)
  # 2. Download and split dataset (once, or when use_test_set / num_workers changes)
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

  # --- Resuming after a Learner Lab session restart ---
  # When a new lab session starts, EC2 instances are restarted with NEW public IPs.
  # Run 'resume' to refresh Terraform state before using status/logs/collect:
  python scripts/aws_deploy.py resume

Usage
-----
  python scripts/aws_deploy.py <command> [args]

Commands
--------
  provision   Create EC2 instances and security group via Terraform
  deploy      Build Docker images, upload dataset partitions, start containers
  collect     Download metrics.csv (and test_result.json) from each worker
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

PROJECT_ROOT = Path(__file__).parent.parent
TERRAFORM_DIR = PROJECT_ROOT / "terraform"
DATA_ROOT = PROJECT_ROOT / "data" / "femnist"

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
# Helpers: Terraform
# ---------------------------------------------------------------------------

def _terraform(*args):
    subprocess.run(["terraform", f"-chdir={TERRAFORM_DIR}", *args], cwd=PROJECT_ROOT, check=True)


def _terraform_output() -> dict:
    result = subprocess.run(
        ["terraform", f"-chdir={TERRAFORM_DIR}", "output", "-json"],
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
        "Dockerfile.worker", "Dockerfile.registry",
        "requirements.worker.txt", "requirements.registry.txt",
        "gossip.proto", "main_worker.py", "registry_server.py", "config.yaml",
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
             "-f /home/ubuntu/build/Dockerfile.registry /home/ubuntu/build")
    else:
        _ssh(ip, key_path,
             "docker build -t fl-worker:latest "
             "-f /home/ubuntu/build/Dockerfile.worker /home/ubuntu/build")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_resume():
    """Refresh Terraform state after a Learner Lab session restart.

    When a lab session ends and restarts, EC2 instances are stopped and
    restarted with new public IPv4 addresses. This command syncs Terraform
    state with the actual AWS state so that status/logs/collect use correct IPs.
    """
    print("Refreshing Terraform state (public IPs may have changed after session restart)...")
    _terraform("apply", "-refresh-only", "-auto-approve")
    out = _terraform_output()
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

    _terraform("init")
    _terraform("apply", "-auto-approve")

    out = _terraform_output()
    print(f"\nRegistry : {out['registry_public_ip']}  (private {out['registry_private_ip']})")
    for i, (pub, priv) in enumerate(zip(out["worker_public_ips"], out["worker_private_ips"])):
        print(f"Worker {i:2d}: {pub}  (private {priv})")


def cmd_deploy(cfg: dict):
    """Build images on all instances, upload partitions, start containers."""
    aws_cfg  = cfg.get("aws", {})
    net_cfg  = cfg["network"]

    key_path    = os.path.expanduser(aws_cfg["key_path"])
    num_workers = net_cfg["num_workers"]
    grpc_port   = net_cfg["grpc_port"]
    reg_port    = net_cfg["registry_port"]
    img_source  = aws_cfg.get("image_source", "build")

    out          = _terraform_output()
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
        _ssh(ip, key_path, "mkdir -p /home/ubuntu/data/femnist")
        _scp_to(ip, key_path, local, "/home/ubuntu/data/femnist/", recursive=True)
        print(f"  worker_{i} → {ip}: done")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(upload, enumerate(worker_pubs)))

    # ------------------------------------------------------------------
    # 4. Start registry container
    # ------------------------------------------------------------------
    print("\n[4/5] Starting registry...")
    _ssh(reg_pub, key_path,
         f"docker rm -f fl-registry 2>/dev/null || true && "
         f"docker run -d --name fl-registry --restart unless-stopped "
         f"-p {reg_port}:{reg_port} "
         f"-e REGISTRY_PORT={reg_port} "
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
        _ssh(pub, key_path,
             f"docker rm -f fl-worker 2>/dev/null || true && "
             f"docker run -d --name fl-worker "
             f"-p {grpc_port}:{grpc_port} "
             f"-v /home/ubuntu/data/femnist/worker_{i}:/app/data/femnist "
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
    """Download metrics.csv and test_result.json from every worker instance."""
    aws_cfg  = cfg.get("aws", {})
    net_cfg  = cfg["network"]
    key_path = os.path.expanduser(aws_cfg["key_path"])
    num_workers = net_cfg["num_workers"]

    out         = _terraform_output()
    worker_pubs = out["worker_public_ips"]

    print("Collecting metrics...")
    for i, ip in enumerate(worker_pubs):
        local_dir = DATA_ROOT / f"worker_{i}"
        local_dir.mkdir(parents=True, exist_ok=True)
        # Files are written to /app/data/femnist/ inside the container, which
        # maps to /home/ubuntu/data/femnist/worker_{i}/ on the host (the mount point).
        remote_dir = f"/home/ubuntu/data/femnist/worker_{i}"
        for fname in ("metrics.csv", "test_result.json", "model_final.pt"):
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

    out      = _terraform_output()
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

    out = _terraform_output()
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
    _terraform("destroy", "-auto-approve")


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
        description="AWS multi-instance FL deployment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("provision", help="Create EC2 instances via Terraform")
    sub.add_parser("deploy",    help="Build images, upload data, start containers")
    sub.add_parser("collect",   help="Download metrics from all workers")
    sub.add_parser("status",    help="Show container status on all instances")
    sub.add_parser("destroy",   help="Terminate all EC2 instances")
    sub.add_parser("resume",    help="Refresh state after a Learner Lab session restart (IPs change)")
    p = sub.add_parser("logs",  help="Tail logs from a worker or registry")
    p.add_argument("id", nargs="?", default="0", help="Worker ID or 'registry' (default: 0)")

    args = parser.parse_args()
    _check_terraform()
    cfg = _load_config()

    if   args.command == "provision": cmd_provision(cfg)
    elif args.command == "deploy":    cmd_deploy(cfg)
    elif args.command == "collect":   cmd_collect(cfg)
    elif args.command == "status":    cmd_status(cfg)
    elif args.command == "destroy":   cmd_destroy()
    elif args.command == "resume":    cmd_resume()
    elif args.command == "logs":      cmd_logs(cfg, args.id)


if __name__ == "__main__":
    main()
