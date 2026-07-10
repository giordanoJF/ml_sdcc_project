#!/usr/bin/env python3

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

def _ssh(ip: str, key_path: str, cmd: str, *, check: bool = True, capture: bool = False, timeout: float | None = None):
    full = ["ssh", *SSH_OPTS, "-i", key_path, f"ubuntu@{ip}", cmd]
    if capture:
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(full, check=check, timeout=timeout)


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
        try:
            r = _ssh(ip, key_path, "docker info", check=False, capture=True, timeout=15)
            if r.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
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


def _create_single_source_archive() -> str:

    src_files = [
        "docker/Dockerfile.worker", "docker/Dockerfile.registry",
        "requirements.worker.txt", "requirements.registry.txt",
        "requirements.debug.txt",
        "proto/gossip.proto", "main_worker.py", "registry_server.py",
        "config.yaml", "docker-compose.yml",
    ]
    src_dirs = ["core", "network", "scripts"]
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
                for item in sorted(p.rglob("*")):
                    if "__pycache__" not in str(item) and item.suffix != ".pyc":
                        tar.add(item, arcname=str(item.relative_to(PROJECT_ROOT)))
    return tmp.name


def _build_on_instance(ip: str, key_path: str, archive: str, role: str):

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

    print(f"\nNext step:")
    print(f"  python scripts/aws_deploy.py deploy_single")


def cmd_deploy_single(cfg: dict):

    aws_cfg  = cfg.get("aws", {})
    key_path = os.path.expanduser(aws_cfg.get("key_path", "~/Downloads/labsuser.pem"))

    out = _terraform_output(TERRAFORM_SINGLE_DIR)
    ip  = out["public_ip"]

    # 1. Wait for SSH + Docker ready
    print("[1/4] Waiting for SSH + Docker...")
    if not _wait_for_docker(ip, key_path):
        sys.exit(f"ERROR: {ip} never became ready (timeout)")
    print(f"  {ip}: ready")

    # 2. Upload source code (Dockerfiles, app code, docker-compose.yml, scripts)
    print("\n[2/4] Uploading source code...")
    archive = _create_single_source_archive()
    try:
        _ssh(ip, key_path, "mkdir -p /home/ubuntu/project")
        _scp_to(ip, key_path, Path(archive), "/home/ubuntu/project.tar.gz")
        _ssh(ip, key_path,
             "tar -xzf /home/ubuntu/project.tar.gz -C /home/ubuntu/project && "
             "rm /home/ubuntu/project.tar.gz")
        print("  source uploaded and extracted")
    finally:
        os.unlink(archive)

    # Install analysis dependencies (needed to run aggregate_metrics.py on EC2)
    _ssh(ip, key_path,
         "pip install -q -r /home/ubuntu/project/requirements.debug.txt")
    print("  analysis deps installed")

    # 3. Upload dataset (clean old data first to avoid stale files)
    print("\n[3/4] Uploading dataset...")
    if not DATA_ROOT.exists():
        sys.exit(f"ERROR: {DATA_ROOT} not found — run split_dataset.py first")
    _ssh(ip, key_path,
         "rm -rf /home/ubuntu/project/data/femnist && "
         "mkdir -p /home/ubuntu/project/data")
    _scp_to(ip, key_path, DATA_ROOT, "/home/ubuntu/project/data/", recursive=True)
    print("  dataset uploaded")

    # 4. Build images and start containers in detached mode (stop any previous run first)
    print("\n[4/4] Building images and starting containers...")
    _ssh(ip, key_path,
         "cd /home/ubuntu/project && docker compose down 2>/dev/null || true && "
         "docker compose up --build -d")
    print("  containers started")

    print("\nDeploy complete.")
    print(f"  Logs    : ssh -i {key_path} ubuntu@{ip} 'cd ~/project && docker compose logs -f'")
    print(f"  Analyze : ssh -i {key_path} ubuntu@{ip} 'cd ~/project && "
          f"python scripts/aggregate_metrics.py && python scripts/save_experiment.py <nome>'")
    print(f"  Destroy : python scripts/aws_deploy.py destroy_single")


def cmd_collect_single(cfg: dict):
    """Download per-worker output files from the single EC2 instance to data/femnist/."""
    aws_cfg     = cfg.get("aws", {})
    net_cfg     = cfg["network"]
    key_path    = os.path.expanduser(aws_cfg["key_path"])
    num_workers = net_cfg["num_workers"]

    out = _terraform_output(TERRAFORM_SINGLE_DIR)
    ip  = out["public_ip"]

    print(f"Collecting from single EC2 ({ip})...")
    for i in range(num_workers):
        local_dir  = DATA_ROOT / f"worker_{i}"
        local_dir.mkdir(parents=True, exist_ok=True)
        # Containers write to /app/data/femnist (bind-mounted from
        # ~/project/data/femnist/worker_{i} on the EC2 host).
        remote_dir = f"/home/ubuntu/project/data/femnist/worker_{i}"
        for fname in ("metrics.csv", "local_test_result.json", "model_best.pt"):
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

    print("\nRun locally to analyze:")
    print("  python scripts/aggregate_metrics.py --plot")
    print("  python scripts/save_experiment.py <name>")


def cmd_destroy_single():
    """Terminate the single EC2 instance and remove its security group."""
    _terraform(TERRAFORM_SINGLE_DIR, "destroy", "-auto-approve")


def cmd_resume_single():

    print("Refreshing Terraform state for single EC2 instance...")
    _terraform(TERRAFORM_SINGLE_DIR, "apply", "-refresh-only", "-auto-approve")
    out = _terraform_output(TERRAFORM_SINGLE_DIR)
    print(f"\nInstance: {out['public_ip']}  (private {out['private_ip']})")


# ---------------------------------------------------------------------------
# Commands — multi-instance
# ---------------------------------------------------------------------------

def cmd_resume():

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
    # 2. Build Docker images on all instances in parallel
    # ------------------------------------------------------------------
    print("\n[2/5] Building Docker images...")
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
         f"docker run -d --name fl-registry --restart on-failure "
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
        # maps to /home/ubuntu/data/femnist/worker_{i}/ on the host .
        remote_dir = f"/home/ubuntu/data/femnist/worker_{i}"
        for fname in ("metrics.csv", "local_test_result.json", "model_best.pt"):
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
    sub.add_parser("deploy_single",    help="Upload project and dataset, build images, start containers (single EC2)")
    sub.add_parser("collect_single",   help="Download per-worker output files from the single EC2 to data/femnist/")
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
    elif args.command == "deploy_single":    cmd_deploy_single(cfg)
    elif args.command == "collect_single":   cmd_collect_single(cfg)
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
