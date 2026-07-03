import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass

import requests
import torch
import yaml

from core.dataset import load_global_test, load_partition
from core.metrics import MetricsWriter
from core.model import FEMNISTModel
from core.trainer import train_step, validate
from network.grpc_client import send_model
from network.grpc_server import AggregationBuffer, start_grpc_server

WORKER_ID = os.environ.get("WORKER_ID", "?")
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [Worker {WORKER_ID}] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    worker_id: str
    total_workers: int
    my_address: str
    registry_url: str
    grpc_port: int
    gossip_fanout: int
    total_rounds: int
    inner_steps: int
    patience: int
    data_dir: str
    batch_size: int
    learning_rate: float
    clip_grad: float
    label_smoothing: float
    dropout_conv: float
    dropout_fc: float
    global_test_dir: str
    metrics_enabled: bool
    metrics_file: str
    drop_prob: float
    crash_prob: float
    grpc_timeout: float
    max_staleness: int


def build_config(cfg: dict) -> WorkerConfig:
    worker_id     = str(os.environ.get("WORKER_ID", "0"))
    total_workers = int(os.environ.get("TOTAL_WORKERS", "3"))
    my_host       = os.environ.get("MY_HOST", f"worker_{worker_id}")

    net_cfg     = cfg["network"]
    fl_cfg      = cfg["federated_learning"]
    ml_cfg      = cfg["machine_learning"]
    fault_cfg   = cfg["fault_injection"]
    metrics_cfg = cfg.get("metrics", {})

    grpc_port  = net_cfg["grpc_port"]

    return WorkerConfig(
        worker_id=worker_id,
        total_workers=total_workers,
        my_address=f"{my_host}:{grpc_port}",
        # REGISTRY_URL can be overridden via env var for AWS multi-instance deploys
        registry_url=os.environ.get("REGISTRY_URL", net_cfg["registry_url"]),
        grpc_port=grpc_port,
        gossip_fanout=net_cfg["gossip_fanout"],
        total_rounds=fl_cfg["total_rounds"],
        inner_steps=fl_cfg["inner_steps_H"],
        patience=fl_cfg["early_stopping_patience"],
        data_dir=ml_cfg["data_dir"],
        batch_size=ml_cfg["batch_size"],
        learning_rate=ml_cfg["learning_rate"],
        clip_grad=ml_cfg.get("clip_grad", 1.0),
        label_smoothing=ml_cfg.get("label_smoothing", 0.1),
        dropout_conv=ml_cfg.get("dropout_conv", 0.25),
        dropout_fc=ml_cfg.get("dropout_fc", 0.5),
        global_test_dir=ml_cfg.get("global_test_dir", "/app/data/femnist/global_test"),
        metrics_enabled=metrics_cfg.get("enabled", True),
        metrics_file=metrics_cfg.get("output_file", "metrics.csv"),
        drop_prob=fault_cfg["drop_probability"],
        crash_prob=fault_cfg["crash_probability"],
        grpc_timeout=fault_cfg["grpc_timeout_seconds"],
        max_staleness=fault_cfg["max_staleness"],
    )


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def register_worker(registry_url: str, worker_id: str, address: str, max_retries: int = 10):
    for attempt in range(max_retries):
        try:
            requests.post(
                f"{registry_url}/register",
                json={"worker_id": worker_id, "address": address},
                timeout=5,
            ).raise_for_status()
            logger.info(f"Registered at {address}")
            return
        except Exception as exc:
            logger.warning(f"Registration attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(3)
    logger.error("Could not register with the Discovery Server. Exiting.")
    sys.exit(1)


def deregister_worker(registry_url: str, worker_id: str, max_retries: int = 10):
    for attempt in range(max_retries):
        try:
            requests.post(
                f"{registry_url}/deregister",
                json={"worker_id": worker_id},
                timeout=5,
            ).raise_for_status()
            return
        except Exception as exc:
            logger.warning(f"Deregistration attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(3)
    logger.error("Could not deregister — registry may have a stale entry for this worker.")


def fetch_peers(registry_url: str, max_retries: int = 3) -> list[str]:
    for attempt in range(max_retries):
        try:
            return requests.get(f"{registry_url}/peers", timeout=5).json()
        except Exception as exc:
            logger.warning(f"Peer fetch attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(1)
    return []


def wait_for_all_peers(
    registry_url: str, total_workers: int, poll_interval: int = 5, max_wait: int = 300
) -> list[str]:
    deadline = time.time() + max_wait
    peers: list[str] = []
    while time.time() < deadline:
        peers = fetch_peers(registry_url)
        if len(peers) >= total_workers:
            logger.info(f"All {total_workers} workers online — peer cache ready")
            return peers
        logger.info(f"Waiting for peers: {len(peers)}/{total_workers} registered ...")
        time.sleep(poll_interval)
    logger.warning(f"Startup timeout: only {len(peers)}/{total_workers} workers registered — proceeding anyway")
    return peers


def infinite_batches(loader):
    # Cycles indefinitely so the training loop can take exactly H steps per round
    # regardless of dataset size.
    while True:
        yield from loader


def main():
    # --- config & device ---
    p = build_config(load_config())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting on {p.my_address} | device={device} | total_workers={p.total_workers}")

    # --- data ---
    train_loader, val_loader, local_test_loader, local_samples = load_partition(p.data_dir, p.batch_size)
    mode = "80/10/10 train/val/local_test" if local_test_loader is not None else "90/10 train/val"
    logger.info(f"Loaded {local_samples} local training samples ({mode})")

    global_test_loader = load_global_test(p.global_test_dir, p.batch_size)
    if global_test_loader is not None:
        logger.info(f"Global test set loaded from {p.global_test_dir} ({len(global_test_loader.dataset)} samples)")

    # --- model / optimizer / metrics ---
    model     = FEMNISTModel(dropout_conv=p.dropout_conv, dropout_fc=p.dropout_fc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p.learning_rate)

    metrics_writer: MetricsWriter | None = None
    if p.metrics_enabled:
        metrics_path   = os.path.join(p.data_dir, p.metrics_file)
        metrics_writer = MetricsWriter(metrics_path, p.worker_id)
        logger.info(f"Metrics logging enabled → {metrics_path}")

    # --- infrastructure: gRPC server + registry + signals ---
    buffer = AggregationBuffer()
    # Written by Thread 2 (Phase C), read by Thread 1 for staleness checks.
    # Plain dict is safe: Python's GIL makes integer assignment atomic for a single writer.
    shared_state = {"current_round": 0}

    grpc_server = start_grpc_server(p.grpc_port, buffer, shared_state, p.max_staleness)
    register_worker(p.registry_url, p.worker_id, p.my_address)

    # SIGTERM (docker stop) and SIGINT (Ctrl+C) are routed through sys.exit so
    # the finally block always runs. SIGKILL cannot be caught — known limitation.
    def _handle_shutdown(signum, frame):
        logger.info(f"Signal {signum} received — shutting down cleanly")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # --- training loop ---
    try:
        # Inside try so SIGTERM during startup polling still triggers finally → deregister.
        peer_cache = wait_for_all_peers(p.registry_url, p.total_workers)

        train_iter       = infinite_batches(train_loader)
        best_val_loss    = float("inf")
        patience_counter = 0

        for round_num in range(1, p.total_rounds + 1):
            round_start = time.time()
            logger.info(f"=== Round {round_num}/{p.total_rounds} ===")

            # --- Phase A: FedAvg aggregation ---
            t_phase_a            = time.time()
            neighbors_aggregated = 0
            with buffer.lock:
                if buffer.received_samples > 0:
                    neighbor_samples     = buffer.received_samples
                    neighbors_aggregated = buffer.messages_received
                    combined_samples     = local_samples + neighbor_samples
                    local_state          = model.state_dict()
                    # FedAvg: new_w = (local_w * local_n + weighted_sum) / combined_n
                    # weighted_sum already holds sum(w_i * sender_n_i) over all received neighbors.
                    new_state = {
                        k: (v.float() * local_samples + buffer.weighted_sum[k].to(v.device)) / combined_samples
                        if v.is_floating_point() else v
                        for k, v in local_state.items()
                    }
                    model.load_state_dict(new_state)
                    buffer.weighted_sum      = None
                    buffer.received_samples  = 0
                    buffer.messages_received = 0
                    logger.info(
                        f"FedAvg applied — local={local_samples}, "
                        f"neighbors={neighbor_samples} ({neighbors_aggregated} models), "
                        f"combined={combined_samples}"
                    )

            val_loss, val_acc = validate(model, val_loader, device)
            logger.info(f"Validation — loss={val_loss:.4f}, accuracy={val_acc:.2%}")

            global_test_acc: float | None = None
            if global_test_loader is not None:
                _, global_test_acc = validate(model, global_test_loader, device)
                logger.info(f"Global test — accuracy={global_test_acc:.2%}")

            phase_a_s = time.time() - t_phase_a

            # 1e-4 threshold avoids saving on negligible improvements
            if val_loss < best_val_loss - 1e-4:
                best_val_loss    = val_loss
                patience_counter = 0
                best_path = os.path.join(p.data_dir, "model_best.pt")
                try:
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"Best model saved → {best_path} (val_loss={val_loss:.4f})")
                except Exception as exc:
                    logger.warning(f"Could not save best model: {exc}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{p.patience}")
                if patience_counter >= p.patience:
                    logger.info("Early stopping triggered. Training loop stopped; gRPC server remains active.")
                    break

            # --- Phase B: local training for H steps ---
            t_phase_b  = time.time()
            total_loss = 0.0
            for _ in range(p.inner_steps):
                total_loss += train_step(
                    model, optimizer, next(train_iter), device,
                    clip_grad=p.clip_grad, label_smoothing=p.label_smoothing,
                )
            train_loss_avg = total_loss / p.inner_steps
            phase_b_s      = time.time() - t_phase_b
            logger.info(f"Local training — avg_loss={train_loss_avg:.4f} over {p.inner_steps} steps")

            if random.random() < p.crash_prob:
                logger.warning("FAULT INJECTION: simulated node crash via sys.exit(1)")
                sys.exit(1)

            # --- Phase C: gossip push ---
            t_phase_c = time.time()
            # Update before sending so the receiver's staleness check sees the correct round.
            shared_state["current_round"] = round_num

            eligible_peers   = [peer for peer in peer_cache if peer != p.my_address]
            targets          = random.sample(eligible_peers, min(p.gossip_fanout, len(eligible_peers))) if eligible_peers else []
            weights_snapshot = model.state_dict()  # snapshot once so all targets get identical weights
            tried            = set(targets)
            sent_count       = 0
            dropped_count    = 0
            failed_targets   = []
            grpc_latencies: list[float] = []

            for target in targets:
                if random.random() < p.drop_prob:
                    dropped_count += 1
                    logger.debug(f"Dropped message to {target}")
                    continue
                t_call  = time.time()
                success = send_model(target, weights_snapshot, round_num, local_samples, p.worker_id, p.grpc_timeout)
                grpc_latencies.append(time.time() - t_call)
                if success:
                    sent_count += 1
                else:
                    failed_targets.append(target)

            # On gRPC failure: refresh peer list and retry once with fresh peers not already tried.
            # Registry traffic is zero in healthy rounds — at most one call per round on failure.
            retried = 0
            if failed_targets:
                fresh_peers  = fetch_peers(p.registry_url)
                if fresh_peers:
                    peer_cache = fresh_peers
                replacements = [peer for peer in peer_cache if peer != p.my_address and peer not in tried]
                for replacement in random.sample(replacements, min(len(failed_targets), len(replacements))):
                    tried.add(replacement)
                    if random.random() < p.drop_prob:
                        dropped_count += 1
                        continue
                    t_call  = time.time()
                    success = send_model(replacement, weights_snapshot, round_num, local_samples, p.worker_id, p.grpc_timeout)
                    grpc_latencies.append(time.time() - t_call)
                    if success:
                        sent_count += 1
                    retried += 1

            phase_c_s = time.time() - t_phase_c
            logger.info(f"Gossip push — sent={sent_count}, dropped={dropped_count}, failed={len(failed_targets)}, retried={retried}")

            grpc_mean_latency_s = sum(grpc_latencies) / len(grpc_latencies) if grpc_latencies else 0.0
            round_duration      = time.time() - round_start

            if metrics_writer is not None:
                metrics_writer.log(
                    round_num=round_num,
                    train_loss_avg=train_loss_avg,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    round_duration_s=round_duration,
                    phase_a_s=phase_a_s,
                    phase_b_s=phase_b_s,
                    phase_c_s=phase_c_s,
                    grpc_mean_latency_s=grpc_mean_latency_s,
                    neighbors_aggregated=neighbors_aggregated,
                    peers_contacted=sent_count,
                    global_test_accuracy=global_test_acc,
                )

    finally:
        # SIGTERM/SIGINT → sys.exit(0) → finally runs. Only SIGKILL bypasses this.
        deregister_worker(p.registry_url, p.worker_id)

    # --- post-training evaluation ---
    # Local test: run once after training, never influences training decisions.
    # Only present when local_test_set: true in config (80/10/10 split mode).
    if local_test_loader is not None:
        local_test_loss, local_test_acc = validate(model, local_test_loader, device)
        logger.info(f"Local test set — loss={local_test_loss:.4f}, accuracy={local_test_acc:.2%}")
        result_path = os.path.join(p.data_dir, "local_test_result.json")
        with open(result_path, "w") as f:
            json.dump({
                "worker_id": p.worker_id,
                "local_test_loss": round(local_test_loss, 6),
                "local_test_accuracy": round(local_test_acc, 6),
            }, f)
        logger.info(f"Local test result saved → {result_path}")

    # 10 s grace: lets in-flight RPCs complete. After deregistration no new peer
    # will select us, so we don't wait indefinitely.
    logger.info("Training complete. Shutting down gRPC server.")
    grpc_server.stop(grace=10)


if __name__ == "__main__":
    main()
