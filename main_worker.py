"""
P2P Worker entry point.

Coordinates two parallel threads:
  - Thread 1 (gRPC Server): started by start_grpc_server(), always listening.
  - Thread 2 (Training Loop): the main thread; runs Phases A, B, C each round.

Round structure
---------------
  Phase A — Weighted FedAvg aggregation with received neighbors' models.
  Phase B — Local training for exactly H inner steps (AdamW optimizer).
  Phase C — Gossip Push: send own weights to M randomly selected peers.
"""
import json
import logging
import os
import random
import signal
import sys
import time

import requests
import torch
import yaml

from core.dataset import load_partition
from core.metrics import MetricsWriter
from core.model import FEMNISTModel
from core.trainer import train_step, validate
from network.grpc_client import send_model
from network.grpc_server import AggregationBuffer, start_grpc_server

# Configure logging early so the worker_id appears in every line
WORKER_ID = os.environ.get("WORKER_ID", "?")
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [Worker {WORKER_ID}] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def register_worker(registry_url: str, worker_id: str, address: str, max_retries: int = 10):
    """
    Register this worker with the Discovery Server.
    Retries with a fixed delay to handle the case where the registry container
    starts slightly after the workers.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{registry_url}/register",
                json={"worker_id": worker_id, "address": address},
                timeout=5,
            )
            response.raise_for_status()
            logger.info(f"Registered at {address}")
            return
        except Exception as exc:
            logger.warning(f"Registration attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(3)
    logger.error("Could not register with the Discovery Server. Exiting.")
    sys.exit(1)


def deregister_worker(registry_url: str, worker_id: str):
    """Best-effort deregistration on clean shutdown (skipped on crash)."""
    try:
        requests.post(
            f"{registry_url}/deregister",
            json={"worker_id": worker_id},
            timeout=5,
        )
    except Exception:
        pass  # non-critical: the registry will eventually serve a stale entry


def fetch_peers(registry_url: str) -> list[str]:
    """Return the list of currently active gRPC addresses from the registry."""
    try:
        return requests.get(f"{registry_url}/peers", timeout=5).json()
    except Exception as exc:
        logger.warning(f"Could not fetch peers: {exc}")
        return []


def infinite_batches(loader):
    """Cycle over a DataLoader indefinitely to allow arbitrary H inner steps."""
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Read identity from environment variables set by docker-compose
    worker_id = str(os.environ.get("WORKER_ID", "0"))
    total_workers = int(os.environ.get("TOTAL_WORKERS", "3"))
    my_host = os.environ.get("MY_HOST", f"worker_{worker_id}")

    net_cfg = cfg["network"]
    fl_cfg = cfg["federated_learning"]
    ml_cfg = cfg["machine_learning"]
    fault_injection_cfg = cfg["fault_injection"]

    # REGISTRY_URL can be overridden via env var for AWS multi-instance deploys
    registry_url = os.environ.get("REGISTRY_URL", net_cfg["registry_url"])
    grpc_port = net_cfg["grpc_port"]
    gossip_fanout = net_cfg["gossip_fanout"]          # k: peers to push weights to each round
    my_address = f"{my_host}:{grpc_port}"

    total_rounds = fl_cfg["total_rounds"]
    inner_steps = fl_cfg["inner_steps_H"]            # H: local steps before gossip
    patience = fl_cfg["early_stopping_patience"]
    data_dir = ml_cfg["data_dir"]
    batch_size = ml_cfg["batch_size"]
    learning_rate = ml_cfg["learning_rate"]
    clip_grad = ml_cfg.get("clip_grad", 1.0)
    label_smoothing = ml_cfg.get("label_smoothing", 0.1)
    dropout_conv = ml_cfg.get("dropout_conv", 0.25)
    dropout_fc = ml_cfg.get("dropout_fc", 0.5)

    metrics_cfg = cfg.get("metrics", {})
    metrics_enabled = metrics_cfg.get("enabled", True)
    metrics_file = metrics_cfg.get("output_file", "metrics.csv")

    drop_prob = fault_injection_cfg["drop_probability"]
    crash_prob = fault_injection_cfg["crash_probability"]
    grpc_timeout = fault_injection_cfg["grpc_timeout_seconds"]
    max_staleness = fault_injection_cfg["max_staleness"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting on {my_address} | device={device} | total_workers={total_workers}")

    # --- Dataset: load this worker's pre-split partition ---
    train_loader, val_loader, test_loader, local_samples = load_partition(data_dir, batch_size)
    # local_samples: number of training examples owned by THIS worker.
    # Kept constant for the entire run; used to weight our contribution in FedAvg.
    if test_loader is not None:
        logger.info(f"Loaded {local_samples} local training samples (80/10/10 mode — test set available)")
    else:
        logger.info(f"Loaded {local_samples} local training samples (90/10 mode — no test set)")

    # --- Model and optimizer ---
    model = FEMNISTModel(dropout_conv=dropout_conv, dropout_fc=dropout_fc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Metrics writer (writes to data_dir/metrics.csv, visible on host) ---
    metrics_writer: MetricsWriter | None = None
    if metrics_enabled:
        metrics_path = os.path.join(data_dir, metrics_file)
        metrics_writer = MetricsWriter(metrics_path, worker_id)
        logger.info(f"Metrics logging enabled → {metrics_path}")

    # --- Shared state between Thread 1 and Thread 2 ---
    buffer = AggregationBuffer()
    # current_round is written by Thread 2 (Phase C) and read by Thread 1
    # for the staleness check; a plain dict is sufficient since Python's GIL
    # makes integer assignment atomic for a single writer.
    shared_state = {"current_round": 0}

    # --- Thread 1: start the gRPC server in background ---
    grpc_server = start_grpc_server(grpc_port, buffer, shared_state, max_staleness)

    # --- Register with the Discovery Server ---
    register_worker(registry_url, worker_id, my_address)

    # --- Signal handlers for clean shutdown ---
    # SIGTERM: sent by `docker stop` / `docker compose down` (10s grace period before SIGKILL).
    # SIGINT:  sent by Ctrl+C, both in an attached terminal and via `docker attach`.
    # Both call sys.exit(0) which raises SystemExit, traversing the finally block below
    # and guaranteeing deregister_worker() and checkpoint save always run.
    # SIGKILL (docker kill, OOM killer) cannot be caught — documented as known limitation.
    def _handle_shutdown(signum, frame):
        logger.info(f"Signal {signum} received — shutting down cleanly")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # --- Thread 2: training loop ---
    train_iter = infinite_batches(train_loader)
    best_val_loss = float("inf")
    patience_counter = 0

    try:
        for round_num in range(1, total_rounds + 1):
            round_start = time.time()
            logger.info(f"=== Round {round_num}/{total_rounds} ===")

            # -----------------------------------------------------------
            # Phase A: Weighted FedAvg aggregation + validation
            # (skipped in baseline mode — phase_a_s = 0 in that case)
            # -----------------------------------------------------------
            t_phase_a = time.time()
            neighbors_aggregated = 0
            with buffer.lock:
                if buffer.received_samples > 0:
                    # neighbor_samples: total training examples contributed by all
                    # neighbors whose updates arrived this round (denominator of
                    # the neighbors' weighted average, already baked into weighted_sum).
                    neighbor_samples = buffer.received_samples
                    neighbors_aggregated = buffer.messages_received
                    # combined_samples: grand total used to weight local vs. neighbors.
                    combined_samples = local_samples + neighbor_samples
                    local_state = model.state_dict()
                    new_state = {}
                    for k, v in local_state.items():
                        if v.is_floating_point():
                            # FedAvg formula:
                            #   new_w = (local_w * local_samples + weighted_sum) / combined_samples
                            # weighted_sum already holds sum(w_i * sender_samples_i)
                            # over all received neighbors, so no intermediate average needed.
                            new_state[k] = (
                                v.float() * local_samples + buffer.weighted_sum[k]
                            ) / combined_samples
                        else:
                            # Non-float buffers (e.g. BatchNorm's num_batches_tracked)
                            # are not averaged; keep the local value.
                            new_state[k] = v
                    model.load_state_dict(new_state)
                    # Reset the buffer so the next round starts fresh
                    buffer.weighted_sum = None
                    buffer.received_samples = 0
                    buffer.messages_received = 0
                    logger.info(
                        f"FedAvg applied — local={local_samples}, "
                        f"neighbors={neighbor_samples} ({neighbors_aggregated} models), "
                        f"combined={combined_samples}"
                    )

            # Validate after aggregation to track convergence for early stopping
            val_loss, val_acc = validate(model, val_loader, device)
            logger.info(f"Validation — loss={val_loss:.4f}, accuracy={val_acc:.2%}")
            phase_a_s = time.time() - t_phase_a

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    # Exit the training loop but keep the gRPC server alive so
                    # other workers can still push their updates to us.
                    logger.info(
                        "Early stopping triggered. "
                        "Training loop stopped; gRPC server remains active."
                    )
                    break

            # -----------------------------------------------------------
            # Phase B: Local training for exactly H inner steps
            # (no network interaction during this phase)
            # -----------------------------------------------------------
            t_phase_b = time.time()
            total_loss = 0.0
            for _ in range(inner_steps):
                batch = next(train_iter)
                total_loss += train_step(
                    model, optimizer, batch, device,
                    clip_grad=clip_grad,
                    label_smoothing=label_smoothing,
                )
            train_loss_avg = total_loss / inner_steps
            phase_b_s = time.time() - t_phase_b
            logger.info(
                f"Local training — avg_loss={train_loss_avg:.4f} "
                f"over {inner_steps} steps"
            )

            # -----------------------------------------------------------
            # Fault injection: random crash simulation
            # -----------------------------------------------------------
            if random.random() < crash_prob:
                # sys.exit raises SystemExit, which is caught by the finally
                # block (deregistration runs) but NOT by wait_for_termination(),
                # so the process actually dies — simulating a real node crash.
                logger.warning("FAULT INJECTION: simulated node crash via sys.exit(1)")
                sys.exit(1)

            # -----------------------------------------------------------
            # Phase C: Gossip Push
            # -----------------------------------------------------------
            sent_count = 0
            phase_c_s = 0.0
            grpc_latencies: list[float] = []
            t_phase_c = time.time()
            # Update current_round before sending so the receiver's staleness
            # check sees the correct value.
            shared_state["current_round"] = round_num

            all_peers = fetch_peers(registry_url)
            # Exclude self to avoid sending to ourselves
            eligible_peers = [p for p in all_peers if p != my_address]
            targets = (
                random.sample(eligible_peers, min(gossip_fanout, len(eligible_peers)))
                if eligible_peers else []
            )

            weights_snapshot = model.state_dict()  # snapshot once, reuse for all targets
            dropped_count = 0
            failed_targets = []
            tried = set(targets)  # all peers attempted, used to avoid duplicates on retry

            for target in targets:
                # Simulate packet loss before attempting the RPC
                if random.random() < drop_prob:
                    dropped_count += 1
                    logger.debug(f"Dropped message to {target}")
                    continue
                t_call = time.time()
                success = send_model(
                    target, weights_snapshot, round_num, local_samples, worker_id, grpc_timeout
                )
                grpc_latencies.append(time.time() - t_call)
                if success:
                    sent_count += 1
                else:
                    failed_targets.append(target)

            # Reactive re-query: if any gRPC push failed (not simulated drops),
            # fetch a fresh peer list and attempt one replacement per failure.
            # This covers the case where a peer crashed and deregistered between
            # our initial fetch_peers() call and the push attempt.
            # At most one extra HTTP call per round, only when failures occur.
            retried = 0
            if failed_targets:
                fresh_peers = fetch_peers(registry_url)
                replacements = [p for p in fresh_peers if p != my_address and p not in tried]
                for replacement in random.sample(replacements, min(len(failed_targets), len(replacements))):
                    tried.add(replacement)
                    if random.random() < drop_prob:
                        dropped_count += 1
                        continue
                    t_call = time.time()
                    success = send_model(
                        replacement, weights_snapshot, round_num, local_samples, worker_id, grpc_timeout
                    )
                    grpc_latencies.append(time.time() - t_call)
                    if success:
                        sent_count += 1
                    retried += 1

            phase_c_s = time.time() - t_phase_c
            logger.info(
                f"Gossip push — sent={sent_count}, dropped={dropped_count}, "
                f"failed={len(failed_targets)}, retried={retried}"
            )

            grpc_mean_latency_s = (sum(grpc_latencies) / len(grpc_latencies)) if grpc_latencies else 0.0

            # -----------------------------------------------------------
            # Log round metrics
            # -----------------------------------------------------------
            round_duration = time.time() - round_start
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
                )

    finally:
        # Save the final model checkpoint to data_dir (visible on host via mount).
        # Used by aggregate_metrics.py to compute inter-worker weight divergence.
        checkpoint_path = os.path.join(data_dir, "model_final.pt")
        try:
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved → {checkpoint_path}")
        except Exception as exc:
            logger.warning(f"Could not save checkpoint: {exc}")

        # Always executed: sys.exit(), SystemExit, KeyboardInterrupt, and normal
        # loop completion all traverse finally. SIGTERM/SIGINT are routed through
        # sys.exit(0) by the signal handlers above. Only SIGKILL bypasses this block.
        deregister_worker(registry_url, worker_id)

    # Final test set evaluation — run once after training, never used for any
    # training decision (not early stopping, not hyperparameter selection).
    # Only present when use_test_set: true in config.yaml (80/10/10 mode).
    if test_loader is not None:
        test_loss, test_acc = validate(model, test_loader, device)
        logger.info(f"Test set evaluation — loss={test_loss:.4f}, accuracy={test_acc:.2%}")
        result_path = os.path.join(data_dir, "test_result.json")
        with open(result_path, "w") as f:
            json.dump({
                "worker_id": worker_id,
                "test_loss": round(test_loss, 6),
                "test_accuracy": round(test_acc, 6),
            }, f)
        logger.info(f"Test result saved → {result_path}")

    # Keep the process alive so Thread 1 can continue serving peers that are
    # still training. Reached only after a clean break from the loop.
    logger.info("Training complete. gRPC server still active for remaining peers.")
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    main()
