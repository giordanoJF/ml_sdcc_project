"""
Per-worker metrics logging.

Each worker writes one CSV row per round to {data_dir}/metrics.csv.
Because data_dir is mounted from the host, the file is immediately
visible on the host without any extra data transfer.

After the experiment, run scripts/aggregate_metrics.py to produce
global statistics across all workers.
"""
import csv
import os
import time


FIELDS = [
    "worker_id",
    "round",
    "timestamp",
    "train_loss_avg",
    "val_loss",
    "val_accuracy",
    "round_duration_s",
    "phase_a_s",             # FedAvg aggregation + validation (Phase A)
    "phase_b_s",             # local training for H steps (Phase B)
    "phase_c_s",             # gossip push to k peers (Phase C)
    "grpc_mean_latency_s",   # mean latency per actual gRPC send_model call (0 if none)
    "neighbors_aggregated",  # distinct models incorporated in Phase A
    "peers_contacted",       # successful gossip pushes in Phase C
]


class MetricsWriter:
    """Appends one CSV row per round to output_path."""

    def __init__(self, output_path: str, worker_id: str):
        self.path = output_path
        self.worker_id = worker_id
        # Write header only if the file does not already exist (supports resume).
        if not os.path.exists(output_path):
            with open(output_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(
        self,
        round_num: int,
        train_loss_avg: float,
        val_loss: float,
        val_accuracy: float,
        round_duration_s: float,
        phase_a_s: float,
        phase_b_s: float,
        phase_c_s: float,
        grpc_mean_latency_s: float,
        neighbors_aggregated: int,
        peers_contacted: int,
    ) -> None:
        row = {
            "worker_id": self.worker_id,
            "round": round_num,
            "timestamp": round(time.time(), 3),
            "train_loss_avg": round(train_loss_avg, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
            "round_duration_s": round(round_duration_s, 3),
            "phase_a_s": round(phase_a_s, 4),
            "phase_b_s": round(phase_b_s, 4),
            "phase_c_s": round(phase_c_s, 4),
            "grpc_mean_latency_s": round(grpc_mean_latency_s, 6),
            "neighbors_aggregated": neighbors_aggregated,
            "peers_contacted": peers_contacted,
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
