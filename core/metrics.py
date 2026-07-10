
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
    "val_macro_precision",   # macro-averaged precision on val set
    "val_macro_recall",      # macro-averaged recall on val set — low value = specific classes missed
    "val_macro_f1",          # macro-averaged F1 on val set — catches per-class imbalance accuracy hides
    "global_test_accuracy",  # accuracy on shared global test set (empty if global_test_set: false)
    "global_test_macro_precision",  # macro-averaged precision on shared global test set (empty if disabled)
    "global_test_macro_recall",     # macro-averaged recall on shared global test set (empty if disabled)
    "global_test_macro_f1",  # macro-averaged F1 on shared global test set (empty if global_test_set: false)
    "round_duration_s",
    "phase_a_s",             # FedAvg aggregation + validation (Phase A)
    "phase_b_s",             # local training for H steps (Phase B)
    "phase_c_s",             # gossip push to k peers (Phase C)
    "grpc_mean_latency_s",   # mean latency per actual gRPC send_model call (0 if none)
    "neighbors_aggregated",  # distinct models incorporated in Phase A
    "peers_contacted",       # successful gossip pushes in Phase C
]


class MetricsWriter:

    def __init__(self, output_path: str, worker_id: str):
        self.path = output_path
        self.worker_id = worker_id
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
        val_macro_precision: float = 0.0,
        val_macro_recall: float = 0.0,
        val_macro_f1: float = 0.0,
        global_test_accuracy: float | None = None,
        global_test_macro_precision: float | None = None,
        global_test_macro_recall: float | None = None,
        global_test_macro_f1: float | None = None,
    ) -> None:
        row = {
            "worker_id": self.worker_id,
            "round": round_num,
            "timestamp": round(time.time(), 3),
            "train_loss_avg": round(train_loss_avg, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
            "val_macro_precision": round(val_macro_precision, 6),
            "val_macro_recall": round(val_macro_recall, 6),
            "val_macro_f1": round(val_macro_f1, 6),
            "global_test_accuracy": round(global_test_accuracy, 6) if global_test_accuracy is not None else "",
            "global_test_macro_precision": round(global_test_macro_precision, 6) if global_test_macro_precision is not None else "",
            "global_test_macro_recall": round(global_test_macro_recall, 6) if global_test_macro_recall is not None else "",
            "global_test_macro_f1": round(global_test_macro_f1, 6) if global_test_macro_f1 is not None else "",
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
