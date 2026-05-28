"""
Worker Thread 1: gRPC Server (The Receiver).

Listens for incoming gossip messages from peers, applies a staleness check,
and merges received weights into a shared buffer via online (streaming) aggregation.
The buffer stores a running weighted sum instead of a list of models, keeping
memory usage O(model_size) regardless of how many messages are received.
"""
import concurrent.futures
import io
import logging
import threading

import grpc
import torch

import gossip_pb2
import gossip_pb2_grpc

logger = logging.getLogger(__name__)


class AggregationBuffer:
    """
    Thread-safe shared state between the gRPC server (Thread 1) and the
    training loop (Thread 2).

    Attributes:
        lock:             Mutex protecting both accumulators.
        weighted_sum:     Running weighted sum of received parameters,
                          {param_name -> Tensor}. None until the first message
                          arrives. Each tensor equals sum(w_i * n_i) over all
                          received neighbors i.
        received_samples: Sum of local_samples contributed by every received
                          neighbor. Acts as the denominator when computing the
                          neighbors' weighted average in Phase A.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.weighted_sum: dict | None = None
        # Counts only samples received from neighbors, NOT this worker's own data.
        self.received_samples: int = 0
        # Number of distinct neighbor models accepted this round (for metrics).
        self.messages_received: int = 0


class GossipServicer(gossip_pb2_grpc.GossipServiceServicer):
    """gRPC servicer that implements the ReceiveModel RPC."""

    def __init__(
        self,
        buffer: AggregationBuffer,
        shared_state: dict,
        max_staleness: int,
    ):
        self.buffer = buffer
        # shared_state["current_round"] is written by Thread 2 at the start of
        # Phase C and read here to decide whether an incoming message is stale.
        self.shared_state = shared_state
        self.max_staleness = max_staleness

    def ReceiveModel(self, request: gossip_pb2.ModelMessage, context) -> gossip_pb2.Ack:
        sender_round = request.round
        current_round = self.shared_state["current_round"]

        # --- Staleness check (one-directional) ---
        # Discard messages that are too old. If sender_round > current_round
        # the difference is negative, so messages "from the future" are always
        # accepted — a more advanced peer's update should never be rejected.
        staleness = current_round - sender_round
        if staleness > self.max_staleness:
            logger.debug(
                f"Stale message dropped from {request.worker_id}: "
                f"sender_round={sender_round}, current_round={current_round}, "
                f"staleness={staleness}"
            )
            return gossip_pb2.Ack(accepted=False)

        # Deserialize received weights (weights_only=True prevents arbitrary
        # code execution via pickle that torch.load would otherwise allow)
        weights: dict = torch.load(
            io.BytesIO(request.weights), map_location="cpu", weights_only=True
        )
        # How many training samples the sender trained on locally.
        # Used to weight this neighbor's contribution proportionally.
        sender_samples: int = request.num_samples

        # --- Online aggregation: maintain a running weighted sum ---
        # Instead of storing every received model in a list (O(N * model_size)),
        # we accumulate: weighted_sum += weights * sender_samples
        # so that weighted_sum / received_samples gives the neighbors' average.
        with self.buffer.lock:
            # Scale each float parameter by the sender's sample count
            weighted = {
                k: v.float() * sender_samples
                for k, v in weights.items()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            }
            if self.buffer.received_samples == 0:
                # First contribution this round: initialise the accumulator
                self.buffer.weighted_sum = weighted
            else:
                # Subsequent contributions: accumulate in place
                for k in self.buffer.weighted_sum:
                    self.buffer.weighted_sum[k] += weighted[k]
            self.buffer.received_samples += sender_samples
            self.buffer.messages_received += 1

        logger.debug(
            f"Accepted from {request.worker_id} "
            f"(round={sender_round}, sender_samples={sender_samples})"
        )
        return gossip_pb2.Ack(accepted=True)


def start_grpc_server(
    port: int,
    buffer: AggregationBuffer,
    shared_state: dict,
    max_staleness: int,
) -> grpc.Server:
    """
    Start the gRPC server in background threads and return the server instance.
    The caller should invoke server.wait_for_termination() to keep the process
    alive after the training loop exits.
    """
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    gossip_pb2_grpc.add_GossipServiceServicer_to_server(
        GossipServicer(buffer, shared_state, max_staleness), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"gRPC server listening on port {port}")
    return server
