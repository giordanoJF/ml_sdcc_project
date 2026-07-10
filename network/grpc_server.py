
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


    def __init__(self):
        self.lock = threading.Lock()
        self.weighted_sum: dict | None = None
        # Counts only samples received from neighbors, NOT this worker's own data.
        self.received_samples: int = 0
        # Number of distinct neighbor models accepted this round (for metrics).
        self.messages_received: int = 0


class GossipServicer(gossip_pb2_grpc.GossipServiceServicer):


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

        # Staleness check (one-directional)
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

        #  Online aggregation: maintain a running weighted sum
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
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    # Links our GossipServicer implementation to the GossipService contract
    # defined in proto/gossip.proto. The generated add_* function registers
    # the servicer so that incoming ReceiveModel RPCs are routed to it.
    gossip_pb2_grpc.add_GossipServiceServicer_to_server(
        GossipServicer(buffer, shared_state, max_staleness), server
    )
    server.add_insecure_port(f"[::]:{port}")  # [::] = all interfaces (IPv4 + IPv6)
    server.start()  # non-blocking: server runs in background thread pool
    logger.info(f"gRPC server listening on port {port}")
    return server
