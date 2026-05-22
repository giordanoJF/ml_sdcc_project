"""
gRPC client for the Gossip Push (Phase C of the training loop).

Serializes model weights into a Protobuf message and sends them to a peer
with a configurable per-call timeout to avoid blocking on crashed nodes.
"""
import io
import logging

import grpc
import torch

import gossip_pb2
import gossip_pb2_grpc

logger = logging.getLogger(__name__)


def serialize_weights(state_dict: dict) -> bytes:
    """Serialize a PyTorch state_dict to bytes using torch.save."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def send_model(
    address: str,
    state_dict: dict,
    round_num: int,
    local_samples: int,
    worker_id: str,
    timeout: float,
) -> bool:
    """
    Send model weights to a peer via gRPC.

    Args:
        address:       Target peer address in "host:port" format.
        state_dict:    Model parameters to transmit.
        round_num:     Current round of the sender (used for staleness check).
        local_samples: Number of training samples this worker trained on locally
                       (used by the receiver for weighted FedAvg).
        worker_id:     Sender identifier (for logging on the receiver side).
        timeout:       Maximum seconds to wait for the RPC to complete.

    Returns:
        True if the peer accepted the message, False on error or rejection.
    """
    try:
        with grpc.insecure_channel(address) as channel:
            stub = gossip_pb2_grpc.GossipServiceStub(channel)
            message = gossip_pb2.ModelMessage(
                weights=serialize_weights(state_dict),
                round=round_num,
                num_samples=local_samples,  # proto field name is fixed by gossip.proto
                worker_id=worker_id,
            )
            ack = stub.ReceiveModel(message, timeout=timeout)
            return ack.accepted
    except grpc.RpcError as e:
        logger.warning(f"Failed to send to {address}: {e.code()} — {e.details()}")
        return False
