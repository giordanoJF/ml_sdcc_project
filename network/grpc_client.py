
import io
import logging

import grpc
import torch

import gossip_pb2
import gossip_pb2_grpc

logger = logging.getLogger(__name__)


def serialize_weights(state_dict: dict) -> bytes:

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

    try:
        with grpc.insecure_channel(
            address,
            options=[("grpc.max_send_message_length", 50 * 1024 * 1024)],
        ) as channel:
            stub = gossip_pb2_grpc.GossipServiceStub(channel)
            message = gossip_pb2.ModelMessage(
                weights=serialize_weights(state_dict),
                round=round_num,
                num_samples=local_samples,  # proto field name is fixed by proto/gossip.proto
                worker_id=worker_id,
            )
            ack = stub.ReceiveModel(message, timeout=timeout)
            return ack.accepted
    except grpc.RpcError as e:
        logger.warning(f"Failed to send to {address}: {e.code()} — {e.details()}")
        return False
