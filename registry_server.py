"""
Discovery Server (Registry).

Sole responsibility: service discovery.
Keeps a live map of {worker_id -> grpc_address} and exposes three REST endpoints.
Does NOT handle model weights, hyperparameters, or training state.
"""
import logging
import os
import threading

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [REGISTRY] %(message)s")

app = Flask(__name__)

# In-memory registry: {worker_id: "host:port"}
_registry: dict[str, str] = {}
_lock = threading.Lock()  # protects concurrent register/deregister calls


@app.post("/register")
def register():
    """Add a worker to the active registry."""
    data = request.get_json()
    worker_id = data["worker_id"]
    address = data["address"]
    with _lock:
        _registry[worker_id] = address
    logging.info(f"Registered: {worker_id} @ {address}")
    return jsonify({"status": "ok"})


@app.post("/deregister")
def deregister():
    """Remove a worker from the active registry."""
    data = request.get_json()
    worker_id = data["worker_id"]
    with _lock:
        _registry.pop(worker_id, None)
    logging.info(f"Deregistered: {worker_id}")
    return jsonify({"status": "ok"})


@app.get("/peers")
def get_peers():
    """Return the list of currently active gRPC addresses."""
    with _lock:
        return jsonify(list(_registry.values()))


@app.get("/health")
def health():
    """Liveness probe used by Docker healthcheck — does not appear in worker traffic."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("REGISTRY_PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
