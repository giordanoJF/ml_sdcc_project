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

# Auto-shutdown: when all expected workers have finished and deregistered, the
# registry exits so docker compose returns to the terminal automatically.
# TOTAL_WORKERS is injected by generate_compose.py; if absent, auto-shutdown is disabled.
_expected_workers: int = int(os.environ.get("TOTAL_WORKERS", "0"))
_peak_registered: int = 0  # max simultaneous registrations seen

# Inactivity watchdog: once all expected workers have been seen, if no deregistration
# happens for _WATCHDOG_TIMEOUT seconds the registry forces shutdown. This handles
# workers that exit ungracefully (SIGKILL, OOM) after registering — they never call
# /deregister, so the registry would otherwise stall indefinitely.
_WATCHDOG_TIMEOUT: int = 3600  # 60 minutes
_watchdog_timer: threading.Timer | None = None


def _watchdog_fire() -> None:
    logging.warning(
        f"WATCHDOG_TIMEOUT: no deregistration in {_WATCHDOG_TIMEOUT // 60} min — "
        "forcing shutdown (ungraceful worker exits suspected)."
    )
    os._exit(1)


def _arm_watchdog() -> None:
    """Start or restart the inactivity watchdog."""
    global _watchdog_timer
    if _watchdog_timer is not None:
        _watchdog_timer.cancel()
    _watchdog_timer = threading.Timer(_WATCHDOG_TIMEOUT, _watchdog_fire)
    _watchdog_timer.daemon = True
    _watchdog_timer.start()


def _cancel_watchdog() -> None:
    global _watchdog_timer
    if _watchdog_timer is not None:
        _watchdog_timer.cancel()
        _watchdog_timer = None


def _schedule_shutdown() -> None:
    """Exit 2 s after the last worker deregisters (gives time to send the HTTP response)."""
    _cancel_watchdog()
    def _exit():
        logging.info("RUN_TERMINATION: NORMAL — all workers deregistered cleanly.")
        os._exit(0)
    threading.Timer(2.0, _exit).start()


@app.post("/register")
def register():
    """Add a worker to the active registry."""
    global _peak_registered
    data = request.get_json()
    worker_id = data["worker_id"]
    address = data["address"]
    with _lock:
        _registry[worker_id] = address
        if len(_registry) > _peak_registered:
            _peak_registered = len(_registry)
            if _expected_workers > 0 and _peak_registered >= _expected_workers:
                # All expected workers are now registered — arm the watchdog.
                _arm_watchdog()
    logging.info(f"Registered: {worker_id} @ {address}")
    return jsonify({"status": "ok"})


@app.post("/deregister")
def deregister():
    """Remove a worker from the active registry."""
    data = request.get_json()
    worker_id = data["worker_id"]
    with _lock:
        _registry.pop(worker_id, None)
        empty = len(_registry) == 0
        all_seen = _expected_workers > 0 and _peak_registered >= _expected_workers
    logging.info(f"Deregistered: {worker_id} — active={len(_registry)}")
    if all_seen:
        if empty:
            _schedule_shutdown()
        else:
            _arm_watchdog()  # reset the timer: still waiting for remaining workers
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
