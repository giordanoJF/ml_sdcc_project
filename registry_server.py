import logging
import os
import threading

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [REGISTRY] %(message)s")

app = Flask(__name__)

_registry: dict[str, str] = {}
_lock = threading.Lock()

_expected: int = int(os.environ.get("TOTAL_WORKERS", "0"))
_peak: int = 0           # max simultaneous registrations; used to detect full-cluster events
_WATCHDOG_TIMEOUT = 3600  # 60 min; handles workers that exit without calling /deregister
_watchdog: threading.Timer | None = None


def _on_watchdog_fire() -> None:
    logging.warning(f"WATCHDOG: no deregistration in {_WATCHDOG_TIMEOUT // 60} min — forcing shutdown.")
    os._exit(1)


def _arm_watchdog() -> None:
    global _watchdog
    if _watchdog:
        _watchdog.cancel()
    _watchdog = threading.Timer(_WATCHDOG_TIMEOUT, _on_watchdog_fire)
    _watchdog.daemon = True
    _watchdog.start()


def _schedule_shutdown() -> None:
    global _watchdog
    if _watchdog:
        _watchdog.cancel()
        _watchdog = None
    def _exit():
        logging.info("RUN_TERMINATION: NORMAL — all workers deregistered cleanly.")
        os._exit(0)
    threading.Timer(2.0, _exit).start()


@app.post("/register")
def register():
    global _peak
    data = request.get_json()
    with _lock:
        _registry[data["worker_id"]] = data["address"]
        if len(_registry) > _peak:
            _peak = len(_registry)
            if _expected > 0 and _peak >= _expected:
                _arm_watchdog()
    logging.info(f"Registered: {data['worker_id']} @ {data['address']}")
    return jsonify({"status": "ok"})


@app.post("/deregister")
def deregister():
    data = request.get_json()
    with _lock:
        _registry.pop(data["worker_id"], None)
        empty = not _registry
        all_seen = _expected > 0 and _peak >= _expected
        active = len(_registry)
    logging.info(f"Deregistered: {data['worker_id']} — active={active}")
    if all_seen:
        _schedule_shutdown() if empty else _arm_watchdog()
    return jsonify({"status": "ok"})


@app.get("/peers")
def get_peers():
    with _lock:
        return jsonify(list(_registry.values()))


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("REGISTRY_PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
