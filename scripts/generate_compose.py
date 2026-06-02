#!/usr/bin/env python3
"""
Generate docker-compose.yml from config.yaml.

Workflow:
    1. Edit  network.num_workers  in config.yaml
    2. Run   python scripts/generate_compose.py
    3. Run   docker compose up --build
"""
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def _healthcheck(registry_port: int) -> str:
    return (
        f'      test: ["CMD-SHELL", "python -c \\"import urllib.request; '
        f'urllib.request.urlopen(\'http://localhost:{registry_port}/health\')\\""]'
    )


def write_local_compose(num_workers: int, registry_port: int, use_gpu: bool) -> None:
    """
    Generate docker-compose.yml for local single-machine development.

    Uses string templating (instead of yaml.dump) so that the healthcheck
    test is rendered as an inline flow sequence, which IDE schema validators
    accept without warnings.
    """
    dockerfile = "docker/Dockerfile.worker.gpu" if use_gpu else "docker/Dockerfile.worker"
    image_tag = "fl-worker-gpu" if use_gpu else "fl-worker"

    gpu_block = [
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        "              count: all",
        "              capabilities: [gpu]",
    ] if use_gpu else []

    worker_blocks = []
    for i in range(num_workers):
        worker_blocks += [
            "",
            f"  worker_{i}:",
            f"    image: {image_tag}",
            "    build:",
            "      context: .",
            f"      dockerfile: {dockerfile}",
            "    environment:",
            f"      - WORKER_ID={i}",
            f"      - TOTAL_WORKERS={num_workers}",
            f"      - MY_HOST=worker_{i}",
            "    depends_on:",
            "      registry:",
            "        condition: service_healthy",
            "    volumes:",
            "      - type: bind",
            f"        source: ./data/femnist/worker_{i}",
            "        target: /app/data/femnist",
            "    networks:",
            "      - fl_net",
        ] + gpu_block

    lines = [
        'version: "3.8"',
        "",
        "services:",
        "",
        "  registry:",
        "    build:",
        "      context: .",
        "      dockerfile: docker/Dockerfile.registry",
        "    environment:",
        f"      - REGISTRY_PORT={registry_port}",
        "    ports:",
        f'      - "{registry_port}:{registry_port}"',
        "    networks:",
        "      - fl_net",
        "    healthcheck:",
        _healthcheck(registry_port),
        "      interval: 5s",
        "      timeout: 3s",
        "      retries: 10",
    ] + worker_blocks + [
        "",
        "networks:",
        "  fl_net:",
        "    driver: bridge",
        "",
    ]

    path = os.path.join(PROJECT_ROOT, "docker-compose.yml")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  docker-compose.yml        ({num_workers} workers)")


def main():
    cfg = _load_config()
    num_workers: int = cfg["network"]["num_workers"]
    grpc_port: int = cfg["network"]["grpc_port"]
    registry_port: int = cfg["network"]["registry_port"]
    use_gpu: bool = cfg["federated_learning"].get("use_gpu", False)

    mode = "GPU (Dockerfile.worker.gpu)" if use_gpu else "CPU (Dockerfile.worker)"
    print(f"Generating docker-compose.yml — {num_workers} workers, gRPC port {grpc_port}, mode={mode} ...")
    write_local_compose(num_workers, registry_port, use_gpu)
    print("Done. Run: docker compose up --build")


if __name__ == "__main__":
    main()
