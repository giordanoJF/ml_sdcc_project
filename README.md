# P2P Federated Learning — FEMNIST

## Requirements

- Docker + Docker Compose
- Python 3.11+

```bash
pip install -r requirements.debug.txt
```

## Quick Start

```bash
# 1. Edit config.yaml (num_workers, gossip_fanout, learning_rate, ...)
#    See comments in config.yaml for candidate values to try.

# 2. Download dataset (once, or when use_test_set changes)
python scripts/download_femnist.py --sf 0.05   # 5% for fast local runs
# python scripts/download_femnist.py           # full dataset (--sf 1.0)

# 3. Split into per-worker partitions (re-run when num_workers or use_test_set changes)
python scripts/split_dataset.py

# 4. Generate Docker Compose files (re-run when num_workers changes)
python scripts/generate_compose.py

# 5. Run
docker compose up --build

# 6. Analyze and save results
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <name>   # e.g. lr_1e-3, fanout_2, baseline
# → archives config.yaml + all metrics to results/<timestamp>_<name>/

# To compare configurations: repeat steps 1 and 5-6, varying one parameter at a time.
# Use use_test_set: false for all search runs.
# Once the best config is found, optionally re-run with use_test_set: true
# for an unbiased final accuracy estimate (requires re-running steps 2-3).
```

See `report.md` for full documentation.
