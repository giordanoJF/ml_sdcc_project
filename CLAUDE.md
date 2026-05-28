# CLAUDE.md — Federated Learning P2P (ML+SDCC)

## Comportamento

- **Mai** eseguire `git commit` o `git push` senza esplicita richiesta esplicita nel turno corrente. Non proporre mai di fare commit — aspettare che sia l'utente a dirlo. Dopo modifiche al codice, presentare solo un sommario e fermarsi.
- Non aggiungere firma, menzione di Claude, o co-author nei messaggi di commit.
- Fare commit per feature o refactoring separati, non tutto insieme.

## Progetto

Progetto universitario (corso ML+SDCC, a.a. 2025-26): Federated Learning decentralizzato P2P ispirato a DiLoCo. Nessun aggregatore globale — i worker si scambiano modelli via gossip gRPC.

**Dataset:** LEAF FEMNIST (28×28 grayscale, 62 classi), partizione non-i.i.d. per worker.

**Stack:** Python 3.11, PyTorch, gRPC, Flask, Docker, docker-compose, Terraform (AWS EC2).

## Architettura

| File | Ruolo |
|---|---|
| `registry_server.py` | Discovery Server Flask — registra IP:Port dei worker |
| `main_worker.py` | Entrypoint worker: lancia thread gRPC server + training loop |
| `network/grpc_server.py` | Riceve modelli in arrivo, staleness check, aggregazione online |
| `network/grpc_client.py` | Gossip push verso peer con timeout |
| `core/dataset.py` | Partizionamento deterministico LEAF non-i.i.d. — carica TUTTO in RAM in `__init__` (non lazy) |
| `core/model.py` | CNN FEMNIST |
| `core/trainer.py` | `train_step` e `validate` |

## Workflow locale

```
config.yaml → scripts/split_dataset.py → scripts/generate_compose.py → docker compose up --build
```

## Deployment AWS (Learner Lab)

Due modalità, entrambe configurate da `config.yaml` (sezione `aws`):

- **Single EC2** (`terraform/single/`): tutti i container su una macchina, docker-compose. Max t3.large (Learner Lab non supporta xlarge o superiori).
- **Multi-instance** (`terraform/`): registry + N worker su istanze separate.

`scripts/aws_deploy.py` legge `config.yaml` e passa tutto a Terraform via tfvars — non modificare i file `.tf` direttamente per i parametri.

**Parametri chiave in `config.yaml`:**
- `availability_zone`: pin tutti in stesso AZ → traffico intra-cluster gratuito
- `volume_size_*`: EBS gp3 esplicito (più economico e veloce di gp2)

## Documentazione

`docs/report.md` è la fonte primaria di documentazione tecnica (architettura, esperimenti, rationale). `README.md` copre il workflow operativo.
