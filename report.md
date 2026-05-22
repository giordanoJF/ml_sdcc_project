# Federated Learning Decentralizzato Peer-to-Peer con Gossip Protocol
## Relazione Tecnica di Progetto

**Corso:** Machine Learning + Sistemi Distribuiti e Cloud Computing — A.A. 2025-26

---

## Abstract

Il presente documento descrive la progettazione e l'implementazione di un sistema di Federated Learning (FL) completamente decentralizzato in modalità peer-to-peer (P2P). L'architettura adottata elimina il ruolo del server aggregatore globale tipico del FL classico, sostituendolo con un protocollo gossip asincrono ispirato al framework DiLoCo [1]. Ciascun nodo partecipante esegue un numero elevato di step di ottimizzazione locale prima di diffondere i propri parametri a un sottoinsieme casuale di vicini, riducendo significativamente il volume di comunicazione rispetto al paradigma federato standard. L'aggregazione dei modelli ricevuti avviene mediante una variante decentralizzata di FedAvg [2] con tecnica di *online aggregation*, che mantiene il consumo di memoria costante rispetto al numero di messaggi ricevuti — O(dimensione del modello) — indipendentemente dal fan-in della rete. Il sistema è implementato interamente in Python 3, containerizzato tramite Docker e progettato per il deployment su istanze AWS EC2 senza modifiche al codice sorgente. Il documento illustra in dettaglio le scelte implementative, le motivazioni architetturali, i trade-off di progettazione e i meccanismi di fault injection adottati per validare la robustezza del sistema.

---

## 1. Introduzione

Il Federated Learning è un paradigma di addestramento distribuito in cui i dati rimangono locali sui dispositivi partecipanti e solo i parametri del modello vengono condivisi con un aggregatore centrale [2]. Nell'architettura classica (FedAvg centralizzato), un server raccoglie i modelli da tutti i client, ne calcola la media pesata e ridistribuisce il modello aggiornato. Questa soluzione, pur semplice da implementare, presenta tre criticità strutturali dal punto di vista dei sistemi distribuiti:

1. **Single point of failure**: il server centrale è l'unico componente in grado di produrre il modello aggregato; il suo guasto interrompe immediatamente il processo di training per l'intera rete.
2. **Collo di bottiglia sulla banda**: tutte le trasmissioni di peso transitano attraverso il server; al crescere del numero di partecipanti o della dimensione del modello, la banda disponibile al server diventa il fattore limitante.
3. **Sincronizzazione globale**: il server deve attendere un quorum di client prima di procedere all'aggregazione, introducendo dipendenze temporali che rendono il sistema sensibile a ritardi e crash parziali.

Il presente progetto adotta un'architettura alternativa interamente decentralizzata, in cui ogni nodo comunica direttamente con i propri vicini senza intermediari di aggregazione. La scoperta dei peer è delegata a un componente di *service discovery* (Discovery Server) che mantiene esclusivamente gli indirizzi di rete e non partecipa mai all'elaborazione dei modelli. La propagazione dei parametri avviene tramite gossip asincrono: ogni worker, al termine di $H$ step di ottimizzazione locale, invia i propri pesi a $M$ vicini selezionati casualmente. Questo schema, ispirato a DiLoCo [1], consente al sistema di operare in modo completamente asincrono e di tollerare guasti parziali senza interruzione del training globale.

---

## 2. Background e Riferimenti

### 2.1 Federated Learning e FedAvg

L'algoritmo Federated Averaging (FedAvg), introdotto da McMahan et al. [2], costituisce la base teorica del meccanismo di aggregazione adottato. Nella sua formulazione originale, un server centrale calcola la media pesata dei parametri ricevuti dai client, dove il peso di ciascun contributo è proporzionale al numero di campioni di training locali:

$$w_{\text{global}} = \frac{\sum_{k=1}^{K} n_k \cdot w_k}{\sum_{k=1}^{K} n_k}$$

dove $w_k$ sono i parametri del modello del nodo $k$, $n_k$ il numero di campioni locali e $K$ il numero totale di partecipanti. La ponderazione per $n_k$ è fondamentale: un nodo con 10.000 campioni deve influenzare il modello aggregato più di uno con 100 campioni, altrimenti la media non riflette la distribuzione reale dei dati nell'intera rete.

Nel contesto decentralizzato del presente sistema, la formula viene adattata: ogni worker non aggrega l'intera rete, ma integra il proprio modello con la media pesata degli aggiornamenti ricevuti dai vicini nel round corrente. La derivazione è descritta in dettaglio nella Sezione 4.2.

### 2.2 DiLoCo e Sparse Communication

DiLoCo [1] propone un paradigma di training distribuito in cui ogni partecipante esegue un numero elevato di step di ottimizzazione locale — denominati *inner steps* — prima di ogni sincronizzazione con gli altri nodi. Questo riduce la frequenza di comunicazione di un fattore $H$ rispetto al training distribuito sincrono standard, dove $H$ è il numero di inner steps configurato. Il principio alla base è che, per modelli con molti parametri, il costo computazionale di un singolo step di ottimizzazione è trascurabile rispetto al costo di trasmissione del modello; conviene quindi ammortizzare il costo di comunicazione su quanti più step locali possibile.

Con $H = 500$, ogni worker trasmette i propri pesi solo al termine di 500 batch di training. Supponendo batch da 32 campioni, ciò equivale a 16.000 esempi elaborati per ogni gossip push. L'impatto sulla qualità del modello aggregato è limitato perché gli inner steps locali producono aggiornamenti nella stessa direzione generale del gradiente globale, convergendo verso una soluzione compatibile con quella degli altri worker.

DiLoCo introduce inoltre la tolleranza esplicita al drop asincrono dei messaggi: un aggiornamento mancante in un round non blocca il training del nodo mittente né quello del ricevente, che proseguono indipendentemente. Questo comportamento è intrinseco all'architettura gossip asincrona adottata: il buffer di aggregazione è semplicemente vuoto al termine del round se nessun vicino ha inviato aggiornamenti.

### 2.3 Dataset LEAF e FEMNIST

Il dataset FEMNIST, distribuito dal framework LEAF [3], è il benchmark standard per il Federated Learning non-i.i.d. Deriva da EMNIST ed è organizzato per autore: ogni utente ha uno stile di scrittura caratteristico, producendo una distribuzione dei dati naturalmente eterogenea tra i partecipanti — proprietà definita *non independent and identically distributed* (non-i.i.d.). Ogni campione è un'immagine in scala di grigi di dimensione $28 \times 28$ pixel, con 62 classi (cifre 0–9 e lettere a–z, A–Z).

#### Struttura degli oggetti di dominio

L'entità fondamentale del dataset è il **writer** (chiamato `user` nel formato LEAF) — una persona reale che ha scritto caratteri a mano. Ogni writer ha uno stile di scrittura proprio e ha prodotto un certo numero di immagini di caratteri diversi. Il dataset completo conta **3.597 writer** nel training set, per un totale di **734.463 immagini**, con una media di circa 204 immagini per writer.

LEAF serializza il dataset in file JSON distribuiti in `train/` e `test/`. Ogni file contiene fino a 100 writer e ha la seguente struttura:

```json
{
  "users": ["f1967_21", "f1968_05", ...],
  "num_samples": [105, 88, ...],
  "user_data": {
    "f1967_21": {
      "x": [
        [0.12, 0.0, 0.87, ..., 0.4],
        [0.0,  0.4, 0.1,  ..., 0.2],
        ...
      ],
      "y": [7, 7, 0, 8, 6, ...]
    },
    ...
  }
}
```

I campi hanno il seguente significato:

- **`users`** — lista degli ID writer presenti in questo file. L'ID (es. `f1967_21`) è un codice anonimizzato assegnato da LEAF.
- **`num_samples`** — numero di immagini per ciascun writer, nell'ordine corrispondente a `users`.
- **`user_data`** — dizionario che mappa ogni writer ai propri dati:
  - **`x`** — lista di immagini. Ogni immagine è un vettore flat di **784 float** in $[0, 1]$, corrispondente ai pixel di un'immagine $28 \times 28$ in scala di grigi normalizzata.
  - **`y`** — lista di etichette intere in $[0, 61]$: 0–9 per le cifre, 10–35 per le maiuscole A–Z, 36–61 per le minuscole a–z.

Il training set completo è distribuito su **36 file JSON**; ogni file copre fino a 100 writer.

#### Trasformazione degli oggetti attraverso la pipeline

I dati subiscono tre trasformazioni successive prima di essere usati dal modello:

**1. Lettura e fusione (`_read_json_shards`).**
Tutti i file JSON di una split (`train/` o `test/`) vengono letti e fusi in due strutture:
- `all_users`: lista flat di tutti i writer nell'ordine originale di LEAF (ordine deterministico).
- `user_data`: dizionario globale `{writer_id → {x, y}}`.

**2. Partizionamento per worker (`split_dataset.py`).**
La lista `all_users` viene divisa in $N$ slice contigue di dimensione $\lfloor |\mathcal{U}|/N \rfloor$, dove $\mathcal{U}$ è l'insieme dei writer e $N$ è `num_workers`. Con $N=3$ e 3.597 writer:

```
Worker 0 → writer    0–1198  (~1.199 writer, ~245.000 immagini)
Worker 1 → writer 1199–2397  (~1.199 writer, ~245.000 immagini)
Worker 2 → writer 2398–3596  (~1.199 writer, ~245.000 immagini)
```

Ogni slice viene scritta in un file `data/femnist/worker_{i}/train/data.json` separato, con la stessa struttura JSON originale ma contenente solo i writer di quel worker.

**3. Appiattimento e tensori (`collect_samples` + `FEMNISTDataset`).**
All'interno del container, `load_partition` legge il proprio `data.json` e appiattisce tutti i campioni di tutti i writer in due liste parallele:
- `train_x`: lista di vettori da 784 float — tutte le immagini del worker.
- `train_y`: lista di etichette corrispondenti.

`FEMNISTDataset` converte queste liste in tensori PyTorch e ridimensiona ogni vettore da flat $(784,)$ a immagine $(1, 28, 28)$, che è il formato atteso dalla CNN:

```python
self.x = torch.tensor(x_data, dtype=torch.float32).view(-1, 1, 28, 28)
self.y = torch.tensor(y_data, dtype=torch.long)
```

#### Perché la non-i.i.d. emerge naturalmente

Poiché i writer vengono assegnati per slice contigue e LEAF li ordina per ID (che codifica il writer reale), ogni worker riceve gli stili di scrittura di un sottoinsieme specifico e distinto di persone. La distribuzione delle classi varia tra worker: uno scrittore potrebbe aver prodotto molte lettere maiuscole e poche cifre, un altro il contrario. Non esiste alcun meccanismo artificiale per garantire la non-i.i.d. — emerge direttamente dalla struttura del dataset, che riflette la variabilità naturale della scrittura umana.

La proprietà non-i.i.d. è cruciale per la valutazione realistica del sistema: un modello che converge su dati non-i.i.d. con comunicazione rara dimostra la robustezza dell'algoritmo di aggregazione in condizioni fedeli a quelle di un deployment reale.

#### Split train/test fisso vs cross-validation

LEAF fornisce uno split train/test predeterminato (90% train, 10% test, configurabile tramite `--tf`). Questo è lo schema adottato da tutte le paper di riferimento sul benchmark FEMNIST — incluse FedAvg [2] e le varianti DiLoCo-inspired — ed è la scelta adottata in questo progetto. La porzione di test viene usata localmente da ogni worker come validation set per il controllo dell'early stopping.

Un'alternativa sarebbe la **k-fold cross-validation**, in cui i dati di ogni worker vengono suddivisi in $k$ fold, il training viene ripetuto $k$ volte usando a rotazione un fold diverso come validation, e i risultati vengono mediati. Il confronto tra i due approcci nel contesto FL è il seguente:

| | Split fisso (adottato) | K-fold cross-validation |
|---|---|---|
| **Costo computazionale** | Training eseguito una sola volta | Training ripetuto $k$ volte per worker |
| **Stima della generalizzazione** | Singola stima, dipendente dal seed dello split | Stima più robusta, riduce la varianza |
| **Proprietà non-i.i.d.** | Preservata: LEAF assegna scrittori interi a train o test | Potenzialmente alterata: rimescolare i dati può mescolare gli stili tra fold |
| **Standard nella letteratura FL** | Sì — approccio universale nelle paper FL | No — raramente usato in FL |
| **Complessità implementativa** | Nessuna | Richiede modifiche a `load_partition` e al training loop |

La cross-validation locale è teoricamente realizzabile: ogni worker dividerebbe indipendentemente la propria partizione in $k$ fold, eseguirebbe $k$ round di training completi e ne medierebbe i risultati. In pratica richiederebbe di: (1) modificare `load_partition` per accettare un parametro `fold_index` e restituire il fold corretto come validation set; (2) avvolgere l'intero training loop in un ciclo esterno su $k$ iterazioni; (3) aggregare le metriche di validazione tra i fold prima di applicare l'early stopping. Il costo computazionale sarebbe $k \times$ quello attuale — ingiustificato per un sistema già distribuito e per l'obiettivo di questo progetto, che è validare la convergenza e non ottimizzare iperparametri.

---

## 3. Architettura del Sistema

Il sistema è composto da due tipologie di componenti con responsabilità nettamente separate: il **Discovery Server** (Registry) e i **nodi Worker**. Questa separazione è un vincolo di progettazione deliberato: il Registry non deve mai conoscere la struttura del modello, i suoi parametri o qualsiasi informazione relativa al training. La Figura 1 illustra l'architettura logica complessiva.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Sistema P2P                                 │
│                                                                      │
│   ┌─────────────┐   register/deregister/get_peers   ┌─────────────┐ │
│   │   Registry  │ ◄────────────────────────────────►│  Worker 0   │ │
│   │   (Flask)   │                                   │ Thread 1 gRPC│ │
│   └─────────────┘                                   │ Thread 2  ML │ │
│                                                     └──────┬───────┘ │
│                                      gossip push (gRPC)   │         │
│                              ┌────────────────────────────┤         │
│                              ▼                            ▼         │
│                    ┌─────────────────┐        ┌─────────────────┐   │
│                    │    Worker 1     │        │    Worker 2     │   │
│                    │  Thread 1 gRPC  │        │  Thread 1 gRPC  │   │
│                    │  Thread 2  ML   │        │  Thread 2  ML   │   │
│                    └─────────────────┘        └─────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```
*Figura 1 — Architettura logica del sistema P2P. Le frecce continue rappresentano comunicazioni gRPC (gossip push); le frecce tratteggiate rappresentano le interazioni REST con il Registry.*

### 3.1 Discovery Server (Registry)

#### Ruolo e responsabilità

Il Discovery Server è implementato come un server HTTP ultra-leggero in Flask (`registry_server.py`). Il suo ruolo è limitato esclusivamente alla *service discovery*: mantiene in memoria una mappa `{worker_id → indirizzo_gRPC}` e la espone tramite tre endpoint REST:

- `POST /register` — registra un worker con il proprio indirizzo `host:porta`;
- `POST /deregister` — rimuove un worker dalla lista attiva (chiamato nel blocco `finally` al termine del processo);
- `GET /peers` — restituisce la lista degli indirizzi gRPC correntemente attivi.

Il vincolo più importante di questo componente è **l'assoluta assenza di logica di training e di topologia**: il Registry non conosce né la struttura del modello, né i suoi parametri, né alcun iperparametro, né le relazioni di vicinanza tra i nodi. La selezione dei peer con cui comunicare è responsabilità esclusiva di ciascun worker (Sezione 4.2, Fase C). Questa separazione garantisce che il componente rimanga un semplice name server, scalabile e rimpiazzabile senza impatto sul processo di apprendimento.

#### Scelta tecnologica: Flask vs alternative

La scelta di Flask rispetto ad alternative come FastAPI o un server gRPC è motivata dalla natura delle operazioni esposte. Il Registry riceve al massimo $N$ chiamate a `/register` all'avvio, $N$ chiamate a `/deregister` alla chiusura, e $N \times R$ chiamate a `/get_peers` durante il training (una per worker per round, con $R$ numero di round). Il carico è quindi **O(N × R)** richieste semplici, con payload JSON di poche decine di byte. In questo scenario Flask — con una singola dipendenza, nessuna configurazione e avvio in meno di un secondo — supera FastAPI per semplicità senza sacrificare le prestazioni.

Un server gRPC per il Registry sarebbe stato più coerente con il resto del sistema ma avrebbe aggiunto complessità (generazione di un secondo file proto, gestione di due porte) senza alcun vantaggio funzionale.

#### Thread safety del dizionario interno

Flask in modalità di sviluppo è single-threaded, ma in produzione (o con `threaded=True`) può gestire più richieste concorrentemente. Il dizionario `_registry` è protetto da un `threading.Lock` per evitare race condition in caso di registrazioni o deregistrazioni simultanee. Senza lock, una sequenza `pop()` + `items()` concorrente potrebbe produrre viste inconsistenti della lista peer.

L'implementazione sceglie consapevolmente di **non persistere lo stato su disco**: il registro è interamente in memoria. Se il Registry crasha, tutti i worker in esecuzione perdono la possibilità di scoprire nuovi peer, ma continuano a comunicare tra loro tramite i peer già noti (memorizzati localmente dalla chiamata precedente a `/get_peers`). La deregistrazione è *best-effort*: il blocco `finally` in `main_worker.py` tenta la chiamata, ma se il Registry è già irraggiungibile l'eccezione viene silenziata — il registro potrebbe contenere entry stantie, ma la lista restituita da `/get_peers` contiene solo indirizzi attivi nella pratica (i crash dei worker riducono la lista per deregistrazione, non per heartbeat).

### 3.2 Architettura del Nodo Worker

#### Modello a due thread

Ogni worker è un processo Python che esegue due thread per l'intera durata della propria vita. La scelta del modello a due thread è dettata da un requisito fondamentale: il server gRPC che riceve aggiornamenti dai vicini deve rimanere **sempre in ascolto**, indipendentemente dallo stato del training loop. Se il receiver fosse single-threaded con il trainer, ogni chiamata gRPC in arrivo durante la Fase B (H inner steps) verrebbe rifiutata o messa in coda indefinitamente, degradando la qualità degli aggiornamenti ricevuti.

La separazione è realizzata così:

- **Thread 1 (gRPC Server)** — avviato da `start_grpc_server()` che restituisce immediatamente. Il server gRPC di grpcio crea internamente un pool di thread (`ThreadPoolExecutor(max_workers=10)`) per gestire richieste concorrenti. Thread 1 è quindi un supervisore del pool, non un singolo thread di I/O.
- **Thread 2 (Training Loop)** — è il thread principale del processo (`main()`), che esegue le tre fasi del round in sequenza.

Questa architettura ha un costo: richiede sincronizzazione tra i due thread sullo stato condiviso. La sincronizzazione è minimizzata: l'unico stato condiviso è l'`AggregationBuffer` (protetto da `threading.Lock`) e il dizionario `shared_state` (con garanzie di atomicità del GIL per l'intero unico writer).

#### Numero fisso di worker e assenza di join dinamici

Il sistema supporta un numero **fisso e preconfigurabile** di worker (`num_workers` in `config.yaml`). I join dinamici non sono supportati per una ragione fondamentale: la partizione del dataset è **deterministica e statica**, calcolata all'avvio in funzione di `WORKER_ID` e `TOTAL_WORKERS`. Se un nuovo worker si aggiungesse a runtime, non esiste un meccanismo per assegnargli una partizione coerente con quelle già attive senza ribilanciare l'intera distribuzione dei dati — operazione incompatibile con il requisito di semplicità e con l'architettura stateless adottata.

Questa scelta semplifica significativamente la gestione della consistenza: non è necessario alcun protocollo di membership dinamica (Paxos, Raft, SWIM, ecc.), rendendo il sistema più comprensibile e manutenibile.

#### Stato condiviso tra Thread 1 e Thread 2

I due thread condividono due strutture:

**`AggregationBuffer`** — contiene gli accumulatori per la media pesata:

```python
class AggregationBuffer:
    lock: threading.Lock        # mutex for exclusive access
    weighted_sum: dict | None   # {param_name: Tensor} = sum(w_i * sender_samples_i)
    received_samples: int       # sum of sender_samples across all received neighbors
```

L'accesso è sempre mediato da `buffer.lock`. Thread 1 scrive (accumula), Thread 2 legge e resetta (in Phase A). La scelta di usare un singolo lock per entrambi i campi — invece di due lock separati — evita il rischio di deadlock e garantisce che la coppia `(weighted_sum, received_samples)` sia sempre letta e scritta atomicamente.

**`shared_state`** — un dizionario Python con una sola chiave: `{"current_round": int}`. Thread 2 scrive il valore corrente del round all'inizio della Fase C; Thread 1 lo legge per il controllo di staleness. Non è protetto da lock perché l'assegnazione di un intero in Python è atomica grazie al GIL (Global Interpreter Lock): un solo writer (Thread 2) garantisce che Thread 1 non legga mai uno stato parzialmente scritto.

### 3.3 Protocollo di Comunicazione: gRPC e Protobuf

#### Definizione del contratto (gossip.proto)

La comunicazione inter-worker è definita dal file `gossip.proto`:

```protobuf
syntax = "proto3";
package gossip;

service GossipService {
    rpc ReceiveModel (ModelMessage) returns (Ack);
}

message ModelMessage {
    bytes  weights     = 1;  // serialized PyTorch state_dict
    int32  round       = 2;  // sender's current round
    int32  num_samples = 3;  // sender's local sample count
    string worker_id   = 4;  // sender identifier
}

message Ack {
    bool accepted = 1;
}
```

Il messaggio `ModelMessage` trasporta quattro campi: i pesi serializzati del modello, il numero del round corrente del mittente, il numero di campioni locali del mittente e il suo identificatore. I campi `round` e `num_samples` sono essenziali per la logica di aggregazione: il primo serve allo staleness check, il secondo alla ponderazione FedAvg.

#### Scelta di gRPC rispetto a REST/HTTP

La motivazione principale per scegliere gRPC è la **serializzazione binaria compatta dei pesi del modello**. Un modello CNN per FEMNIST ha tipicamente nell'ordine di $10^5$–$10^6$ parametri float32. In JSON ogni float occupa mediamente 8–12 caratteri (es. `0.0034521`), per un totale di 4–12 MB per messaggio. Con `torch.save()` + Protobuf il payload è 4 byte per float, circa 400 KB–4 MB — un risparmio di 2–3× rispetto a JSON.

Ulteriori motivazioni:
- **Stub autogenerati**: `grpc_tools.protoc` produce codice client/server Python da `gossip.proto`, eliminando la necessità di scrivere manualmente il codice di serializzazione e routing.
- **Timeout per chiamata**: ogni `stub.ReceiveModel(message, timeout=T)` solleva `grpc.RpcError` se il server non risponde entro `T` secondi, senza bisogno di gestione manuale di socket timeout.
- **Evoluzione del protocollo**: Protobuf supporta l'aggiunta di nuovi campi con retro-compatibilità garantita; aggiungere metadati al messaggio (es. versione del modello, loss locale) richiede solo una modifica al `.proto`.

#### Generazione degli stub a build time

I file `gossip_pb2.py` e `gossip_pb2_grpc.py` sono generati dal compilatore `protoc` nel `Dockerfile.worker`, **prima** del `COPY` del sorgente applicativo:

```dockerfile
COPY gossip.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gossip.proto
COPY config.yaml main_worker.py ./
COPY core/ ./core/
COPY network/ ./network/
```

Questo ordine è critico: i file generati si trovano nella directory di lavoro del container prima che il sorgente venga copiato sopra. Poiché `.gitignore` esclude i file `pb2` dal repository, la `COPY` successiva non li sovrascrive. Il vantaggio aggiuntivo è il **riutilizzo del layer Docker**: il layer contenente la compilazione Protobuf viene invalidato solo se `gossip.proto` cambia, rendendo le rebuild successive molto più veloci.

#### Sicurezza nella deserializzazione

La deserializzazione dei pesi usa `torch.load(..., weights_only=True)`:

```python
weights = torch.load(io.BytesIO(request.weights), map_location="cpu", weights_only=True)
```

L'opzione `weights_only=True` è necessaria perché `torch.load` sfrutta `pickle` internamente; senza questa restrizione, un messaggio malevolo potrebbe eseguire codice arbitrario sul ricevente al momento della deserializzazione. Con `weights_only=True` vengono accettati solo tensori e tipi primitivi, eliminando questo vettore di attacco.

---

## 4. Algoritmo di Training Federato

### 4.1 Partizionamento del Dataset (Non-i.i.d.)

#### Pipeline di preparazione del dataset (pre-deployment)

La preparazione del dataset avviene interamente sull'host, **prima** della creazione dei container, attraverso due script eseguiti in sequenza.

**`scripts/download_femnist.py`** — scarica e preprocessa FEMNIST tramite il framework LEAF. I passi interni sono:

1. Clone del repository LEAF da GitHub (saltato se già presente).
2. Patch di compatibilità su `data_to_json.py` di LEAF: `Image.ANTIALIAS → Image.LANCZOS` (vedi Sezione 11).
3. Installazione delle dipendenze di preprocessing di LEAF (`tensorflow-cpu`, `Pillow`, `numpy`) nell'ambiente Python corrente.
4. Esecuzione di `preprocess.sh` con split non-i.i.d. per scrittore, 90% train / 10% test.
5. Copia **selettiva** di sole `train/` e `test/` in `data/femnist/data/`. Le directory intermedie prodotte da LEAF (immagini raw EMNIST, file `.pkl`, dati campionati) non vengono copiate: occuperebbero gigabyte inutili poiché non servono al training. Il dataset finale pesa ~2–4 GB.
6. Rimozione automatica dell'intera directory `leaf/` (~20 GB). Una volta che `data/femnist/data/` esiste, il repository LEAF non serve più — se necessario verrà riclonato automaticamente da GitHub alla prossima esecuzione dello script.

**`scripts/split_dataset.py`** — partiziona `data/femnist/data/` in slice per-worker, scrivendo `data/femnist/worker_{i}/{train,test}/data.json` per ciascun worker $i \in [0, N)$. Lo script adotta una strategia a **due passate con scrittura immediata su disco** per mantenere il consumo di RAM costante indipendentemente dalla dimensione del dataset. Il dataset completo occupa ~4 GB su disco ma si espanderebbe a 40–80 GB come oggetti Python se caricato interamente in memoria — dimensione insostenibile su un portatile.

- **Passata 1 (solo ID):** legge esclusivamente il campo `users` di ogni shard JSON, senza caricare i pixel. Produce la lista globale ordinata di tutti i writer, calcola la mappa `writer_id → worker_index` e raggruppa gli ID per worker. Consumo RAM: trascurabile (solo stringhe).
- **Passata 2 (streaming con scrittura immediata):** apre tutti i file di output dei worker simultaneamente; legge un shard alla volta; per ogni writer nel shard, scrive l'entry `user_id: {x, y}` direttamente nel file del worker corretto in quel momento, senza accumularla in memoria. Alla fine del shard, esegue `del shard` + `gc.collect()` per liberare subito la RAM prima del shard successivo. Il picco di RAM è **un singolo shard** (~1–2 GB come oggetti Python) indipendentemente dal numero di worker o dalla dimensione totale del dataset.

Lo script rimuove automaticamente le directory `worker_*` esistenti all'avvio, rendendo ogni esecuzione idempotente: se interrotto a metà, basta rilanciarla da capo senza rischio di dati inconsistenti. Il sorgente `data/femnist/data/` non viene mai modificato.

La motivazione di eseguire entrambi gli step su host anziché dentro i container è fondamentale per la correttezza dello scenario federato: ogni container riceve in mount **esclusivamente la propria porzione di dati**, senza possibilità di accedere a quelli degli altri worker. Questo rispecchia fedelmente la realtà del Federated Learning, dove ogni dispositivo ha accesso fisico solo ai propri dati locali — non è necessario alcun meccanismo software per isolare le partizioni, è l'architettura stessa del filesystem a garantirlo.

#### Strategia di partizione statica pre-deployment

Il dataset FEMNIST viene partizionato in modo **deterministico e statico** prima dell'avvio dei container, dallo script `scripts/split_dataset.py`. Lo script legge i file JSON prodotti da LEAF, estrae la lista globale degli utenti ordinata, e assegna a ciascun worker uno slice contiguo:

$$\text{start}_k = k \cdot \left\lfloor \frac{|\mathcal{U}|}{N} \right\rfloor, \quad \text{end}_k = \begin{cases} \text{start}_k + \lfloor |\mathcal{U}|/N \rfloor & \text{se } k < N-1 \\ |\mathcal{U}| & \text{se } k = N-1 \end{cases}$$

dove $\mathcal{U}$ è l'insieme totale degli utenti e $N$ è `num_workers` in `config.yaml`. La partizione del worker $k$ viene scritta su host in `data/femnist/worker_k/{train,test}/data.json` e montata nel suo container tramite bind mount Docker:

```
./data/femnist/worker_k  →  /app/data/femnist  (dentro il container k)
```

Ogni container ha accesso **esclusivo e isolato** alla propria partizione: il filesystem del container non contiene alcun dato appartenente ad altri worker. Questo rispecchia fedelmente uno scenario federato reale, dove ogni dispositivo ha accesso solo ai propri dati locali.

#### Garanzia della proprietà non-i.i.d.

La partizione è non-i.i.d. per costruzione: LEAF organizza i dati per autore, ciascuno con uno stile di scrittura caratteristico. Assegnare utenti contigui a un worker garantisce che la sua distribuzione di classi rifletta gli stili di un sottoinsieme specifico di scrittori — diverso da quello di ogni altro worker. Questo simula fedelmente lo scenario FL reale in cui i dispositivi partecipanti hanno dati generati da utenti diversi con abitudini proprie.

#### Motivazione della scelta statica vs dinamica

Una partizione dinamica (che ribilancia i dati al join di nuovi worker) avrebbe garantito partizioni di dimensione uniforme anche in caso di variazioni del numero di nodi. Tuttavia, introdurrebbe una dipendenza globale: ogni ribilanciamento richiederebbe un coordinatore che conosce l'intera distribuzione degli utenti — contraddittorio con l'approccio puramente P2P adottato. La scelta statica mantiene il sistema autonomo: ogni worker carica semplicemente i file presenti nella propria directory montata, senza conoscere `WORKER_ID` o `TOTAL_WORKERS` a livello di dataset.

#### Caricamento nel worker

`core/dataset.py` espone la funzione `load_partition(data_dir, batch_size)` che legge semplicemente tutti i file JSON presenti in `data_dir/train/` e `data_dir/test/` — la stessa interfaccia di lettura indipendentemente da quanti worker esistano. Il splitting è già avvenuto su host; il container non sa nulla della topologia globale.

#### Gestione del ciclo infinito sui batch

Per permettere esattamente $H$ inner steps indipendentemente dalla dimensione della partizione locale, viene utilizzato un generatore infinito:

```python
def infinite_batches(loader):
    while True:
        yield from loader
```

Se la partizione di un worker contiene meno di $H$ batch, il loader ricomincia dall'inizio, ripetendo i dati. Questo è equivalente ad aumentare artificialmente il numero di epoche locali. L'impatto sulla convergenza è limitato perché il numero di ripetizioni è piccolo (tipicamente meno di una volta completa con $H=500$ e partizioni ragionevoli).

### 4.2 Ciclo di Training — Le Tre Fasi

Ad ogni round, il Thread 2 esegue sequenzialmente le tre fasi seguenti. L'ordine A→B→C è deliberato: si aggrega prima (beneficio delle informazioni ricevute nel round precedente), poi si allena, poi si propaga.

#### Fase A — Aggregazione FedAvg Pesata

**Meccanismo.** Thread 2 acquisisce il `Lock` sull'`AggregationBuffer`. Se `received_samples > 0` — ovvero almeno un vicino ha inviato un aggiornamento dall'inizio del round precedente — viene eseguita l'aggregazione. La formula implementata è:

$$w_{\text{new}}[k] = \frac{w_{\text{local}}[k] \cdot \texttt{local\_samples} + \texttt{weighted\_sum}[k]}{\texttt{local\_samples} + \texttt{received\_samples}}$$

dove `combined_samples = local_samples + received_samples`. Questa forma è equivalente a calcolare prima la media dei vicini e poi mediare con il modello locale, ma evita un'allocazione intermedia. La derivazione è:

$$w_{\text{new}} = \frac{w_{\text{local}} \cdot n_{\text{local}} + \bar{w}_{\text{neighbors}} \cdot n_{\text{neighbors}}}{n_{\text{local}} + n_{\text{neighbors}}}$$

dove $\bar{w}_{\text{neighbors}} = \texttt{weighted\_sum} / \texttt{received\_samples}$. Sostituendo:

$$w_{\text{new}} = \frac{w_{\text{local}} \cdot n_{\text{local}} + (\texttt{weighted\_sum} / \texttt{received\_samples}) \cdot \texttt{received\_samples}}{n_{\text{local}} + \texttt{received\_samples}} = \frac{w_{\text{local}} \cdot n_{\text{local}} + \texttt{weighted\_sum}}{n_{\text{local}} + \texttt{received\_samples}}$$

**Trattamento dei parametri non-float.** Solo i parametri floating-point vengono aggregati. I buffer interi presenti nello `state_dict` — come `num_batches_tracked` nei layer BatchNorm, che conta il numero di batch visti — mantengono il valore locale. Mediare contatori interi non ha senso semantico e potrebbe produrre valori inconsistenti.

**Reset del buffer.** Dopo l'aggregazione, `weighted_sum` viene posto a `None` e `received_samples` a `0`. Il reset avviene con il lock acquisito, garantendo che nessun messaggio in arrivo (Thread 1) possa modificare il buffer nel breve intervallo tra la lettura e il reset.

**Caso base: nessun vicino ha inviato.** Se `received_samples == 0`, la Fase A viene saltata e il worker procede direttamente alla Fase B con il proprio modello invariato. Questo è il comportamento corretto in caso di assenza di aggiornamenti (nessun vicino attivo, tutti i messaggi droppati o stantii): il training locale prosegue autonomamente.

**Early stopping post-aggregazione.** Immediatamente dopo l'aggregazione (o dopo il suo skip), il modello viene validato sul validation set locale. Se la validation loss non si riduce per `early_stopping_patience` round consecutivi, Thread 2 esce dal loop. Thread 1 rimane attivo: il processo non termina e il server gRPC continua a servire i peer che sono ancora in training. Questo comportamento è ottenuto chiamando `grpc_server.wait_for_termination()` dopo il break, che blocca il thread principale finché il server gRPC non viene fermato esternamente.

L'early stopping è **locale e indipendente** per ogni worker: non esiste coordinamento globale. Worker diversi possono convergere in round diversi, e quelli che convergono prima continuano a servire gli altri come destinatari passivi di gossip push.

#### Fase B — Training Locale (H Inner Steps)

**Meccanismo.** Il worker esegue esattamente `inner_steps_H` passi di ottimizzazione locale usando l'ottimizzatore AdamW con learning rate configurabile. Durante questa fase, Thread 1 continua ad accumulare i messaggi ricevuti nell'`AggregationBuffer`, ma Thread 2 non li legge: la sincronizzazione avviene solo all'inizio del round successivo (Fase A).

**Scelta dell'ottimizzatore: AdamW vs SGD.** AdamW è preferito a SGD per la sua robustezza ai learning rate: richiede meno tuning del learning rate rispetto a SGD con momentum, che è critico in un contesto distribuito dove non c'è un tutor centrale che aggiusta i parametri. AdamW introduce la weight decay direttamente sull'aggiornamento dei pesi (non sul gradiente come L2 regularization), il che tende a produrre modelli con generalizzazione migliore.

**Scelta di H=500.** Il valore $H=500$ è ispirato direttamente a DiLoCo [1] e rappresenta un trade-off tra qualità dell'aggregazione e costo di comunicazione. Con $H$ piccolo (es. 1), ogni aggiornamento è quasi un gradiente puro e l'aggregazione è equivalente al SGD distribuito sincrono — ottima qualità ma alta frequenza di comunicazione. Con $H$ grande (es. 10.000), ogni worker diverge significativamente dagli altri prima di sincronizzarsi — comunicazione rara ma aggregazione degradata. $H=500$ mantiene i worker sufficientemente allineati da rendere l'aggregazione FedAvg efficace, pur riducendo la frequenza di comunicazione di due ordini di grandezza rispetto al training sincrono.

#### Fase C — Gossip Push

**Meccanismo.** Prima dell'invio, `shared_state["current_round"]` viene aggiornato al valore del round corrente, rendendolo visibile a Thread 1 per i successivi controlli di staleness. Il worker interroga il Discovery Server tramite `GET /peers`, esclude il proprio indirizzo dalla lista, e seleziona casualmente `min(gossip_fanout, len(eligible_peers))` vicini. Per ciascun target viene applicata la logica di fault injection (Sezione 8), poi viene invocato `send_model()`.

**Frequenza di interrogazione del registry.** Il registry viene interrogato **una volta per round**, all'inizio della Fase C. Questa scelta è coerente con il requisito di minimizzare il traffico di rete: con H=500 inner steps un round dura tipicamente diversi minuti, quindi aggiornare la lista peer più spesso produrrebbe overhead HTTP senza benefici concreti sulla freschezza. Interrogare il registry meno spesso (es. ogni K round) ridurrebbe ulteriormente il traffico al costo di una visione più stale della topologia. Il trade-off è bilanciato al valore attuale: un'interrogazione per round mantiene la lista allineata con i cambiamenti topologici (worker che si registrano o deregistrano) senza generare traffico aggiuntivo apprezzabile rispetto ai push gRPC che domina il volume totale.

**Re-query reattivo dopo fallimento gRPC.** Un push fallito (codice `UNAVAILABLE` o `DEADLINE_EXCEEDED`) è un segnale che il peer potrebbe essere crashato e potrebbe essersi già deregistrato. Il worker sfrutta questa informazione: se almeno un push gRPC fallisce (esclusi i drop simulati, che sono intenzionali), viene eseguita immediatamente una seconda chiamata a `GET /peers` per ottenere una lista aggiornata. Per ogni peer irraggiungibile viene tentato un sostitutivo scelto tra i peer freschi non ancora contattati in questo round (tracciati nel set `tried`). Questo meccanismo costa **al massimo una HTTP call extra per round**, emessa solo quando si verificano fallimenti reali. Copre il caso più comune: peer crashato con deregistrazione pulita (via `finally` o signal handler). Non risolve il caso di hard crash (SIGKILL, OOM) dove il peer rimane nel registry — documentato come known limitation in Sezione 8.4. Il log di fine Fase C riporta `failed=N, retried=M` per osservabilità diretta.

**Snapshot unico dei pesi.** Il modello viene snapshotted una volta — `weights_snapshot = model.state_dict()` — prima del loop sui target. Tutti i vicini ricevono la stessa versione del modello. Questo evita che modifiche al modello durante l'invio (impossibili in questo design, ma buona pratica) producano incoerenze.

**Selezione casuale dei vicini.** La selezione di $M$ vicini casuali a ogni round implementa la variante **k-push** del gossip protocol (anche nota come *push-based k-fan-out*): ogni nodo invia a $k$ peer scelti a caso in un singolo hop, senza che i destinatari facciano forwarding del messaggio. La casualità garantisce che nel lungo periodo tutti i worker ricevano aggiornamenti da tutti gli altri (con probabilità crescente con il numero di round), anche con $M \ll N-1$. Questo produce una connettività media della rete dell'ordine di $M$ archi uscenti per nodo, sufficiente per la propagazione dell'informazione in reti sparse.

**K-push vs. rumor mongering: analisi comparativa.** Il k-push non è l'unica variante di gossip esistente. L'alternativa classica è il *rumor mongering* (o gossip epidemico): ogni nodo che riceve un messaggio decide probabilisticamente se propagarlo ulteriormente, generando una catena di inoltri multi-hop. La tabella seguente confronta le due varianti nel contesto FL.

| Aspetto | K-push (adottato) | Rumor mongering |
|---|---|---|
| **Hop per round** | 1 — il mittente originale invia, i destinatari non forwardano | Multipli — ogni ricevente può diventare mittente |
| **Traffico** | Deterministico: esattamente $N 	imes k$ messaggi per round | Variabile e imprevedibile; può crescere esponenzialmente su reti piccole |
| **Diffusione** | Lenta per $N$ grande: raggiunge $k$ peer per round | Rapida: copertura $O(\log N)$ hop con alta probabilità |
| **Semantica FedAvg** | Corretta: ogni contributo arriva al più da un percorso | Rischio di duplicati: gli stessi pesi possono arrivare via percorsi distinti e venire aggregati più volte |
| **Deduplicazione** | Non necessaria | Obbligatoria: il buffer deve tracciare `(sender_id, round)` già visti |
| **Amplificazione stale update** | Contenuta: un update stantio raggiunge al più $k$ nodi | Pericolosa: un update stantio che sfugge al filtro locale si propaga a tutta la rete |
| **Complessità implementativa** | Bassa | Media — richiede seen-set, logica di forwarding, deduplicazione nel buffer |

**Perché k-push è la scelta corretta in questo progetto.** La ragione primaria è un **requisito esplicito della traccia di progetto**: il sistema deve mantenere basso il traffico di rete. Il k-push soddisfa questo requisito per costruzione: il volume di comunicazione per round è deterministico e pari a $N 	imes k 	imes 	ext{model\_size}$, indipendentemente dallo stato della rete. Con il rumor mongering il traffico è invece imprevedibile e può crescere molto di più — ogni update si propaga a cascata, e su reti piccole (N=3–20) questo produce ridondanza elevata senza benefici reali di diffusione. Il k-push permette di calibrare con precisione il trade-off tra traffico e qualità dell'aggregazione agendo su un solo parametro (`gossip_fanout`), in linea con l'obiettivo del progetto di analizzare sperimentalmente questo trade-off.

Il secondo motivo è semantico: FedAvg richiede che ogni contributo venga contato *esattamente una volta* per round. Il k-push garantisce questo per costruzione — ogni worker invia i propri pesi, i destinatari non li ritrasmettono. Il rumor mongering rompe questa invariante e richiederebbe un fix esplicito nel `AggregationBuffer`, aggiungendo complessità senza vantaggi a questa scala.

**Quando il rumor mongering sarebbe preferibile.** In sistemi reali con $N$ nell'ordine delle migliaia — reti di sensori IoT, sistemi di membership distribuiti (es. SWIM protocol), DHT come Chord o Kademlia — il k-push con $k$ piccolo lascia zone della rete non raggiunte per decine di round, rendendo la convergenza globale molto lenta. Il rumor mongering garantisce in quei contesti copertura quasi totale in $O(\log N)$ round indipendentemente da $k$, un vantaggio decisivo. In ambito FL, sarebbe applicabile con una variante modificata che deduplicherebbe gli aggiornamenti per `(sender_id, round)` prima dell'aggregazione, accettando il costo di complessità e traffico extra in cambio di convergenza più rapida su reti sparse e molto grandi.

**Perché la selezione avviene nel worker, non nel Registry.** Una progettazione alternativa potrebbe delegare la selezione dei vicini al Discovery Server: il worker chiede "dammi M peer casuali" e il Registry risponde con la lista già filtrata. Questa alternativa è stata esplicitamente scartata per due ragioni distinte.

La prima è il **rispetto del ruolo del Registry**: il Discovery Server è progettato come un *name server* puro — conosce solo indirizzi, non topologia. Delegargli la selezione dei peer significherebbe introdurre logica di routing, rendendo il componente più complesso, più fragile e più difficile da sostituire. Il vincolo architetturale è deliberato: il Registry non deve mai contenere logica che riguardi il training o la comunicazione tra modelli.

La seconda è la **distribuzione della conoscenza topologica**: in un sistema P2P, ogni nodo mantiene una propria visione locale della rete — la lista di peer ottenuta dall'ultima chiamata a `GET /peers`. La selezione casuale operata localmente è coerente con questo principio: ciascun nodo decide autonomamente con chi comunicare, senza dipendere dalla disponibilità del Registry per ogni singolo round. Un Registry temporaneamente irraggiungibile durante la Fase C non impedisce il gossip push verso i peer già noti; impedirebbe solo la scoperta di *nuovi* nodi entrati nel sistema.

### 4.3 Online Aggregation nel gRPC Server (Thread 1)

#### Il problema: memoria O(N × model_size)

L'approccio naïve all'aggregazione consisterebbe nel salvare ogni modello ricevuto in una lista e calcolare la media pesata in Fase A. Con $K$ vicini attivi e un modello di dimensione $S$ byte, questo richiede $O(K \cdot S)$ memoria — proporzionale al numero di messaggi ricevuti. Per modelli grandi o reti dense, questo approccio è impraticabile.

#### La soluzione: accumulatore a running weighted sum

Il Thread 1 mantiene invece un **accumulatore a somma ponderata corrente** che richiede $O(S)$ memoria indipendentemente da quanti messaggi vengono ricevuti. L'invariante dell'accumulatore è:

$$\texttt{weighted\_sum}[k] = \sum_{i \in \text{received}} w_i[k] \cdot n_i, \quad \texttt{received\_samples} = \sum_{i \in \text{received}} n_i$$

dove $w_i$ sono i pesi del messaggio $i$-esimo e $n_i$ il numero di campioni del mittente. Alla ricezione di ogni nuovo messaggio con pesi $w_{\text{new}}$ e campioni $n_{\text{sender}}$:

$$\texttt{weighted}[k] = w_{\text{received}}[k] \cdot \texttt{sender\_samples}$$

$$\texttt{weighted\_sum}[k] \mathrel{+}= \texttt{weighted}[k], \qquad \texttt{received\_samples} \mathrel{+}= \texttt{sender\_samples}$$

Il caso base (`received_samples == 0`) inizializza il buffer con il primo contributo. La prova che questo accumulatore produce lo stesso risultato dell'approccio batch è diretta per linearità della somma: $\sum_i (w_i \cdot n_i) = $ running sum step-by-step.

#### Correttezza con accesso concorrente

Thread 1 può ricevere messaggi da più sender concorrentemente (il pool di thread interno a gRPC gestisce connessioni parallele). Ogni invocazione di `ReceiveModel` acquisisce il lock prima di modificare l'accumulatore, garantendo serializzazione degli aggiornamenti. L'overhead del lock è trascurabile rispetto al costo di deserializzazione dei pesi (operazione dominante).

### 4.4 Staleness Check (Unidirezionale)

#### Il problema: aggiornamenti stantii

In un sistema gossip asincrono, la latenza di rete e la differenza di velocità tra worker possono causare l'arrivo di messaggi con un ritardo di molti round. Un worker che ha già effettuato 50 round potrebbe ricevere pesi calcolati al round 30 da un peer lento. Incorporare questo aggiornamento degraderebbe la qualità del modello: i pesi vecchi codificano informazioni superate sul gradiente.

#### Implementazione del check

Thread 1 applica il seguente controllo prima di ogni aggregazione:

$$\text{discard if} \quad (r_{\text{current}} - r_{\text{sender}}) > \Delta_{\max}$$

dove $r_{\text{current}}$ è il round corrente del ricevente (letto da `shared_state["current_round"]`), $r_{\text{sender}}$ è il campo `round` del messaggio, e $\Delta_{\max}$ è il parametro `max_staleness` (default: 10). Il messaggio viene scartato restituendo `Ack(accepted=False)` senza modificare il buffer.

#### Unidirezionalità: perché non scartare anche i messaggi "dal futuro"

Il check è volutamente **unidirezionale**: se $r_{\text{sender}} > r_{\text{current}}$, la differenza è negativa e il check non scatta. I messaggi provenienti da worker più avanzati vengono sempre accettati. La motivazione è asimmetrica:

- **Messaggi dal passato** ($r_{\text{sender}} \ll r_{\text{current}}$): i pesi riflettono un modello che ha visto $r_{\text{sender}} \cdot H$ batch in meno — la direzione di update è stantia e potrebbe peggiorare la convergenza del ricevente.
- **Messaggi dal futuro** ($r_{\text{sender}} > r_{\text{current}}$): i pesi riflettono un modello più aggiornato — incorporarli anticipa la convergenza del ricevente senza penalità.

Scartare i messaggi dal futuro sarebbe controproducente: un worker lento che riceve da uno veloce perderebbe informazioni preziose.

#### Scelta di $\Delta_{\max} = 10$

Il valore 10 rappresenta un trade-off: troppo basso (es. 1) scarterebbe molti aggiornamenti validi in presenza di variabilità di rete, riducendo il numero effettivo di contributi per round. Troppo alto (es. 100) accetterebbe aggiornamenti molto vecchi che potrebbero degradare la convergenza. 10 round di tolleranza — corrispondenti a 5.000 batch locali ($10 \times H = 10 \times 500$) — rappresentano un ritardo accettabile nella maggior parte degli scenari di rete reale.

---

## 5. Modello di Machine Learning

### 5.1 Scelta dell'Architettura: CNN per FEMNIST

Il task di apprendimento è la classificazione di caratteri scritti a mano in 62 classi su immagini $28 \times 28$ in scala di grigi. La scelta dell'architettura neurale è vincolata da tre fattori:

1. **Dimensione dell'input**: $28 \times 28$ pixel in scala di grigi — un'immagine piccola rispetto agli standard moderni di computer vision.
2. **Distribuzione locale dei dati**: ogni worker possiede un sottoinsieme non-i.i.d. di scrittori con potenzialmente poche centinaia di campioni per classe. L'overfitting locale è il rischio principale.
3. **Costo di comunicazione gossip**: il modello viene trasmesso in rete ad ogni round di gossip. Ridurre il numero di parametri riduce direttamente il payload dei messaggi gRPC.

**CNN vs MLP.** Un MLP applicato a pixel flat tratta ogni pixel come feature indipendente, perdendo la struttura spaziale dell'immagine. Due pixel adiacenti in un carattere scritto a mano hanno una correlazione spaziale fondamentale per riconoscere tratti, curve e angoli — correlazione che la convoluzione sfrutta tramite kernel condivisi. Le CNN superano i MLP di 5–15% di accuracy su FEMNIST per questa ragione.

**CNN leggera vs ResNet.** Su immagini $28 \times 28$, le architetture residuali profondi aggiungono parametri e complessità senza beneficio proporzionale: a questa risoluzione, i tratti del carattere sono già ben rappresentati con 2–3 layer convolutivi. Un ResNet-18 conta ~11M parametri contro i ~1.7M dell'architettura adottata — 6× più pesante da trasmettere via gossip e più difficile da addestrare su partizioni locali piccole. Il vincolo della traccia di progetto (NN come modello base con valutazione LEAF) è pienamente soddisfatto da questa scelta.

### 5.2 Architettura: CNN a Doppio Blocco Convolutivo

L'architettura è organizzata in tre componenti sequenziali: due blocchi convolutivi e un classificatore fully-connected.

```
┌────────────────────────────────────────────────────────────┐
│  Input: (N, 1, 28, 28)                                     │
└──────────────────────────┬─────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │        Blocco 1         │
              │  Conv(1→32, 3×3, p=1)  │  ← same padding: 28×28 invariato
              │  BatchNorm2d + ReLU     │
              │  Conv(32→32, 3×3, p=1) │
              │  BatchNorm2d + ReLU     │
              │  MaxPool2d(2) → 14×14  │
              │  Dropout2d(p=0.25)      │
              └────────────┬────────────┘
                           │ (N, 32, 14, 14)
              ┌────────────▼────────────┐
              │        Blocco 2         │
              │  Conv(32→64, 3×3, p=1) │  ← same padding: 14×14 invariato
              │  BatchNorm2d + ReLU     │
              │  Conv(64→64, 3×3, p=1) │
              │  BatchNorm2d + ReLU     │
              │  MaxPool2d(2) → 7×7    │
              │  Dropout2d(p=0.25)      │
              └────────────┬────────────┘
                           │ (N, 64, 7, 7)
              ┌────────────▼────────────┐
              │      Classificatore     │
              │  Flatten → (N, 3136)    │
              │  Linear(3136 → 512)     │
              │  BatchNorm1d + ReLU     │
              │  Dropout(p=0.5)         │
              │  Linear(512 → 62)       │
              └────────────┬────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │  Output: (N, 62) — logits grezzi     │
        └─────────────────────────────────────┘
```

**Conteggio dei parametri:**

| Componente | Parametri |
|---|---:|
| Blocco 1: Conv(1→32) + Conv(32→32) | ~18.7K |
| Blocco 2: Conv(32→64) + Conv(64→64) | ~55.4K |
| Tutti i layer BatchNorm | ~1.3K |
| FC1: Linear(3136→512) | ~1.607M |
| FC2: Linear(512→62) | ~31.8K |
| **Totale** | **~1.72M** |

Il classificatore contribuisce il 94% dei parametri (FC1 domina), come tipico nelle CNN piccole: la feature extraction è parsimoniosa grazie ai kernel condivisi, mentre i layer FC non condividono pesi.

### 5.3 Motivazione di Ogni Scelta di Progetto

#### Same Padding (`padding=1`)

Il modello placeholder usava valid padding (nessun padding): ogni `Conv2d(3×3)` riduce la dimensione spaziale di 2 pixel per lato. Con same padding (`padding=1`), le dimensioni restano invariate fino al MaxPool (28→28→14→14→7→7). Questo preserva più informazione spaziale nei layer iniziali e permette ai kernel di vedere i bordi dell'immagine, dove i tratti dei caratteri spesso iniziano o terminano.

#### Double Conv Block (stile VGG)

Il placeholder applicava un solo conv prima di ogni pool. Due conv consecutivi prima del pooling offrono:
- **Campo ricettivo equivalente a Conv(5×5)** con meno parametri: due Conv(3×3) hanno campo ricettivo 5×5 ma usano 2×9 = 18 vs 25 pesi per coppia canali input/output, con una non-linearità intermedia aggiuntiva.
- **Feature hierarchy più ricca**: il primo conv estrae feature elementari (bordi, tratti), il secondo combina queste in pattern più complessi (curve, intersezioni di tratti) prima della riduzione di risoluzione.

#### BatchNorm2d dopo ogni Conv

BatchNorm normalizza l'output di ogni conv layer al batch corrente: sottrae la media del batch e divide per la deviazione standard, poi applica i parametri appresi $\gamma$ (scala) e $\beta$ (shift). I benefici principali sono:
- **Stabilizzazione del training**: riduce l'*internal covariate shift* (cambiamento della distribuzione degli input ai layer successivi durante il training), permettendo learning rate più alti e convergenza più rapida.
- **Effetto regolarizzante**: la normalizzazione introduce rumore basato sulle statistiche del batch corrente, con effetto simile ma più debole del Dropout.

**Nota critica — BatchNorm in Federated Learning.** BatchNorm ha un comportamento delicato in FL. I parametri appresi ($\gamma$ e $\beta$), essendo parte dello `state_dict`, vengono aggregati normalmente da FedAvg. I buffer `running_mean` e `running_var` — aggiornati ad ogni forward pass con le statistiche del batch locale — **non vengono aggregati** e rimangono specifici del worker.

Questo genera un disallineamento post-aggregazione: dopo FedAvg, i parametri $\gamma$ e $\beta$ riflettono una media tra worker, mentre le running stats sono ancora quelle del worker locale. Le prime iterazioni post-aggregazione possono produrre predizioni leggermente degradate fino a quando le running stats si riallineano con i nuovi $\gamma$ e $\beta$.

Per questo progetto il disallineamento è accettabile: la validazione (early stopping) è sempre locale, quindi le running stats e i parametri sono sempre coerenti a livello di worker; le prime iterazioni post-aggregazione costituiscono solo una piccola frazione degli H=500 inner steps. In letteratura (FedBN, Li et al. 2021) questo comportamento è documentato come limitazione nota e, in alcuni contesti, è stato persino studiato come meccanismo di personalizzazione locale.

*Alternative considerate:* GroupNorm e LayerNorm non hanno running statistics e si comportano identicamente in training e inference, risolvendo il problema FL. Tuttavia GroupNorm è meno standard e richiederebbe la scelta del numero di gruppi; LayerNorm è più adatto a sequence models. BatchNorm resta la scelta dominante nella letteratura FL e viene adottata qui con la limitazione esplicitamente documentata.

#### Dropout2d (Spatial Dropout) nei Blocchi Conv — p=0.25

`nn.Dropout2d` azzera interi canali (feature map) durante il training. Per un tensore `(N, C, H, W)`, un intero canale viene azzerato con probabilità `p`. Questo è più efficace del Dropout scalare per layer conv, perché i pixel adiacenti nella stessa feature map sono altamente correlati: azzerare pixel singoli non rompe la correlazione, mentre azzerare l'intera feature map forza la rete a non dipendere da alcun singolo filtro per la classificazione.

La probabilità `p=0.25` è conservativa rispetto al Dropout FC: i layer conv hanno già un effetto regolarizzante intrinseco (condivisione dei pesi, riduzione di risoluzione), quindi necessitano di meno regolarizzazione esterna.

#### Dropout(p=0.5) nel Classificatore FC

Il layer `Linear(3136→512)` è il layer più denso e il principale rischio di overfitting su partizioni locali piccole. Con `p=0.5`, la metà delle unità viene azzerata casualmente a ogni passo di training, forzando la rete a sviluppare rappresentazioni ridondanti e distribuite. In fase di inferenza (`model.eval()`), Dropout è disabilitato e tutti i neuroni contribuiscono con i pesi originali (senza il fattore di scala $1/p$ perché PyTorch usa *inverted dropout* di default).

Il valore `p=0.5` è il valore classico proposto da Srivastava et al. (2014) per layer FC in classificazione.

#### Gradient Clipping in `train_step` — max\_norm=1.0

Il gradient clipping limita la norma L2 del gradiente aggregato su tutti i parametri a `max_norm=1.0` prima di ogni `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

In FL, il gradient clipping svolge un ruolo specifico nel mitigare il **client drift**: su dati non-i.i.d., il gradiente locale può divergere significativamente dalla direzione del gradiente globale, specialmente dopo molti inner steps H. Gradienti grandi amplificano questa divergenza, rendendo il modello aggregato in Fase A meno stabile. Limitare la norma dei gradienti locali contiene la divergenza massima tra worker e migliora la qualità dell'aggregazione FedAvg. Il valore `max_norm=1.0` è una scelta conservativa standard in letteratura FL; con learning rate 0.001 i gradienti sono tipicamente già nell'ordine di $10^{-3}$–$10^{-1}$, quindi il clipping interviene solo in casi di gradiente esplosivo.

#### Label Smoothing — $\epsilon = 0.1$

Con 62 classi, molte visivamente simili (`0`/`O`, `1`/`l`/`I`, `5`/`S`, `c`/`C`), la cross-entropy standard allena il modello a produrre distribuzioni dove quasi tutta la massa di probabilità è concentrata sulla classe corretta. Questo porta a modelli *sovra-confidenti*.

Label smoothing sostituisce il target hard $\delta_{k,y}$ con un target morbido:

$$\tilde{y}_k = (1 - \epsilon) \cdot \delta_{k,y} + \frac{\epsilon}{K}$$

dove $\epsilon = 0.1$ e $K = 62$. La probabilità target della classe corretta diventa 0.90 invece di 1.0, distribuendo 0.10 uniformemente tra tutte le classi. I benefici sono:
- **Riduzione dell'over-confidence** su classi ambigue, con output probabilistici meglio calibrati.
- **Miglioramento della generalizzazione post-aggregazione**: un modello calibrato su dati non-i.i.d. locali generalizza meglio quando i suoi pesi vengono mediati con quelli di worker con distribuzioni diverse.

*Nota*: label smoothing aumenta leggermente la loss di training (target meno estremi), ma riduce la validation loss — questo è il segnale atteso di miglioramento della generalizzazione. I confronti di accuracy tra configurazioni devono essere fatti sulla validation loss senza smoothing per comparabilità.

### 5.4 Confronto con il Modello Placeholder

| Caratteristica | Placeholder | Modello Proposto |
|---|---|---|
| Conv layers totali | 2 (valid padding) | 4 (same padding) |
| Feature map dopo pool finale | 64 × 12 × 12 = 9.216 | 64 × 7 × 7 = 3.136 |
| Parametri totali | ~2.4M | ~1.72M |
| BatchNorm | No | Sì (dopo ogni conv + FC) |
| Dropout conv | No | Spatial Dropout 25% |
| Dropout FC | No | 50% |
| Gradient clipping | No | Sì (max\_norm = 1.0) |
| Label smoothing | No | 10% |
| Dimensione messaggio gossip (float32) | ~9.6 MB | ~6.9 MB |

Il modello proposto ha **meno parametri** del placeholder nonostante abbia il doppio dei layer conv. La ragione è che il placeholder, con valid padding, produce una feature map 64×12×12 = 9.216 elementi dopo il pool; il modello proposto, con same padding e doppio pool, produce 64×7×7 = 3.136 elementi. Il layer FC1 (dominante) è quindi 3× più piccolo: $9.216 \times 256 \approx 2.4\text{M}$ vs $3.136 \times 512 \approx 1.6\text{M}$. Il ridotto numero di parametri ha un beneficio diretto sul sistema FL: ogni messaggio gossip trasporta i pesi del modello serializzato, e il risparmio del 28% per messaggio si moltiplica per il numero di round e di worker.

### 5.5 Early Stopping Locale

L'early stopping è implementato nel training loop principale (`main_worker.py`) e si basa sulla validation loss locale calcolata da `validate()`. La validazione avviene **dopo la Phase A** (aggregazione FedAvg), quindi misura la qualità del modello aggregato — non solo del modello locale pre-training. Se la validation loss non migliora di almeno $10^{-4}$ per `early_stopping_patience` round consecutivi (default: 10), il training locale si arresta.

Il comportamento post-early stopping è intenzionalmente non-terminante: il thread di training (Thread 2) si ferma, ma il **server gRPC (Thread 1) rimane attivo**. Questo comportamento è ottenuto chiamando `grpc_server.wait_for_termination()` dopo il break dal loop, che blocca il thread principale finché il server non viene fermato esternamente.

#### Differenza semantica rispetto all'early stopping centralizzato

In ML centralizzato, l'early stopping misura la loss sul validation set **globale**: se peggiora, il modello sta overfittando l'intero dataset di training. La decisione è globale e coordinata.

Nel nostro sistema la decisione è **locale e indipendente**: ogni worker misura la propria val_loss sulla propria partizione locale. Questo crea due problemi specifici del contesto FL:

1. **Convergenza locale ≠ convergenza globale.** Un worker con una partizione "facile" può raggiungere un plateau locale al round 30 mentre la rete FL globale non ha ancora raggiunto consenso. Fermare quel worker priva gli altri di un peer attivo nei round successivi.

2. **Perdita di un vicino gossip.** Quando un worker si ferma, chiama `deregister_worker()` nel blocco `finally` e sparisce dalla lista peer del Discovery Server. Gli altri worker non lo trovano più come target per i push di Phase C, riducendo il `gossip_fanout` effettivo della rete. Con 3 worker totali, la perdita di uno riduce il fanout disponibile da 2 a 1 — impatto significativo.

Il fatto che la validation avvenga dopo la Phase A mitiga parzialmente il primo problema: la loss misurata include il contributo degli aggiornamenti ricevuti dai vicini, non solo quello del training locale. Tuttavia non elimina il rischio di stopping prematuro.

#### Raccomandazione per gli esperimenti

Per i **confronti controllati** (Esperimenti 1–4 del piano sperimentale), è consigliabile disabilitare l'early stopping impostando `early_stopping_patience` a un valore superiore a `total_rounds` (es. `9999`). Questo garantisce che tutti i worker eseguano esattamente lo stesso numero di round, rendendo i confronti di accuratezza e convergenza direttamente comparabili.

Per **run di produzione** o esperimenti esplorativi dove si vuole evitare compute inutile su worker già convergenti, l'early stopping può rimanere abilitato con `patience: 10`.

### 5.6 Selezione degli Iperparametri

Il sistema non usa cross-validation (motivata in Sezione 2.3). L'approccio adottato per la selezione degli iperparametri è il **grid search su hold-out fisso**, che è lo stesso approccio usato nei paper di riferimento LEAF e FedAvg:

1. **Subset veloce per l'esplorazione.** Si lancia `download_femnist.py --sf 0.05` per ottenere il 5% del dataset (~170 scrittori). Con questa dimensione, ogni esperimento completa in pochi minuti anche su CPU.
2. **Grid search.** Si varia un iperparametro alla volta mantenendo gli altri ai valori di default. I candidati principali sono: `learning_rate` (1e-4, 1e-3, 5e-3), `inner_steps_H` (100, 500, 1000), `gossip_fanout` (1, 2, N-1). Per ogni configurazione si lancia `docker compose up` e si osserva la convergenza.
3. **Metrica di confronto.** Dopo ogni esperimento, `python scripts/aggregate_metrics.py` produce le statistiche globali. La metrica principale è la **mean accuracy finale** (media tra worker), con la **std accuracy** come indicatore di equità (worker che convergono uniformemente sono preferibili a quelli dove un worker eccelle e gli altri no).
4. **Conferma su dataset completo.** La configurazione migliore trovata sul subset 5% viene rieseguita su `--sf 1.0` per verificare che i risultati si scalino correttamente.

Questo procedimento è interamente abilitato dal sistema di metriche descritto nella Sezione 6.

---

## 6. Metriche di Prestazione

### 6.1 Architettura del Sistema di Metriche

In un sistema P2P decentralizzato, non esiste un nodo centrale che osservi le prestazioni globali in tempo reale. Il sistema di metriche adottato sfrutta la struttura dei **bind mount Docker**: ogni worker scrive le proprie metriche su `{data_dir}/metrics.csv`, che — essendo `data_dir` montata dall'host — è immediatamente visibile sul filesystem dell'host senza alcun trasferimento dati aggiuntivo.

```
Container Worker 0               Host
/app/data/femnist/ ←────────── ./data/femnist/worker_0/
   └── metrics.csv                   └── metrics.csv  ← leggibile durante e dopo l'esperimento

Container Worker 1               Host
/app/data/femnist/ ←────────── ./data/femnist/worker_1/
   └── metrics.csv                   └── metrics.csv

...

scripts/aggregate_metrics.py ── legge worker_*/metrics.csv
                              ── scrive global_metrics.csv
                              ── scrive summary.txt
```

Questo approccio non richiede alcuna modifica al Registry né alcun canale di comunicazione aggiuntivo tra worker: le metriche rimangono dati locali del worker, mai condivisi in rete.

### 6.2 Metriche Raccolte Per Worker

`core/metrics.py` implementa `MetricsWriter`, che appende una riga CSV al termine di ogni round. I campi registrati sono:

| Campo | Tipo | Descrizione |
|---|---|---|
| `worker_id` | string | Identificatore del worker |
| `round` | int | Numero del round corrente |
| `timestamp` | float | Unix timestamp (per analisi temporale reale) |
| `train_loss_avg` | float | Loss media su H inner steps della Fase B |
| `val_loss` | float | Loss sul test set locale dopo la Fase A |
| `val_accuracy` | float | Accuracy sul test set locale [0, 1] |
| `round_duration_s` | float | Durata totale del round (Fase A + B + C) in secondi |
| `neighbors_aggregated` | int | Numero di modelli vicini incorporati in Fase A (0 = nessuna aggregazione) |
| `peers_contacted` | int | Push gossip con successo in Fase C |

La riga viene scritta **dopo** le fasi A, B e C, incluse le durate di rete. La `round_duration_s` misura quindi il tempo reale di ogni ciclo completo del training loop.

### 6.3 Aggregazione Globale Post-Esperimento

`scripts/aggregate_metrics.py` legge tutti i file `worker_*/metrics.csv` e, se presenti, i checkpoint `worker_*/model_final.pt`. Produce:

**1. Tabella per round** — per ogni round, aggrega le metriche di tutti i worker attivi:

| Colonna | Significato |
|---|---|
| `mean_accuracy` | Accuracy media tra tutti i worker — indicatore della qualità del modello globale |
| `std_accuracy` | Deviazione standard dell'accuracy — misura di convergenza tra worker |
| `min_accuracy` / `max_accuracy` | Worker peggiore/migliore — identifica outlier |
| `workers_reporting` | Quanti worker erano ancora attivi (non early-stopped) in quel round |

**2. Riassunto per worker** — rounds completati, accuracy finale, accuracy migliore, media di peer contattati, media di vicini aggregati.

**3. Divergenza dei pesi (weight divergence)** — se i checkpoint finali `model_final.pt` sono presenti (salvati automaticamente al termine di ogni worker), lo script carica tutti i modelli, appiattisce i parametri float in un vettore 1-D e calcola la distanza L2 tra ogni coppia:

$$d(w_i, w_j) = \|w_i - w_j\|_2$$

Questa è la misura diretta di convergenza verso lo stesso punto: una distanza piccola indica che i worker hanno trovato soluzioni simili nello spazio dei pesi — il FL ha funzionato. Una distanza grande indica divergenza, causata tipicamente da troppo pochi round di gossip, valore di H eccessivo o distribuzione dei dati troppo eterogenea.

**4. Volume di comunicazione** — totale messaggi gossip inviati con successo × ~6.9 MB per messaggio = volume totale di dati trasferiti in rete.

I risultati vengono salvati in `data/femnist/global_metrics.csv` e `data/femnist/summary.txt`.

### 6.4 Baseline Senza Gossip (Confronto Isolamento vs FL)

Per verificare che il gossip apporti un contributo reale alla convergenza, il sistema supporta una **modalità baseline** configurabile:

```yaml
federated_learning:
  gossip_enabled: false   # disabilita Fase A e Fase C
```

Con `gossip_enabled: false`, ogni worker addestra il proprio modello **in completo isolamento**: non invia né riceve modelli dagli altri. Questo replica lo scenario in cui ogni dispositivo allena una rete solo sui propri dati locali, senza alcuna forma di apprendimento federato.

Il confronto atteso è:
- **Con gossip**: ogni worker migliora anche su classi poco rappresentate nei suoi dati, grazie alla conoscenza ricevuta dai vicini. La `std_accuracy` finale dovrebbe essere bassa (convergenza uniforme).
- **Senza gossip**: ogni worker è bravo sulle classi dei propri scrittori, ma generalizza male sulle classi degli altri. La `mean_accuracy` finale sarà inferiore e la `std_accuracy` più alta.

La differenza quantitativa tra i due esperimenti dimostra empiricamente il valore aggiunto del Federated Learning nell'architettura implementata.

### 6.5 Analisi della Scalabilità

La metrica di scalabilità principale è: **come cambiano accuracy e tempo di convergenza al variare di `num_workers`?** Il workflow per questa analisi è:

```bash
# Esperimento 1: 2 worker
#   1. Modificare num_workers: 2 in config.yaml
#   2. python scripts/split_dataset.py && python scripts/generate_compose.py
#   3. docker compose up --build
#   4. python scripts/aggregate_metrics.py
#   5. Copiare/rinominare global_metrics.csv → results/global_metrics_2w.csv

# Ripetere per num_workers = 3, 5, 10 ...
```

Le variabili di interesse per lo studio di scalabilità sono:

- **Accuracy a convergenza** (`mean_accuracy` all'ultimo round): tende a migliorare con più worker perché si esplora una distribuzione di dati più ampia.
- **Rounds a convergenza** (round in cui `mean_accuracy` si stabilizza): può aumentare con più worker perché i modelli aggregati partono da punti più distanti.
- **Deviazione standard dell'accuracy** (`std_accuracy`): misura l'equità — con molti worker non-i.i.d., alcuni possono convergere molto più lentamente di altri.
- **Volume totale di comunicazione** (messaggi × 6.9 MB): scala con O(N × R × M), dove N è num_workers, R i round, M gossip_fanout.
- **Durata per round** (`round_duration_s`): domina la Fase B (training locale), quasi indipendente da N — questo è il principale vantaggio del gossip P2P rispetto al FL centralizzato, dove l'aggregazione diventa un collo di bottiglia all'aumentare di N.

---

## 7. Piano Sperimentale

Questa sezione descrive l'intera metodologia sperimentale adottata per validare il sistema: cosa misurare, in quale ordine, e come interpretare i risultati. Gli esperimenti sono organizzati in cinque fasi progressive, ciascuna costruita sui risultati della precedente. Lo strumento principale di analisi è `scripts/aggregate_metrics.py`, che aggrega i file `metrics.csv` prodotti dai worker e calcola statistiche globali.

### 7.1 Struttura Complessiva dello Studio

```
Fase 0 — Preparazione
  └── Setup, download dataset, verifica installazione

Fase 1 — Baseline No-FL
  └── Esp. 1: training in isolamento (gossip_enabled: false)
  └── Obiettivo: misurare cosa ottiene ogni worker senza cooperazione

Fase 2 — FL Standard
  └── Esp. 2: FL con configurazione di default
  └── Obiettivo: dimostrare che il gossip migliora rispetto alla baseline

Fase 3 — Ricerca degli Iperparametri
  └── Esp. 3a: variare learning_rate
  └── Esp. 3b: variare inner_steps_H
  └── Esp. 3c: variare gossip_fanout
  └── Obiettivo: trovare la configurazione ottimale

Fase 4 — Analisi della Scalabilità
  └── Esp. 4: variare num_workers (2, 3, 5, 10)
  └── Obiettivo: misurare come cambiano accuracy e costo al crescere dei nodi

Fase 5 — Robustezza alla Fault Injection
  └── Esp. 5: variare drop_probability e crash_probability
  └── Obiettivo: trovare la soglia di tolleranza del sistema

Fase 6 — Esperimento Finale (Dataset Completo)
  └── Esp. 6: configurazione ottimale su --sf 1.0
  └── Obiettivo: risultati definitivi da riportare nella relazione
```

### 7.2 Fase 0 — Preparazione dell'Ambiente

Prima di qualsiasi esperimento, verificare che il sistema funzioni correttamente su un subset piccolo.

```bash
# 1. Scaricare il 5% del dataset (fast subset per tutti gli esperimenti di sviluppo)
python scripts/download_femnist.py --sf 0.05

# 2. Configurare config.yaml: num_workers: 3, tutti i default
python scripts/split_dataset.py
python scripts/generate_compose.py

# 3. Avviare il sistema e verificare che tutti e 3 i worker si registrino e partano
docker compose up --build
# Attendersi nei log: "[Worker 0] Registered", "[Worker 1] Registered", ecc.
# Attendersi: "=== Round 1/200 ===" nei log di ciascun worker

# 4. Al termine, verificare che esistano i file attesi:
ls data/femnist/worker_*/metrics.csv
ls data/femnist/worker_*/model_final.pt
python scripts/aggregate_metrics.py
```

Se tutto funziona correttamente, il sistema è pronto per gli esperimenti.

**Nota:** Per tutti gli esperimenti di sviluppo (Fasi 1–5) usare `--sf 0.05`. Il dataset al 5% contiene circa 170 scrittori per split, sufficienti per osservare convergenza in decine di round invece che centinaia. Il tempo di esecuzione passa da ore a minuti.

### 7.3 Esperimento 1 — Baseline No-FL (Training in Isolamento)

**Obiettivo:** misurare le prestazioni di ogni worker quando allena il modello esclusivamente sui propri dati, senza nessuna forma di cooperazione. Questi risultati costituiscono il **limite inferiore** di riferimento.

**Configurazione:**
```yaml
federated_learning:
  gossip_enabled: false   # ← unica modifica rispetto al default
  total_rounds: 100
```

**Procedura:**
```bash
# Modificare config.yaml come sopra, poi:
docker compose up --build
python scripts/aggregate_metrics.py
# Salvare i risultati:
cp data/femnist/global_metrics.csv results/exp1_baseline_no_fl.csv
cp data/femnist/summary.txt results/exp1_summary.txt
# Pulire per il prossimo esperimento:
rm data/femnist/worker_*/metrics.csv data/femnist/worker_*/model_final.pt
```

**Risultati attesi:**
- Ogni worker converge localmente: la sua `val_accuracy` aumenta nel tempo.
- La `std_accuracy` rimane alta o non tende a zero: i worker hanno performance diverse perché ognuno conosce solo i propri scrittori.
- La `mean_accuracy` finale sarà il benchmark da battere con il FL.
- Il `mean pairwise L2 distance` dei pesi finali sarà elevato: i modelli hanno divergito, ognuno specializzandosi sul proprio subset.

**Cosa dimostra:** senza gossip, il sistema riduce a K addestramenti indipendenti. I worker con più varietà di classi nei loro dati otterranno accuracy migliore; quelli con subset sbilanciati risulteranno più deboli su certe classi.

### 7.4 Esperimento 2 — FL con Configurazione di Default

**Obiettivo:** dimostrare che il gossip P2P migliora la convergenza rispetto alla baseline. Questo è il confronto fondamentale che valida l'utilità del Federated Learning nell'architettura implementata.

**Configurazione:**
```yaml
federated_learning:
  gossip_enabled: true    # ← ripristinato
  total_rounds: 100
  inner_steps_H: 500
  early_stopping_patience: 10
network:
  gossip_fanout: 3
```

**Procedura:**
```bash
docker compose up --build
python scripts/aggregate_metrics.py
cp data/femnist/global_metrics.csv results/exp2_fl_default.csv
rm data/femnist/worker_*/metrics.csv data/femnist/worker_*/model_final.pt
```

**Risultati attesi e confronto con Esp. 1:**

| Metrica | Esp. 1 (no FL) | Esp. 2 (FL) | Interpretazione |
|---|---|---|---|
| `mean_accuracy` finale | valore base | **più alta** | il gossip porta conoscenza dagli altri worker |
| `std_accuracy` finale | alta | **più bassa** | i worker convergono a performance simili |
| `L2 weight distance` | alta | **più bassa** | i modelli si sono avvicinati nello spazio dei pesi |
| Rounds a convergenza | — | valore di riferimento | baseline temporale per confronti futuri |

Se `mean_accuracy` FL > `mean_accuracy` no-FL e `std_accuracy` FL < `std_accuracy` no-FL, il Federated Learning ha dimostrato il suo valore su questa architettura.

**Cosa monitorare nei log durante l'esecuzione:**
- "FedAvg applied" compare nei log ad ogni round in cui si ricevono aggiornamenti.
- `neighbors_aggregated` nel CSV deve essere > 0 per la maggior parte dei round.
- La `val_accuracy` deve crescere più velocemente che nel baseline.

### 7.5 Esperimento 3 — Ricerca degli Iperparametri

**Obiettivo:** trovare la configurazione ottimale degli iperparametri di training. Si varia un parametro alla volta mantenendo gli altri al valore dell'Esp. 2. La metrica di confronto è la `mean_accuracy` finale da `aggregate_metrics.py`.

Tutti gli esperimenti di questa fase usano `--sf 0.05` e `total_rounds: 50` per velocità.

#### 3a — Learning Rate

| Config | `learning_rate` | Risultato atteso |
|---|---|---|
| A (default) | 0.001 | riferimento |
| B | 0.0001 | convergenza più lenta ma stabile |
| C | 0.005 | convergenza rapida, rischio instabilità |

```bash
# Per ogni valore, modificare config.yaml e lanciare:
docker compose up --build && python scripts/aggregate_metrics.py
cp data/femnist/global_metrics.csv results/exp3a_lr_VALORE.csv
rm data/femnist/worker_*/metrics.csv data/femnist/worker_*/model_final.pt
```

#### 3b — Inner Steps H

| Config | `inner_steps_H` | Effetto atteso |
|---|---|---|
| A | 100 | aggregazione frequente, convergenza più solida ma più comunicazione |
| B (default) | 500 | bilanciamento ottimale (ispirato a DiLoCo) |
| C | 1000 | meno comunicazione, ma i modelli divergono di più localmente |

Questo è il parametro che bilancia **qualità dell'aggregazione** vs **costo di comunicazione**. Con H grande, ogni worker si allontana di più dagli altri tra un gossip e l'altro; la media pesata sarà meno accurata ma il numero di messaggi scambiati sarà inferiore.

#### 3c — Numero di Gossip Peers (M)

| Config | `gossip_fanout` | Effetto atteso |
|---|---|---|
| A | 1 | propagazione lenta: serve più round per diffondere informazioni |
| B (default) | 2 | buon compromesso |
| C | N−1 | massima connettività ma volume di comunicazione più alto |

Con M = N−1, ogni worker invia il modello a tutti gli altri ad ogni round. Con M = 1, la diffusione è più lenta ma il traffico è minimo.

**Scelta della configurazione ottimale:** al termine di Esp. 3, selezionare la combinazione `(lr, H, M)` con la `mean_accuracy` più alta su `--sf 0.05`. Questa diventa la **configurazione fissa** per tutti gli esperimenti successivi.

### 7.6 Esperimento 4 — Analisi della Scalabilità

**Obiettivo:** misurare come le prestazioni del sistema cambiano al variare del numero di worker. Questo è il requisito sperimentale esplicito della traccia di progetto ("analisi della scalabilità").

**Configurazione:** usare la configurazione ottimale trovata in Esp. 3. Variare solo `num_workers`.

**Procedura per ogni valore di N:**
```bash
# 1. Modificare num_workers in config.yaml
# 2. Ripartizionare il dataset (obbligatorio ad ogni cambio di num_workers)
python scripts/split_dataset.py
python scripts/generate_compose.py
# 3. Avviare e raccogliere risultati
docker compose up --build
python scripts/aggregate_metrics.py
cp data/femnist/global_metrics.csv results/exp4_scalability_Nw.csv
rm data/femnist/worker_*/metrics.csv data/femnist/worker_*/model_final.pt
```

**Valori di N da testare:** 2, 3, 5, 10 (e oltre se le risorse lo permettono su AWS).

**Metriche di scalabilità da raccogliere e riportare:**

| Metrica | Come cambia con N crescente | Perché |
|---|---|---|
| `mean_accuracy` finale | tendenza a crescere fino a un plateau | più worker = più varietà di dati distribuita |
| `std_accuracy` finale | tendenza a crescere | più worker = distribuzione non-i.i.d. più eterogenea |
| Rounds a convergenza | tende a crescere | i modelli devono mediare contributi più diversi |
| `round_duration_s` media | quasi invariata | la Fase B (training locale) domina e non dipende da N |
| Volume comunicazione totale | cresce con O(N × R × M) | più worker × più round × stessi M peers |

Il vantaggio del gossip P2P emerge chiaramente nell'ultimo punto: il volume di comunicazione scala linearmente con N (ogni worker invia a M peer fissi), non quadraticamente come nel FL centralizzato dove il server riceve da tutti N i worker ad ogni round.

### 7.7 Esperimento 5 — Robustezza alla Fault Injection

**Obiettivo:** misurare la resistenza del sistema a condizioni di rete avverse e crash di nodo. Trovare le soglie oltre le quali il training non converge più.

**Configurazione:** num_workers ottimale da Esp. 4, configurazione ottimale da Esp. 3.

#### 5a — Variare drop_probability

Testare: 0.0 (nessuna perdita), 0.2 (default), 0.5, 0.8.

Con `drop_probability: 0.8` e `gossip_fanout: 2`, il numero atteso di messaggi inviati per round è solo $2 \times 0.2 = 0.4$ — meno di uno per round in media. Il sistema dovrebbe mostrare convergenza più lenta o divergenza.

#### 5b — Variare crash_probability

Testare: 0.0, 0.05 (default), 0.1, 0.2.

Con `crash_probability: 0.2` ogni worker ha una probabilità del 20% di crashare ad ogni round. Con 3 worker, la probabilità che almeno uno crashi ad ogni round è $1 - 0.8^3 \approx 49\%$ — quasi un crash a round. Verificare che i worker superstiti continuino il training.

**Risultati attesi:** il sistema è robusto fino a una certa soglia (quella propria del gossip asincrono), poi degrada gradualmente piuttosto che fallire improvvisamente. Questo comportamento di *graceful degradation* è una proprietà fondamentale dell'architettura P2P.

### 7.8 Esperimento Finale — Dataset Completo

**Obiettivo:** produrre i risultati definitivi da riportare nella relazione, su dataset completo.

```bash
# 1. Scaricare il dataset completo (solo se non già presente)
python scripts/download_femnist.py --sf 1.0

# 2. Configurare la configurazione ottimale (da Esp. 3) + num_workers ottimale (da Esp. 4)
#    + fault injection ai valori di default

# 3. Partizionare e avviare
python scripts/split_dataset.py
python scripts/generate_compose.py
docker compose up --build   # o su AWS

# 4. Raccogliere i risultati definitivi
python scripts/aggregate_metrics.py
```

Questo esperimento può richiedere ore su CPU locale; è il principale candidato per il deployment su AWS EC2.

### 7.9 Analisi e Visualizzazione dei Risultati

I file `global_metrics.csv` prodotti da ogni esperimento contengono tutte le informazioni necessarie per i grafici della relazione. Di seguito i plot consigliati:

**Plot 1 — Curva di convergenza (Esp. 1 vs Esp. 2)**
`round` (asse X) vs `mean_accuracy` (asse Y), una linea per "no-FL" e una per "FL". Mostra visivamente che il gossip accelera la convergenza e raggiunge un'accuracy più alta.

**Plot 2 — Deviazione standard dell'accuracy nel tempo**
`round` (asse X) vs `std_accuracy` (asse Y). Con FL funzionante, questa curva dovrebbe tendere verso lo zero: i worker convergono a soluzioni sempre più simili.

**Plot 3 — Scalabilità: accuracy vs num_workers**
Barre o punti: un punto per ogni valore di N, con `mean_accuracy` finale (asse Y). Mostra la curva di rendimento marginale decrescente: aggiungere worker aiuta fino a un certo punto.

**Plot 4 — Scalabilità: volume comunicazione vs num_workers**
`num_workers` (asse X) vs `total_gossip_messages × 6.9 MB` (asse Y). Mostra la crescita lineare del costo di comunicazione — vantaggio del gossip P2P rispetto al FL centralizzato.

**Plot 5 — Robustezza: accuracy finale vs drop_probability**
`drop_probability` (asse X) vs `mean_accuracy` finale (asse Y). Identifica la soglia di tolleranza oltre la quale le prestazioni degradano significativamente.

Tutti questi grafici possono essere generati in Python con `matplotlib` leggendo direttamente i CSV salvati da ogni esperimento.

---

## 8. Tolleranza ai Guasti e Fault Injection

Il sistema include tre meccanismi di fault injection configurabili, progettati per simulare le condizioni avverse di una rete reale distribuita. I parametri sono raggruppati nella sezione `fault_injection` di `config.yaml`.

### 8.1 Message Drop (Perdita di Messaggi)

#### Implementazione

Prima di ogni tentativo di gossip push nella Fase C, viene estratto un valore casuale $u \sim \mathcal{U}(0, 1)$. Se $u < p_{\text{drop}}$ (default: 0.20), la trasmissione verso quel vicino viene saltata senza tentare la connessione gRPC:

```python
if random.random() < drop_prob:
    dropped_count += 1
    continue   # skip send_model entirely
```

Il drop avviene **prima** della chiamata gRPC, non dopo. Questo modella la perdita di pacchetti a livello di rete (il messaggio non viene nemmeno inviato) piuttosto che un rifiuto esplicito del ricevente.

#### Robustezza intrinseca dell'algoritmo

L'algoritmo è **robusto per costruzione** al message drop: la Fase A aggrega esclusivamente i modelli effettivamente ricevuti nel buffer. Se un round produce zero messaggi ricevuti (tutti droppati, nessun vicino attivo), la Fase A viene semplicemente saltata e il worker procede con il suo modello invariato. Non esiste alcuna dipendenza su una soglia minima di messaggi ricevuti per procedere.

Con $p_{\text{drop}} = 0.20$ e $M = 3$ vicini, il numero atteso di messaggi inviati con successo per round è $3 \times (1 - 0.20) = 2.4$. Il numero di messaggi ricevuti da ogni worker dipende da quanti vicini lo abbiano selezionato come target: con $N=3$ worker, ogni worker è selezionato in media da $2 \times (M/2) \times (1 - p_{\text{drop}}) \approx 1.6$ peer per round.

### 8.2 Node Crash (Crash del Nodo)

#### Implementazione

Ad ogni round, dopo il completamento della Fase B e prima della Fase C, viene estratto $u \sim \mathcal{U}(0, 1)$. Se $u < p_{\text{crash}}$ (default: 0.05), il processo esegue `sys.exit(1)`:

```python
if random.random() < crash_prob:
    logger.warning("FAULT INJECTION: simulated node crash via sys.exit(1)")
    sys.exit(1)
```

#### Semantica di sys.exit(1) e il blocco finally

La scelta di `sys.exit(1)` è deliberata e ha implicazioni precise:

1. `sys.exit(1)` solleva `SystemExit`, un'eccezione Python che **attraversa** i blocchi `finally`. Questo garantisce che il blocco `finally` in `main()` — che chiama `deregister_worker()` — venga eseguito prima che il processo termini. Il Registry riceve la deregistrazione e il checkpoint viene salvato.

2. `SystemExit` non viene catturata da `grpc_server.wait_for_termination()`, che non viene mai raggiunta. Il processo termina effettivamente — simulando un crash reale piuttosto che una terminazione pulita.

3. Il Docker container si arresta con exit code 1, il che (in assenza di `restart: always` nel compose) lascia il servizio down — comportamento intenzionale.

Lo stesso meccanismo `finally` è sfruttato dai signal handler descritti in Sezione 8.4: SIGTERM e SIGINT vengono intercettati e reindirizzati a `sys.exit(0)`, garantendo la stessa sequenza di cleanup (deregistrazione + checkpoint) anche per shutdown manuali e `docker stop`.

#### Gestione del nodo crashato dagli altri worker

I worker che tentano di contattare il nodo crashato ricevono un `grpc.RpcError` con codice `UNAVAILABLE` o `DEADLINE_EXCEEDED`. La funzione `send_model()` cattura questo errore e restituisce `False` senza propagare l'eccezione:

```python
except grpc.RpcError as e:
    logger.warning(f"Failed to send to {address}: {e.code()} — {e.details()}")
    return False
```

Il training loop prosegue verso i vicini successivi: il crash di un nodo non interrompe il round degli altri.

### 8.3 gRPC Timeout

#### Il problema: blocking indefinito

Senza timeout, una chiamata gRPC verso un nodo irraggiungibile attende indefinitamente la risposta del server TCP, bloccando il thread che ha effettuato la chiamata. Nel training loop, questo serializzerebbe la Fase C sul tempo di attesa della rete — potenzialmente infinito.

#### Implementazione

Ogni chiamata `stub.ReceiveModel()` include un timeout esplicito:

```python
ack = stub.ReceiveModel(message, timeout=timeout)  # timeout = grpc_timeout_seconds
```

Se il server non risponde entro `grpc_timeout_seconds` (default: 5.0 s), gRPC solleva `grpc.RpcError` con codice `DEADLINE_EXCEEDED`, che viene gestita come un failure silenzioso.

#### Trade-off nella scelta del timeout

Un timeout troppo basso (es. 0.5 s) potrebbe rifiutare connessioni legittime verso nodi lenti ma attivi, degradando artificialmente il numero di aggiornamenti ricevuti. Un timeout troppo alto (es. 60 s) renderebbe il round lento in presenza di nodi crashati. Il valore di 5 secondi è sufficiente per reti locali (latenza <1 ms) e ragionevole per EC2 nella stessa region (latenza tipica 1–10 ms), lasciando ampio margine per serializzazione e deserializzazione dei pesi.

### 8.4 Graceful Shutdown: Signal Handling

#### Il problema: hard crash vs shutdown gestito

Il meccanismo di fault injection (Sezione 8.2) simula crash tramite `sys.exit(1)`, che attraversa il blocco `finally` e deregistra il worker pulitamente. Nella realtà esistono però scenari di terminazione che non passano per il codice Python:

| Evento | Segnale | Intercettabile? | Deregistrazione |
|---|---|:---:|:---:|
| `docker stop` / `docker compose down` | SIGTERM | ✅ | ✅ con handler |
| Ctrl+C (terminale o `docker attach`) | SIGINT | ✅ | ✅ con handler |
| `docker kill` | SIGKILL | ❌ | ❌ |
| OOM killer del kernel | SIGKILL | ❌ | ❌ |

SIGTERM e SIGINT sono intercettabili in Python tramite `signal.signal()`. SIGKILL è inviato direttamente dal kernel al processo e non può essere catturato in nessun linguaggio — è il meccanismo di terminazione forzata di Unix.

#### Implementazione

All'avvio, dopo la registrazione presso il Discovery Server, vengono installati due handler:

```python
def _handle_shutdown(signum, frame):
    logger.info(f"Signal {signum} received — shutting down cleanly")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)
```

Entrambi chiamano `sys.exit(0)`, che solleva `SystemExit` e attraversa il blocco `finally` — la stessa sequenza del crash simulato, ma con exit code 0 (terminazione normale). Il risultato è:

1. `deregister_worker()` viene chiamato → il worker sparisce dalla lista peer
2. Il checkpoint `model_final.pt` viene salvato
3. Il processo termina con exit code 0

#### Known limitation: SIGKILL e OOM

Se il container viene terminato con `docker kill` o dall'OOM killer del kernel, il processo riceve SIGKILL e termina istantaneamente senza eseguire alcun codice Python. In questo caso:
- Il worker rimane nel registry fino al successivo riavvio (entry stale)
- Gli altri worker continueranno a tentare push verso di esso, ricevendo `UNAVAILABLE`
- Il meccanismo di re-query reattivo (Sezione 4.2) attiva automaticamente la ricerca di peer sostitutivi

Una soluzione completa richiederebbe un meccanismo di **heartbeat con TTL** nel registry: i worker inviano periodicamente un segnale di vita, e il registry rimuove automaticamente chi non si fa vivo da T secondi. Questo è il pattern adottato in protocolli di membership production-grade come SWIM. Per il perimetro di questo progetto, dove i crash SIGKILL non fanno parte del modello di fault injection, il meccanismo di re-query reattivo costituisce una mitigazione sufficiente.

---

## 9. Implementazione e Deployment

### 9.1 Struttura dei File

```
ml_sdcc_project/
├── gossip.proto              # gRPC service and message definitions
├── registry_server.py        # Discovery Server (Flask)
├── main_worker.py            # Worker entry point — training loop + gRPC server
├── config.yaml               # Single source of truth for all parameters
├── .dockerignore             # Excludes data/, scripts/, docs from build context
├── requirements.registry.txt # Registry dependencies (Flask only)
├── requirements.worker.txt   # Worker dependencies (PyTorch, gRPC, ...)
├── Dockerfile.registry       # Minimal image: no PyTorch, no grpcio
├── Dockerfile.worker         # Full image: PyTorch + gRPC + proto compilation
├── docker-compose.yml        # [GENERATED] Local deployment — do not edit manually
├── docker-compose.aws.yml    # [GENERATED] AWS EC2 deployment — do not edit manually
├── scripts/
│   ├── download_femnist.py      # LEAF dataset download and preprocessing
│   ├── split_dataset.py         # Splits dataset into per-worker partitions
│   ├── generate_compose.py      # Generates compose files from config.yaml
│   └── aggregate_metrics.py     # Aggregates per-worker CSVs into global stats
├── core/
│   ├── dataset.py               # LEAF data loading (no splitting logic)
│   ├── model.py                 # CNN for FEMNIST (VGG-style double-block architecture)
│   ├── trainer.py               # train_step (clip_grad + label_smoothing) and validate
│   └── metrics.py               # MetricsWriter: per-round CSV logging per worker
└── network/
    ├── grpc_server.py           # Thread 1: receiver + online aggregation
    └── grpc_client.py           # Gossip push with configurable timeout
```

### 9.2 Containerizzazione e Build Docker

#### Due immagini separate

Il sistema usa due immagini Docker distinte, in accordo con il principio di separazione delle responsabilità:

- **`Dockerfile.registry`** — immagine minimale: solo `python:3.11-slim` + Flask. Non contiene PyTorch, grpcio o il codice worker. Dimensione tipica: ~80 MB.
- **`Dockerfile.worker`** — immagine completa: PyTorch CPU, grpcio, grpcio-tools. Dimensione tipica: ~1.5 GB.

La separazione riduce significativamente i tempi di rebuild del registry (nessuna dipendenza pesante) e minimizza la superficie di attacco dell'immagine registry.

#### Ottimizzazione del layer caching

Docker costruisce le immagini a strati: ogni istruzione (`FROM`, `RUN`, `COPY`) produce un layer immutabile identificato da un hash. Se alla build successiva l'hash di un layer coincide con quello in cache, Docker lo riutilizza senza rieseguire il comando. L'invalidazione è **a cascata**: modificare un layer invalida automaticamente tutti quelli successivi, indipendentemente dal loro contenuto.

La regola pratica che ne discende è ordinare le istruzioni dal più stabile al più volatile: le dipendenze pesanti in cima, il codice sorgente in fondo. Entrambi i Dockerfile rispettano questo principio.

**`Dockerfile.worker` — sequenza dei layer:**

```dockerfile
# Layer 1 — base image: invalido solo al cambio di versione Python
FROM python:3.11-slim

# Layer 2 — compilatori di sistema: invalido solo se cambia il comando apt
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Layer 3 — dipendenze Python (~1.5 GB, richiede diversi minuti):
#   invalido solo se cambia requirements.worker.txt
COPY requirements.worker.txt .
RUN pip install --no-cache-dir -r requirements.worker.txt

# Layer 4 — compilazione Protobuf: invalido solo se cambia gossip.proto
COPY gossip.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gossip.proto

# Layer 5 — sorgente applicativo: invalido ad ogni modifica al codice
COPY config.yaml main_worker.py ./
COPY core/ ./core/
COPY network/ ./network/
```

Il layer più costoso è il Layer 3 (installazione di PyTorch): viene rieseguito solo se `requirements.worker.txt` cambia. Ogni modifica al codice Python invalida esclusivamente il Layer 5 — la rebuild richiede secondi invece di minuti. Lo stesso principio vale per `Dockerfile.registry`: prima `requirements.registry.txt`, poi `registry_server.py`.

I file generati dalla compilazione Protobuf (`gossip_pb2.py`, `gossip_pb2_grpc.py`) esistono nel container prima che il sorgente venga copiato; il `COPY` successivo non li sovrascrive perché sono esclusi dal repository tramite `.gitignore`.

**Build context e `.dockerignore`.** Durante `docker compose up --build`, Docker trasferisce l'intera directory di progetto al daemon come *build context* prima ancora di valutare la cache. Senza `.dockerignore`, `data/femnist/` (potenzialmente diversi GB) verrebbe trasferita ad ogni build anche con tutti i layer in cache — annullando il vantaggio del caching. Il file `.dockerignore` esclude `data/`, `leaf/`, `scripts/`, i file di documentazione e la cache Python, riducendo il build context al solo sorgente necessario.

#### Healthcheck e dipendenze tra servizi

Il `docker-compose.yml` configura ogni worker con dipendenza condizionale dal registry:

```yaml
depends_on:
  registry:
    condition: service_healthy
```

Il healthcheck del registry verifica che `/peers` risponda correttamente prima di avviare i worker. Questo evita race condition allo startup: senza healthcheck, un worker potrebbe tentare la registrazione mentre Flask non ha ancora completato l'inizializzazione.

Il meccanismo di retry in `register_worker()` — con `max_retries=10` tentativi e pausa di 3 secondi tra l'uno e l'altro — costituisce un secondo livello di resilienza per gestire variabilità nei tempi di avvio dei container.

### 9.3 Gestione Dinamica del Numero di Worker

Il numero di worker non è hardcoded nei compose file ma letto da `config.yaml`. Due script cooperano per mantenere il sistema coerente: `split_dataset.py` prepara i dati su host, `generate_compose.py` configura i container. I compose file sono **artefatti generati** e non vanno editati manualmente.

```bash
# Workflow per modificare il numero di worker:
# 1. Modificare network.num_workers in config.yaml
# 2. Rieseguire il partizionamento (sovrascrive le slice precedenti)
python scripts/split_dataset.py
# 3. Rigenerare i compose file
python scripts/generate_compose.py
# 4. Riavviare il sistema
docker compose up --build
```

`generate_compose.py` produce `docker-compose.yml` e `docker-compose.aws.yml` con il numero corretto di servizi. Il registry riceve `REGISTRY_PORT` come variabile d'ambiente, in modo che la porta su cui ascolta sia sempre coerente con quella configurata in `config.yaml`. Ogni worker riceve `WORKER_ID=i` e `TOTAL_WORKERS=num_workers` come variabili d'ambiente, e monta esclusivamente la propria partizione tramite **bind mount** Docker (`type: bind`, sintassi lunga esplicita) — isolamento dei dati garantito a livello di filesystem.

In AWS, ogni worker ottiene anche una porta host distinta (`grpc_port + i`) per evitare conflitti nel caso di deployment su istanza singola (Opzione A).

### 9.4 Deploy su AWS EC2

Il file `docker-compose.aws.yml` gestisce due modalità di deployment:

**Opzione A — Istanza singola**: tutti i container girano sullo stesso EC2. I worker si trovano tramite nomi Docker interni (`worker_0`, `worker_1`, ecc.) ma registrano il loro IP pubblico nel Discovery Server per compatibilità con l'Opzione B.

**Opzione B — Istanza per worker**: ogni worker gira su un EC2 separato. Ogni nodo ha un IP pubblico distinto, configurato tramite la variabile d'ambiente `EC2_PUBLIC_IP` (usata come `MY_HOST` per la registrazione). Il registry è raggiungibile tramite `REGISTRY_EC2_IP`.

La sovrascrittura dell'URL del registry via variabile d'ambiente evita di dover modificare `config.yaml` tra deploy locale e cloud:

```python
registry_url = os.environ.get("REGISTRY_URL", cfg["network"]["registry_url"])
```

I Security Group AWS devono consentire:
- Porta **5000** (TCP) — registry HTTP, accessibile da tutti i worker;
- Porte **`grpc_port` … `grpc_port + num_workers − 1`** — gRPC workers, accessibili tra le istanze.

---

## 10. Parametri di Configurazione

Tutti i parametri operativi del sistema sono centralizzati in `config.yaml`, unica sorgente di verità per l'intera infrastruttura. La modifica di un parametro in questo file si propaga automaticamente a tutti i componenti al successivo `docker compose up --build`.

### 10.1 Tabella dei Parametri

| Sezione | Parametro | Default | Descrizione |
|---|---|:---:|---|
| `network` | `registry_url` | `http://registry:5000` | Endpoint del Discovery Server; sovrascrivibile via `REGISTRY_URL` env var |
| `network` | `registry_port` | `5000` | Porta HTTP su cui il registry ascolta; iniettata come `REGISTRY_PORT` env var nel container |
| `network` | `grpc_port` | `50051` | Porta gRPC esposta da ogni worker |
| `network` | `num_workers` | `3` | Numero totale di worker; modifica + `python scripts/generate_compose.py` |
| `network` | `gossip_fanout` (M) | `3` | Vicini contattati per round nella Fase C |
| `federated_learning` | `total_rounds` | `200` | Tetto massimo di round; può terminare prima per early stopping |
| `federated_learning` | `inner_steps_H` | `500` | Step di training locale per round (Fase B) |
| `federated_learning` | `early_stopping_patience` | `10` | Round consecutivi senza miglioramento prima di fermare il training |
| `federated_learning` | `gossip_enabled` | `true` | `false` = training in isolamento totale (baseline no-FL per confronto) |
| `machine_learning` | `batch_size` | `32` | Dimensione del mini-batch per AdamW |
| `machine_learning` | `learning_rate` | `0.001` | Learning rate dell'ottimizzatore AdamW |
| `machine_learning` | `clip_grad` | `1.0` | Max norma L2 per gradient clipping (0 = disabilitato) |
| `machine_learning` | `label_smoothing` | `0.1` | Label smoothing sulla cross-entropy (0 = disabilitato) |
| `machine_learning` | `dropout_conv` | `0.25` | Probabilità dropout spaziale nei blocchi conv |
| `machine_learning` | `dropout_fc` | `0.5` | Probabilità dropout prima del layer FC finale |
| `metrics` | `enabled` | `true` | Abilita/disabilita il logging CSV delle metriche per worker |
| `metrics` | `output_file` | `metrics.csv` | Nome del file CSV scritto in `data_dir/` |
| `fault_injection` | `drop_probability` | `0.20` | Probabilità di saltare un gossip push verso un vicino |
| `fault_injection` | `crash_probability` | `0.05` | Probabilità di crash simulato (`sys.exit(1)`) per round |
| `fault_injection` | `grpc_timeout_seconds` | `5.0` | Timeout massimo per ogni chiamata gRPC client |
| `fault_injection` | `max_staleness` ($\Delta_{\max}$) | `10` | Round massimi di ritardo accettati dallo staleness check |

### 10.2 File di Configurazione Completo

```yaml
network:
  registry_url: "http://registry:5000"  # Overridable via REGISTRY_URL env var
  registry_port: 5000                   # Port the registry server listens on
  grpc_port: 50051
  gossip_fanout: 3
  num_workers: 3                        # Change here, then run: python scripts/generate_compose.py

federated_learning:
  total_rounds: 200           # Maximum number of rounds
  inner_steps_H: 500          # Local training steps between two gossip rounds
  aggregation_strategy: "FedAvg"
  early_stopping_patience: 10 # Rounds without improvement before stopping training

machine_learning:
  dataset: "femnist"
  data_dir: "/app/data/femnist"
  batch_size: 32
  learning_rate: 0.001
  optimizer: "AdamW"
  clip_grad: 1.0          # max L2 norm for gradient clipping (0 = disabled)
  label_smoothing: 0.1    # cross-entropy label smoothing (0 = disabled)
  dropout_conv: 0.25      # spatial dropout probability in conv blocks
  dropout_fc: 0.5         # dropout probability before the final FC layer

metrics:
  enabled: true
  output_file: "metrics.csv"  # written to data_dir/metrics.csv (mounted on host)

fault_injection:
  drop_probability: 0.20      # Probability of dropping a gossip push
  crash_probability: 0.05     # Probability of simulated crash per round
  grpc_timeout_seconds: 5.0   # Timeout for each gRPC call
  max_staleness: 10           # Discard updates older than N rounds
```

---

## 11. Istruzioni di Esecuzione

Il workflow segue quattro passi in sequenza. I passi 1 e 2 sono una-tantum; i passi 3 e 4 vanno ripetuti ogni volta che si cambia `num_workers`.

> **Nota di compatibilità — Pillow ≥ 10.0.**
> Lo script di preprocessing di LEAF (`leaf/data/femnist/preprocess/data_to_json.py`) usa `Image.ANTIALIAS`, rimosso in Pillow 10.0 (2023) in favore di `Image.LANCZOS`. I due identificano lo stesso filtro di ricampionamento (sinc di Lanczos): la sostituzione è puramente nominale e **non altera in alcun modo il dataset prodotto**. `download_femnist.py` applica automaticamente questa patch subito dopo il clone di LEAF, prima di avviare il preprocessing — non è richiesto alcun intervento manuale.

```bash
# Passo 1 — Scarica e preprocessa il dataset FEMNIST (una sola volta)
python scripts/download_femnist.py --sf 1.0
# --sf 0.05 per un subset veloce (5%); 1.0 per il dataset completo (~2-4 GB)

# Passo 2 — Imposta i parametri in config.yaml
#   In particolare: num_workers, gossip_fanout, total_rounds, ecc.

# Passo 3 — Partiziona il dataset e rigenera i compose file
python scripts/split_dataset.py      # crea data/femnist/worker_{i}/
python scripts/generate_compose.py   # rigenera docker-compose.yml e docker-compose.aws.yml

# Passo 4a — Avvia il sistema in locale
docker compose up --build

# Passo 4b — Deploy su singola istanza EC2
EC2_PUBLIC_IP=<ip-pubblico-ec2> \
docker compose -f docker-compose.aws.yml up --build -d

# Passo 4c — Deploy multi-istanza EC2 (un worker per istanza)
# Sul nodo registry:
docker compose -f docker-compose.aws.yml up registry
# Su ogni nodo worker (esempio per worker_0):
REGISTRY_EC2_IP=<ip-registry> EC2_PUBLIC_IP=<ip-questo-nodo> \
docker compose -f docker-compose.aws.yml up worker_0

# Passo 5 — Analisi delle metriche (al termine dell'esperimento o in corso)
# Ogni worker ha scritto data/femnist/worker_{i}/metrics.csv durante il training.
python scripts/aggregate_metrics.py
# Output: tabella per round (mean/std/min/max accuracy), riassunto per worker,
#         data/femnist/global_metrics.csv, data/femnist/summary.txt

# Per confrontare due configurazioni diverse:
#   - Prima di ogni esperimento, eliminare i vecchi metrics.csv:
#     rm data/femnist/worker_*/metrics.csv
#   - Dopo ogni esperimento, salvare i risultati:
#     cp data/femnist/global_metrics.csv results/global_metrics_<config>.csv
```

---

## Riferimenti

[1] Douillard, A., Feng, Q., Ruder, S., Dieleman, S., Bousquet, O., & Houlsby, N. (2023). *DiLoCo: Distributed Low-Communication Training of Language Models*. arXiv:2311.08105.

[2] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017.

[3] Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Konečný, J., McMahan, H. B., Smith, V., & Talwalkar, A. (2018). *LEAF: A Benchmark for Federated Settings*. arXiv:1812.01097.
