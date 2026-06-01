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

#### Perché la ponderazione per $n_k$ è l'unica scelta corretta

La ponderazione non è una convenzione arbitraria: deriva direttamente dall'obiettivo di minimizzare la loss globale sul dataset complessivo. La loss globale si scrive come:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i; \theta), y_i) = \sum_{k=1}^{K} \frac{n_k}{N} \mathcal{L}_k(\theta)$$

dove $N = \sum_k n_k$ è il totale dei campioni e $\mathcal{L}_k$ è la loss locale del worker $k$. La media pesata dei parametri ottimali locali — sotto l'ipotesi semplificativa che ogni worker abbia trovato il proprio ottimo locale $w_k^*$ — è l'approssimazione di primo ordine all'ottimo globale $w^* = \arg\min \mathcal{L}(\theta)$. Una media non pesata equivarrebbe a minimizzare $\frac{1}{K}\sum_k \mathcal{L}_k$ — una loss uniforme per worker che dà lo stesso peso a un worker con 100 campioni e uno con 100.000, producendo un modello sbilanciato verso le distribuzioni rappresentate dai worker con meno dati.

#### Convergenza di FedAvg in presenza di eterogeneità

Li et al. (2020) hanno analizzato la convergenza di FedAvg in setting non-i.i.d. e hanno dimostrato che, sotto ipotesi di *bounded gradient dissimilarity*, l'algoritmo converge a una *neighborhood* dell'ottimo globale, non all'ottimo esatto. La misura di eterogeneità è:

$$G^2 = \frac{1}{K} \sum_{k=1}^{K} \left\| \nabla \mathcal{L}_k(\theta^*) \right\|^2$$

dove $\theta^*$ è il minimizzatore globale. Quando i dati sono i.i.d., $\nabla \mathcal{L}_k(\theta^*)= 0$ per tutti $k$ e $G^2 = 0$: FedAvg converge esattamente all'ottimo. Con dati eterogenei, $G^2 > 0$: il gradiente locale di ogni worker non si annulla all'ottimo globale — ogni worker "vuole" continuare ad allontanarsi dall'ottimo globale nella direzione del proprio ottimo locale. L'errore di convergenza è proporzionale a $G^2 \cdot H$: più eterogeneità e più inner steps, maggiore il divario tra l'output di FedAvg e l'ottimo globale reale. Questo rende rigorous il trade-off H grande/piccolo discusso in Sezione 2.2: $H$ non è solo una leva sul traffico di rete ma anche un moltiplicatore dell'errore indotto dall'eterogeneità.

### 2.2 DiLoCo e Sparse Communication

DiLoCo [1] propone un paradigma di training distribuito in cui ogni partecipante esegue un numero elevato di step di ottimizzazione locale — denominati *inner steps* — prima di ogni sincronizzazione con gli altri nodi. Questo riduce la frequenza di comunicazione di un fattore $H$ rispetto al training distribuito sincrono standard, dove $H$ è il numero di inner steps configurato. Il principio alla base è che, per modelli con molti parametri, il costo computazionale di un singolo step di ottimizzazione è trascurabile rispetto al costo di trasmissione del modello; conviene quindi ammortizzare il costo di comunicazione su quanti più step locali possibile.

Con $H = 500$, ogni worker trasmette i propri pesi solo al termine di 500 batch di training. Supponendo batch da 32 campioni, ciò equivale a 16.000 esempi elaborati per ogni gossip push. L'impatto sulla qualità del modello aggregato è limitato perché gli inner steps locali producono aggiornamenti nella stessa direzione generale del gradiente globale, convergendo verso una soluzione compatibile con quella degli altri worker.

> **Inner steps vs epoche.** In letteratura FL alcuni paper (in particolare quelli basati su FedAvg) esprimono la computazione locale in *epoche* $E$ — cioè passaggi completi sul dataset locale. DiLoCo e questo progetto usano invece *gradient steps* $H$, che è un'unità più precisa e più controllabile. Con Worker 0 che ha ~209.700 campioni di training e `batch_size=32`, un'epoca corrisponde a circa 6.553 step; $H = 500$ equivale quindi a circa 0.08 epoche per round. La scelta degli step rispetto alle epoche non è arbitraria: con dataset non-i.i.d. le partizioni dei worker hanno dimensioni diverse (nel setup a 3 worker variano da ~210k a ~273k campioni), quindi un'epoca dura un tempo diverso per ogni worker. Esprimere $H$ in step garantisce che tutti i worker facciano esattamente la stessa quantità di computazione per round, indipendentemente dalla dimensione della loro partizione, mantenendo il traffico di rete prevedibile e uniforme.

DiLoCo introduce inoltre la tolleranza esplicita al drop asincrono dei messaggi: un aggiornamento mancante in un round non blocca il training del nodo mittente né quello del ricevente, che proseguono indipendentemente. Questo comportamento è intrinseco all'architettura gossip asincrona adottata: l'accumulatore di aggregazione è semplicemente a zero al termine del round se nessun vicino ha inviato aggiornamenti.

#### DiLoCo vs questo progetto: differenze algoritmiche chiave

La lettura del paper rivela che DiLoCo *non è equivalente a FedAvg con H grande*. La differenza fondamentale risiede nell'**ottimizzatore esterno** (outer optimizer). In DiLoCo, l'aggiornamento del modello condiviso tra worker non è una semplice media dei pesi locali, ma un processo in due fasi distinte:

1. **Outer gradient** — al termine degli $H$ inner steps, ogni worker calcola il proprio *delta* nello spazio dei pesi rispetto al punto di partenza del round: $\Delta_k^{(B)} = \theta^{(B-1)} - \theta_k^{(B)}$. La media di questi delta tra tutti i $K$ worker è l'*outer gradient*: $\Delta^{(B)} = \frac{1}{K}\sum_{k=1}^K \Delta_k^{(B)}$.

2. **Outer optimizer** — il modello condiviso viene aggiornato applicando l'outer gradient attraverso un ottimizzatore esterno: $\theta^{(B)} = \text{OuterOpt}(\theta^{(B-1)}, \Delta^{(B)})$. DiLoCo usa **Nesterov momentum** ($\eta_{\text{outer}} = 0.7$, $\beta_{\text{outer}} = 0.9$) come outer optimizer.

Il paper confronta esplicitamente diversi outer optimizer e conclude:

> *"We found that using as outer optimizer SGD (equivalent to FedAvg) or Adam performed poorly [...] We found Nesterov optimizer to perform the best."*

Usando SGD come outer optimizer con learning rate 1, DiLoCo si riduce esattamente a FedAvg: la media dei delta è equivalente alla media dei pesi finali quando tutti i worker partono dallo stesso punto. È precisamente questa la strategia adottata in questo progetto: **FedAvg con $H = 500$** è equivalente a DiLoCo con outer optimizer SGD ($\eta = 1$).

**Perché non possiamo implementare l'outer optimizer di DiLoCo.** La ragione non è solo una scelta di semplicità: l'outer optimizer di DiLoCo è **architetturalmente incompatibile con il requisito di sistema completamente decentralizzato** richiesto dalla traccia del progetto. L'aggiornamento del modello condiviso (Algorithm 1, linea 14) richiede per costruzione:
1. Che tutti i worker trasmettano i propri delta a un'entità centrale che calcoli la media globale;
2. Che quella stessa entità mantenga lo stato del momentum di Nesterov *tra round* — uno stato che deve essere unico e persistente;
3. Che il modello aggiornato venga redistribuito da quell'entità a tutti i worker per il round successivo.

Nel nostro sistema gossip P2P, nessun nodo vede tutti i contributi in un singolo round. L'aggregazione di ogni worker è parziale e asincrona: si integrano solo i modelli ricevuti casualmente via gossip, non l'intera rete. Non esiste nessun nodo che possa accumulare lo stato del momentum globale né redistribuire il risultato — sono esattamente le responsabilità che la traccia richiede di eliminare per ottenere un sistema P2P privo di aggregatore centrale.

FedAvg (media pesata dei pesi) è la variante di aggregazione che si adatta naturalmente al gossip asincrono: ogni worker può calcolare localmente la propria media con qualsiasi sottoinsieme di modelli ricevuti, senza dipendere da una visione globale del round. È per questo che tutti i sistemi FL P2P esistenti usano FedAvg o varianti equivalenti come aggregazione locale, e non le formulazioni con outer optimizer centralizzato.

**Differenza strutturale: P2P vs centralizzato.** Indipendentemente dall'outer optimizer, DiLoCo ha una struttura *centralizzata*: tutti i worker trasmettono i propri delta a un aggregatore centrale che applica l'outer optimizer e redistribuisce il modello aggiornato. Questo progetto ha invece una struttura **completamente decentralizzata**: ogni worker invia i propri pesi direttamente a un sottoinsieme casuale di peer (gossip k-push), e ogni worker aggrega *localmente* solo i modelli ricevuti via gossip. Non esiste nessun nodo centrale che veda tutti i contributi in un singolo round — la FedAvg di ciascun worker è parziale e asincrona.

**Risultati quantitativi di DiLoCo rilevanti per questo progetto:**

- *Ablation su H*: comunicare ogni H ∈ {50, 100, 250, **500**, 1000, 2000} step mostra che H=500 è il punto di rendimento marginale decrescente — il vantaggio di comunicare più frequentemente (H < 500) è marginale, mentre H=1000 aumenta la perplexity di solo ~2.9% rispetto a H=50. Questo **valida direttamente la scelta H=500** adottata in questo progetto.
- *Ablation su numero di worker*: più worker migliorano la generalizzazione con rendimento decrescente dopo 8 worker (perplexity: 1 worker → 16.23, 4 → 15.18, 8 → 15.02, 16 → 14.91, 64 → 14.96). L'impatto di aggiungere worker oltre 8 è quasi nullo, confermando il range 3–8 come sufficientemente rappresentativo per la campagna sperimentale di questo progetto.
- *i.i.d. vs non-i.i.d.*: DiLoCo mostra che il non-i.i.d. non degrada significativamente la performance finale — solo la velocità di convergenza nei round iniziali è più lenta. Questo è consistente con quanto osservato nei nostri run di sviluppo su FEMNIST.
- *Comunicazione ridotta di 500×*: DiLoCo su 8 worker ottiene prestazioni migliori del baseline sincrono con batch 8× più grande, comunicando 500× meno. Il vantaggio relativo è ancora più marcato nel contesto di questo progetto, dove la comunicazione avviene su rete TCP/IP reale tra EC2 distinte anziché su interconnessioni ad alta banda tra acceleratori.

#### Confronto qualitativo con i risultati di DiLoCo

Un confronto diretto sui numeri è precluso dalla diversità dei task (classificazione CNN su FEMNIST vs language modeling su C4), delle metriche (accuracy vs perplexity) e delle scale (1.7M vs 60–400M parametri). Il confronto significativo è invece *qualitativo*: le tendenze osservate nei nostri esperimenti sono coerenti con le previsioni teoriche di DiLoCo?

**Cosa ci svantaggia rispetto a DiLoCo:**

Il punto di svantaggio più rilevante è l'aggregazione. DiLoCo mostra che FedAvg (= SGD outer, il nostro metodo) "performed poorly" rispetto a Nesterov. Tuttavia questa conclusione è tratta su LLM da centinaia di milioni di parametri con un gradient landscape profondamente non-convesso: il momentum esterno è particolarmente utile quando i delta degli inner steps sono rumorosi e variabili, come accade su sequenze testuali di lunghezza 1024 token con un transformer. Su una CNN da 1.7M parametri su un task di classificazione d'immagini — molto più "regolare" dal punto di vista dell'ottimizzazione — la differenza tra FedAvg e Nesterov esterno è presumibilmente più contenuta. Non è possibile quantificarla senza implementare entrambe le varianti, ma la limitazione è documentata.

Il secondo svantaggio è il punto di partenza: DiLoCo usa sempre un modello pretrainato (24k step) come inizializzazione — tutti i risultati sono di fine-tuning. Il paper mostra che partire da zero degrada la perplexity finale di ~0.1 PPL, un impatto piccolo ma non nullo. Noi alleniamo sempre from scratch, che è il setting più difficile e teoricamente il più lontano da qualsiasi ottimo.

**Cosa ci avvantaggia rispetto a DiLoCo:**

Il contributo architetturale di questo progetto — la decentralizzazione completa via gossip P2P — è qualcosa che DiLoCo non affronta. DiLoCo è *ispirazione* per i meccanismi di sparse communication, ma rimane centralizzato nel suo aggregatore. Il nostro sistema tolera la perdita di worker, opera su rete TCP/IP reale con latenza variabile, e non ha nessun single point of failure per il training. Questi sono requisiti di sistemi distribuiti che DiLoCo non si pone.

**Cosa è veramente comparabile:**

La metrica più utile per il confronto non è l'accuracy assoluta ma il **guadagno relativo del gossip rispetto al training isolato** (Esperimento 1 vs Esperimento 2 del piano sperimentale). DiLoCo mostra che l'architettura sparse communication porta benefici significativi — migliore generalizzazione e meno comunicazione — rispetto al training isolato. Se i nostri esperimenti mostrano un delta positivo significativo tra no-FL e FL gossip, si valida la stessa intuizione su un dominio diverso e con vincoli di sistema più stringenti. Questo delta è la metrica principale da confrontare con le affermazioni qualitative di DiLoCo.

### 2.3 Dataset LEAF e FEMNIST

Il dataset FEMNIST, distribuito dal framework LEAF [3], è il benchmark standard per il Federated Learning non-i.i.d. Deriva da EMNIST ed è organizzato per autore: ogni utente ha uno stile di scrittura caratteristico, producendo una distribuzione dei dati naturalmente eterogenea tra i partecipanti — proprietà definita *non independent and identically distributed* (non-i.i.d.). Ogni campione è un'immagine in scala di grigi di dimensione $28 \times 28$ pixel, con 62 classi (cifre 0–9 e lettere a–z, A–Z).

#### Struttura degli oggetti di dominio

L'entità fondamentale del dataset è il **writer** (chiamato `user` nel formato LEAF) — una persona reale che ha scritto caratteri a mano. Ogni writer ha uno stile di scrittura proprio e ha prodotto un certo numero di immagini di caratteri diversi. Il dataset completo conta **3.597 writer** per un totale di **734.463 immagini**, con una media di circa 204 immagini per writer.

LEAF serializza il dataset in file JSON distribuiti in due cartelle: `train/` e `test/`. **Entrambe le cartelle contengono gli stessi writer**: lo split non divide le persone, ma i campioni di ogni persona — il 90% dei campioni di ogni writer va in `train/`, il 10% in `test/`. Ogni file contiene fino a 100 writer e ha la seguente struttura:

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

**Non esistono cartelle per scrittore.** L'output di `download_femnist.py` è semplicemente `data/femnist/data/train/` e `data/femnist/data/test/` — due cartelle piatte con file JSON. Non c'è una sottocartella per `f1967_21` o per nessun altro scrittore. La suddivisione per scrittore non sparisce: è preservata *dentro* i JSON nella chiave `user_data`, dove ogni writer_id mantiene le proprie immagini separate. È questa struttura che `split_dataset.py` legge per distribuire scrittori ai worker: estrae la lista `users`, prende una fetta contigua per ogni worker, e scrive solo i writer_id di quella fetta nella cartella del worker corrispondente.

#### Trasformazione degli oggetti attraverso la pipeline

I dati subiscono tre trasformazioni successive prima di essere usati dal modello:

**1. Lettura e fusione (`_read_json_shards`).**
Tutti i file JSON di una split (`train/` o `val/`) vengono letti e fusi in due strutture:
- `all_users`: lista flat di tutti i writer nell'ordine originale di LEAF (ordine deterministico).
- `user_data`: dizionario globale `{writer_id → {x, y}}`.

**2. Partizionamento per worker (`split_dataset.py`).**
La lista `all_users` viene divisa in $N$ slice contigue di dimensione $\lfloor |\mathcal{U}|/N \rfloor$, dove $\mathcal{U}$ è l'insieme dei writer e $N$ è `num_workers`. Con $N=3$ e 3.597 writer:

```
Worker 0 → writer    0–1198  (~1.199 writer, ~245.000 immagini)
Worker 1 → writer 1199–2397  (~1.199 writer, ~245.000 immagini)
Worker 2 → writer 2398–3596  (~1.199 writer, ~245.000 immagini)
```

Lo stesso partizionamento viene applicato **separatamente** sia a `train/` che a `test/` di LEAF: ogni worker riceve una slice contigua di writer da entrambe le cartelle. Il risultato sono due file per worker: `data/femnist/worker_{i}/train/data.json` e `data/femnist/worker_{i}/val/data.json`. La cartella sorgente `test/` di LEAF viene rinominata `val/` nelle cartelle worker per riflettere l'uso reale: non è un test set tenuto fuori dal training, ma il validation set usato per l'early stopping ad ogni round. Il rapporto 90/10 per campione dentro ogni writer — stabilito da LEAF — è preservato intatto.

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

LEAF fornisce uno split predeterminato configurabile tramite `--tf` (default 0.9 = 90% train, 10% validation). Lo split avviene **per campione dentro ogni scrittore**: entrambe le cartelle `train/` e `test/` contengono gli stessi writer, con campioni diversi. Questo è lo schema adottato da tutte le paper di riferimento sul benchmark FEMNIST — incluse FedAvg [2] e le varianti DiLoCo-inspired — ed è la scelta adottata in questo progetto.

**Nota sul naming**: LEAF chiama la seconda cartella `test/`, ma nel nostro sistema essa è usata come **validation set** — misurata ad ogni round dopo la Fase A per l'early stopping e le metriche di convergenza. Non è un test set tenuto fuori dal training. Per evitare ambiguità, `split_dataset.py` rinomina `test/` in `val/` nelle cartelle worker: il codice riflette l'uso reale. `dataset.py` carica `val/` nel `val_loader` e nel resto di questo documento si usa il termine *validation set* (o *validation loss*).

**Assenza di un test set separato.** In ML classico si distinguono tre set con ruoli distinti:

- **Training set**: il modello ci fa backpropagation sopra. I pesi vengono aggiornati direttamente su questi dati. Il modello li "vede" molte volte e rischia di adattarsi eccessivamente a essi (overfitting).
- **Validation set**: il modello *non* ci fa backpropagation, ma le decisioni di training sono prese in base alle sue performance — quando fermarsi (early stopping), quale configurazione di iperparametri scegliere, quale checkpoint salvare. Il modello non impara direttamente da questo set, ma viene scelto perché funziona bene su di esso: lo "vede" indirettamente.
- **Test set**: usato *una sola volta*, dopo che tutte le decisioni di training e selezione del modello sono state prese. Non influenza nessuna scelta. Serve a dare una stima onesta di quanto il modello generalizza su dati completamente nuovi. Se si usa lo stesso set sia per l'early stopping che per la valutazione finale, la metrica risultante è **ottimistica**: si sta misurando quanto bene il modello ha "imparato" quel set, non quanto generalizza su dati mai visti.

Il nostro sistema ha solo train e val, usando quest'ultimo per **due ruoli distinti**:
1. **Early stopping** — durante ogni run, la val loss decide quando interrompere il training.
2. **Selezione degli iperparametri** — tra le run, la val accuracy finale è la metrica usata per scegliere la configurazione migliore nella grid search (es. `learning_rate` ∈ {1e-4, 1e-3, 5e-3}, `inner_steps_H` ∈ {100, 500, 1000}, `gossip_fanout` ∈ {1, 2, N-1}).

Entrambi questi usi si basano sullo **stesso identico `val/`** — non esiste una suddivisione interna tra "val per early stopping" e "val per confronto". Questo genera un bias che si accumula su due livelli:

1. **Livello checkpoint**: l'early stopping scatta nel round in cui `val/` è al picco — il modello salvato è già quello ottimizzato per quel set specifico.
2. **Livello configurazione**: tra le run, si sceglie la configurazione con il val_accuracy più alto — che è già il picco scelto al livello precedente.

L'ordine concreto è:

```
Run A (lr=1e-3):  training → early stopping su val/ → val_accuracy finale = 0.73
Run B (lr=1e-4):  training → early stopping su val/ → val_accuracy finale = 0.68
Confronto: A vince → si sceglie lr=1e-3
```

Con un test set separato il flusso sarebbe invece:

```
val/  → usato solo per early stopping (decide quando fermarsi)
test/ → misurato una sola volta alla fine di ogni run, per confrontare le configurazioni
```

I due set sarebbero indipendenti: la metrica di confronto non sarebbe influenzata dalle decisioni di stopping. Questo introduce un **bias ottimistico** nelle metriche assolute riportate: i valori finali sono leggermente gonfiati rispetto a quelli che si otterrebbero su un test set indipendente.

**Confronto tra i due approcci possibili:**

| | Train + Val (adottato) | Train + Val + Test |
|---|---|---|
| **Metrica finale** | Ottimistica — val usato per early stopping e per scegliere la configurazione | Onesta — test mai visto in nessuna decisione |
| **Dati per il training** | Maggiori (es. 90% train, 10% val) | Minori (es. 80% train, 10% val, 10% test) |
| **Validità per confronti relativi** | Sì — il bias è sistematico e si cancella nei confronti tra configurazioni | Sì, con stima assoluta più affidabile |
| **Standard nella letteratura FL** | Sì — LEAF e FedAvg usano questo schema | No — non adottato nelle paper di riferimento |

Il bias ottimistico è un problema se si vuole affermare "il modello raggiunge X% di accuracy assoluta". Non è un problema per l'obiettivo di questo progetto, che è confrontare configurazioni relative (FL vs no-FL, diversi `gossip_fanout`, diversi `H`): il bias è identico per tutte le configurazioni e si annulla nel confronto.

L'assenza di un test set separato è quindi accettabile per tre ragioni concrete:

1. **Dimensione dei set**: il 10% dei campioni di ogni scrittore, già diviso tra i worker, è una partizione piccola. Suddividerla ulteriormente in val + test produrrebbe set troppo ridotti per stime statisticamente affidabili.
2. **Bias trascurabile rispetto alla varianza FL**: la fonte di rumore dominante nelle metriche FL è la varianza inter-worker dovuta ai dati non-i.i.d., non il bias da early stopping. Il bias sistematico si cancella nei confronti tra configurazioni, che è l'obiettivo degli esperimenti.
3. **Confrontabilità con la letteratura**: usare lo stesso schema di split permette di confrontare i risultati direttamente con i valori riportati nelle paper di riferimento.

**Perché non usare la k-fold cross-validation.** La k-fold divide i dati in $k$ fold e ripete il training $k$ volte, usando ogni volta un fold diverso come validation. Rispetto allo split fisso, offre una stima più robusta perché ogni campione appare sia in training che in validation, ma presenta tre problemi nel contesto FL su FEMNIST:

1. **Costo computazionale**: $k$ training completi per ogni worker — $k \times$ il tempo attuale.
2. **Violazione della proprietà non-i.i.d.**: una k-fold standard rimescola i campioni tra fold, potendo distribuire i campioni di un writer in fold diversi. Ogni fold perderebbe così la struttura "per scrittore" che è il punto centrale del benchmark. Una k-fold stratificata per writer — che mantiene ogni writer intero in un solo fold — sarebbe tecnicamente corretta ma richiederebbe di riscrivere `load_partition` con un parametro `fold_index` e modificare il training loop.
3. **Non standard in FL**: nessuna paper FL su FEMNIST usa k-fold — adottarla impedirebbe di confrontare i risultati con la letteratura.

Il costo computazionale sarebbe $k \times$ quello attuale — ingiustificato per un sistema già distribuito il cui obiettivo è validare la convergenza, non ottimizzare iperparametri con la massima precisione statistica.

---

### 2.4 Eterogeneità dei Dati: Conseguenze e Mitigazione

Il non-i.i.d. non è una caratteristica innocua del dataset: è la principale fonte di difficoltà sia nell'apprendimento locale di ogni worker sia nella qualità dell'aggregazione federata. Questa sezione tratta sistematicamente le conseguenze dell'eterogeneità dei dati e le strategie — teoriche ed implementate — per mitigarle.

#### 2.4.1 Conseguenze in ML Centralizzato (singolo modello)

In un contesto di ML classico senza federazione, addestrare un modello su dati non-i.i.d. significa addestrarlo su una distribuzione *biased*: i campioni provengono da una distribuzione $P_{\text{local}}$ che differisce dalla distribuzione target $P_{\text{global}}$. Le conseguenze principali sono:

**Covariate shift e domain shift.** Le feature di input hanno distribuzione diversa tra training e test set. Un modello addestrato solo sugli scrittori del Worker 0 apprende rappresentazioni ottimizzate per quegli stili di scrittura specifici. Applicato a scrittori del Worker 1 — con un'altra calligrafia — le sue feature map attivano pattern diversi da quelli attesi, degradando la predizione. Questo è un caso specifico di *covariate shift*: $P(\mathbf{x})$ cambia tra source e target, anche se $P(y|\mathbf{x})$ resta simile (la `a` è sempre `a`, ma visivamente diversa tra scrittori).

**Overfitting sulla distribuzione locale.** Con $H$ inner steps su dati non-i.i.d., il modello si specializza progressivamente sulla propria partizione. Nei dataset FEMNIST per scrittori, questo significa che la loss locale scende regolarmente, ma la capacità di generalizzare su scrittori mai visti peggiora — esattamente il trade-off tra bias (per la distribuzione locale) e varianza (su quella globale).

**Classi squilibrate.** Non tutti i writer producono tutti i caratteri con la stessa frequenza. Un worker i cui scrittori hanno scritto raramente cifre avrà rappresentazioni deboli per le classi 0–9. Senza accesso a dati di altri worker, il modello non può colmare questa lacuna.

#### 2.4.2 Conseguenze Specifiche del Federated Learning

Il FL introduce problemi aggiuntivi che non esistono nel caso centralizzato, perché l'aggregazione unisce modelli addestrati su distribuzioni eterogenee:

**1. Client drift.** È il problema centrale del FL non-i.i.d. Durante gli $H$ inner steps, ogni worker ottimizza nella direzione del gradiente locale $\nabla \mathcal{L}_k(\theta)$, che punta verso l'ottimo della propria distribuzione locale. Con più worker, questi gradienti locali divergono tra loro e si allontanano tutti dal gradiente globale $\nabla \mathcal{L}(\theta) = \frac{1}{K}\sum_k \nabla \mathcal{L}_k(\theta)$. Dopo $H$ step, ogni modello si trova in una regione diversa dello spazio dei pesi — la media FedAvg produce un modello che non è ottimo per nessuna delle partizioni. Con H grande e dati molto eterogenei, il drift può essere così pronunciato che FedAvg produce un modello peggiore del training isolato.

**2. Degrado della qualità di FedAvg.** FedAvg calcola una media pesata dei pesi: $w_{\text{agg}} = \sum_k \frac{n_k}{n} w_k$. Questo è ottimale quando i $w_k$ si trovano nello stesso bacino di attrazione dello spazio di loss. Se invece i modelli hanno divergito verso bacini diversi (scenario comune con dati non-i.i.d. e H grande), la loro media cade in un punto di loss elevata per tutti — un "compromesso" che non funziona bene su nessuna partizione. In geometria dell'ottimizzazione, questo corrisponde a mediare punti su versanti opposti di una valle — il risultato è la cima della cresta, non la valle.

**3. Accuracy valley dopo la prima FedAvg.** È il fenomeno più visibile nei run di sviluppo su FEMNIST: l'accuracy crolla significativamente subito dopo la prima aggregazione (da ~75% a ~3% in un caso estremo). La causa è esattamente il punto 2: il modello locale aveva imparato feature ottimizzate per i propri scrittori; la media con un modello da scrittori completamente diversi produce un ibrido che non funziona bene su nessuna delle due partizioni. L'accuracy recupera nei round successivi man mano che il training locale "riadatta" il modello aggregato alla distribuzione locale — ma questo richiede diversi round, durante i quali il sistema appare regredire.

**4. Staleness dell'ottimizzatore dopo FedAvg.** AdamW accumula momenti di primo ordine ($m_t$, proporzionale alla media mobile dei gradienti) e di secondo ordine ($v_t$, proporzionale alla media mobile del quadrato dei gradienti). Questi momenti sono calibrati sulla traiettoria di ottimizzazione del modello locale. Dopo FedAvg, i pesi del modello cambiano significativamente, ma i momenti rimangono quelli del modello pre-aggregazione: il primo step post-aggregazione applica una direzione di aggiornamento calibrata su un punto dello spazio dei pesi completamente diverso da quello attuale. Questo contribuisce direttamente all'instabilità osservata nei round immediatamente dopo l'aggregazione e amplifica la severity dell'accuracy valley.

**5. Disallineamento delle statistiche BatchNorm.** I parametri appresi di BatchNorm ($\gamma$, $\beta$) vengono aggregati via FedAvg e riflettono una media tra worker. Le running statistics ($\mu_{\text{run}}$, $\sigma^2_{\text{run}}$) invece non vengono aggregate e rimangono quelle del training locale. Nei primi step post-aggregazione, i parametri di scaling/shift ($\gamma$, $\beta$) sono calibrati su una distribuzione diversa da quella rappresentata dalle running stats — producendo normalizzazione errata fino a quando le running stats convergono al nuovo regime. L'impatto è solitamente limitato (pochi batch), ma amplifica l'instabilità del round post-aggregazione.

**6. Asimmetria delle velocità e accumulo multi-round nel buffer.** In un sistema gossip asincrono, i worker più veloci (partizioni più piccole, meno step per epoch) completano più round prima che i worker lenti abbiano finito il loro. Il buffer di aggregazione del worker lento accumula più messaggi dal worker veloce che da quello lento — non per migliore qualità del modello, ma per pura differenza di velocità. Questo genera un'asimmetria di contributo: Worker 0 (209k campioni) è strutturalmente sovra-rappresentato nell'aggregazione di Worker 1 (272k campioni) perché può inviare 2–3 push mentre Worker 1 completa un singolo round.

**7. Early stopping prematuro per effetto FedAvg.** Il contatore di patience dell'early stopping misura round consecutivi senza miglioramento della val loss *locale*. Poiché FedAvg può peggiorare temporaneamente la val loss locale (punti 2 e 4 sopra), il contatore può avanzare anche quando il sistema FL sta convergendo globalmente — non per overfitting, ma per il rimescolamento dei pesi dovuto all'aggregazione. Con patience=5, un worker potrebbe fermarsi proprio durante la fase di recovery post-aggregazione, producendo un risultato peggiore di quello che si otterrebbe con patience più alta.

#### 2.4.3 Strategie di Mitigazione

La letteratura FL ha sviluppato diverse strategie per affrontare questi problemi. Le distinguiamo in categorie per chiarezza:

**Mitigazione del client drift:**

- **FedProx** (Li et al., 2020): aggiunge un termine prossimale alla loss locale che penalizza la distanza dal modello globale: $\mathcal{L}_k^{\text{prox}}(\theta) = \mathcal{L}_k(\theta) + \frac{\mu}{2}\|\theta - \theta^{(B-1)}\|^2$. Il termine $\mu > 0$ limita quanto il modello locale può allontanarsi dal punto di partenza, controllando il drift. Il parametro $\mu$ bilancia aderenza al gradiente locale (basso $\mu$) e contenimento del drift (alto $\mu$).
- **SCAFFOLD** (Karimireddy et al., 2020): introduce *control variates* — termini correttivi per ogni worker che stimano la differenza tra il gradiente locale e quello globale. Ogni worker mantiene un vettore $c_k$ che corregge il gradiente locale: $g_k^{\text{corr}} = \nabla \mathcal{L}_k(\theta) - c_k + c$, dove $c$ è la media globale dei control variates. SCAFFOLD elimina teoricamente il client drift in condizioni i.i.d. e lo riduce significativamente in condizioni non-i.i.d.
- **FedNova** (Wang et al., 2020): normalizza gli aggiornamenti locali prima dell'aggregazione per tener conto del numero effettivo di step compiuti da ciascun worker. Questo risolve l'asimmetria generata da worker con un numero diverso di campioni locali (e quindi un numero diverso di step per epoch).
- **H ridotto**: la soluzione più diretta — comunicare più frequentemente riduce il numero di step durante i quali i modelli possono divergere. Con H=100 invece di H=500, il drift è 5× inferiore per round. Il costo è proporzionalmente maggiore in termini di traffico di rete.

**Mitigazione dell'instabilità post-aggregazione:**

- **Reset dell'ottimizzatore dopo FedAvg**: azzerare i momenti $m_t$ e $v_t$ di AdamW al momento dell'aggregazione elimina la staleness del punto 4. Il costo è che i primi step del round successivo ripartono senza il beneficio del momentum accumulato — essenzialmente un warm-up implicito. Non implementato in questo progetto per non alterare la comparabilità degli esperimenti.
- **Learning rate warm-up post-aggregazione**: applicare un piccolo learning rate nei primi $W$ step dopo ogni FedAvg, poi tornare al lr nominale. DiLoCo stesso osserva spike di perplexity dopo ogni outer step e li attribuisce a questo effetto — il warm-up mitiga i picchi.

**Mitigazione dei problemi BatchNorm:**

- **FedBN** (Li et al., 2021): non aggrega i parametri BatchNorm ($\gamma$, $\beta$, running_mean, running_var) — ogni worker li mantiene localmente. Questo elimina il disallineamento del punto 5 a costo di una leggera perdita di potere aggregante per i layer di normalizzazione. Nel nostro sistema le running statistics non vengono già aggregate (solo i parametri appresi $\gamma$ e $\beta$ entrano nella FedAvg) — questa è una forma parziale di FedBN.
- **GroupNorm / LayerNorm**: alternative a BatchNorm che non usano running statistics e si comportano identicamente in training e inference. GroupNorm è comunemente usato in letteratura FL come sostituto diretto di BatchNorm.

**Strategie implementate in questo progetto:**

| Strategia | Implementata | Dove | Effetto sul non-i.i.d. |
|---|:---:|---|---|
| Gradient clipping (max_norm=1.0) | ✅ | `trainer.py` | Limita la norma del gradiente locale → riduce il drift per step |
| Running stats BatchNorm non aggregate | ✅ | `grpc_server.py` — solo float aggregati | Mitigazione parziale FedBN |
| Label smoothing (ε=0.1) | ✅ | `trainer.py` | Riduce l'over-confidence su distribuzioni locali sbilanciate |
| Staleness check (max_staleness=10) | ✅ | `grpc_server.py` | Scarta modelli troppo vecchi che amplificano il drift |
| FedProx | ❌ | — | Direzione di miglioramento futura |
| SCAFFOLD | ❌ | — | Incompatibile con gossip asincrono (richiede control variates globali) |
| Reset optimizer post-FedAvg | ❌ | — | Scelta deliberata: non alterare la comparabilità degli esperimenti |
| H variabile (grid search) | ✅ (in piano) | Esperimento 3b | Esplora il trade-off drift vs comunicazione |

> **Nota su SCAFFOLD in un sistema P2P.** SCAFFOLD richiede che i control variates siano sincronizzati tra tutti i worker ad ogni round di aggregazione — un'operazione intrinsecamente centralizzata. In un sistema gossip asincrono dove ogni worker aggrega solo i modelli che riceve casualmente, non è possibile mantenere control variates globali coerenti. SCAFFOLD è quindi architetturalmente incompatibile con il nostro design, per la stessa ragione per cui l'outer optimizer di DiLoCo non può essere implementato.

---

### 2.5 Fondamenti Teorici: Gossip FL come Discesa del Gradiente Decentralizzata

Questa sezione colloca il sistema implementato all'interno della teoria del *Decentralized Stochastic Gradient Descent* (DSGD) e spiega perché il gossip P2P converge, a quali condizioni, e come la scelta dei parametri di sistema si riflette sulle garanzie teoriche.

#### 2.5.1 DSGD e matrice di mixing

Nell'ottimizzazione decentralizzata classica (Lian et al., 2017; Koloskova et al., 2019), ogni nodo $k$ mantiene una propria copia dei parametri $\theta_k$ e aggiorna periodicamente il suo stato mescolando con i vicini tramite una *matrice di mixing* $W \in \mathbb{R}^{N \times N}$:

$$\theta_k^{(t+1)} = \sum_{j=1}^{N} W_{kj} \cdot \theta_j^{(t)} - \eta \nabla \mathcal{L}_k(\theta_k^{(t)})$$

dove $W_{kj} > 0$ se $j$ è un vicino di $k$ (o $j = k$) e $W_{kj} = 0$ altrimenti. Perché la media decentralizzata converga alla media globale, $W$ deve essere *doubly stochastic* ($\mathbf{1}^T W = \mathbf{1}^T$ e $W \mathbf{1} = \mathbf{1}$) e connessa. Il gossip k-push produce implicitamente una matrice di mixing stocastica in senso spettrale: in attesa, ogni worker riceve aggiornamenti da tutti gli altri con probabilità positiva, garantendo la connettività del grafo di comunicazione.

La velocità di convergenza del mixing è governata dal **gap spettrale** $\gamma = 1 - \lambda_2(W)$, dove $\lambda_2(W)$ è il secondo autovalore più grande della matrice (in valore assoluto). Un gap spettrale grande (vicino a 1) significa che il mixing è rapido — le informazioni si propagano in pochi round. Un gap piccolo (vicino a 0) significa che il mixing è lento — servono molti round per che ogni nodo abbia "visto" indirettamente i contributi di tutti gli altri.

Con gossip k-push su $N$ nodi, il gap spettrale atteso cresce con $k$: più peer contattati per round → matrice più densa → gap più grande → mixing più rapido. Questo è il fondamento teorico del parametro `gossip_fanout`: non è solo una leva empirica sul traffico, ma determina la velocità di mixing del sistema e quindi la velocità di convergenza teorica dell'algoritmo.

Con $N=3$ e `gossip_fanout=1`, il grafo di comunicazione è uno sparse random graph con 1 arco uscente per nodo per round. Il *mixing time* atteso — il numero di round perché la distribuzione dell'informazione sia $\epsilon$-vicina all'uniforme — è $O(\log N / k) = O(\log 3 / 1) \approx 1.6$ round: una notizia si propaga all'intera rete in meno di 2 round. Questa è la ragione per cui il sistema può funzionare anche con fanout=1 su reti piccole: la velocità di mixing è già molto alta per $N$ piccolo.

Con $N=8$ e `gossip_fanout=1`, il mixing time cresce a $O(\log 8) = 3$ round. Con `gossip_fanout=3`, si riduce a 1 round. Il vantaggio del fanout alto diventa più marcato al crescere di $N$ — confermando che gli esperimenti di scalabilità (Esperimento 4) sono quelli dove l'effetto di `gossip_fanout` è più interessante da studiare.

#### 2.5.2 Linear Mode Connectivity e perché FedAvg funziona

FedAvg calcola una media lineare nello spazio dei pesi. Per un'interpolazione lineare tra due modelli $\theta_A$ e $\theta_B$ abbia senso, è necessario che il *segmento* $\{(1-\alpha)\theta_A + \alpha\theta_B, \alpha \in [0,1]\}$ nello spazio dei pesi non attraversi regioni di loss elevata — ovvero che i due modelli si trovino nello stesso *bacino* di attrazione della loss landscape.

Frankle et al. (2020) hanno osservato empiricamente che modelli addestrati con la stessa architettura su dati diversi tendono a convergere verso punti connessi linearmente nella loss landscape, con barriere di loss trascurabili lungo il segmento che li unisce. Questo fenomeno — chiamato *linear mode connectivity* — è la ragione profonda per cui FedAvg funziona: la media dei pesi di modelli convergenti è anch'essa un buon modello.

Tuttavia la connettività si indebolisce sotto forte eterogeneità: modelli addestrati su distribuzioni molto diverse possono convergere verso bacini distanti, separati da una "cresta" di loss elevata. La media dei due modelli cade sulla cresta — non nel bacino di nessuno dei due. Questo spiega perché:

1. **L'accuracy valley post-FedAvg** è tanto più profonda quanto più eterogenee sono le partizioni. I worker con distribuzioni più distanti producono modelli in bacini più lontani; la loro media è più lontana da entrambi.
2. **H grande amplifica il problema**: con più inner steps, ogni modello si allontana di più dal punto di partenza comune, aumentando la distanza tra i bacini finali. Convergenza verso bacini distanti → media nella cresta.
3. **Il warm-up post-aggregazione funziona**: partendo dalla cresta (alta loss), i primi inner steps scendono rapidamente verso il bacino più vicino — che è quello corrispondente alla distribuzione locale. È questo "scivolamento" verso il bacino locale che produce il recupero dell'accuracy nei round successivi alla FedAvg.

> **Corollario pratico.** La dimensione e la profondità dell'accuracy valley è un indicatore della distanza tra i bacini dei modelli aggregati — e quindi dell'eterogeneità effettiva delle distribuzioni locali. Un sistema con dati i.i.d. non mostrerebbe questo fenomeno (i bacini coincidono); un sistema con dati molto eterogenei mostra un calo profondo e un recupero lento. FEMNIST è un caso intermedio: la variabilità tra scrittori è reale ma contenuta — tutti scrivono le stesse 62 classi, con variazioni di stile ma non di semantica.

#### 2.5.3 Consenso decentralizzato e convergenza al modello globale

In DSGD, sotto le ipotesi standard (smoothness della loss, varianza bounded dei gradienti, mixing sufficientemente rapido), è possibile dimostrare che tutti i nodi convergono *allo stesso punto*:

$$\frac{1}{K}\sum_k \|\theta_k^{(T)} - \theta^*\|^2 \xrightarrow{T \to \infty} 0$$

dove $\theta^*$ è il minimizzatore della loss globale $\mathcal{L}$. La velocità di convergenza dipende dal gap spettrale $\gamma$ del gossip graph: maggiore è $\gamma$, più rapida è la convergenza. Nel regime non-i.i.d., il punto limite non è l'ottimo esatto di $\mathcal{L}$ ma una sua neighborhood, con raggio proporzionale a $G^2/\gamma$ (eterogeneità dei dati divisa per il gap spettrale del grafo).

Questo mette in relazione diretta i due parametri principali del sistema:
- `inner_steps_H` → controlla $G^2$: H più grande = modelli che divergono di più = maggiore "gradiente dissimilarity" al momento dell'aggregazione
- `gossip_fanout` → controlla $\gamma$: fanout più grande = gap spettrale più alto = convergenza più rapida e neighborhood più piccola

L'obiettivo degli esperimenti (Sezione 7) è empiricamente verificare queste relazioni e trovare il punto operativo ottimale nel piano (H, fanout) per questo specifico task e dataset.

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

> **Il Registry come unico punto di centralizzazione.** Il Discovery Server è l'unico componente del sistema con un ruolo centralizzato, ed è una centralizzazione intenzionale e limitata al piano di rete — non al piano del learning. L'analogia corretta è il DNS: un server DNS è un single point of failure per la risoluzione dei nomi, ma non per il traffico applicativo. Allo stesso modo, se il Registry diventa irraggiungibile durante il training, i worker continuano a comunicare tra loro usando la lista peer memorizzata localmente dall'ultima chiamata a `/get_peers` — il training non si interrompe. La claim di decentralizzazione del sistema si riferisce al protocollo di apprendimento: nessun nodo vede i gradienti o i pesi degli altri se non tramite gossip diretto, nessun aggregatore centrale produce il modello finale. Questa proprietà è indipendente dall'esistenza del Registry.

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

La comunicazione inter-worker è definita dal file `proto/gossip.proto`:

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
- **Stub autogenerati**: `grpc_tools.protoc` produce codice client/server Python da `proto/gossip.proto`, eliminando la necessità di scrivere manualmente il codice di serializzazione e routing.
- **Timeout per chiamata**: ogni `stub.ReceiveModel(message, timeout=T)` solleva `grpc.RpcError` se il server non risponde entro `T` secondi, senza bisogno di gestione manuale di socket timeout.
- **Evoluzione del protocollo**: Protobuf supporta l'aggiunta di nuovi campi con retro-compatibilità garantita; aggiungere metadati al messaggio (es. versione del modello, loss locale) richiede solo una modifica al `.proto`.

#### Generazione degli stub a build time

I file `gossip_pb2.py` e `gossip_pb2_grpc.py` sono generati dal compilatore `protoc` nel `docker/Dockerfile.worker`, **prima** del `COPY` del sorgente applicativo:

```dockerfile
COPY proto/gossip.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gossip.proto
COPY config.yaml main_worker.py ./
COPY core/ ./core/
COPY network/ ./network/
```

Questo ordine è critico: i file generati si trovano nella directory di lavoro del container prima che il sorgente venga copiato sopra. Poiché `.gitignore` esclude i file `pb2` dal repository, la `COPY` successiva non li sovrascrive. Il vantaggio aggiuntivo è il **riutilizzo del layer Docker**: il layer contenente la compilazione Protobuf viene invalidato solo se `proto/gossip.proto` cambia, rendendo le rebuild successive molto più veloci.

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
2. **Patch `data_to_json.py` — compatibilità Pillow ≥ 10.0** *(modifica a codice di terze parti — vedi nota sotto)*  
   `Image.ANTIALIAS` è stato rimosso in Pillow 10.0 (ottobre 2023); `Image.LANCZOS` è il nome ufficiale dello stesso filtro Lanczos dal 2013. La patch sostituisce l'unica occorrenza in `leaf/data/femnist/preprocess/data_to_json.py`. Output pixel: identico.
3. **Patch `get_data.sh` — sostituzione `unzip`** *(modifica a codice di terze parti — vedi nota sotto)*  
   `get_data.sh` scarica prima entrambi i file (`by_class.zip` ~984 MB, `by_write.zip` ~542 MB) ed esegue poi `unzip <file>` per estrarli (senza flag `-q`: l'estrazione è silenziosa). `unzip` non è preinstallato di default in molte distribuzioni Linux e in ambienti WSL. La patch sostituisce ogni occorrenza di `unzip <file>` con `python3 -c "import zipfile; zipfile.ZipFile('<file>').extractall('.')"`, che usa la libreria standard Python — sempre disponibile — e produce output identico. L'estrazione avviene in silenzio (nessun log di avanzamento): è normale che lo script rimanga fermo per 5–10 minuti durante questo passo.
4. Installazione delle dipendenze di preprocessing di LEAF (`tensorflow-cpu`, `Pillow`, `numpy`) nell'ambiente Python corrente.
5. Esecuzione di `preprocess.sh` con split non-i.i.d. per scrittore, 90% train / 10% test.
6. Copia **selettiva** di sole `train/` e `test/` in `data/femnist/data/`. Le directory intermedie prodotte da LEAF (immagini raw EMNIST, file `.pkl`, dati campionati) non vengono copiate: occuperebbero gigabyte inutili poiché non servono al training. Il dataset finale pesa ~2–4 GB.
7. Rimozione automatica dell'intera directory `leaf/` (~20 GB). Una volta che `data/femnist/data/` esiste, il repository LEAF non serve più — se necessario verrà riclonato automaticamente da GitHub alla prossima esecuzione dello script.

> **Nota sulle modifiche a codice LEAF di terze parti.**  
> LEAF (Caldas et al., 2018) è un repository accademico non più attivamente mantenuto per la compatibilità con Python e librerie di sistema moderne. Le due patch sopra non alterano l'algoritmo di preprocessing né la struttura dei dati prodotti — modificano esclusivamente chiamate di sistema o di libreria diventate obsolete o non portabili. Le patch vengono applicate programmaticamente da `download_femnist.py` alla copia locale clonata, e scompaiono insieme all'intera directory `leaf/` al passo 7: non è necessario mantenere un fork. Ad ogni nuova esecuzione di `download_femnist.py`, LEAF viene riclonato e riprotato da zero.

**`scripts/split_dataset.py`** — partiziona `data/femnist/data/` in slice per-worker. Il comportamento dipende da `use_test_set` in `config.yaml`: con `false` (default) scrive `data/femnist/worker_{i}/{train,val}/data.json` rinominando la `test/` di LEAF in `val/`; con `true` scrive anche `data/femnist/worker_{i}/test/data.json` dividendo il 20% di LEAF al 50/50 per scrittore (10% val + 10% test). Lo script adotta una strategia a **due passate con scrittura immediata su disco** per mantenere il consumo di RAM costante indipendentemente dalla dimensione del dataset. Il dataset completo occupa ~4 GB su disco ma si espanderebbe a 40–80 GB come oggetti Python se caricato interamente in memoria — dimensione insostenibile su un portatile.

- **Passata 1 (solo ID):** legge esclusivamente il campo `users` di ogni shard JSON, senza caricare i pixel. Produce la lista globale ordinata di tutti i writer, calcola la mappa `writer_id → worker_index` e raggruppa gli ID per worker. Consumo RAM: trascurabile (solo stringhe).
- **Passata 2 (streaming con scrittura immediata):** apre tutti i file di output dei worker simultaneamente; legge un shard alla volta; per ogni writer nel shard, scrive l'entry `user_id: {x, y}` direttamente nel file del worker corretto in quel momento, senza accumularla in memoria. Alla fine del shard, esegue `del shard` + `gc.collect()` per liberare subito la RAM prima del shard successivo. Il picco di RAM è **un singolo shard** (~1–2 GB come oggetti Python) indipendentemente dal numero di worker o dalla dimensione totale del dataset.

Lo script rimuove automaticamente le directory `worker_*` esistenti all'avvio, rendendo ogni esecuzione idempotente: se interrotto a metà, basta rilanciarla da capo senza rischio di dati inconsistenti. Il sorgente `data/femnist/data/` non viene mai modificato.

La motivazione di eseguire entrambi gli step su host anziché dentro i container è fondamentale per la correttezza dello scenario federato: ogni container riceve in mount **esclusivamente la propria porzione di dati**, senza possibilità di accedere a quelli degli altri worker. Questo rispecchia fedelmente la realtà del Federated Learning, dove ogni dispositivo ha accesso fisico solo ai propri dati locali — non è necessario alcun meccanismo software per isolare le partizioni, è l'architettura stessa del filesystem a garantirlo.

#### Strategia di partizione statica pre-deployment

Il dataset FEMNIST viene partizionato in modo **deterministico e statico** prima dell'avvio dei container, dallo script `scripts/split_dataset.py`. Lo script legge i file JSON prodotti da LEAF, estrae la lista globale degli utenti ordinata, e assegna a ciascun worker uno slice contiguo:

$$\text{start}_k = k \cdot \left\lfloor \frac{|\mathcal{U}|}{N} \right\rfloor, \quad \text{end}_k = \begin{cases} \text{start}_k + \lfloor |\mathcal{U}|/N \rfloor & \text{se } k < N-1 \\ |\mathcal{U}| & \text{se } k = N-1 \end{cases}$$

dove $\mathcal{U}$ è l'insieme totale degli utenti e $N$ è `num_workers` in `config.yaml`. La partizione del worker $k$ viene scritta su host in `data/femnist/worker_k/{train,val}/data.json` e montata nel suo container tramite bind mount Docker:

```
./data/femnist/worker_k  →  /app/data/femnist  (dentro il container k)
```

Ogni container ha accesso **esclusivo e isolato** alla propria partizione: il filesystem del container non contiene alcun dato appartenente ad altri worker. Questo rispecchia fedelmente uno scenario federato reale, dove ogni dispositivo ha accesso solo ai propri dati locali.

#### Garanzia della proprietà non-i.i.d.

La partizione è non-i.i.d. per costruzione: LEAF organizza i dati per autore, ciascuno con uno stile di scrittura caratteristico. Assegnare utenti contigui a un worker garantisce che la sua distribuzione di classi rifletta gli stili di un sottoinsieme specifico di scrittori — diverso da quello di ogni altro worker. Questo simula fedelmente lo scenario FL reale in cui i dispositivi partecipanti hanno dati generati da utenti diversi con abitudini proprie.

**Perché le partizioni hanno dimensioni diverse.** Lo split è per *scrittore*, non per *campione*: ogni worker riceve circa $|\mathcal{U}|/N$ scrittori, ma ogni scrittore ha un numero diverso di immagini (alcuni hanno scritto 200 caratteri, altri 400 o più). Di conseguenza, anche con lo stesso numero di scrittori assegnati, il totale dei campioni varia tra worker. Con 3 worker sul dataset completo, a titolo indicativo:

```
Worker 0 → ~1166 scrittori → ~210k campioni
Worker 1 → ~1166 scrittori → ~273k campioni
Worker 2 → ~1165 scrittori → ~252k campioni
```

Questa asimmetria è intenzionale e realistica: in un deployment FL reale i dispositivi hanno quantità di dati eterogenee. Il meccanismo di FedAvg con ponderazione per `num_samples` compensa parzialmente questa differenza nel calcolo della media pesata dei modelli.

**Come si realizza il non-i.i.d.** Tutti i worker hanno tutte e 62 le classi — non è che Worker 0 abbia solo le lettere A–M e Worker 1 solo N–Z. Il non-i.i.d. emerge dagli *stili di scrittura*: le `a` di un gruppo di scrittori assegnati a Worker 0 hanno un aspetto diverso dalle `a` degli scrittori di Worker 1. Il modello di ogni worker impara feature visive specifiche del proprio gruppo di scrittori, rendendo i modelli eterogenei tra loro anche a parità di classi — esattamente la condizione che il gossip FL deve saper gestire.

#### Motivazione della scelta statica vs dinamica

Una partizione dinamica (che ribilancia i dati al join di nuovi worker) avrebbe garantito partizioni di dimensione uniforme anche in caso di variazioni del numero di nodi. Tuttavia, introdurrebbe una dipendenza globale: ogni ribilanciamento richiederebbe un coordinatore che conosce l'intera distribuzione degli utenti — contraddittorio con l'approccio puramente P2P adottato. La scelta statica mantiene il sistema autonomo: ogni worker carica semplicemente i file presenti nella propria directory montata, senza conoscere `WORKER_ID` o `TOTAL_WORKERS` a livello di dataset.

#### Caricamento nel worker

`core/dataset.py` espone la funzione `load_partition(data_dir, batch_size)` che legge semplicemente tutti i file JSON presenti in `data_dir/train/` e `data_dir/val/` — la stessa interfaccia di lettura indipendentemente da quanti worker esistano. Il splitting è già avvenuto su host; il container non sa nulla della topologia globale.

#### Immutabilità dei dati durante il training

I dati di train, val e test sono caricati **una sola volta** all'avvio del worker e rimangono invariati per tutta la run. Non esiste nessun meccanismo che ricarichi, rimescoli o sostituisca i campioni tra un round e l'altro.

- **Train**: stessi campioni per tutti i round. Il `DataLoader` ha `shuffle=True`, quindi l'ordine dei batch cambia ad ogni epoch, ma il pool di immagini è sempre quello della partizione assegnata a quel worker. L'iteratore infinito garantisce esattamente $H$ step per round indipendentemente dalla dimensione della partizione.
- **Val**: stessi campioni ogni round, stesso ordine (`shuffle=False`). La val loss al round $r$ è calcolata sulle stesse immagini del round 1 — l'unica variabile è il modello, che nel frattempo ha aggiornato i pesi.
- **Test** (se `use_test_set: true`): stessi campioni, valutati una sola volta alla fine del training.

Questa immutabilità è un requisito, non una limitazione. Se i dati di validation cambiassero tra round, le val loss di round diversi non sarebbero comparabili e l'early stopping — che decide di fermarsi confrontando la val loss attuale con quella dei round precedenti — non avrebbe senso. La stabilità del set di valutazione è ciò che rende il confronto inter-round significativo.

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

**Reset dell'accumulatore.** Dopo l'aggregazione, `weighted_sum` viene posto a `None` e `received_samples` a `0`. Il reset avviene con il lock acquisito, garantendo che nessun messaggio in arrivo (Thread 1) possa modificare l'accumulatore nel breve intervallo tra la lettura e il reset.

**Caso base: nessun vicino ha inviato.** Se `received_samples == 0`, la Fase A viene saltata e il worker procede direttamente alla Fase B con il proprio modello invariato. Questo è il comportamento corretto in caso di assenza di aggiornamenti (nessun vicino attivo, tutti i messaggi droppati o stantii): il training locale prosegue autonomamente.

**Early stopping post-aggregazione.** Immediatamente dopo l'aggregazione (o dopo il suo skip), il modello viene validato sul validation set locale. Se la validation loss non si riduce per `early_stopping_patience` round consecutivi, Thread 2 esce dal loop. Thread 1 rimane attivo: il processo non termina e il server gRPC continua a servire i peer che sono ancora in training. Questo comportamento è ottenuto chiamando `grpc_server.wait_for_termination()` dopo il break, che blocca il thread principale finché il server gRPC non viene fermato esternamente.

L'early stopping è **locale e indipendente** per ogni worker: non esiste coordinamento globale. Worker diversi possono convergere in round diversi, e quelli che convergono prima continuano a servire gli altri come destinatari passivi di gossip push.

#### Fase B — Training Locale (H Inner Steps)

**Meccanismo.** Il worker esegue esattamente `inner_steps_H` passi di ottimizzazione locale usando l'ottimizzatore AdamW con learning rate configurabile. Durante questa fase, Thread 1 continua ad accumulare i messaggi ricevuti nell'`AggregationBuffer`, ma Thread 2 non li legge: la sincronizzazione avviene solo all'inizio del round successivo (Fase A).

**Scelta dell'ottimizzatore: AdamW vs SGD.** AdamW è preferito a SGD per la sua robustezza ai learning rate: richiede meno tuning del learning rate rispetto a SGD con momentum, che è critico in un contesto distribuito dove non c'è un tutor centrale che aggiusta i parametri. AdamW introduce la weight decay direttamente sull'aggiornamento dei pesi (non sul gradiente come L2 regularization), il che tende a produrre modelli con generalizzazione migliore.

Nell'ambito del FL non-i.i.d., AdamW presenta un ulteriore vantaggio rispetto a SGD: i tassi di apprendimento adattativi *per parametro*. SGD applica lo stesso learning rate scalare a tutti i parametri in tutti i worker — ma in un setting non-i.i.d. il gradiente locale ha magnitude e direzione sistematicamente diverse tra worker, a causa delle distribuzioni eterogenee. Un learning rate che garantisce convergenza per Worker 0 può essere troppo alto per Worker 1 (causando oscillazioni) o troppo basso per Worker 2 (convergenza lenta). AdamW adatta implicitamente il learning rate effettivo di ogni parametro in base alla storia dei gradienti di quel worker: parametri con gradienti consistentemente grandi ricevono passi più piccoli; parametri con gradienti piccoli ricevono passi più grandi. Questo effetto di normalizzazione rende la traiettoria di ottimizzazione locale più stabile e più comparabile tra worker con distribuzioni diverse, migliorando la qualità dell'aggregazione FedAvg.

Un secondo vantaggio di AdamW in FL è la velocità di recovery post-aggregazione. Dopo FedAvg, i pesi cambiano bruscamente e i momenti ($m_t$, $v_t$) di AdamW si trovano disallineati con il nuovo punto dello spazio dei pesi. Tuttavia AdamW aggiorna i momenti ad ogni step: dopo pochi batch post-aggregazione, $m_t$ e $v_t$ convergono alle statistiche del nuovo regime, permettendo una discesa efficiente verso il bacino locale. SGD con momentum richiede più tempo per "dimenticare" il momentum del round precedente, amplificando l'instabilità post-aggregazione.

**Scelta di H=500.** Il valore $H=500$ è ispirato direttamente a DiLoCo [1] e rappresenta un trade-off tra qualità dell'aggregazione e costo di comunicazione. Con $H$ piccolo (es. 1), ogni aggiornamento è quasi un gradiente puro e l'aggregazione è equivalente al SGD distribuito sincrono — ottima qualità ma alta frequenza di comunicazione. Con $H$ grande (es. 10.000), ogni worker diverge significativamente dagli altri prima di sincronizzarsi — comunicazione rara ma aggregazione degradata. $H=500$ mantiene i worker sufficientemente allineati da rendere l'aggregazione FedAvg efficace, pur riducendo la frequenza di comunicazione di due ordini di grandezza rispetto al training sincrono.

#### Fase C — Gossip Push

**Meccanismo.** Prima dell'invio, `shared_state["current_round"]` viene aggiornato al valore del round corrente, rendendolo visibile a Thread 1 per i successivi controlli di staleness. Il worker interroga il Discovery Server tramite `GET /peers`, esclude il proprio indirizzo dalla lista, e seleziona casualmente `min(gossip_fanout, len(eligible_peers))` vicini. Per ciascun target viene applicata la logica di fault injection (Sezione 8), poi viene invocato `send_model()`.

**Frequenza di interrogazione del registry.** Il registry viene interrogato **una volta per round**, all'inizio della Fase C. Questa scelta è coerente con il requisito di minimizzare il traffico di rete: con H=500 inner steps un round dura tipicamente diversi minuti, quindi aggiornare la lista peer più spesso produrrebbe overhead HTTP senza benefici concreti sulla freschezza. Interrogare il registry meno spesso (es. ogni K round) ridurrebbe ulteriormente il traffico al costo di una visione più stale della topologia. Il trade-off è bilanciato al valore attuale: un'interrogazione per round mantiene la lista allineata con i cambiamenti topologici (worker che si registrano o deregistrano) senza generare traffico aggiuntivo apprezzabile rispetto ai push gRPC che domina il volume totale.

**Re-query reattivo dopo fallimento gRPC.** Un push fallito (`UNAVAILABLE` o `DEADLINE_EXCEEDED`) è un segnale che il peer potrebbe essere crashato e deregistrato tra la chiamata iniziale a `GET /peers` e il tentativo di push. Il meccanismo funziona in due fasi distinte.

*Fase normale:* il worker campiona `k` peer casuali, tenta l'invio a ciascuno, e accumula in `failed_targets` i peer che hanno restituito `RpcError`. Il set `tried` viene popolato con tutti i target originali fin dall'inizio, indipendentemente dall'esito.

*Re-query:* solo se `failed_targets` non è vuota, viene eseguita **una singola** chiamata aggiuntiva a `GET /peers`. Dalla lista fresca vengono esclusi tutti i peer già presenti in `tried` — sia quelli che hanno risposto che quelli falliti — per evitare di ritentare nodi già irraggiungibili. Dai rimanenti si campionano esattamente `min(len(failed_targets), len(replacements))` sostituti: uno per ogni fallimento, non di più.

```python
if failed_targets:
    fresh_peers = fetch_peers(registry_url)
    replacements = [p for p in fresh_peers if p != my_address and p not in tried]
    for replacement in random.sample(replacements, min(len(failed_targets), len(replacements))):
        tried.add(replacement)
        ...
        send_model(replacement, ...)
```

Esempio con `gossip_fanout=3`, peer disponibili A, B, C, D, E:

```
Target iniziali:  [A, B, C]       tried = {A, B, C}
A → successo
B → RpcError     failed_targets = [B]
C → successo

fresh_peers = [A, B, D, E]        ← B nel frattempo potrebbe essersi deregistrato
replacements = [D, E]              ← A e B esclusi da tried
campionati = [D]                   ← min(1 fallimento, 2 disponibili) = 1 sostituto

D → successo    sent_count += 1
```

Il costo totale è **al massimo una HTTP call extra per round**, emessa solo in presenza di fallimenti reali (non drop simulati, che sono intenzionali). Se anche i sostituti falliscono non c'è un secondo livello di re-query: il round prosegue con i push riusciti. Il log di fine Fase C riporta `failed=N, retried=M` per osservabilità diretta. Il meccanismo copre il caso più comune (crash con deregistrazione pulita via `finally` o signal handler); il caso di hard crash SIGKILL è documentato come known limitation in Sezione 8.4.

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

Il caso base (`received_samples == 0`) inizializza l'accumulatore con il primo contributo. La prova che questo accumulatore produce lo stesso risultato dell'approccio batch è diretta per linearità della somma: $\sum_i (w_i \cdot n_i) = $ running sum step-by-step.

#### Correttezza con accesso concorrente

Thread 1 può ricevere messaggi da più sender concorrentemente (il pool di thread interno a gRPC gestisce connessioni parallele). Ogni invocazione di `ReceiveModel` acquisisce il lock prima di modificare l'accumulatore, garantendo serializzazione degli aggiornamenti. L'overhead del lock è trascurabile rispetto al costo di deserializzazione dei pesi (operazione dominante).

### 4.4 Staleness Check (Unidirezionale)

#### Il problema: aggiornamenti stantii

In un sistema gossip asincrono, la latenza di rete e la differenza di velocità tra worker possono causare l'arrivo di messaggi con un ritardo di molti round. Un worker che ha già effettuato 50 round potrebbe ricevere pesi calcolati al round 30 da un peer lento. Incorporare questo aggiornamento degraderebbe la qualità del modello: i pesi vecchi codificano informazioni superate sul gradiente.

#### Implementazione del check

Thread 1 applica il seguente controllo prima di ogni aggregazione:

$$\text{discard if} \quad (r_{\text{current}} - r_{\text{sender}}) > \Delta_{\max}$$

dove $r_{\text{current}}$ è il round corrente del ricevente (letto da `shared_state["current_round"]`), $r_{\text{sender}}$ è il campo `round` del messaggio, e $\Delta_{\max}$ è il parametro `max_staleness` (default: 10). Il messaggio viene scartato restituendo `Ack(accepted=False)` senza modificare l'accumulatore.

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

Nel contesto FL, il Dropout svolge un ruolo aggiuntivo rispetto alla semplice regolarizzazione: forza il modello a sviluppare **rappresentazioni distribuite e ridondanti**. Se il modello non può fare affidamento su nessun singolo filtro (perché viene azzerato con probabilità 0.25 ad ogni step), impara a codificare la stessa feature visiva su più filtri contemporaneamente. Questa ridondanza produce modelli più *compatibili* per la media FedAvg: se Worker A e Worker B codificano entrambi la feature "curva superiore della lettera `a`" su 4–5 filtri ciascuno (grazie al Dropout), la loro media mantiene quella feature su 4–5 filtri; se invece ciascuno la codificasse su un solo filtro (overfitting locale), la media potrebbe ridurla o cancellarla se i due filtri dominanti non coincidono per indice. La ridondanza aumenta la probabilità che i filtri "utili" di un worker sopravvivano nella media con quelli dell'altro.

#### Dropout(p=0.5) nel Classificatore FC

Il layer `Linear(3136→512)` è il layer più denso e il principale rischio di overfitting su partizioni locali piccole. Con `p=0.5`, la metà delle unità viene azzerata casualmente a ogni passo di training, forzando la rete a sviluppare rappresentazioni ridondanti e distribuite. In fase di inferenza (`model.eval()`), Dropout è disabilitato e tutti i neuroni contribuiscono con i pesi originali (senza il fattore di scala $1/p$ perché PyTorch usa *inverted dropout* di default).

Il valore `p=0.5` è il valore classico proposto da Srivastava et al. (2014) per layer FC in classificazione. Nel contesto non-i.i.d. di FEMNIST, la ratio è ulteriormente rafforzata: con partizioni da ~210k–273k campioni e 62 classi, ogni worker ha mediamente ~3.400–4.400 campioni per classe. Su questa quantità, un layer FC da 1.6M parametri è a forte rischio di overfitting locale, specializzandosi sulle peculiarità visive degli scrittori di quella partizione. Un modello altamente overfit alla propria partizione produce pesi in zone remote dello spazio dei parametri, lontane dai pesi degli altri worker — rendendo la media FedAvg meno efficace. Il Dropout contrasta questo effetto mantenendo il modello in una regione dello spazio dei pesi più "centrale" e condivisa.

#### Gradient Clipping in `train_step` — max\_norm=1.0

Il gradient clipping limita la norma L2 del gradiente aggregato su tutti i parametri a `max_norm=1.0` prima di ogni `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

In FL, il gradient clipping svolge un ruolo specifico nel mitigare il **client drift**: su dati non-i.i.d., il gradiente locale può divergere significativamente dalla direzione del gradiente globale, specialmente dopo molti inner steps H. Gradienti grandi amplificano questa divergenza, rendendo il modello aggregato in Fase A meno stabile. Limitare la norma dei gradienti locali contiene la divergenza massima tra worker e migliora la qualità dell'aggregazione FedAvg. Il valore `max_norm=1.0` è una scelta conservativa standard in letteratura FL; con learning rate 0.001 i gradienti sono tipicamente già nell'ordine di $10^{-3}$–$10^{-1}$, quindi il clipping interviene solo in casi di gradiente esplosivo.

Il gradient clipping fornisce inoltre una **garanzia di bounded drift** verificabile. Se la norma del gradiente è clippata a `max_norm` e il learning rate è $\eta$, ogni singolo passo di ottimizzazione può spostare i parametri di al più $\eta \cdot \text{max\_norm} = 0.001 \times 1.0 = 10^{-3}$ in norma L2. Dopo $H = 500$ inner steps, il drift massimo dal punto di partenza del round è limitato superiormente da $H \cdot \eta \cdot \text{max\_norm} = 0.5$. Questo upper bound — nella pratica molto più ottimistico perché i passi non sono allineati — garantisce che i pesi dei diversi worker non possano diverg oltre una distanza controllata prima della prossima aggregazione. È questa garanzia che rende FedAvg teoricamente fondata nel nostro sistema: la distanza L2 tra due modelli da aggregare è bounded, la loro interpolazione lineare cade quindi in una regione di peso space che è "vicina" a entrambi i punti di partenza.

#### Label Smoothing — $\epsilon = 0.1$

Con 62 classi, molte visivamente simili (`0`/`O`, `1`/`l`/`I`, `5`/`S`, `c`/`C`), la cross-entropy standard allena il modello a produrre distribuzioni dove quasi tutta la massa di probabilità è concentrata sulla classe corretta. Questo porta a modelli *sovra-confidenti*.

Label smoothing sostituisce il target hard $\delta_{k,y}$ con un target morbido:

$$\tilde{y}_k = (1 - \epsilon) \cdot \delta_{k,y} + \frac{\epsilon}{K}$$

dove $\epsilon = 0.1$ e $K = 62$. La probabilità target della classe corretta diventa 0.90 invece di 1.0, distribuendo 0.10 uniformemente tra tutte le classi. I benefici sono:
- **Riduzione dell'over-confidence** su classi ambigue, con output probabilistici meglio calibrati.
- **Miglioramento della generalizzazione post-aggregazione**: un modello calibrato su dati non-i.i.d. locali generalizza meglio quando i suoi pesi vengono mediati con quelli di worker con distribuzioni diverse.

La motivazione è particolarmente forte su FEMNIST per due ragioni legate alla struttura del task. Prima: le 62 classi includono molte coppie visivamente ambigue (`0`/`O`, `1`/`l`/`I`, `b`/`d`/`p`/`q`, `c`/`C`, `s`/`S`, `v`/`V`, `x`/`X`). Su questi caratteri, la "risposta corretta" è meno netta che su, ad esempio, MNIST con sole 10 cifre ben distinte. Addestrare il modello a produrre probabilità 1.0 sulla classe corretta lo porta a tracciare frontiere di decisione molto strette e fragili in prossimità di queste coppie ambigue. Con label smoothing ε=0.1, il modello impara invece a mantenere una probabilità residua sulle classi simili — comportamento più robusto alle variazioni di stile tra scrittori.

Seconda: in un sistema FL non-i.i.d., i modelli di worker diversi sviluppano *gerarchie di confidenza* diverse sulle stesse classi, perché i loro scrittori scrivono le stesse lettere in modo leggermente diverso. Un modello molto over-confident su Worker 0 (che assegna p=0.99 alla classe `a` per certi tratti) e uno altrettanto over-confident su Worker 1 (che assegna p=0.99 alla classe `a` per tratti leggermente diversi) producono, dopo la media FedAvg dei pesi, un modello con logit inconsistenti — perché le regioni di attivazione che ciascun modello considera "definitivamente `a`" non coincidono nello spazio delle feature. Modelli con distribuzioni di output più morbide sono intrinsecamente più compatibili per la media: la loro interpolazione lineare nello spazio dei pesi produce logit che conservano la gerarchia di confidenza su entrambe le distribuzioni.

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

#### Scopo originale dell'early stopping e adattamento al contesto FL

In ML centralizzato, l'early stopping nasce per rilevare l'**overfitting**: quando la train loss scende ancora ma la val loss smette di migliorare o peggiora, il modello sta memorizzando il training set invece di generalizzare. Si ferma il training al minimo della val loss — il punto in cui il modello generalizza meglio su dati mai visti — e si scarta tutto il training successivo. La val loss è il segnale affidabile perché è calcolata su campioni che il modello non ha mai usato per aggiornarsi.

Nel nostro sistema la stessa meccanica si applica a livello di round: se la val loss locale non migliora di almeno $10^{-4}$ per `early_stopping_patience` round **consecutivi** (non totali — il contatore si azzera ad ogni miglioramento), il worker ferma il proprio training. L'unità temporale è il round invece dell'epoca.

#### Differenza semantica rispetto all'early stopping centralizzato

In ML centralizzato, l'early stopping misura la loss sul validation set **globale**: se peggiora, il modello sta overfittando l'intero dataset di training. La decisione è globale e coordinata.

Nel nostro sistema la decisione è **locale e indipendente**: ogni worker misura la propria val_loss sulla propria partizione locale. Questo crea due problemi specifici del contesto FL:

1. **Convergenza locale ≠ convergenza globale.** Un worker con una partizione "facile" può raggiungere un plateau locale al round 30 mentre la rete FL globale non ha ancora raggiunto consenso. Fermare quel worker priva gli altri di un peer attivo nei round successivi.

2. **Perdita di un vicino gossip.** Quando un worker si ferma, chiama `deregister_worker()` nel blocco `finally` e sparisce dalla lista peer del Discovery Server. Gli altri worker non lo trovano più come target per i push di Phase C, riducendo il `gossip_fanout` effettivo della rete. Con 3 worker totali, la perdita di uno riduce il fanout disponibile da 2 a 1 — impatto significativo.

Il fatto che la validation avvenga dopo la Phase A mitiga parzialmente il primo problema: la loss misurata include il contributo degli aggiornamenti ricevuti dai vicini, non solo quello del training locale. Tuttavia non elimina il rischio di stopping prematuro.

3. **FedAvg come fonte di rumore sul contatore.** In FL non-i.i.d., la FedAvg può causare un peggioramento temporaneo della val loss anche quando il sistema sta convergendo globalmente — il modello riceve pesi da worker con distribuzioni di dati diverse e impiega qualche round di training locale per "riadattarsi". Questo significa che il contatore di patience può salire non per overfitting ma per il normale rimescolamento dei pesi dovuto all'aggregazione. Nei run sperimentali su FEMNIST si osserva questa dinamica nei round iniziali: accuracy che crolla dopo la prima FedAvg (da ~75% a ~3% nel caso estremo) e poi risale gradualmente. Un early stopping con patience bassa (es. 5) potrebbe fermare il worker proprio durante questa fase di recovery, producendo un risultato peggiore di quanto si otterrebbe lasciandolo continuare.

#### Raccomandazione per gli esperimenti

Per i **confronti controllati** (Esperimenti 1–4 del piano sperimentale), è consigliabile disabilitare l'early stopping impostando `early_stopping_patience` a un valore superiore a `total_rounds` (es. `9999`). Questo garantisce che tutti i worker eseguano esattamente lo stesso numero di round, rendendo i confronti di accuratezza e convergenza direttamente comparabili.

Per **run di produzione** o esperimenti esplorativi dove si vuole evitare compute inutile su worker già convergenti, l'early stopping può rimanere abilitato con `patience: 10`.

### 5.6 Selezione degli Iperparametri

Il sistema non usa cross-validation (motivata in Sezione 2.3) né ottimizzazione automatica degli iperparametri (Bayesian optimization, Optuna, ecc.). I parametri in `config.yaml` sono fissi per tutta la durata di ogni run: non cambiano durante il training e non vengono aggiustati in risposta alla val loss. La ricerca è **manuale**: si esegue una run per ogni configurazione, si legge la val accuracy finale, e si sceglie la configurazione migliore a mano.

#### Tassonomia dei parametri

I parametri del sistema si dividono in tre categorie con ruoli distinti:

**Iperparametri ML** — influenzano direttamente la qualità del modello. Sono quelli ottimizzati con la grid search:

| Parametro | Candidati | Default | Effetto |
|---|---|---|---|
| `learning_rate` | 1e-4, 1e-3, 5e-3 | 1e-3 | Velocità di convergenza e stabilità del gradiente |
| `inner_steps_H` | 100, 500, 1000 | 500 | Drift locale tra worker; meno step = meno drift ma più traffico |
| `batch_size` | 16, 32, 64 | 32 | Stabilità del gradiente e velocità per step |

**Parametri di sistema** — influenzano le metriche ML ma sono determinati dall'architettura del deployment, non ottimizzati come iperparametri. Si studiano negli esperimenti di scalabilità (Sezione 9):

| Parametro | Candidati | Default | Effetto |
|---|---|---|---|
| `gossip_fanout` | 1, 2, 3, N-1 | 3 | Trade-off traffico/qualità aggregazione: fanout alto = più aggregazioni per round = convergenza più rapida ma volume di rete proporzionale |
| `num_workers` | 3, 5, 8 | 3 | Dimensione delle partizioni locali e numero di peer disponibili per l'aggregazione |

`gossip_fanout` è il parametro centrale del progetto: quantifica esattamente il trade-off traffico/convergenza che il sistema intende studiare, ed è il soggetto principale degli esperimenti comparativi. `num_workers` è fisso per ogni deployment ed è trattato in due fasi distinte: durante la ricerca degli iperparametri (Esp. 3) rimane fisso a un valore medio (es. 3 o 5), per isolare l'effetto degli iperparametri ML; successivamente, nella fase di scalabilità (Esp. 4), la configurazione ottimale trovata viene mantenuta fissa e si varia solo `num_workers` (3 → 5 → 8) per misurare come l'accuracy e il tempo di convergenza cambiano con la dimensione della rete.

**Parametri strutturali** — fissi per design, non si variano negli esperimenti:

| Parametro | Valore | Motivazione |
|---|---|---|
| `aggregation_strategy` | FedAvg | Algoritmo di riferimento della letteratura FL |
| `max_staleness` | 10 | Trade-off accettazione/qualità degli aggiornamenti |
| `drop_probability`, `crash_probability` | 0.20, 0.05 | Fault injection calibrata per gli esperimenti di robustezza |

#### Metriche per worker e metriche globali

Ogni worker misura le proprie metriche **localmente**: la `val_accuracy` di ogni round è calcolata sul `val/` di quel worker, con il suo modello dopo la FedAvg. Non esiste un nodo che osservi le prestazioni globali in tempo reale. `aggregate_metrics.py` aggrega i CSV post-run:

```
Per round (vista globale — media tra tutti i worker):
  Round 10 | mean_acc=0.71 | std_acc=0.04 | min=0.65 | max=0.76
  Round 11 | mean_acc=0.73 | ...

Per worker (vista individuale):
  Worker 0: final_acc=0.76 | best_acc=0.78
  Worker 1: final_acc=0.68 | best_acc=0.71
  Worker 2: final_acc=0.74 | best_acc=0.75
```

La metrica principale per confrontare le configurazioni è la **mean val accuracy finale** (media tra worker all'ultimo round). La **std accuracy** indica equità di convergenza: std bassa significa che tutti i worker beneficiano delle aggregazioni in modo uniforme — risultato atteso in un sistema FL sano. Std alta indica che alcuni worker convergono bene e altri no, spesso sintomo di fanout troppo basso o dati troppo sbilanciati.

#### Flusso di grid search

```bash
python scripts/download_femnist.py   # dataset completo (default, --sf 1.0)
python scripts/split_dataset.py && python scripts/generate_compose.py
```

Si varia un parametro alla volta mantenendo gli altri ai valori di default. Per ogni configurazione:

```bash
# 1. Modifica il parametro in config.yaml

# 2. Pulisci i risultati precedenti
rm -f data/femnist/worker_*/metrics.csv \
      data/femnist/worker_*/model_final.pt \
      data/femnist/worker_*/test_result.json

# 3. Lancia la run
# IMPORTANTE — quando usare --build:
#   Qualsiasi modifica a config.yaml o a file .py → sempre --build
#   (config.yaml è copiato nell'immagine durante il build, non montato)
#   Stesso codice e stessa config → --build è opzionale (l'immagine esistente è riusata)
#   Cambio di num_workers o use_test_set → ri-eseguire anche split_dataset.py
#   e generate_compose.py prima del --build
docker compose up --build

# 4. Analizza e archivia prima di passare alla prossima configurazione
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <nome>   # es: lr_1e-3, fanout_2, baseline
# → salva config.yaml + metriche + log container in results/<timestamp>_<nome>/
# IMPORTANTE: eseguire PRIMA di docker compose down — i log vengono rimossi con i container
docker compose down
```

Lo script `save_experiment.py` copia in `results/` i `metrics.csv` di ogni worker, il `global_metrics.csv`, il `summary.txt`, gli eventuali `test_result.json`, il `config.yaml` usato, e i log di ogni container Docker (`logs/<service>.log`). Il salvataggio dei log avviene tramite `docker compose logs` prima che i container vengano rimossi: Docker mantiene i log di un container finché il container non viene eliminato — anche se è crashato — quindi il file di log include l'output fino al momento del crash.

**Cosa fa `docker compose down` e quando usarlo.** `docker compose down` ferma i container e li **rimuove** (lo stato "exited" viene eliminato insieme ai log Docker), rimuove le reti create dal compose, ma **non** rimuove le immagini Docker (rimangono in cache) né i file su disco (`data/femnist/worker_*/` rimane intatto). Va eseguito tra ogni run perché `config.yaml` è copiato nell'immagine al build time — non è montato come volume — quindi un run successivo con `config.yaml` modificato ma senza `docker compose down` + `--build` userebbe la config precedente baked nell'immagine.

Quando cambia `num_workers` il ciclo è più lungo perché il dataset va ripartizionato e il compose rigenerato:

```bash
python scripts/save_experiment.py <nome>
docker compose down
# modifica num_workers in config.yaml
python scripts/split_dataset.py      # ripartiziona i dati per il nuovo numero di worker
python scripts/generate_compose.py   # rigenera docker-compose.yml con N servizi
docker compose up --build
```

Quando cambia solo un parametro con `num_workers` invariato:

```bash
python scripts/save_experiment.py <nome>
docker compose down
# modifica il parametro in config.yaml
docker compose up --build            # --build sempre necessario se config.yaml è cambiato
```

**Fase 2 — Conferma sul dataset completo**

```bash
python scripts/download_femnist.py             # dataset completo (default)
python scripts/split_dataset.py && python scripts/generate_compose.py
# Imposta la configurazione migliore trovata in Fase 1
docker compose up --build
python scripts/aggregate_metrics.py
python scripts/save_experiment.py best_config_full
```

La configurazione migliore trovata sul 5% viene rieseguita su `--sf 1.0` per verificare che i risultati si scalino correttamente.

**Fase 3 — Valutazione finale (opzionale, con test set)**

Se si vuole una stima non influenzata dalle decisioni di early stopping:

```bash
# Imposta use_test_set: true in config.yaml
python scripts/download_femnist.py             # re-download necessario (--tf diverso)
python scripts/split_dataset.py && python scripts/generate_compose.py
docker compose up --build
python scripts/aggregate_metrics.py
python scripts/save_experiment.py best_config_with_test
# → riporta val_accuracy (early stopping) + test_accuracy (stima onesta)
```

**Nota importante sul confronto tra Run A e Run B.** La Run B con `use_test_set: true` allena il modello su **80% dei dati** invece del 90% della Run A. Questo significa che la `test_accuracy` di Run B sarà probabilmente leggermente più bassa della `val_accuracy` di Run A per due motivi sovrapposti: (1) meno dati di training, effetto reale e non eliminabile; (2) assenza del bias ottimistico, che è quello che si vuole misurare. Non è possibile separare i due contributi con precisione.

Ciò che Run B garantisce comunque: la `test_accuracy` è una stima onesta della generalizzazione di quella specifica configurazione con 80% di training data. Se la differenza con la `val_accuracy` di Run A è piccola, il bias era trascurabile; se è grande, parte della differenza è bias e parte è l'effetto del training set più piccolo. Per l'obiettivo di questo progetto — validare la convergenza del sistema FL, non pubblicare un benchmark ML — questa ambiguità è accettabile.

Questo procedimento è interamente abilitato dal sistema di metriche descritto nella Sezione 6.

---

## 6. Metriche di Prestazione

### 6.1 Architettura del Sistema di Metriche

In un sistema P2P decentralizzato, non esiste un nodo centrale che osservi le prestazioni globali in tempo reale. Il sistema di metriche adottato sfrutta la struttura dei **bind mount Docker**: ogni worker scrive le proprie metriche su `{data_dir}/metrics.csv`, che — essendo `data_dir` montata dall'host — è immediatamente visibile sul filesystem dell'host senza alcun trasferimento dati aggiuntivo.

> **Osservabilità dall'host e decentralizzazione.** Il fatto che `aggregate_metrics.py` venga lanciato dall'host al termine del training non contraddice la natura decentralizzata del sistema. Gli script di analisi sono strumenti di osservabilità *post-hoc* — leggono i risultati dopo che il training è concluso, senza influenzare né coordinare il processo di apprendimento. La decentralizzazione riguarda il protocollo di training (nessun aggregatore centrale, gossip P2P), non gli strumenti di analisi dei risultati. In qualsiasi sistema FL reale — inclusi quelli descritti in letteratura — la valutazione finale avviene su un'infrastruttura separata dai nodi di training. Containerizzare `aggregate_metrics.py` non aggiungerebbe nulla alla claim di decentralizzazione: sarebbe un container che legge file CSV, non un partecipante al protocollo.

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
| `val_loss` | float | Loss sul validation set locale dopo la Fase A |
| `val_accuracy` | float | Accuracy sul validation set locale [0, 1] |
| `round_duration_s` | float | Durata totale del round (Fase A + B + C) in secondi |
| `neighbors_aggregated` | int | Numero di modelli vicini incorporati in Fase A (0 = nessuna aggregazione) |
| `peers_contacted` | int | Push gossip con successo in Fase C |

La riga viene scritta **dopo** le fasi A, B e C, incluse le durate di rete. La `round_duration_s` misura quindi il tempo reale di ogni ciclo completo del training loop.

### 6.3 Aggregazione Globale Post-Esperimento

`scripts/aggregate_metrics.py` legge tutti i file `worker_*/metrics.csv` e, se presenti, gli snapshot finali `worker_*/model_final.pt`. Produce:

**1. Tabella per round** — per ogni round, aggrega le metriche di tutti i worker attivi:

| Colonna | Significato |
|---|---|
| `mean_accuracy` | Accuracy media tra tutti i worker — indicatore della qualità del modello globale |
| `std_accuracy` | Deviazione standard dell'accuracy — misura di convergenza tra worker |
| `min_accuracy` / `max_accuracy` | Worker peggiore/migliore — identifica outlier |
| `workers_reporting` | Quanti worker erano ancora attivi (non early-stopped) in quel round |

**2. Riassunto per worker** — rounds completati, accuracy finale, accuracy migliore, media di peer contattati, media di vicini aggregati.

**3. Divergenza dei pesi (weight divergence)** — se gli snapshot finali `model_final.pt` sono presenti, lo script carica tutti i modelli, appiattisce i parametri float in un vettore 1-D e calcola la distanza L2 tra ogni coppia.

> **Nota su `model_final.pt`:** non è un checkpoint di ripristino — non salva lo stato dell'optimizer né il round corrente, quindi non permette di riprendere il training. È uno snapshot una-tantum dei pesi del modello salvato nel blocco `finally` alla fine del training (o in caso di terminazione pulita). Il suo unico scopo è questo calcolo di divergenza.

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

# Ripetere per num_workers = 3, 5, 8 ...
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
  └── Esp. 4: variare num_workers (3, 5, 8) con config ottimale da Esp. 3
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
# 1. Scaricare il dataset (completo per default; aggiungere --sf 0.05 solo per test rapidi)
python scripts/download_femnist.py

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

**Nota:** Tutti gli esperimenti usano il dataset completo (`--sf 1.0`, default). L'opzione `--sf 0.05` (5% del dataset, ~170 scrittori per split) è disponibile come scorciatoia per verifiche rapide di installazione o debugging del codice, ma non produce risultati rappresentativi da riportare.

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

Tutti gli esperimenti di questa fase usano il dataset completo (`--sf 1.0`, default). Se necessario velocizzare le iterazioni di sviluppo è possibile usare `--sf 0.05`, ma i risultati finali da riportare devono essere ottenuti su dataset completo.

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

**Scelta della configurazione ottimale:** al termine di Esp. 3, selezionare la combinazione `(lr, H, M)` con la `mean_accuracy` più alta sul dataset completo. Questa diventa la **configurazione fissa** per tutti gli esperimenti successivi.

### 7.6 Esperimento 4 — Analisi della Scalabilità

**Obiettivo:** misurare come le prestazioni del sistema cambiano al variare del numero di worker. Questo è il requisito sperimentale esplicito della traccia di progetto ("analisi della scalabilità"). L'esperimento presuppone che la configurazione ottimale sia già stata trovata in Esp. 3 con un valore di `num_workers` fisso (es. 3 o 5): qui si varia **solo** `num_workers`, tenendo fissi tutti gli altri parametri.

> **Nota sul deploy:** per misurare il *tempo* di convergenza in modo significativo, questo esperimento va eseguito in modalità **AWS multi-instance** (`aws_deploy.py`) dove ogni worker gira su un'istanza EC2 separata con latenza di rete reale. In locale i worker comunicano via loopback e i tempi non sono confrontabili tra configurazioni diverse di N. La valutazione dell'accuracy è invece identica nei due ambienti.

**Procedura per ogni valore di N:**
```bash
# 1. Modificare num_workers in config.yaml (es. 3, poi 5, poi 8)
# 2. Ripartizionare il dataset (obbligatorio ad ogni cambio di num_workers)
python scripts/split_dataset.py
python scripts/generate_compose.py

# Modalità locale (accuracy, non tempi reali):
docker compose up --build
python scripts/aggregate_metrics.py
python scripts/save_experiment.py scalability_N3   # o N5, N8

# Modalità AWS multi-instance (accuracy + tempi reali):
python scripts/aws_deploy.py provision   # re-provisionare: il numero di istanze cambia
python scripts/aws_deploy.py deploy
python scripts/aws_deploy.py collect
python scripts/aggregate_metrics.py
python scripts/save_experiment.py scalability_aws_N3
python scripts/aws_deploy.py destroy     # IMPORTANTE: distruggere prima di cambiare N
```

**Valori di N da testare:** 3, 5, 8 (massimo per AWS Learner Lab: 9 istanze totali, 1 usata dal registry).

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
python scripts/download_femnist.py

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

### 7.10 Osservazioni Empiriche da Documentare (TODO — completare con dati reali)

I punti seguenti sono fenomeni osservati durante i run di sviluppo o attesi dalla teoria FL. Vanno arricchiti con i valori numerici reali prodotti dagli esperimenti completi e integrati nella discussione finale.

**1. Accuracy valley dopo la prima FedAvg.**
Nei run su FEMNIST non-i.i.d. si osserva un crollo drastico dell'accuracy subito dopo la prima aggregazione (nel run di sviluppo: da ~75% a ~3% su Worker 1 al round 3). Il modello locale ha imparato feature specifiche del proprio gruppo di scrittori; la media con un modello addestrato su scrittori completamente diversi produce un ibrido che non funziona bene su nessuna delle due partizioni. L'accuracy recupera nei round successivi man mano che il training locale "riadatta" il modello aggregato. Da documentare: entità del crollo, numero di round per il recovery, confronto con il caso i.i.d. (baseline teorica).

**2. Optimizer state staleness dopo FedAvg.**
AdamW accumula momenti di primo e secondo ordine ($m_t$, $v_t$) basati sui gradienti del modello locale. Dopo FedAvg i pesi cambiano significativamente ma i momenti restano quelli del modello pre-aggregazione — il primo step post-aggregazione applica una direzione di aggiornamento calibrata su un modello diverso da quello attuale. Questo contribuisce all'instabilità dei round immediatamente successivi all'aggregazione. La soluzione standard sarebbe resettare l'optimizer dopo ogni FedAvg, ma non è implementata per non alterare la dinamica comparativa degli esperimenti. Da documentare: confronto loss dei primi N step post-aggregazione vs step a regime.

**3. Accumulo multi-round nel buffer asincrono.**
Un worker più veloce (meno campioni per partizione) può completare più round mentre un worker lento è ancora nel suo round corrente, inviando più messaggi che si accumulano nel buffer del ricevente. Nel run di sviluppo: `neighbors=713235 (3 models)` = Worker 0 (209k) + Worker 2 (251k) × 2 messaggi in un singolo round di Worker 1 (272k). Questo crea un'asimmetria di contributo implicita: i worker più veloci pesano di più nell'aggregazione per pura differenza di velocità, non per qualità del modello. Da documentare: frequenza di questo fenomeno al variare di `num_workers` e `inner_steps_H`.

**4. Client drift e tensione con inner_steps_H.**
Con H elevato e dati non-i.i.d., i modelli locali "derivano" progressivamente lontano da qualsiasi ottimo comune — ogni worker ottimizza per la propria distribuzione locale, allontanandosi dagli altri. La tensione è: H grande → meno traffico di rete ma più drift → FedAvg meno efficace; H piccolo → più comunicazione ma modelli più allineati. Da documentare: curva di val accuracy finale vs H (Esperimento 3), con interpretazione in termini di drift.

**5. Propagazione dell'informazione con fanout=1.**
In una rete di N nodi con fanout=1, un modello aggiornato da un worker raggiunge tutti gli altri in almeno $\lceil \log_2 N \rceil$ round nel caso ottimo, ma la selezione casuale dei peer introduce alta varianza. Con 3 worker, è possibile che un worker non riceva aggiornamenti da un certo peer per molti round consecutivi per pura casualità. Da documentare: distribuzione empirica degli intervalli tra ricezioni da ogni peer, confronto tra fanout=1 e fanout=2.

**6. Early stopping come amplificatore delle asimmetrie di rete.**
Quando un worker raggiunge il plateau locale e si deregistra, il fanout effettivo degli altri worker si riduce. Con 3 worker totali, perderne uno dimezza i peer disponibili per il gossip — impatto sproporzionato rispetto a quanto succederebbe con 8 worker. Da documentare: round a cui scatta l'early stopping per ogni worker, impatto sulla convergenza dei worker rimanenti.

**7. Costo reale di comunicazione per configurazione.**
Ogni messaggio gossip trasporta il modello serializzato (~6.5 MB float32). Il volume totale per un run completo è: `gossip_fanout × rounds × num_workers × 6.5 MB`. Con fanout=1, 200 round, 3 worker: ~3.9 GB totali. Da documentare: tabella comparativa del volume per ogni configurazione testata, con confronto al FL centralizzato equivalente (dove ogni round trasferisce N × model_size).

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

L'algoritmo è **robusto per costruzione** al message drop: la Fase A aggrega esclusivamente i modelli effettivamente ricevuti nell'accumulatore. Se un round produce zero messaggi ricevuti (tutti droppati, nessun vicino attivo), la Fase A viene semplicemente saltata e il worker procede con il suo modello invariato. Non esiste alcuna dipendenza su una soglia minima di messaggi ricevuti per procedere.

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

1. `sys.exit(1)` solleva `SystemExit`, un'eccezione Python che **attraversa** i blocchi `finally`. Questo garantisce che il blocco `finally` in `main()` — che chiama `deregister_worker()` — venga eseguito prima che il processo termini. Il Registry riceve la deregistrazione e lo snapshot finale `model_final.pt` viene salvato.

2. `SystemExit` non viene catturata da `grpc_server.wait_for_termination()`, che non viene mai raggiunta. Il processo termina effettivamente — simulando un crash reale piuttosto che una terminazione pulita.

3. Il Docker container si arresta con exit code 1, il che (in assenza di `restart: always` nel compose) lascia il servizio down — comportamento intenzionale.

Lo stesso meccanismo `finally` è sfruttato dai signal handler descritti in Sezione 8.4: SIGTERM e SIGINT vengono intercettati e reindirizzati a `sys.exit(0)`, garantendo la stessa sequenza di cleanup (deregistrazione + salvataggio snapshot finale) anche per shutdown manuali e `docker stop`.

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
2. Lo snapshot finale `model_final.pt` viene salvato
3. Il processo termina con exit code 0

#### Known limitation: SIGKILL e OOM

Se il container viene terminato con `docker kill` o dall'OOM killer del kernel, il processo riceve SIGKILL e termina istantaneamente senza eseguire alcun codice Python. In questo caso:
- Il worker rimane nel registry fino al successivo riavvio (entry stale)
- Gli altri worker continueranno a tentare push verso di esso, ricevendo `UNAVAILABLE`
- Il meccanismo di re-query reattivo (Sezione 4.2) attiva automaticamente la ricerca di peer sostitutivi

Una soluzione completa richiederebbe un meccanismo di **heartbeat con TTL** nel registry: i worker inviano periodicamente un segnale di vita, e il registry rimuove automaticamente chi non si fa vivo da T secondi. Questo è il pattern adottato in protocolli di membership production-grade come SWIM. Per il perimetro di questo progetto, dove i crash SIGKILL non fanno parte del modello di fault injection, il meccanismo di re-query reattivo costituisce una mitigazione sufficiente.

#### Comportamento del sistema alla perdita di un worker

La tabella seguente riassume tutti gli scenari di terminazione e il loro impatto sul sistema:

| Causa | Segnale | `finally` | Deregistrazione | Impatto sugli altri worker |
|---|---|:---:|:---:|---|
| `docker stop` / Ctrl+C | SIGTERM / SIGINT | ✅ | ✅ | Dal round successivo non compare più in `/get_peers`; fanout effettivo si riduce |
| Crash simulato (`crash_probability`) | `sys.exit(1)` → SystemExit | ✅ | ✅ | Identico al caso sopra |
| Early stopping | loop `break` → SystemExit | ✅ | ✅ | Deregistrato ma gRPC server ancora attivo; può ricevere push ma non li processa |
| `docker kill` / OOM killer | SIGKILL | ❌ | ❌ | Entry stale nel registry; altri worker ricevono `UNAVAILABLE` e timeout da 5s per round |

**Adattamento del sistema.** In tutti i casi di terminazione pulita (SIGTERM, SIGINT, crash simulato, early stopping), la riduzione del numero di worker è trasparente: il registry aggiorna la lista, e dalla successiva chiamata a `GET /peers` gli altri worker ottengono una lista senza il nodo uscente. Il `gossip_fanout` effettivo diventa `min(gossip_fanout, peer_disponibili)` — automaticamente, senza nessuna riconfigurazione. I dati e i pesi già aggregati nei round precedenti restano incorporati nei modelli dei worker superstiti: la perdita di un nodo non annulla il lavoro già fatto.

**Degradazione graduale, non catastrofica.** Con 3 worker e `gossip_fanout=2`, la perdita di uno riduce il fanout disponibile a 1 — ogni worker ha un solo peer a cui inviare. La convergenza rallenta ma il training prosegue. Con 2 worker rimasti, ogni worker riceve aggiornamenti da 1 vicino per round invece che da 2: le aggregazioni sono meno ricche ma il sistema non si ferma. Questo comportamento di *graceful degradation* è una proprietà fondamentale dell'architettura P2P — non esiste un coordinatore centrale la cui perdita blocchi l'intero sistema.

**Caso SIGKILL: costo per round.** Se un worker muore senza deregistrarsi, ogni round gli altri worker sprecano `grpc_timeout_seconds` (5s) tentando di raggiungerlo. Con 3 worker e `gossip_fanout=2`, se uno è morto via SIGKILL: ogni round i due superstiti tentano il push, uno fallisce con timeout dopo 5s, attiva il re-query reattivo, ottiene la stessa lista stale, probabilmente fallisce di nuovo. Il costo è ~10s extra per round per worker — non bloccante ma rilevante. La soluzione completa (heartbeat con TTL nel registry) è documentata come known limitation; per il modello di fault injection di questo progetto, dove i crash avvengono via `sys.exit(1)` con deregistrazione pulita, il caso SIGKILL non è nel perimetro degli esperimenti.

### 8.5 Semantiche di Consegna dei Messaggi

#### Premessa: le semantiche si discutono sul failure path, non sul success path

Le semantiche di consegna (at-most-once, at-least-once, exactly-once) descrivono il comportamento del sistema **quando la rete fallisce** — connessione interrotta, timeout, nodo irraggiungibile. Non descrivono il caso nominale: quando TCP funziona e la RPC completa con successo, il messaggio è consegnato esattamente una volta per definizione. La domanda rilevante è: *cosa succede se la consegna fallisce a metà?* È quella risposta che determina la semantica.

#### Garanzie fornite automaticamente da gRPC e Flask

Entrambi i canali di comunicazione usano TCP come trasporto. TCP garantisce **ordine e integrità dei byte** all'interno di una singola connessione: se la trasmissione completa senza eccezione, il payload è arrivato integro e nell'ordine corretto. Questo vale sia per le chiamate gRPC (gossip push) sia per le richieste HTTP al registry (Flask).

Le garanzie a livello applicativo — quante volte un messaggio viene consegnato in caso di errore — dipendono invece dalla logica implementata sopra TCP.

#### Ricezione concorrente da più peer

Uno scenario comune durante la Fase C è che due o più peer inviano i propri pesi allo stesso worker quasi contemporaneamente. Ogni peer apre una propria connessione TCP separata: TCP non sa nulla delle altre connessioni e le gestisce indipendentemente, quindi entrambi i messaggi arrivano integralmente senza interferenze a livello di rete.

Lato receiver, il server gRPC è avviato con un thread pool:

```python
server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
```

Ogni chiamata `ReceiveModel` in arrivo viene assegnata a un thread del pool. Due peer concorrenti producono due thread che eseguono `ReceiveModel` in parallelo — nessuno viene scartato o messo in coda indefinitamente.

Il rischio reale è la **scrittura concorrente sull'accumulatore**: senza sincronizzazione, i due thread potrebbero leggere `weighted_sum` nello stesso istante, calcolare i propri incrementi separatamente, e uno sovrascrivere il contributo dell'altro. Il `threading.Lock` in `AggregationBuffer` serializza gli aggiornamenti:

```python
with self.buffer.lock:
    # un solo thread alla volta modifica weighted_sum e received_samples
    self.buffer.weighted_sum[k] += weighted[k]
    self.buffer.received_samples += sender_samples
    self.buffer.messages_received += 1
```

Il secondo thread attende che il primo rilasci il lock, poi accumula il suo contributo. Entrambi gli aggiornamenti vengono registrati correttamente: la concorrenza non causa perdita di dati.

#### Timeout gRPC e il ruolo dell'ACK

Ogni chiamata `ReceiveModel` include un timeout esplicito configurabile (`grpc_timeout_seconds`, default 5.0 s):

```python
ack = stub.ReceiveModel(message, timeout=timeout)
```

Se il server non risponde entro questo limite, gRPC solleva `RpcError` con codice `DEADLINE_EXCEEDED`. Il timeout serve a evitare che il training loop rimanga bloccato su un nodo irraggiungibile: senza di esso, una chiamata verso un peer crashato aspetterebbe indefinitamente la risposta TCP.

L'`Ack` restituito dal server (`Ack(accepted=True/False)`) **non è un meccanismo separato di acknowledgment**: è semplicemente la risposta del metodo RPC, come qualsiasi valore di ritorno di una funzione remota. Quando `stub.ReceiveModel` ritorna senza eccezione, significa che la connessione TCP è rimasta aperta per tutto il ciclo richiesta-risposta e che il server ha eseguito `ReceiveModel` fino in fondo. Il campo `accepted` indica se il messaggio ha superato lo staleness check (Sezione 4.4), non se è arrivato fisicamente.

#### Gossip push (gRPC): semantica at-most-once e il caso limite dell'ACK perso

Il client effettua **una sola chiamata RPC** per destinatario, senza retry:

```python
success = send_model(target, weights_snapshot, round_num, local_samples, worker_id, grpc_timeout)
if success:
    sent_count += 1
else:
    failed_targets.append(target)
```

Il meccanismo di re-query reattivo (Sezione 4.2) cerca un **peer sostitutivo**, non riprova lo stesso destinatario. La semantica è **at-most-once per peer**.

Il caso limite che giustifica questo nome è il seguente: il server riceve il messaggio, lo accumula correttamente, poi la connessione TCP cade prima che l'`Ack` raggiunga il sender. Il sender riceve `RpcError` e marca la consegna come fallita — ma il messaggio era già stato processato. Senza retry, il messaggio risulta consegnato zero volte dal punto di vista del sender ma una volta dal punto di vista del receiver. Questo è esattamente at-most-once: il sender non riprova, quindi il messaggio è processato **al più una volta** (zero in caso di errore, uno in caso di successo). Per garantire almeno-una-volta servirebbe un retry, ma come discusso nella sezione "Cosa richiederebbe exactly-once", questo introdurrebbe duplicati senza deduplicazione.

gRPC non implementa retry automatici di default. Esiste una *retry policy* configurabile via service config, ma richiede deduplicazione lato server ed è disabilitata per scelta in questo progetto.

#### Timeout e retry su Flask: comportamento per endpoint

I tre endpoint del registry hanno comportamenti diversi perché hanno priorità diverse:

**`/register` — at-least-once, retry attivo**

```python
for attempt in range(max_retries):   # default max_retries=10
    try:
        response = requests.post(f"{registry_url}/register", ..., timeout=5)
        response.raise_for_status()
        return
    except Exception:
        time.sleep(3)
```

La registrazione è critica: senza di essa il worker non è raggiungibile dagli altri peer e non riceve gossip push. Il retry con `timeout=5` per chiamata e `time.sleep(3)` tra tentativi copre il caso in cui il registry container non sia ancora avviato al momento del primo tentativo. L'operazione è idempotente (`_registry[worker_id] = address` sovrascrive silenziosamente), quindi un doppio invio non causa inconsistenze.

**`/peers` — at-most-once, nessun retry**

```python
def fetch_peers(registry_url: str) -> list[str]:
    try:
        return requests.get(f"{registry_url}/peers", timeout=5).json()
    except Exception as exc:
        logger.warning(f"Could not fetch peers: {exc}")
        return []
```

Un fallimento restituisce una lista vuota: il worker salta la Fase C per quel round e riproverà al round successivo. Non è critico: perdere una query dei peer in un round non compromette la correttezza — al massimo quel round non produce gossip push.

**`/deregister` — best-effort, nessun retry**

```python
def deregister_worker(registry_url: str, worker_id: str):
    try:
        requests.post(f"{registry_url}/deregister", ..., timeout=5)
    except Exception:
        pass  # non-critical
```

La deregistrazione è best-effort: se fallisce, il registry mantiene un'entry stale fino al prossimo riavvio. Gli altri worker riceveranno `UNAVAILABLE` dal gRPC e il meccanismo di re-query reattivo (Sezione 4.2) troverà peer alternativi. Non giustifica un retry perché l'effetto di un fallimento è limitato e temporaneo.

#### Perché at-most-once è la scelta corretta per il gossip push

In un sistema transazionale (pagamenti, database) la perdita di un messaggio è un errore grave che richiede retry, deduplicazione e garanzie exactly-once. Nel federated learning il modello è diverso: i pesi inviati durante il gossip push sono **aggiornamenti statistici approssimati**, non operazioni atomiche con stato persistente.

La robustezza al message drop è già documentata in Sezione 8.1: l'accumulatore di aggregazione in Fase A opera su qualunque sottoinsieme di messaggi ricevuti — se un round produce zero contributi da peer, la Fase A viene semplicemente saltata e il worker procede con il proprio modello invariato. Il round successivo riceverà nuovi aggiornamenti. Non esiste alcuna dipendenza su una soglia minima di messaggi ricevuti per garantire la correttezza dell'algoritmo.

At-most-once è anche coerente con il **requisito di basso traffico di rete** della traccia di progetto: nessun retry implica volume di comunicazione deterministico, pari a $N \times k \times S_{\text{model}}$ per round al massimo.

#### Cosa richiederebbe exactly-once

Per garantire la consegna exactly-once servirebbe:

- un **sequence number per (sender, round)** nel messaggio
- un registro di deduplicazione lato ricevente (es. set degli `(worker_id, round)` già processati)
- un meccanismo di retry lato sender fino a conferma esplicita

Oltre alla complessità implementativa, questa soluzione introdurrebbe un problema semantico nel contesto FL: un retry che arriva nel round successivo porterebbe pesi appartenenti al round precedente nell'accumulatore del round corrente, violando la semantica dello staleness check (Sezione 4.4) e potenzialmente peggiorando la convergenza. At-most-once è quindi non solo più semplice, ma **semanticamente più corretto** per questo dominio.

---

## 9. Implementazione e Deployment

### 9.1 Struttura dei File

```
ml_sdcc_project/
├── registry_server.py        # Discovery Server (Flask)
├── main_worker.py            # Worker entry point — training loop + gRPC server
├── config.yaml               # Single source of truth for all parameters
├── .dockerignore             # Excludes data/, scripts/, docs from build context
├── requirements.registry.txt # Registry dependencies (Flask only)
├── requirements.worker.txt   # Worker dependencies (PyTorch, gRPC, ...)
├── docker-compose.yml        # [GENERATED] Local + Single EC2 deployment — do not edit manually
├── docker/
│   ├── Dockerfile.registry      # Minimal image: no PyTorch, no grpcio
│   └── Dockerfile.worker        # Full image: PyTorch + gRPC + proto compilation
├── proto/
│   └── gossip.proto             # gRPC service and message definitions
├── scripts/
│   ├── download_femnist.py      # LEAF dataset download and preprocessing
│   ├── split_dataset.py         # Splits dataset into per-worker partitions
│   ├── generate_compose.py      # Generates docker-compose.yml from config.yaml
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

- **`docker/Dockerfile.registry`** — immagine minimale: solo `python:3.11-slim` + Flask. Non contiene PyTorch, grpcio o il codice worker. Dimensione tipica: ~80 MB.
- **`docker/Dockerfile.worker`** — immagine completa: PyTorch CPU, grpcio, grpcio-tools. Dimensione tipica: ~1.5 GB.

La separazione riduce significativamente i tempi di rebuild del registry (nessuna dipendenza pesante) e minimizza la superficie di attacco dell'immagine registry.

#### Ottimizzazione del layer caching

Docker costruisce le immagini a strati: ogni istruzione (`FROM`, `RUN`, `COPY`) produce un layer immutabile identificato da un hash. Se alla build successiva l'hash di un layer coincide con quello in cache, Docker lo riutilizza senza rieseguire il comando. L'invalidazione è **a cascata**: modificare un layer invalida automaticamente tutti quelli successivi, indipendentemente dal loro contenuto.

La regola pratica che ne discende è ordinare le istruzioni dal più stabile al più volatile: le dipendenze pesanti in cima, il codice sorgente in fondo. Entrambi i Dockerfile rispettano questo principio.

**`docker/Dockerfile.worker` — sequenza dei layer:**

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

# Layer 4 — compilazione Protobuf: invalido solo se cambia proto/gossip.proto
COPY proto/gossip.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gossip.proto

# Layer 5 — sorgente applicativo: invalido ad ogni modifica al codice
COPY config.yaml main_worker.py ./
COPY core/ ./core/
COPY network/ ./network/
```

Il layer più costoso è il Layer 3 (installazione di PyTorch): viene rieseguito solo se `requirements.worker.txt` cambia. Ogni modifica al codice Python invalida esclusivamente il Layer 5 — la rebuild richiede secondi invece di minuti. Lo stesso principio vale per `docker/Dockerfile.registry`: prima `requirements.registry.txt`, poi `registry_server.py`.

**Condivisione dell'immagine tra N worker.** Tutti i container worker (`worker_0`, `worker_1`, ..., `worker_N`) sono istanze della **stessa immagine Docker** — non viene costruita una immagine separata per ognuno. `docker compose up --build` con 10 worker esegue `docker build` una sola volta, producendo una singola immagine con i suoi layer. I 10 container vengono poi istanziati da quell'unica immagine: i layer read-only (incluso il Layer 3 con PyTorch, ~750 MB) sono condivisi in memoria e su disco tra tutti i container. Ogni container ha solo un sottile layer scrivibile per i propri file di runtime (log, metriche, checkpoint), che è trascurabile rispetto al Layer 3.

Il risultato pratico è che PyTorch viene scaricato e installato **una volta sola**, indipendentemente da quanti worker si lanciano:

| Operazione | Costo |
|---|---|
| Prima build (nessuna cache) | ~250 MB download, ~5 min |
| Rebuild dopo modifica al codice sorgente | ~secondi (solo Layer 5 invalido) |
| Rebuild dopo aggiunta dipendenza Python | ~250 MB download, ~5 min (Layer 3 invalido) |
| `docker compose up` con N=10 worker | stesso costo di N=3 — stessa immagine |

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

Il numero di worker non è hardcoded nel compose file ma letto da `config.yaml`. Due script cooperano per mantenere il sistema coerente: `split_dataset.py` prepara i dati su host, `generate_compose.py` configura i container. Il compose file è un **artefatto generato** e non va editato manualmente.

```bash
# Workflow per modificare il numero di worker:
# 1. Modificare network.num_workers in config.yaml
# 2. Rieseguire il partizionamento (sovrascrive le slice precedenti)
python scripts/split_dataset.py
# 3. Rigenerare il compose file
python scripts/generate_compose.py
# 4. Riavviare il sistema
docker compose up --build
```

`generate_compose.py` produce `docker-compose.yml` con il numero corretto di servizi. Il registry riceve `REGISTRY_PORT` come variabile d'ambiente, in modo che la porta su cui ascolta sia sempre coerente con quella configurata in `config.yaml`. Ogni worker riceve `WORKER_ID=i` e `TOTAL_WORKERS=num_workers` come variabili d'ambiente, e monta esclusivamente la propria partizione tramite **bind mount** Docker (`type: bind`, sintassi lunga esplicita) — isolamento dei dati garantito a livello di filesystem.

### 9.4 Deploy su AWS EC2

Il sistema supporta tre modalità di deployment, tutte governate dal medesimo `config.yaml`:

| Modalità | Comando | Quando usarla |
|---|---|---|
| **Locale** | `docker compose up --build` | Sviluppo, debug, grid search degli iperparametri |
| **AWS singola istanza** | `docker compose up --build` (su EC2 via SSH) | Test su cloud, stesso flusso del locale |
| **AWS multi-istanza** | `python scripts/aws_deploy.py deploy` | Esperimenti di convergenza con latenza di rete reale |

#### Perché multi-istanza per misurare la convergenza

In modalità locale (tutti i container sullo stesso host), la comunicazione gRPC avviene via loopback (`127.0.0.1`) con latenza < 0.1 ms e banda limitata solo dalla CPU locale. In produzione reale — e nei termini della specifica del progetto — ogni nodo è una macchina separata. Con Docker su singolo host si misura il comportamento algoritmico del gossip (convergenza in termini di round), ma non il tempo di convergenza reale, che dipende dalla latenza di rete tra i nodi.

Deployando ogni worker su un'istanza EC2 separata, i messaggi gRPC viaggiano su TCP/IP tra macchine fisicamente distinte (latenza tipica inter-EC2 stesso availability zone: 0.2–1 ms), rendendo le misure di convergenza temporale significative e confrontabili tra configurazioni diverse di `gossip_fanout` e `num_workers`.

#### Architettura AWS multi-istanza

```
Macchina locale (orchestratore)
    │
    ├─ terraform apply  →  VPC / Security Group / EC2 registry / N EC2 worker
    ├─ aws_deploy.py    →  build immagini + upload dataset + start container
    └─ aws_deploy.py    →  collect metrics → aggregate_metrics.py
                                │
                    ┌───────────┼───────────────────────┐
                    │           │                       │
              EC2 registry  EC2 worker_0  ...  EC2 worker_N-1
              :5000 HTTP    :50051 gRPC        :50051 gRPC
                    │           │    \  gossip  /   │
                    └───────────┴─────────────────┘
                        tutti nella stessa VPC
                        comunicazione via IP privati
```

Ogni worker registra il proprio **IP privato** come indirizzo gRPC (variabile `MY_HOST`). Le connessioni inter-worker rimangono all'interno della VPC, senza uscire su Internet: latenza più bassa e nessun costo di trasferimento dati. Tutte le istanze sono pinate alla **stessa Availability Zone** (parametro `aws.availability_zone` in `config.yaml`): il traffico IP privato intra-AZ è gratuito in AWS, mentre il traffico cross-AZ costa $0.01/GB per direzione — con gossip_fanout=3 e 200 round l'importo sarebbe ~$0.33 su 8 worker, evitabile a costo zero. L'orchestratore (macchina locale) accede alle istanze via SSH tramite i loro IP pubblici solo per deploy, monitoring e raccolta metriche.

#### Provisioning con Terraform

La directory `terraform/` definisce l'intera infrastruttura come codice:

```
terraform/
    main.tf        — provider AWS, security group, istanze EC2 con user_data
    variables.tf   — dichiarazioni delle variabili
    outputs.tf     — IP pubblici e privati di tutte le istanze
    terraform.tfvars  (generato da aws_deploy.py, non versionato)
```

`terraform/terraform.tfvars` viene generato automaticamente da `scripts/aws_deploy.py provision` leggendo `config.yaml`: il numero di worker, i tipi di istanza, la regione e le porte sono sempre sincronizzati con il file di configurazione senza intervento manuale.

Il `user_data` di ogni istanza installa Docker all'avvio:

```bash
#!/bin/bash
apt-get update -y && apt-get install -y docker.io
systemctl enable docker && systemctl start docker
usermod -aG docker ubuntu
```

Questo avviene in parallelo per tutte le istanze durante `terraform apply`. Lo script `aws_deploy.py` aspetta che Docker sia operativo prima di procedere con la build.

Il Security Group apre esattamente tre porte verso l'esterno:

| Porta | Protocollo | Scopo |
|---|---|---|
| 22 | TCP | SSH dall'orchestratore locale |
| 5000 | TCP | Registry HTTP (health check e peer discovery) |
| 50051 | TCP | gRPC worker (una sola porta: un worker per istanza) |

Tutto il traffico tra istanze dello stesso security group è permesso senza restrizioni (`self = true`), rendendo possibile la comunicazione gRPC su IP privati.

#### Flusso di deploy (`aws_deploy.py`)

Lo script `scripts/aws_deploy.py` è l'unico punto di controllo per l'intero ciclo di vita del cluster AWS. Espone sei sottocomandi:

```
provision  →  genera tfvars + terraform apply
deploy     →  [1] attende SSH+Docker su tutte le istanze
              [2] build immagini Docker in parallelo (SCP sorgente + docker build)
              [3] SCP partizioni dataset ai worker in parallelo
              [4] avvia registry container + attende healthcheck
              [5] avvia worker container in parallelo
collect    →  SCP metrics.csv, test_result.json, model_final.pt da ogni worker
status     →  mostra docker ps su ogni istanza
logs <id>  →  docker logs -f worker_<id> o registry (via SSH interattivo)
destroy    →  terraform destroy (elimina tutte le istanze)
```

La fase `[2]` costruisce l'immagine `fl-worker` sui nodi worker e `fl-registry` sul nodo registry. Il codice sorgente e `config.yaml` vengono compressi in un archivio `.tar.gz` e copiati via SCP; `docker build` gira in parallelo su tutte le istanze, sfruttando la CPU di ogni EC2. Con `t3.small`, una build da zero richiede circa 5–8 minuti (dominata dall'installazione di PyTorch); le build successive sono veloci grazie al layer caching di Docker. Il comando `python scripts/aws_deploy.py destroy` va eseguito al termine di ogni sessione per fermare la fatturazione.

#### AWS Learner Lab — vincoli e note operative

Il Learner Lab impone limiti precisi che determinano le scelte architetturali e di configurazione del sistema.

**Limiti sulle istanze EC2**

| Vincolo | Valore | Impatto sul progetto |
|---|---|---|
| Istanze concorrenti per regione | **max 9** | Con 1 registry → max **8 worker** |
| vCPU concorrenti per regione | max 32 | t3.small usa 2 vCPU → 9 × 2 = 18 vCPU, entro il limite |
| Tipi di istanza supportati | nano, micro, small, medium, large | **xlarge e superiori non sono supportati** |
| Istanze on-demand | sì | Spot instances non disponibili |
| Superare i limiti | istanze eccedenti terminate | 20+ istanze → disattivazione immediata account |

`aws_deploy.py provision` verifica che `num_workers + 1 ≤ 9` prima di lanciare Terraform e interrompe con un errore esplicito se il vincolo è violato.

**Scelta delle istanze**

Per i worker è stato scelto `t3.small` (2 vCPU, 2 GB RAM). Il collo di bottiglia di RAM è il dataset: `FEMNISTDataset.__init__` converte l'intero split di training in tensori PyTorch in memoria all'avvio (non caricamento lazy). La stima per worker è circa:

| `num_workers` | Immagini/worker | Tensori train (float32) | PyTorch overhead | Totale |
|:---:|:---:|:---:|:---:|:---:|
| 3 | ~267k | ~830 MB | ~300 MB | ~1.1 GB |
| 5 | ~160k | ~500 MB | ~300 MB | ~800 MB |
| 8 | ~100k | ~310 MB | ~300 MB | ~610 MB |

I pesi del modello sono trascurabili (~7 MB). `t3.small` (2 GB) è sufficiente per tutti i valori di `num_workers` con il dataset completo, con margine. Per batch size > 64 o modelli più grandi, `t3.medium` (4 GB) offre maggiore sicurezza.

Per il registry `t3.micro` (1 GB RAM) è più che sufficiente: il Discovery Server è un server Flask in-memory con traffico minimo.

**Voci di costo AWS**

AWS addebita quattro voci distinte; tutte sono rilevanti per questo progetto:

| Voce | Tariffa | Note |
|---|---|---|
| EC2 compute | $0.021/hr per t3.small, $0.042 per t3.medium | Solo istanze *running*; istanze *stopped* non addebitano compute |
| **IPv4 pubblici** | **$0.005/hr per IP** (dal feb 2024) | Si applica a ogni istanza running; spesso dimenticato — aggiunge ~25% al compute su 9 istanze |
| EBS (disco) | $0.08/GB/mese (gp3) | Addebitato anche su istanze *stopped*; 20 GB ≈ $0.002/hr; rischio se si dimentica `destroy` |
| Trasferimento dati | $0.01/GB tra AZ diverse (IP privati) | Gratis nella stessa AZ; il deployment pinna tutte le istanze alla stessa AZ per azzerare questo costo |

I worker comunicano via IP privati VPC (non Internet), quindi non si applicano tariffe egress Internet ($0.09/GB). L'orchestratore locale scarica solo i CSV di metriche (KB totali).

**Stima costo totale per configurazione (run da 30 minuti)**

| Config | Compute | IPv4 | EBS | Totale/run | 25 run |
|---|---|---|---|---|---|
| 3 worker t3.small + 1 t3.micro | $0.018 | $0.010 | $0.001 | ~$0.029 | ~$0.73 |
| 5 worker t3.small + 1 t3.micro | $0.028 | $0.015 | $0.002 | ~$0.045 | ~$1.13 |
| 8 worker t3.small + 1 t3.micro | $0.043 | $0.023 | $0.003 | ~$0.069 | ~$1.73 |
| 8 worker t3.medium + 1 t3.micro | $0.085 | $0.023 | $0.003 | ~$0.111 | ~$2.78 |

Il budget Learner Lab è di $100: l'intera campagna sperimentale (grid search iperparametri + scalabilità + test set) rimane abbondantemente sotto i $10. Il rischio principale non è il costo per run, ma dimenticare `destroy` e lasciare le istanze accese tra sessioni — l'EBS continua ad accumularsi finché le istanze non vengono terminate.

**Key pair e accesso SSH**

In us-east-1, il Learner Lab mette a disposizione una key pair predefinita chiamata `vockey`. Non è necessario creare una nuova key pair:
1. Nel pannello del lab cliccare **AWS Details**
2. Cliccare **Download PEM** → salva `labsuser.pem`
3. Impostare in `config.yaml`: `key_name: "vockey"`, `key_path: "~/Downloads/labsuser.pem"`

In us-west-2, invece, la vockey non è disponibile: occorre creare una nuova key pair dall'EC2 Console e aggiornarla in `config.yaml`.

**Regioni disponibili**: us-east-1 (default, con vockey) e us-west-2.

**Comportamento tra sessioni e IP pubblici**

Quando la sessione del lab scade, le istanze EC2 vengono **stoppate** (non terminate) e riavviate automaticamente all'inizio della sessione successiva. Questo comporta tre conseguenze:

1. **Le credenziali scadono** ma le istanze rimangono. Occorre esportare nuove credenziali all'inizio di ogni sessione.
2. **Gli IP pubblici cambiano** a ogni riavvio: le istanze ottengono un nuovo IPv4 pubblico, rendendo stale lo stato di Terraform. Dopo aver riavviato una sessione lab con istanze già in esecuzione, eseguire:

```bash
python scripts/aws_deploy.py resume   # → terraform apply -refresh-only
```

Questo aggiorna lo stato di Terraform con i nuovi IP pubblici senza modificare l'infrastruttura. Gli **IP privati** non cambiano tra stop/start e continuano a funzionare per la comunicazione interna tra worker.

3. **Le istanze ripartono automaticamente alla sessione successiva** e riprendono a consumare budget. Le istanze che erano in esecuzione quando la sessione è terminata vengono riavviate automaticamente all'inizio della sessione successiva — anche se non si intende usarle. **Distruggerle** (`destroy`) è il modo sicuro per evitare spese impreviste.

**Estendere la sessione durante il training.** La sessione dura 4 ore, ma può essere estesa cliccando nuovamente **Start Lab** *prima* che il timer raggiunga 0:00. Se si avvia un training lungo, ricordarsi di rinnovare la sessione a metà run evita del tutto la situazione di sessione scaduta e rende `resume` non necessario.

**Budget monitoring — ritardo di 8-12 ore.** Il pannello del lab mostra il credito residuo aggiornato da AWS Budgets, che si aggiorna tipicamente ogni 8-12 ore. Il saldo visualizzato può quindi non riflettere le spese più recenti. Non fare affidamento esclusivo su quel valore: stimare i costi a priori con la tabella sopra e distruggere le istanze al termine di ogni sessione.

**IMPORTANTE**: eseguire sempre `python scripts/aws_deploy.py destroy` al termine di ogni sessione di lavoro.

**SSH user.** Le istruzioni del Learner Lab mostrano il comando `ssh -i labsuser.pem ec2-user@<ip>`, dove `ec2-user` è l'utente predefinito per le AMI Amazon Linux. Le nostre istanze usano Ubuntu 22.04 (AMI Canonical), dove l'utente SSH è `ubuntu`. `aws_deploy.py` usa già correttamente `ubuntu@<ip>` in tutte le sue connessioni SSH/SCP.

**Elastic IP (opzionale).** Il Learner Lab supporta Elastic IP per mantenere un IP pubblico fisso tra stop/start. Per i nostri esperimenti di convergenza non è necessario (il `resume` command gestisce il cambio di IP), ma può essere utile per ambienti long-running dove si vogliono evitare aggiornamenti manuali delle credenziali SSH.

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
| `aws` | `region` | `us-east-1` | Regione AWS; Learner Lab supporta `us-east-1` (default, con vockey) e `us-west-2` |
| `aws` | `availability_zone` | `us-east-1a` | AZ di tutte le istanze; stesso AZ = traffico IP privato gratuito; cross-AZ = $0.01/GB |
| `aws` | `instance_type_worker` | `t3.small` | Tipo istanza EC2 per i worker (multi-instance); t3.small (2 GB) regge tutti i `num_workers` |
| `aws` | `instance_type_registry` | `t3.micro` | Tipo istanza EC2 per il registry (server Flask, <50 MB RAM) |
| `aws` | `instance_type_single` | `t3.large` | Tipo istanza single-EC2; t3.large (8 GB) regge fino a 8 worker con dataset completo |
| `aws` | `volume_size_worker` | `20` | Disco EBS worker in GB (multi-instance); range consigliato: 15–30 GB |
| `aws` | `volume_size_registry` | `8` | Disco EBS registry in GB; 8 GB sempre sufficiente (range: 8–15 GB) |
| `aws` | `volume_size_single` | `20` | Disco EBS single-EC2 in GB; range consigliato: 20–30 GB |
| `aws` | `key_name` | `vockey` | Nome della key pair EC2; in us-east-1 Learner Lab usa la `vockey` predefinita |
| `aws` | `key_path` | `~/Downloads/labsuser.pem` | Path locale al `.pem` scaricato dal pannello AWS Details |
| `aws` | `image_source` | `build` | `build` = docker build su EC2; `dockerhub` = docker pull |
| `aws` | `dockerhub_image` | `""` | Immagine DockerHub (solo se `image_source: dockerhub`) |

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

aws:
  region: "us-east-1"                    # us-east-1 (vockey) or us-west-2
  availability_zone: "us-east-1a"        # pin all instances to same AZ → free intra-AZ traffic
  instance_type_worker: "t3.small"       # multi-instance: 2 vCPU, 2 GB — sufficient for all num_workers
  instance_type_registry: "t3.micro"     # lightweight Flask server, <50 MB RAM
  instance_type_single: "t3.large"       # single-EC2: 2 vCPU, 8 GB — handles up to 8 workers
  volume_size_worker: 20                 # EBS per worker EC2 in GB (range: 15–30)
  volume_size_registry: 8               # EBS for registry EC2 in GB (range: 8–15)
  volume_size_single: 20                 # EBS for single-EC2 in GB (range: 20–30)
  # IMPORTANT: num_workers + 1 <= 9 (Learner Lab hard limit)
  key_name: "vockey"                     # pre-existing key pair in us-east-1
  key_path: "~/Downloads/labsuser.pem"   # downloaded from AWS Details panel
  image_source: "build"                  # "build" = docker build on EC2 (recommended)
  dockerhub_image: ""                    # only used when image_source: "dockerhub"
```

---

## 11. Istruzioni di Esecuzione

Le tre modalità di deploy condividono gli stessi passi di setup iniziale (download e partizionamento del dataset), che vengono eseguiti sempre sulla **macchina locale** dell'operatore. Differiscono nel passo di avvio e nella raccolta delle metriche.

| Modalità | Istanze EC2 | Container | Comunicazione |
|---|:---:|:---:|---|
| Locale | 0 | `num_workers` + 1 | rete Docker interna (loopback) |
| Singola EC2 | 1 | `num_workers` + 1 | rete Docker interna (loopback) |
| Multi-instance EC2 | `num_workers` + 1 | 1 per istanza | TCP/IP reale tra istanze VPC |

> **Nota di compatibilità — patch automatiche a codice LEAF.**  
> `download_femnist.py` applica due patch a file di LEAF subito dopo il clone, prima di eseguire il preprocessing — non è richiesto alcun intervento manuale:
> - **`data_to_json.py`**: `Image.ANTIALIAS → Image.LANCZOS` (rimosso in Pillow 10.0, ottobre 2023).
> - **`get_data.sh`**: `unzip <file>` → `python3 -c "import zipfile; zipfile.ZipFile(...).extractall('.')"` (`unzip` assente su alcuni sistemi Linux/WSL; estrazione silenziosa, attesa normale di 5–10 minuti).
>
> Le patch sono transenti: scompaiono con la directory `leaf/` al termine del preprocessing. Vedere la nota nella descrizione di `download_femnist.py` per i dettagli.

---

### 11.1 Setup Iniziale (tutte le modalità — gira sulla macchina locale)

**Prerequisiti della macchina locale:** Docker + Docker Compose, Python 3.11+, `git` (usato dal Passo 2 per clonare il repository LEAF). Per le modalità AWS è richiesto anche Terraform (v. Sezioni 11.3 e 11.4).

Questi passi vanno ripetuti ogni volta che cambia `num_workers` o `use_test_set`.

**Passo 1 — Configurazione**

Editare `config.yaml`:
- `num_workers`: numero di worker (es. 3 per la ricerca iperparametri, poi 5 e 8 per la scalabilità)
- `use_test_set`: `false` = split 90/10 train/val; `true` = split 80/10/10 train/val/test
- tutti gli altri iperparametri (learning rate, fanout, ecc.)

**Passo 2 — Download dataset** *(una-tantum, o quando cambia `use_test_set`)*

```bash
# Eseguito sulla macchina locale — scarica FEMNIST da LEAF (~900 MB immagini)
# e produce data/femnist/data/train/*.json e data/femnist/data/test/*.json
python scripts/download_femnist.py
# Con --sf 0.05 per verifiche rapide di installazione (non per risultati da riportare)
```

`download_femnist.py` clona LEAF, applica due patch di compatibilità a codice LEAF (`data_to_json.py` per Pillow ≥ 10 e `get_data.sh` per sistemi senza `unzip`), lancia `preprocess.sh` con i parametri letti da `config.yaml` (`--tf 0.9` o `0.8` in base a `use_test_set`), copia i JSON in `data/femnist/data/`, e rimuove LEAF.

**Passo 3 — Partizionamento e generazione compose** *(ripetere se `num_workers` o `use_test_set` cambia)*

```bash
# split_dataset.py legge data/femnist/data/ e produce una directory per worker:
#   data/femnist/worker_0/train/data.json
#   data/femnist/worker_0/val/data.json   (e test/ se use_test_set: true)
#   data/femnist/worker_1/...
python scripts/split_dataset.py

# generate_compose.py legge config.yaml e genera docker-compose.yml
# con N servizi worker + 1 servizio registry, bind mount corretti, healthcheck
python scripts/generate_compose.py
```

I dati vengono divisi tra i worker per scrittore (non-i.i.d.): ogni worker possiede un sottoinsieme di writer con il loro stile di scrittura, senza sovrapposizioni.

---

### 11.1.1 Ciclo degli Esperimenti

La campagna sperimentale si articola in **tre fasi annidate**. La tabella seguente indica quali passi del setup vanno ri-eseguiti in funzione di cosa cambia in `config.yaml` — tutto il resto viene riutilizzato dal run precedente:

| Cosa cambia in `config.yaml` | `download_femnist.py` | `split_dataset.py` | `generate_compose.py` |
|---|:---:|:---:|:---:|
| Solo parametri ML (`lr`, `H`, `fanout`, `batch_size`, ecc.) | no | no | no |
| `num_workers` | no | **sì** | **sì** |
| `use_test_set` | **sì** | **sì** | **sì** |

> **Multi-instance EC2**: ogni variazione di `num_workers` richiede anche `aws_deploy.py destroy` → `provision` per ricreare le istanze nel numero corretto prima di `deploy`.

---

**Fase 1 — Ricerca iperparametri** (`num_workers` fisso, es. 3; `use_test_set: false`)

Ripetere per ogni combinazione di iperparametri (griglia su `lr`, `gossip_fanout`, `inner_steps_H`, ecc.):

| Passo | Chi | Dove | Locale / Singola EC2 | Multi-instance EC2 |
|---|---|---|---|---|
| 1. Configura | Operatore | locale | Editare `config.yaml` — variare solo parametri ML | identico |
| 2. [Setup] | — | — | Nessun re-setup: dati e compose già validi | identico |
| 3. Avvia training | Operatore | locale | `docker compose up --build` | `python scripts/aws_deploy.py deploy` |
| 4. Training | Container × N | locale / N EC2 | automatico (round: A → B → C) | identico |
| 5. Fine training | Container × N | locale / N EC2 | automatico: checkpoint + deregistra | identico |
| 6. Collect | — | — | — *(metriche già in `data/femnist/worker_*/`)* | `python scripts/aws_deploy.py collect` |
| 7. Aggrega | Operatore | locale | `python scripts/aggregate_metrics.py` | identico |
| 8. Archivia | Operatore | locale | `python scripts/save_experiment.py <nome>` *(es. `lr_1e-3_fanout3`)* | identico |
| 9. Ripeti | Operatore | — | tornare al passo 1 con la prossima combinazione | identico |

`save_experiment.py` archivia `config.yaml` + metriche + log container in `results/<timestamp>_<nome>/` e rimuove i CSV dalla directory di lavoro, così il prossimo run parte da zero. Va eseguito **prima** di `docker compose down` per garantire che i log dei container siano ancora accessibili.

---

**Fase 2 — Studio di scalabilità** (config ottimale; `num_workers` varia 3 → 5 → 8; `use_test_set: false`)

Una volta individuata la configurazione migliore dalla Fase 1, ripetere per ciascun valore di `num_workers`:

| Passo | Chi | Dove | Locale / Singola EC2 | Multi-instance EC2 |
|---|---|---|---|---|
| 1. Configura | Operatore | locale | `config.yaml`: `num_workers: <N>` | identico |
| 2. Re-partiziona | Operatore | locale | `python scripts/split_dataset.py` | identico |
| 3. Rigenera compose | Operatore | locale | `python scripts/generate_compose.py` | identico |
| 4. Infrastruttura | — | — | — | `aws_deploy.py destroy` → `provision` |
| 5. Avvia training | Operatore | locale | `docker compose up --build` | `python scripts/aws_deploy.py deploy` |
| 6. Training | Container × N | locale / N EC2 | automatico | identico |
| 7. Collect | — | — | — | `python scripts/aws_deploy.py collect` |
| 8. Aggrega | Operatore | locale | `python scripts/aggregate_metrics.py` | identico |
| 9. Archivia | Operatore | locale | `python scripts/save_experiment.py scalability_N<N>` | identico |
| 10. Ripeti | Operatore | — | tornare al passo 1 con il prossimo N | identico |

---

**Fase 3 — Valutazione finale con test set** (config e `num_workers` ottimali; una sola volta a campagna conclusa)

Il test set è tenuto fuori da ogni decisione di training e hyperparameter selection. Va eseguito **una sola volta** dopo aver completato le Fasi 1 e 2:

| Passo | Chi | Dove | Locale / Singola EC2 | Multi-instance EC2 |
|---|---|---|---|---|
| 1. Abilita test set | Operatore | locale | `config.yaml`: `use_test_set: true` | identico |
| 2. Re-download | Operatore | locale | `python scripts/download_femnist.py` *(cambia `--tf` LEAF: 0.9 → 0.8)* | identico |
| 3. Re-partiziona | Operatore | locale | `python scripts/split_dataset.py` | identico |
| 4. Rigenera compose | Operatore | locale | `python scripts/generate_compose.py` | identico |
| 5. Avvia training | Operatore | locale | `docker compose up --build` | `python scripts/aws_deploy.py deploy` |
| 6. Training | Container × N | locale / N EC2 | automatico — al termine: `test_result.json` | identico |
| 7. Collect | — | — | — | `python scripts/aws_deploy.py collect` |
| 8. Aggrega | Operatore | locale | `python scripts/aggregate_metrics.py` *(stampa val + test accuracy)* | identico |
| 9. Archivia | Operatore | locale | `python scripts/save_experiment.py final_with_test` | identico |

`test_result.json` (scritto da ogni worker alla fine del training) e `test_accuracy` nell'output di `aggregate_metrics.py` sono la metrica definitiva da riportare — non influenzata da nessuna decisione di training o selezione degli iperparametri.

---

### 11.2 Modalità Locale

**Chi esegue cosa e in che ordine:**

| Passo | Chi | Comando | Risultato |
|---|---|---|---|
| Setup | Operatore (locale) | passi 1–3 sopra | dataset partizionato, compose generato |
| Avvio | Docker Engine (locale) | `docker compose up --build` | build immagine `fl-worker` (una sola), avvio N+1 container |
| Training | Container worker (N) | automatico | ogni worker allena, fa gossip gRPC con gli altri via rete Docker |
| Discovery | Container registry (1) | automatico | Flask server, gestisce register/deregister/peers |
| Fine | Container worker (N) | automatico | ogni worker scrive checkpoint e si deregistra |
| Analisi | Operatore (locale) | `aggregate_metrics.py` | legge le metriche, produce statistiche globali |

**Avvio:**
```bash
docker compose up --build
```

Docker costruisce l'immagine `fl-worker` una sola volta (condivisa da tutti i worker) e avvia i container. Il registry parte per primo; i worker aspettano il suo healthcheck prima di registrarsi.

**Dove finiscono le metriche:**

Ogni worker scrive `metrics.csv` in `/app/data/femnist/` dentro il container. Grazie al bind mount (`./data/femnist/worker_i → /app/data/femnist`), il file appare immediatamente sull'host in:
```
data/femnist/worker_0/metrics.csv
data/femnist/worker_1/metrics.csv
...
data/femnist/worker_0/model_final.pt   ← snapshot finale dei pesi (solo per weight divergence)
data/femnist/worker_0/test_result.json ← solo se use_test_set: true
```

**Raccolta e analisi metriche:**
```bash
python scripts/aggregate_metrics.py
# Legge tutti i data/femnist/worker_*/metrics.csv
# Produce: data/femnist/global_metrics.csv  (per-round mean/std/min/max accuracy)
#          data/femnist/summary.txt          (riassunto per worker)

python scripts/save_experiment.py <nome>
# Archivia config.yaml + tutte le metriche in results/<timestamp>_<nome>/
# Pulisce i metrics.csv per il prossimo esperimento
```

---

### 11.3 Modalità Singola EC2

Il workflow è **identico al locale** — stessi script, stessa immagine Docker, stessa rete Docker interna. La differenza è solo dove girano i container. Come per la modalità multi-instance, l'istanza è gestita tramite **Terraform** (`terraform/single/`): creazione, installazione di Docker e distruzione avvengono automaticamente senza toccare la console AWS.

**Prerequisiti:**
- Sessione Learner Lab attiva (indicatore verde nel pannello AWS Academy)
- Credenziali AWS esportate nella shell locale (pannello AWS Academy → AWS Details → Show):
  ```bash
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_SESSION_TOKEN=...
  ```
- Key pair: `vockey` (us-east-1) o nuova key pair in us-west-2; PEM scaricato da AWS Details → Download PEM
- Terraform installato sulla macchina locale
- `git` installato sulla macchina locale (usato da `scripts/download_femnist.py` — Passo 2 — per clonare LEAF)
- `config.yaml`: parametri `aws.*` rilevanti per questa modalità:

  | Parametro | Default | Note |
  |---|:---:|---|
  | `aws.key_name` | `vockey` | Nome key pair EC2 |
  | `aws.key_path` | `~/Downloads/labsuser.pem` | Path locale al PEM |
  | `aws.region` | `us-east-1` | Regione AWS |
  | `aws.availability_zone` | `us-east-1a` | AZ dell'istanza; non influisce su costi (tutto il traffico è sulla rete Docker interna) |
  | `aws.instance_type_single` | `t3.large` | Tipo istanza; t3.large (8 GB) regge fino a 8 worker con dataset completo |
  | `aws.volume_size_single` | `20` | Disco EBS in GB; 20 GB copre tutti i casi; aumentare a 30 GB con 8 worker e dataset completo |

> **Nota:** `aggregate_metrics.py` e `save_experiment.py` girano direttamente **sull'host EC2** (fuori dai container) e richiedono `pip install -r requirements.debug.txt` sull'host.

**Chi esegue cosa e in che ordine:**

| Passo | Chi | Dove | Comando |
|---|---|---|---|
| Setup (passi 1–3) | Operatore | **macchina locale** | `download_femnist.py`, `split_dataset.py`, `generate_compose.py` |
| `provision_single` | `aws_deploy.py` + Terraform | **locale → AWS** | crea 1 istanza EC2 (Ubuntu 22.04, `t3.large`), installa Docker via `user_data`, attende SSH ready |
| Upload progetto | Operatore | **locale → EC2** | `scp -r . ubuntu@<ip>:~/project` |
| Dipendenze host | Operatore (via SSH) | **EC2** | `pip install -r requirements.debug.txt` |
| Avvio | Operatore (via SSH) | **EC2** | `docker compose up --build` |
| Training | Container (N+1) | **EC2** | automatico, identico al caso locale |
| Analisi | Operatore (via SSH) | **EC2** | `aggregate_metrics.py`, `save_experiment.py` |
| `destroy_single` | `aws_deploy.py` + Terraform | **locale → AWS** | termina l'istanza EC2, rimuove il security group |

```bash
# Esportare le credenziali (ogni sessione Learner Lab)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

# Provisioning: Terraform crea l'istanza EC2 e installa Docker automaticamente
python scripts/aws_deploy.py provision_single

# Upload del progetto (inclusa la cartella data/ già partizionata) e avvio
scp -r . ubuntu@<ip>:~/project
ssh -i ~/Downloads/labsuser.pem ubuntu@<ip>
cd ~/project

# Installa dipendenze host (una sola volta per istanza) — servono per gli script di analisi
pip install -r requirements.debug.txt

# Avvio training
docker compose up --build

# A fine training: analisi direttamente sull'host EC2
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <nome>

# Distruggere l'istanza per fermare la fatturazione
python scripts/aws_deploy.py destroy_single
```

**Sessione scaduta durante il training (`resume_single`):** la sessione dura 4 ore ma può essere rinnovata cliccando **Start Lab** di nuovo *prima* che il timer scada — questo è il modo più semplice per evitare interruzioni su run lunghi. Se la sessione scade comunque prima di `destroy_single`, AWS stoppa l'istanza — i dati su disco (dataset, `metrics.csv` parziale) sopravvivono, ma i container Docker si fermano. Alla riapertura della sessione l'istanza riparte con un nuovo IP pubblico:

```bash
python scripts/aws_deploy.py resume_single   # aggiorna tfstate, stampa nuovo IP

# Caso A — training finito: analizza e distruggi
ssh -i ~/Downloads/labsuser.pem ubuntu@<nuovo_ip>
  cd ~/project && python scripts/aggregate_metrics.py && python scripts/save_experiment.py <nome>
python scripts/aws_deploy.py destroy_single

# Caso B — training era in corso (stato modello perso, nessun checkpoint): riparte dal round 1
ssh -i ~/Downloads/labsuser.pem ubuntu@<nuovo_ip>
  cd ~/project && docker compose up
python scripts/aws_deploy.py destroy_single
```

**Security group:** solo la porta 22 (SSH) deve essere esposta verso l'esterno. Le comunicazioni gRPC tra worker e registry avvengono sulla rete bridge interna di Docker — non richiedono regole ingress aggiuntive.

---

### 11.4 Modalità Multi-Instance EC2

Questa è l'unica modalità in cui i worker comunicano su **TCP/IP reale** tra macchine fisicamente separate, rendendo le misure di tempo di convergenza significative.

**Prerequisiti:**
- **Sessione Learner Lab attiva** (indicatore verde nel pannello AWS Academy → Start Lab)
- **Credenziali AWS** (da esportare all'inizio di ogni sessione Learner Lab): pannello AWS Academy → AWS Details → Show → copiare `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` ed esportarli nella shell locale:
  ```bash
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_SESSION_TOKEN=...
  ```
- **Key pair**: in us-east-1 usare la `vockey` predefinita (AWS Details → Download PEM → `~/Downloads/labsuser.pem`); in us-west-2 creare una nuova key pair dalla console EC2
- **Terraform** installato sulla macchina locale
- **`git`** installato sulla macchina locale (usato da `scripts/download_femnist.py` — Passo 2 — per clonare LEAF)
- `config.yaml`: parametri `aws.*` rilevanti per questa modalità:

  | Parametro | Default | Note |
  |---|:---:|---|
  | `aws.key_name` | `vockey` | Nome key pair EC2 |
  | `aws.key_path` | `~/Downloads/labsuser.pem` | Path locale al PEM |
  | `aws.region` | `us-east-1` | Regione AWS |
  | `aws.availability_zone` | `us-east-1a` | AZ di tutte le istanze; intra-AZ è gratuito, cross-AZ costa $0.01/GB |
  | `aws.instance_type_worker` | `t3.small` | Tipo istanza worker; t3.small (2 GB) è sufficiente per tutti i `num_workers` |
  | `aws.instance_type_registry` | `t3.micro` | Tipo istanza registry; t3.micro (1 GB) è sempre sufficiente |
  | `aws.volume_size_worker` | `20` | Disco EBS worker in GB; range consigliato: 15–30 GB |
  | `aws.volume_size_registry` | `8` | Disco EBS registry in GB; 8 GB è ampiamente sufficiente |

- Le istanze e Docker **non vanno creati manualmente**: `aws_deploy.py provision` invoca Terraform che crea le istanze EC2 e installa Docker automaticamente via `user_data`

**Chi esegue cosa e in che ordine:**

| Passo | Chi | Dove | Cosa succede |
|---|---|---|---|
| Setup (passi 1–3) | Operatore | **locale** | dataset scaricato e partizionato come sempre |
| `provision` | `aws_deploy.py` + Terraform | **locale → AWS** | crea `num_workers + 1` istanze EC2, security group, installa Docker via `user_data` |
| `deploy` [1/5] | `aws_deploy.py` | **locale → EC2** | aspetta SSH + Docker ready su tutte le istanze |
| `deploy` [2/5] | `aws_deploy.py` | **locale → EC2** | SCP del codice sorgente, `docker build` in parallelo su tutte le istanze (~5-8 min prima volta) |
| `deploy` [3/5] | `aws_deploy.py` | **locale → EC2 worker** | SCP della partizione `worker_i/` sull'EC2 corrispondente |
| `deploy` [4/5] | `aws_deploy.py` | **EC2 registry** | avvia container registry, attende healthcheck `/peers` |
| `deploy` [5/5] | `aws_deploy.py` | **EC2 worker × N** | avvia container worker su ogni EC2, con mount della propria partizione e IP privato come `MY_HOST` |
| Training | Container worker (N) | **N EC2 distinte** | allena localmente, fa gossip gRPC tra EC2 via IP privati VPC |
| Discovery | Container registry (1) | **EC2 registry** | gestisce peer list durante il training |
| `collect` | `aws_deploy.py` | **EC2 → locale** | SCP di `metrics.csv`, `model_final.pt`, `test_result.json` da ogni EC2 worker |
| Analisi | Operatore | **locale** | `aggregate_metrics.py`, `save_experiment.py` |
| `destroy` | `aws_deploy.py` + Terraform | **locale → AWS** | termina tutte le istanze EC2, rimuove security group |

```bash
# Esportare le credenziali (ogni sessione Learner Lab)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...

# Provisioning: Terraform crea num_workers+1 istanze EC2 e installa Docker
python scripts/aws_deploy.py provision

# Deploy: build immagini + upload dati + avvio container (tutto automatico)
python scripts/aws_deploy.py deploy

# Monitoring durante il training (opzionale)
python scripts/aws_deploy.py status         # docker ps su ogni istanza
python scripts/aws_deploy.py logs 0         # tail log worker_0 (Ctrl+C per uscire)
python scripts/aws_deploy.py logs registry  # tail log registry

# Raccolta metriche a fine training
python scripts/aws_deploy.py collect        # SCP metrics.csv da ogni EC2 → locale

# Analisi (identica alle altre modalità — le metriche sono ora in locale)
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <nome>    # es. scalability_aws_N5

# Distruggere le istanze per fermare la fatturazione
python scripts/aws_deploy.py destroy
```

**Dove finiscono le metriche:**

Durante il training ogni worker scrive in `/app/data/femnist/` dentro il suo container → per bind mount in `/home/ubuntu/data/femnist/worker_i/` sull'EC2. Il comando `collect` trasferisce questi file sulla macchina locale in `data/femnist/worker_i/`, esattamente dove se li aspetta `aggregate_metrics.py` — il passo di analisi è quindi identico per tutte e tre le modalità.

**Sessione Learner Lab scaduta durante il training (`resume`):**

La sessione dura circa 4 ore ma può essere rinnovata cliccando **Start Lab** di nuovo prima che il timer scada — il modo più semplice per non interrompere un training lungo. Se la sessione scade comunque prima di `destroy`, AWS **stoppa** automaticamente le istanze (non le termina: i dati su disco EBS sopravvivono). Alla riapertura della sessione, le istanze vengono riavviate con **nuovi IP pubblici**. Il file `terraform.tfstate` contiene ancora i vecchi IP, quindi i comandi `collect`, `status` e `logs` si connetterebbero agli indirizzi sbagliati.

`resume` esegue `terraform apply -refresh-only`: interroga AWS, aggiorna il file di stato con i nuovi IP e li stampa. Non modifica l'infrastruttura e non riavvia nessun container.

Esempio concreto:

```
Lunedì 14:00  provision + deploy → training avviato
               worker_0: 54.1.2.3 | worker_1: 54.4.5.6 | registry: 54.7.8.9
               (salvati in terraform.tfstate)

Lunedì 18:00  sessione Learner Lab scade (limite 4h)
               → AWS stoppa le istanze automaticamente
               → container Docker fermati; metrics.csv parziale su disco (EBS)

Martedì       nuova sessione → istanze riavviate con NUOVI IP
               worker_0: 18.9.8.7 | worker_1: 18.2.3.4 | registry: 18.5.6.7
               (terraform.tfstate dice ancora i vecchi IP)

$ python scripts/aws_deploy.py resume
  → aggiorna tfstate | stampa: worker_0: 18.9.8.7, worker_1: 18.2.3.4, ...
```

Da qui si procede in base a cosa è successo durante la sessione scaduta:

```bash
# Caso A — il training era già finito prima della scadenza
#          (metrics.csv completo su disco, collect funziona normalmente)
python scripts/aws_deploy.py collect
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <nome>
python scripts/aws_deploy.py destroy

# Caso B — il training era ancora in corso (stato del modello in RAM: perso)
#          Non esiste checkpointing: il training deve ripartire dal round 1.
python scripts/aws_deploy.py deploy    # re-upload sorgenti, riavvia container
# ... aspetta fine training ...
python scripts/aws_deploy.py collect
python scripts/aws_deploy.py destroy
```

> Se si esegue sempre `destroy` prima che la sessione scada, `resume` non è mai necessario. È uno strumento di recupero per i casi in cui il training superi il limite di sessione.

---

### 11.5 Analisi delle Metriche (comune a tutte le modalità)

```bash
python scripts/aggregate_metrics.py
python scripts/save_experiment.py <nome>
```

`aggregate_metrics.py` produce in output (su stdout e in `summary.txt`):

| Sezione | Cosa mostra |
|---|---|
| **Per-round table** | `round` \| `mean_acc` \| `std_acc` \| `min_acc` \| `max_acc` \| `PhaseA(s)` \| `PhaseB(s)` \| `PhaseC(s)` |
| **Per-worker summary** | accuracy finale e migliore, total training time (somma `round_duration_s`), breakdown medio per fase, latenza gRPC media |
| **System convergence** | per ogni worker: *converged at round X* oppure *hit round limit* + wall-clock reale dal timestamp; poi: verdetto del sistema (*YES — all workers converged* o *PARTIAL*) e wall-clock totale del sistema (dal primo worker start all'ultimo worker end) |
| **Weight divergence** | distanza L2 tra i pesi finali di ogni coppia di worker (se i `model_final.pt` sono presenti) |
| **Test set results** | `test_accuracy` per worker, solo se `use_test_set: true` |

`global_metrics.csv` contiene le stesse colonne per-round e può essere usato per grafici di convergenza.

`save_experiment.py` archivia in `results/<timestamp>_<nome>/`: `config.yaml`, `global_metrics.csv`, `summary.txt`, `worker_*/metrics.csv`, `worker_*/test_result.json`, `logs/<service>.log` per ogni container — poi pulisce la directory di lavoro per il prossimo run. Va eseguito prima di `docker compose down`.

### Confronto tra approccio 90/10 e 80/10/10

Per quantificare il bias ottimistico introdotto dall'assenza di un test set separato, eseguire due run con la stessa configurazione di iperparametri cambiando solo `use_test_set`:

```bash
# Run A — solo val (approccio di default)
# config.yaml: use_test_set: false
python scripts/download_femnist.py   # dataset completo
python scripts/split_dataset.py && python scripts/generate_compose.py
docker compose up --build
python scripts/aggregate_metrics.py  # riporta val_accuracy

# Run B — con test set indipendente
# config.yaml: use_test_set: true
python scripts/download_femnist.py   # re-download necessario — vedi nota sotto
python scripts/split_dataset.py && python scripts/generate_compose.py
docker compose up --build
python scripts/aggregate_metrics.py  # riporta val_accuracy + test_accuracy
```

**Perché il re-download è obbligatorio quando si cambia `use_test_set`.** Il rapporto train/test non è un parametro di `split_dataset.py` ma di LEAF stesso: `download_femnist.py` invoca lo script LEAF con `--tf 0.9` (con `use_test_set: false`) o `--tf 0.8` (con `use_test_set: true`), e LEAF bake il rapporto dentro i file JSON prodotti — ogni writer ha già le proprie immagini pre-assegnate a `train/` o `test/` nel momento in cui i JSON vengono scritti su disco. Con `--tf 0.9`, la cartella `data/femnist/data/test/` contiene esattamente il 10% dei campioni di ogni writer; con `--tf 0.8`, ne contiene il 20%. `split_dataset.py` con `use_test_set: true` divide questa seconda cartella al 50/50 per scrittore per ottenere 10% val + 10% test. Se si cambia `use_test_set` senza re-download, `split_dataset.py` opererebbe su una cartella `test/` costruita con il rapporto sbagliato: i JSON su disco non rifletterebbero la proporzione richiesta, e il risultato sarebbe silenziosamente errato (es. 5% val + 5% test invece di 10% + 10%).

**Motivazione ML.** Il re-download non è solo un dettaglio implementativo: riflette un principio fondamentale della valutazione in ML. Con `use_test_set: false` il sistema usa lo stesso 10% di LEAF per due scopi distinti — early stopping round per round e confronto finale tra configurazioni — introducendo un doppio bias ottimistico. Aggiungere un test set separato richiede necessariamente di sottrarre dati al training (da 90% a 80%): non esiste un modo per avere un test set indipendente senza ridurre i dati di addestramento, perché i campioni totali sono fissi. Il trade-off è inevitabile: più dati al training → metriche finali più ottimistiche (bias non eliminato); meno dati al training → metriche finali più oneste ma modello potenzialmente meno capace. Il re-download rende questo trade-off esplicito e controllato, invece di lasciarlo implicito nella scelta di `use_test_set`.

La differenza tra `val_accuracy` (Run A) e `test_accuracy` (Run B) è indicativa del bias ottimistico, ma non lo misura con precisione: Run B allena su **80% dei dati** invece del 90% di Run A, quindi la `test_accuracy` sarà probabilmente più bassa per due motivi sovrapposti — meno dati di training (effetto reale) e assenza del bias ottimistico (effetto che si vuole isolare). I due contributi non sono separabili. Ciò che Run B garantisce è che la `test_accuracy` è una stima onesta della generalizzazione di quella configurazione su dati mai visti in nessuna decisione di training.

---

## 12. Target di Accuracy e Scalabilità Attesa

Questa sezione raccoglie i valori di riferimento dalla letteratura, le aspettative teoriche per ogni parametro del sistema, e lo schema delle tabelle che verranno popolate con i risultati sperimentali reali al completamento della campagna di esperimenti (Sezione 7).

### 12.1 Valori di Riferimento dalla Letteratura

#### FEMNIST: difficoltà del task

FEMNIST è il benchmark FL non-i.i.d. più usato in letteratura. Con 62 classi (10 cifre + 26 maiuscole + 26 minuscole) e alta variabilità interstile, è intrinsecamente più difficile di MNIST (10 classi, scrittura più uniforme). A titolo di confronto:

- **Accuracy umana su EMNIST-62**: ~96–98% (con tempo sufficiente per disambiguare classi simili come `0`/`O`, `1`/`l`/`I`)
- **CNN single-device su FEMNIST completo** (no FL, dati i.i.d.): ~85–92%, a seconda dell'architettura e del training budget
- **CNN single-device su dati non-i.i.d. locali** (1 solo worker, nessun gossip): ~72–82%, perché il modello è esposto a un sottoinsieme di stili di scrittura

#### Valori riportati in letteratura per FL su FEMNIST

| Metodo | Setting | Accuracy riportata | Note |
|---|---|:---:|---|
| FedAvg [2] | 100 round, 2 epoche locali, 10% partecipazione | ~77–80% | LEAF split 90/10, non-i.i.d. per writer |
| FedProx (Li et al., 2020) | stesso setup di FedAvg | ~79–83% | μ=0.01 proximal term |
| SCAFFOLD (Karimireddy et al., 2020) | riduzione del client drift | ~82–87% | controllo varianza del gradiente |
| Local (no FL, LEAF paper [3]) | training isolato per client | ~60–70% | baseline LEAF su subset ridotto |
| **Questo progetto** | **3 worker, H=500, fanout=1, 200 round** | **TODO** | **popolate con i risultati reali** |

> **Nota metodologica.** I valori in letteratura sono spesso ottenuti su configurazioni diverse (numero di client, frazione di partecipazione, dimensione dei dati locali). Il confronto diretto richiede cautela: la nostra configurazione (3 worker, tutto il dataset diviso in 3) differisce significativamente da un deployment con 100+ client su subset piccoli. L'obiettivo non è superare lo stato dell'arte, ma dimostrare che il gossip P2P converge a risultati comparabili al FL centralizzato su questa scala.

#### Osservazione dai run di sviluppo

Dai run di sviluppo su dataset completo con 3 worker, al round 14-16 l'accuracy è già ~84–85%. Questo suggerisce che la configurazione attuale è ben calibrata e i valori finali si attesteranno plausibilmente nella fascia **85–88%** — nella norma per un FL su FEMNIST con training sufficientemente lungo e architettura ben regolarizzata.

Un risultato **superiore a 80%** è da considerarsi buono e competitivo con FedAvg centralizzato su questa scala. Un risultato **superiore a 85%** è eccellente e dimostra che il protocollo gossip P2P non perde qualità rispetto all'aggregazione centralizzata con N=3 worker.

---

### 12.2 Scaling con il Numero di Worker (`num_workers`)

**Teoria:** aggiungere worker ha due effetti opposti.

**Effetto positivo:** ogni worker copre una porzione diversa dello spazio degli stili di scrittura. Più worker → copertura più ampia → ogni modello, dopo l'aggregazione FedAvg, ha "visto" (indirettamente, via gossip) feature di più scrittori → migliore generalizzazione. Il modello finale tende verso una soluzione più vicina all'ottimo globale su tutti i 3.597 writer.

**Effetto negativo:** con più worker, le partizioni locali diventano più piccole e più eterogenee. La distanza tra le distribuzioni locali cresce: il modello di Worker 0 e quello di Worker 7 (su un dataset a 8 worker) hanno visto stili completamente diversi. La media FedAvg di modelli molto divergenti produce un ibrido che non funziona bene su nessuna partizione — il **client drift** si amplifica.

**Rendimento marginale decrescente:** il beneficio di aggiungere il 4° worker è inferiore a quello del 3°, e così via. Con N molto grande (e fanout piccolo), i modelli locali divergono così tanto che le aggregazioni potrebbero non convergere in un numero finito di round.

**Attese quantitative per la nostra configurazione (dataset completo):**

| `num_workers` | Campioni/worker | `mean_accuracy` finale attesa | Rounds a convergenza | Volume comunicazione (fanout=1, 200 round) |
|:---:|:---:|:---:|:---:|:---:|
| 3 | ~245k | **85–88%** | ~20–50 round | ~3.9 GB |
| 5 | ~147k | **84–87%** | ~25–60 round | ~6.5 GB |
| 8 | ~92k | **82–86%** | ~30–80 round | ~10.4 GB |

> **TODO:** Sostituire con i valori reali degli Esperimenti 4a (N=3), 4b (N=5), 4c (N=8).

**Come interpretare la `std_accuracy`:** con più worker e dati più eterogenei, ci si aspetta una deviazione standard leggermente più alta. Un sistema ben calibrato mantiene `std_accuracy < 5%` anche con 8 worker; valori superiori al 10% indicano che alcuni worker convergono bene e altri no — segnale di fanout troppo basso o H troppo alto rispetto all'eterogeneità dei dati.

**Durata per round vs N:** la durata di un round è dominata dalla Fase B (H inner steps di training locale) e non dipende da N — questo è il principale vantaggio del gossip P2P rispetto al FL centralizzato. In FL centralizzato il server deve aggregare N modelli ad ogni round, diventando un collo di bottiglia: il tempo per round cresce con N. Nel gossip P2P ogni worker aggrega solo i modelli che riceve (al più `gossip_fanout` per round), indipendentemente da N.

---

### 12.3 Scaling con il Gossip Fanout (`gossip_fanout`)

`gossip_fanout` è il parametro centrale del progetto: controlla esattamente il trade-off traffico/qualità di aggregazione.

**Teoria — velocità di propagazione dell'informazione:**

Con N=3 worker e fanout=1, ogni worker invia a 1 peer casuale per round. La probabilità che un modello aggiorni tutti gli altri worker cresce lentamente: in attesa che ogni worker venga raggiunto. Con fanout=N-1=2, ogni worker invia a entrambi gli altri ad ogni round — propagazione massima, ogni worker aggrega da tutti gli altri ogni round. Il vantaggio del fanout alto si riduce all'aumentare di N, dove N-1 diventa costoso.

**Attese quantitative (N=3, H=500, dataset completo):**

| `gossip_fanout` | Messaggi/round per worker | `mean_accuracy` attesa | Rounds a convergenza | Volume totale (200 round) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | **85–87%** | ~30–60 round | ~3.9 GB |
| 2 (= N-1) | 2 | **86–88%** | ~15–35 round | ~7.8 GB |

> Con N=3, i valori significativi di fanout sono solo 1 e 2 (N-1). Fanout=3 con N=3 è equivalente a mandare a tutti + sé stesso — non ha senso.  
> **TODO:** Sostituire con i valori reali degli Esperimenti 3c (fanout=1, fanout=2).

**Differenza attesa tra fanout=1 e fanout=N-1:** con N=3 la differenza di fanout è solo 2× nel numero di messaggi, ma la qualità di aggregazione può variare significativamente nei round iniziali. Con fanout=1 è possibile che un worker non riceva aggiornamenti per 2-3 round consecutivi (per pura casualità della selezione random), rallentando la convergenza. Con fanout=N-1 ogni worker aggrega sempre tutti gli altri: convergenza più rapida nei primi round, poi entrambe le configurazioni tendono allo stesso valore asintotico.

**Il "knee" del trade-off:** il punto di rendimento marginale decrescente su `gossip_fanout` è di grande interesse pratico — è la configurazione che massimizza l'accuracy ottenuta per unità di traffico di rete. Con N=3 la curva ha solo 2 punti, ma con N=8 (Esperimento 4) diventa possibile tracciare la curva completa: fanout ∈ {1, 2, 3, 4, 7} producono accuracy crescente e traffico crescente, e il punto in cui il guadagno marginale di accuracy si azzera è la configurazione ottimale per deployment su rete vincolata.

---

### 12.4 Scaling con gli Inner Steps (`inner_steps_H`)

**Teoria — client drift:**

H è il numero di gradient steps locali tra due gossip push. Con H grande, il modello di ogni worker si muove lungo la direzione del gradiente locale per molti passi prima di sincronizzarsi: i modelli divergono significativamente nello spazio dei pesi. La media FedAvg di modelli molto divergenti è meno accurata della media di modelli vicini — fenomeno noto come **client drift** (deriva del client).

La tensione è:
- **H piccolo** → modelli allineati, aggregazione di qualità alta, ma traffico $\propto 1/H$ più alto
- **H grande** → risparmio di comunicazione, ma drift crescente e qualità dell'aggregazione in calo

Con dati non-i.i.d. (come FEMNIST) il drift è amplificato rispetto al caso i.i.d.: ogni worker ottimizza per la distribuzione dei *propri* scrittori, e la direzione del gradiente locale può essere opposta a quella di un altro worker.

**Attese quantitative (N=3, fanout=1, dataset completo):**

| `inner_steps_H` | Epoche equiv. (Worker 0, ~210k campioni) | Drift atteso | `mean_accuracy` attesa | Volume/round |
|:---:|:---:|:---:|:---:|:---:|
| 100 | ~0.015 epoche | basso | **85–88%** | ~6.5 MB |
| 500 (default) | ~0.076 epoche | medio | **85–87%** | ~6.5 MB |
| 1000 | ~0.153 epoche | alto | **83–86%** | ~6.5 MB |

> Il volume per round non cambia con H: ogni gossip push trasmette lo stesso modello (~6.5 MB) indipendentemente da quanti step ha compiuto. Quello che cambia è la *frequenza* del push, non la dimensione. Il traffico totale per ottenere un certo numero di campioni elaborati cambia: con H=100, per 50.000 step occorrono 500 push; con H=1000 bastano 50 push. Con H=100, il beneficio alla convergenza (modelli sempre allineati) potrebbe non compensare il costo di 10× più push.

> **TODO:** Sostituire con i valori reali degli Esperimenti 3b (H=100, H=500, H=1000). La curva `mean_accuracy` vs `H` è uno dei grafici chiave della relazione.

---

### 12.5 Sintesi: Cosa Costituisce un Buon Risultato

Tenendo conto della letteratura e delle aspettative teoriche, i criteri di valutazione sono:

**Criterio 1 — Accuracy assoluta:**

| Livello | `mean_accuracy` finale | Giudizio |
|---|:---:|---|
| Eccellente | ≥ 86% | Competitivo con FL centralizzato su questa scala |
| Buono | 82–85% | Nella norma per gossip FL non-i.i.d. con N=3 |
| Accettabile | 78–81% | Inferiore al FL centralizzato ma superiore al no-FL baseline |
| Insufficiente | < 78% | La gossip aggregation non porta beneficio significativo rispetto al training isolato |

**Criterio 2 — Vantaggio rispetto alla baseline no-FL:**
Il gossip deve apportare un miglioramento misurabile rispetto al training in isolamento (Esperimento 1). L'entità attesa del vantaggio è **+5–15% di mean_accuracy** e una riduzione della `std_accuracy` tra worker di almeno il 30–50%.

**Criterio 3 — Equità della convergenza (`std_accuracy`):**
Un sistema FL sano produce modelli simili su tutti i worker. Con N=3 e dataset full, `std_accuracy < 3%` al termine indica convergenza uniforme. Valori tra 3% e 7% sono accettabili; oltre il 7% segnalano un'aggregazione inefficace o un fanout troppo basso.

**Criterio 4 — Graceful degradation sotto fault injection:**
Il sistema deve mantenere `mean_accuracy > 80%` con `drop_probability: 0.2`. Un crollo significativo dell'accuracy (> 5%) a `drop_probability: 0.2` indicherebbe dipendenza eccessiva dalla continuità delle comunicazioni — comportamento che il gossip asincrono dovrebbe proprio evitare.

---

### 12.6 Tabelle Risultati (TODO — completare con dati reali)

**Esperimento 1 vs 2 — Baseline no-FL vs FL gossip:**

| Metrica | Esp. 1 (no gossip) | Esp. 2 (gossip default) | Delta |
|---|:---:|:---:|:---:|
| `mean_accuracy` finale | TODO | TODO | TODO |
| `std_accuracy` finale | TODO | TODO | TODO |
| Round a convergenza | TODO | TODO | TODO |
| L2 divergenza pesi | TODO | TODO | TODO |

**Esperimento 3c — Effetto gossip_fanout (N=3, H=500):**

| `gossip_fanout` | `mean_accuracy` | `std_accuracy` | Round a conv. | Vol. totale (GB) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | TODO | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO | TODO |

**Esperimento 3b — Effetto inner_steps_H (N=3, fanout=1):**

| `inner_steps_H` | `mean_accuracy` | `std_accuracy` | Round a conv. | Msg push totali |
|:---:|:---:|:---:|:---:|:---:|
| 100 | TODO | TODO | TODO | TODO |
| 500 | TODO | TODO | TODO | TODO |
| 1000 | TODO | TODO | TODO | TODO |

**Esperimento 4 — Scalabilità num_workers (config ottimale, fanout=best):**

| `num_workers` | `mean_accuracy` | `std_accuracy` | Round a conv. | Vol. totale (GB) | Durata/round (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | TODO | TODO | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO | TODO | TODO |

**Esperimento 5 — Robustezza fault injection (config ottimale):**

| `drop_probability` | `mean_accuracy` | Note |
|:---:|:---:|---|
| 0.0 | TODO | riferimento pulito |
| 0.2 | TODO | default — deve reggere |
| 0.5 | TODO | soglia critica |
| 0.8 | TODO | degradazione attesa |

---

## 13. Piano Sperimentale Completo

Questa sezione descrive in modo strutturato l'intero piano degli esperimenti: quali parametri variano, in quale ordine, e perché ogni run è necessario. La struttura usa una notazione a cicli annidati per rendere esplicite le dipendenze tra esperimenti.

### 13.1 Spazio dei Parametri

La tabella distingue i parametri fissi (identici in tutti i run) da quelli esplorati.

**Parametri fissi in tutti i run:**

| Parametro | Valore fisso | Motivazione |
|---|:---:|---|
| `batch_size` | 32 | bilanciamento gradiente / velocità |
| `learning_rate` | 0.001 | AdamW con lr=1e-3 già calibrato su FEMNIST |
| `clip_grad` | 1.0 | drift bound garantito (sezione 5.3) |
| `label_smoothing` | 0.1 | calibrazione 62 classi (sezione 5.3) |
| `dropout_conv` | 0.25 | regularizzazione validata |
| `dropout_fc` | 0.5 | standard per classificatore FC |
| `aggregation_strategy` | FedAvg | unica strategia implementata |
| `early_stopping_patience` | 10 | stesso criterio di arresto per tutti i run comparativi |
| `drop_probability` | 0.0 | nessuna fault injection (tranne Fase 4) |
| `crash_probability` | 0.0 | nessuna fault injection (tranne Fase 4) |
| `max_staleness` | 10 | ampio margine, mai attivo senza fault injection |

**Parametri esplorati (il valore in grassetto è il valore di controllo — quello del run già completato):**

| Parametro | Valori esplorati | Controllo | Fase |
|---|:---:|:---:|:---:|
| `gossip_enabled` | false, **true** | true | 0 |
| `gossip_fanout` | **1**, 2 | 1 | 1 |
| `inner_steps_H` | 100, **500**, 1000 | 500 | 2 |
| `num_workers` | **3**, 5, 8 | 3 | 3 |
| `drop_probability` | **0.0**, 0.2, 0.5 | 0.0 | 4 |
| `crash_probability` | **0.0**, 0.05 | 0.0 | 4 |
| `use_test_set` | false, **true** | false | 5 |
| `learning_rate` *(opz.)* | **0.001**, 0.0001 | 0.001 | — |

### 13.2 Struttura degli Esperimenti: Pseudocodice

L'intero piano si legge come un programma. Ogni blocco corrisponde a una fase; le frecce indicano dipendenze (un blocco può iniziare solo quando i precedenti sono completati e analizzati). Il run ✓ è il punto di riferimento comune condiviso da Fase 0, 1 e 2 — non va ripetuto.

```
# ── RIFERIMENTO COMUNE ───────────────────────────────────────────────────────
run ✓:  N=3, gossip=True,  fanout=1, H=500          # già completato

# ── FASE 0 — FL vs no-FL ─────────────────────────────────────────────────────
# Isola gossip_enabled; tutto il resto uguale al run ✓.
# Nessuna dipendenza: si può fare subito.
run B0: N=3, gossip=False, fanout=–, H=500

# ── FASE 1 — Effetto fanout ───────────────────────────────────────────────────
# Isola gossip_fanout; H=500 e N=3 come in run ✓.
# Nessuna dipendenza: si può fare subito.
run F1: N=3, gossip=True, fanout=2, H=500

# ── FASE 2 — Effetto H (inner steps) ─────────────────────────────────────────
# Isola inner_steps_H; fanout=1 e N=3 come in run ✓.
# Nessuna dipendenza: si può fare subito.
run H1: N=3, gossip=True, fanout=1, H=100,  total_rounds=300
run H2: N=3, gossip=True, fanout=1, H=1000, total_rounds=100

# ── ANALISI BLOCCO A ─────────────────────────────────────────────────────────
# Dopo B0, F1, H1, H2: confronta mean_accuracy, round di convergenza, L2 divergenza.
# Scegli la configurazione che massimizza l'accuracy media:
best_fanout ← argmax over {1, 2}
best_H      ← argmax over {100, 500, 1000}

# ── FASE 3 — Scalabilità ─────────────────────────────────────────────────────
# Prerequisito per ogni N: aggiorna num_workers in config.yaml,
# poi esegui split_dataset.py + generate_compose.py.
# Dipende da: best_fanout e best_H (da Blocco A).
for N in [5, 8]:               # N=3 già coperto da run ✓
    run S_N: N, gossip=True, fanout=best_fanout, H=best_H

# ── FASE 4 — Fault tolerance (opzionale) ─────────────────────────────────────
# N=3, config ottimale. No risplit necessario.
# Dipende da: best_fanout e best_H (da Blocco A).
for drop_prob in [0.2, 0.5]:
    run D_p: N=3, fanout=best_fanout, H=best_H, drop_probability=drop_prob
run C1: N=3, fanout=best_fanout, H=best_H, crash_probability=0.05

# ── FASE 5 — Valutazione finale unbiased ─────────────────────────────────────
# Prerequisito: download_femnist.py (flag --tf diverso a LEAF) + split_dataset.py.
# Dipende da: tutti i run precedenti (usa la config migliore trovata).
run T0: N=3, gossip=True, fanout=best_fanout, H=best_H, use_test_set=True

# ── OPZIONALE — Tuning learning rate ─────────────────────────────────────────
# Indipendente: si può fare in qualsiasi momento nel Blocco A.
run L1: N=3, gossip=True, fanout=1, H=500, lr=0.0001
```

### 13.3 Lista Completa dei Run

| Run | Nome da salvare | Cosa varia | Valore | Dipende da | Risplit? |
|---|---|---|:---:|---|:---:|
| ✓ | `fanout1_h500_lr1e3` | — | riferimento | — | no |
| B0 | `no_fl_baseline` | `gossip_enabled` | false | — | no |
| F1 | `fanout2_h500` | `gossip_fanout` | 2 | — | no |
| H1 | `fanout1_h100` | `inner_steps_H` | 100 | — | no |
| H2 | `fanout1_h1000` | `inner_steps_H` | 1000 | — | no |
| S1 | `best_config_5w` | `num_workers` | 5 | F1, H1, H2 | **sì** |
| S2 | `best_config_8w` | `num_workers` | 8 | F1, H1, H2 | **sì** |
| D1 | `fault_drop20` | `drop_probability` | 0.2 | F1, H1, H2 | no |
| D2 | `fault_drop50` | `drop_probability` | 0.5 | F1, H1, H2 | no |
| C1 | `fault_crash5` | `crash_probability` | 0.05 | F1, H1, H2 | no |
| T0 | `final_test_eval` | `use_test_set` | true | tutti | **sì** |
| L1 | `lr_1e4` *(opz.)* | `learning_rate` | 0.0001 | — | no |

*I run con "Risplit? = sì" richiedono di aggiornare `config.yaml` e rieseguire `split_dataset.py` + `generate_compose.py` prima del `docker compose up`.*

### 13.4 Ordine di Esecuzione

```
┌─────────────────────────────────────────────────────────────┐
│  BLOCCO A — stessa partizione N=3, eseguibili in parallelo  │
│                                                             │
│  B0   F1   H1   H2   L1(opz.)                              │
└───────────────────┬─────────────────────────────────────────┘
                    │  analisi → scegli best_fanout, best_H
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCCO B — scalabilità (risplit per ogni N)                │
│                                                             │
│  S1 (N=5)  →  S2 (N=8)                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  BLOCCO C — fault tolerance, N=3, eseguibili in parallelo  │
│                                                             │
│  D1   D2   C1                                              │
└───────────────────┬─────────────────────────────────────────┘
                    │  tutti i run completati
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCCO D — valutazione finale (risplit con use_test_set)   │
│                                                             │
│  T0                                                         │
└─────────────────────────────────────────────────────────────┘
```

Blocchi B e C sono indipendenti tra loro: si possono eseguire in qualsiasi ordine dopo il Blocco A. Blocco D è sempre l'ultimo.

### 13.5 Razionale per Fase

**Fase 0 — B0 (baseline no-FL)**: senza questo confronto non è possibile quantificare il contributo del gossip. Se i worker ottengono 87% in isolamento e 87% con FL, il protocollo non aggiunge valore. Ci aspettiamo un gap di 3–7 punti e una divergenza L2 finale molto maggiore (modelli che non si sincronizzano mai). Questo run stabilisce il pavimento assoluto.

**Fase 1 — F1 (fanout=2)**: con N=3, fanout=2 equivale al broadcast completo — ogni worker invia il modello a entrambi i peer ogni round. È il confronto diretto con il run ✓ (fanout=1). Ci aspettiamo che il gap di accuracy tra worker 0 (~90%) e worker 1/2 (~86%) si riduca, che la divergenza L2 collassi verso zero, e che la convergenza sia più rapida. Il costo è il raddoppio del volume di traffico gossip.

**Fase 2 — H1/H2 (ablazione su H)**: H=100 aumenta la frequenza di gossip mantenendo i modelli più allineati ma richiede più round per elaborare la stessa quantità di dati. H=1000 riduce la comunicazione ma lascia divergere i modelli localmente: il FedAvg agisce su modelli più distanti, potenzialmente causando accuracy valley più profonde dopo ogni aggregazione. L'obiettivo è verificare empiricamente se H=500 è effettivamente il sweet spot, come osservato da DiLoCo su LLM. Poiché il nostro dataset è non-i.i.d. e più piccolo, il sweet spot potrebbe spostarsi verso H più piccoli.

**Fase 3 — S1/S2 (scalabilità)**: al crescere di N le partizioni diventano più piccole e più eterogenee (più writer, stili di scrittura più diversi per worker). Ci aspettiamo che l'accuracy media peggiori leggermente ma che il sistema rimanga funzionale fino a N=8. La durata per round decresce (meno dati per worker), ma la convergenza in numero di round potrebbe peggiorare. In modalità multi-instance AWS si misura anche la latenza di rete reale tra istanze EC2 nella stessa AZ.

**Fase 4 — D1/D2/C1 (fault tolerance)**: verifica la resilienza del design P2P. Con `drop_probability=0.2` ogni worker perde in media il 20% dei gossip push in uscita; il sistema dovrebbe compensare con i messaggi ricevuti dagli altri round. Con `crash_probability=0.05` ogni worker ha un'aspettativa di vita di 20 round; il registry lo rimuove automaticamente e i peer sopravvissuti continuano a fare gossip tra loro senza coordinazione centralizzata. Questa proprietà — continuare a funzionare senza un coordinator — è il vantaggio fondamentale dell'architettura P2P rispetto a FedAvg centralizzato.

**Fase finale — T0 (test set unbiased)**: tutti i run precedenti usano la validation accuracy come metrica finale, il che introduce un piccolo bias ottimistico perché l'early stopping ha osservato quella stessa metrica. T0 usa la partizione test separata (split 80/10/10) che non ha mai influenzato né il training né l'early stopping. L'accuracy riportata qui è la stima più onesta delle capacità generalizzative del sistema.

---

## Riferimenti

[1] Douillard, A., Feng, Q., Ruder, S., Dieleman, S., Bousquet, O., & Houlsby, N. (2023). *DiLoCo: Distributed Low-Communication Training of Language Models*. arXiv:2311.08105.

[2] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017.

[3] Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Konečný, J., McMahan, H. B., Smith, V., & Talwalkar, A. (2018). *LEAF: A Benchmark for Federated Settings*. arXiv:1812.01097.
