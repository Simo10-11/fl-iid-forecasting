# Federated Forecasting del traffico di rete (CESNET-TimeSeries24)

Previsione del traffico di rete con **federated learning**, usando [Flower](https://flower.ai) e PyTorch sul dataset [CESNET-TimeSeries24](https://github.com/koumajos/CESNET-TimeSeries24). Ogni client allena un modello LSTM **bidirezionale univariato** su uno split **IID**: le finestre di tutte le istituzioni vengono mescolate e divise in blocchi uguali tra i client, così ognuno riceve un mix rappresentativo del pool.

---

## Requisiti

- Python 3.10+
- Dipendenze principali (vedi `pyproject.toml`): `flwr[simulation]`, `torch`, `cesnet-tszoo`, `scikit-learn`, `numpy`
- ~150 MB liberi su disco: alla prima esecuzione `cesnet-tszoo` scarica automaticamente il dataset

---

## Installazione

```bash
git clone <url-del-repo>
cd fl-iid-forecasting

# crea e attiva un virtual environment
python -m venv flwr-env
source flwr-env/bin/activate      # Windows: flwr-env\Scripts\activate

# installa il progetto e le sue dipendenze
pip install -e .
```

---

## Esecuzione

Avvia la simulazione federata con i parametri di default:

```bash
flwr run . --stream
```

`--stream` mostra i log in tempo reale.

Il numero di client simulati e le risorse per client si passano a parte, come `--federation-config`:

```bash
flwr run . --stream --federation-config 'num-supernodes=5 client-resources-num-cpus=6'
```

---

## Configurazione

Tutti gli iperparametri stanno in `pyproject.toml`, sotto `[tool.flwr.app.config]`:

- **Dataset**: feature target, dimensione delle finestre di input/predizione, split train/test, seed
- **Modello**: dimensioni della LSTM, learning rate, batch size
- **Federated learning**: numero di round, epoche locali per round, frazione di client per training/valutazione, minimo di client richiesti, salvataggio del modello finale

Per un run singolo si possono sovrascrivere senza modificare il file:

```bash
flwr run . --stream --run-config 'num-server-rounds=20 learning-rate=0.005'
```

---

## Output atteso

Durante l'esecuzione vengono stampati:

- il pool di istituzioni usato per lo split IID, una sola volta a inizio run
- un log per ogni fase di ogni round (`train -> N client coinvolti`, `evaluate -> N client coinvolti`)
- le metriche aggregate di training (`train_mse`) e di valutazione (`mse`, `rmse`, `r2`, `harmonic`)

Se `save-model = true` (default), a fine run il modello globale finale viene salvato come `final_model.pt` nella cartella del progetto.

---

## Struttura del progetto

```
fl-iid-forecasting
├── fl_iid_netforecast
│   ├── task.py          # modello LSTM, caricamento/preparazione dati, training e metriche
│   ├── client_app.py    # ClientApp: training e valutazione locale di ogni client
│   └── server_app.py    # ServerApp: strategia FedAvg e ciclo dei round
├── pyproject.toml       # dipendenze e configurazione (dataset, modello, federated learning)
└── README.md
```
