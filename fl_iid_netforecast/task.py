"""Federated Learning su CESNET-TimeSeries24"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, root_mean_squared_error

from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import AgreggationType, SourceType

TARGET_FEATURE_INDEX = 0    # uso solo la feature target, quindi l'indice è 0, (modello univariato)


class LSTMForecast(nn.Module):
    """LSTM per forecasting: prende in input una finestra di training e predice gli step della finestra di predizione"""

    # I valori qui sotto sono solo DEFAULT della firma: build_model li passa sempre tutti
    # espliciti, letti dal run_config. 
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, output_size=1):  # costruisce l'architettura della rete
        # output_size = prediction_window_size
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,  # i dati in ingresso hanno forma (batch_size, seq_len, input_size)
            dropout=dropout,   # non applicabile con un solo layer (serve per ridurre overfitting, spegnendo dei nodi)
            bidirectional=True,  # Legge la finestra di input anche a ritroso (scelta del benchmark di riferimento).
        )

        # Bidirezionale: l'LSTM produce 2 * hidden_size feature per timestep
        self.fc = nn.Linear(hidden_size * 2, output_size)

    #prende un batch di finestre e produce le predizioni    
    def forward(self, x):
        " x: (batch_size, seq_len, input_size) -> out: (batch_size, output_size)"
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   


#run_config pesco la configurazione di run dal server, che la passa a tutti i client ( vedi [tool.flwr.app.config] in pyproject.toml)
def build_model(run_config:dict):
    """Costruisce SOLO l'architettura, senza allenarla.
    Legge gli iperparametri dal run_config e restituisce una LSTM nuova,
    con pesi casuali. 
    Seed viene fissato una volta sola dal server (i client non inizializzano mai riceveranno i pesi globali)
    """
    return LSTMForecast(
        input_size=1,  # modello univariato: una sola feature (target-feature)
        hidden_size=int(run_config["hidden-size"]),
        num_layers=int(run_config["num-layers"]),
        dropout=float(run_config["dropout"]),
        output_size=int(run_config["prediction-window-size"]),
    )




def apri_dataset(run_config:dict):
    """Apre il dataset CESNET-TimeSeries24.
    Non carica ancora i dati: serve poi 
    Sta in una funzione perché la chiameranno in altre funzioni (istituzioni_pool_iid e load_data).
    """
    return CESNET_TimeSeries24.get_dataset(
        data_root=str(run_config["data-root"]),
        source_type=SourceType.INSTITUTIONS,
        aggregation=AgreggationType.AGG_1_HOUR,
        dataset_type="time_based",
        display_details=False,  # niente output verboso: qui i client sono centinaia
    )


def istituzioni_pool_iid(num_partitions: int, run_config: dict):
    """Id reali delle prime num_partitions istituzioni che formano il pool condiviso
    da tutti i client (il pool ha sempre dimensione num_partitions: un'istituzione
    in più per ogni client aggiunto alla federazione)."""
    ids = apri_dataset(run_config).get_available_ts_indices()["id_institution"]
    return ids[:num_partitions].tolist() #prendo i primi num_partitions id istituzioni reali, che formano il pool condiviso da tutti i client


def concatena_finestre(loader):
    """Scorre un DataLoader di cesnet_tszoo (una finestra alla volta) e concatena
    tutte le finestre in due unici array numpy: (n_finestre, window_size, 1)."""
    X_all, Y_all = [], []
    for X, Y in loader: #il loader mi da una finestra alla volta, le inglobo rispettivamente in un contenitore X e Y
        X_all.append(X)
        Y_all.append(Y)
    return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0) #concateno tutte le finestre in un unico array numpy (n_finestre, window_size, 1)


_load_data_cache: dict[tuple[int, int, str], list] = {}  # cache in-memory dei batch già costruiti, per client e split


def load_data(partition_id: int, num_partitions: int, run_config: dict, split: str):
    """
    Dato l'indice di un client, costruisce il suo dataset IID per lo split richiesto
    ("train" o "test"): mette insieme le finestre di TUTTE le istituzioni del pool
    (istituzioni_pool_iid), le mescola con un seed fisso e ne prende un blocco da assegnare a quel client,
    così ogni client riceve un mix casuale ma riproducibile di tutto il pool, non
    solo di una singola istituzione.

    Le finestre di TUTTE le istituzioni del pool vengono normalizzate in un'unica
    inizializzazione, ciascuna con il SUO scaler MinMax individuale i
    volumi di traffico variano di ordini di grandezza tra istituzioni.

    Il risultato viene cachato per (partition_id, num_partitions, split): ricostruirlo
    ad ogni round rilegge e riscorre i loader per istituzione trattiene memoria non rilasciata ad ogni chiamata. 
    Con la cache, eseguo una sola volta per client per l'intera durata della run,
    invece che una volta per round.
    """
    # cahce per client e split: se il client ha già caricato i dati in un round precedente, li riutilizza senza rileggerli da cesnet_tszoo
    cache_key = (partition_id, num_partitions, split)   
    if cache_key in _load_data_cache:
        return _load_data_cache[cache_key]  # hit: nei round successivi salta subito il ricaricamento da cesnet_tszoo

    seed = int(run_config["random-state"])
    pool = istituzioni_pool_iid(num_partitions, run_config) # ottiene la lista di id istituzioni che formano il pool IID

    dataset = apri_dataset(run_config)
    config = TimeBasedConfig(
        ts_ids=pool,    #lista di più isitituzioni
        train_time_period=float(run_config["train-time-period"]),
        test_time_period=float(run_config["test-time-period"]),
        features_to_take=[str(run_config["target-feature"])],
        sliding_window_size=int(run_config["training-window-size"]),
        sliding_window_prediction_size=int(run_config["prediction-window-size"]),
        sliding_window_step=int(run_config["prediction-window-size"]),
        random_state=seed,
        transform_with="min_max_scaler",    # scaler viene fittato solo sul training set 
        include_ts_id=False,
        include_time=False,
    )
    # I valori mancanti vengono riempiti automaticamente con 0 (default_values="default"),
    # come descritto nel paper per le metriche di volume (n_flows, n_packets, n_bytes)
    dataset.set_dataset_config_and_initialize(config, display_config_details=None)

    # sceglie il tipo di loader richiesto (train o test) 
    get_loader = dataset.get_train_dataloader if split == "train" else dataset.get_test_dataloader

    X_parts, Y_parts = [], []   # accumulano le finestre di ciascuna istituzione, prima di unirle tutte insieme
    for institution_id in pool:
        X, Y = concatena_finestre(get_loader(ts_id=institution_id))  #trasforma il loader in due array numpy (n_finestre, window_size, 1) di input e target
        X_parts.append(X)
        Y_parts.append(Y)

    # unisco insieme le finestre delle istituzioni del pool in due unici array,
    # poi mescola l'intero pool con un seed fisso e prende solo il blocco di questo client:
    # ogni client riceve così un mix casuale ma riproducibile di finestre di istituzioni diverse.
    #il mescolamento è necessario perché le finestre di ciascuna istituzione sono ordinate temporalmente, e se un client ricevesse solo finestre di una istituzione non avrebbe un training set rappresentativo del pool.
    #il mescolamento avviene ogni round (non ad ogni epoca locale)
    X_pool = np.concatenate(X_parts)
    Y_pool = np.concatenate(Y_parts)
    rng = np.random.default_rng(seed)   # fisso il seed per avere la stessa mescolazione casuale ma identica per ogni run
    blocco = np.array_split(rng.permutation(len(X_pool)), num_partitions)[partition_id] #len(X_pool) è il numero totale di finestre nel pool, rng.permutation(N) genera una permutazione casuale di 0..N-1,
    # np.array_split divide la sequenza in num_partitions (client) blocchi  uguali, e ne prende solo quello di questo client
    X, Y = X_pool[blocco], Y_pool[blocco]   # ritorno le finestre di training/test di questo client, che sono un mix casuale ma riproducibile di tutte le istituzioni del pool
    batch_size = int(run_config["batch-size"])

    #Taglia X e Y in blocchi consecutivi da batch_size finestre (l'ultimo può essere più corto), scorrendo gli indici di partenza 0, batch_size, 2*batch_size, ....
    #Restituisce la lista di questi blocchi come coppie (X_batch, Y_batch)
    result = [(X[i:i + batch_size], Y[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    _load_data_cache[cache_key] = result  # miss: memorizza il risultato, così viene calcolato una sola volta per (client, split)
    return result




def harmonic_score(rmse, r2):
    """Combina RMSE e R2 in un unico punteggio: media armonica, più basso è meglio.

    Il clipping evita che un client con R2 o RMSE enorme ( magari varianza del target quasi nulla)
    domini qualunque confronto o media. I valori di clipping sono forniti dal paper di riferimento
    """
    rmse_clipped = min(rmse, 11.0)
    r2_clipped = max(r2, -10.0)
    r2_term = abs(r2_clipped - 1)
    return 2 * (rmse_clipped * r2_term) / (rmse_clipped + r2_term)




def train_one_epoch(model, loader, criterion, optimizer, device):
    """Un passaggio completo sui dati di training locali, con un aggiornamento dei pesi."""
    model.train()
    losses = []  # loss di ogni mini-batch, per farne la media a fine epoca

    for X, Y in loader:  # loader dà già mini-batch pronti (vedi load_data)
        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).float().to(device)[:, :, TARGET_FEATURE_INDEX]

        optimizer.zero_grad()       # azzero i gradienti accumulati dal mini-batch precedente
        preds = model(X)            # forward pass sul mini-batch
        loss = criterion(preds, Y)  # MSE tra predizioni e valori reali
        loss.backward()             # calcolo i gradienti rispetto ai pesi del modello
        optimizer.step()            # aggiorno i pesi

        losses.append(loss.item())

    return torch.tensor(losses).mean().item()  # loss media di tutti i mini-batch dell'epoca


def train(model, train_loader, epochs, lr, device):
    """Il client allena il modello locale per epoche partendo dai pesi ricevuti dal server """
    model.to(device)
    criterion = nn.MSELoss()  # loss per regressione: minimizzo la MSE tra predizioni e valori reali (seguo il paper, anche se sensibile agli outlier)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # lr fisso che arriva dal server

    avg_loss = 0.0      # loss media dell'epoca corrente, che viene aggiornata a ogni epoca e restituita alla fine
    for _ in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    return avg_loss  # loss media dell'ULTIMA epoca locale


def test(model, loader, device):
    """Valuta il modello sul test set locale senza allenarlo (chiamata dal client). """
    model.to(device)
    model.eval()
    preds, trues = [], []

    with torch.no_grad():  # non calcolo i gradienti, inutile durante la valutazione
        for X, Y in loader:  # loader dà già mini-batch pronti (vedi load_data)
            X = torch.from_numpy(X).float().to(device)
            # (batch, prediction_window): tutti gli step, non solo il primo
            Y = torch.from_numpy(Y).float().to(device)[:, :, TARGET_FEATURE_INDEX]
            preds.append(model(X))
            trues.append(Y)

    # concateno tutte le predizioni e i valori reali in un unico tensore (N, prediction_window)
    preds = torch.cat(preds)
    trues = torch.cat(trues)

    # Appiattisco le matrici (n_finestre, prediction_window) in un unico vettore, cosi' le
    # metriche sono calcolate su tute le predizioni insieme. 
    trues_np = trues.cpu().numpy().flatten()
    preds_np = preds.cpu().numpy().flatten()

    # metriche in scala normalizzata 
    mse = F.mse_loss(preds, trues)
    rmse = root_mean_squared_error(trues_np, preds_np)
    r2 = r2_score(trues_np, preds_np)
    h_score = harmonic_score(rmse, r2)

    return mse.item(), float(rmse), float(r2), float(h_score), trues, preds
