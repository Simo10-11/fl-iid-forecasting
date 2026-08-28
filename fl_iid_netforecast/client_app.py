"""ClientApp: ogni client riceve uno split IID (mix casuale ma riproducibile di tutto il pool di istituzioni)."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_iid_netforecast.task import build_model, get_device, load_data, statistiche_additive
from fl_iid_netforecast.task import test as test_fn
from fl_iid_netforecast.task import train as train_fn

app = ClientApp()   #  È l'oggetto che il pyproject.toml cerca (clientapp = "fl_iid_netforecast.client_app:app")


def model_from_message(msg: Message, context: Context):
    """Costruisce un modello locale con i pesi del modello globale inviati dal server.
    Il modello locale non viene mai inizializzato con pesi random: riparte sempre da
    quelli ricevuti dal server, altrimenti il federated averaging non convergerebbe.
    
    msg: contiene i pesi del modello globale inviati dal server
    context: contiene la configurazione del run (architettura, numero di layer)
    """

    model = build_model(context.run_config) # costruisce un modello LSTM vuoto con l'architettura specificata nel run_config (pesi per ora casuali)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())  # carica i pesi del modello globale inviati dal server nel modello LSTM vuoto appena costruito
    return model        #ritorno modello LSTM inizializzato 


@app.train()
def train(msg: Message, context: Context):
    """Riceve il modello globale, lo allena sul blocco IID locale di questo client, e rimanda i pesi aggiornati con le metriche"""
    model = model_from_message(msg, context)
    device = get_device()

    partition_id = context.node_config["partition-id"]  # node_config contiene informazioni sul client, tra cui l'id della partizione
    num_partitions = context.node_config["num-partitions"]  # dimensione reale della federazione, sempre sincronizzata con --num-supernodes
    train_loader = load_data(partition_id, num_partitions, context.run_config, split="train")

    train_loss = train_fn(
        model,
        train_loader,
        int(context.run_config["local-epochs"]),
        float(msg.content["config"]["lr"]),
        device,
    )

    # num-examples = numero di FINESTRE di training (non di mini-batch). È la chiave con cui
    # FedAvg pesa l'aggregazione dei pesi e delle metriche, ed è obbligatoria.
    # Split IID: il pool viene mescolato tutto insieme e diviso in blocchi quasi
    # uguali, quindi tutti i client hanno (a meno di arrotondamenti) lo stesso
    # numero di finestre, e la media pesata coincide con la media semplice.
    num_examples = sum(len(X) for X, _ in train_loader)
    metrics = {
        "train_mse": train_loss,    #loss media dell'ultima epoca locale, vedo se train funziona
        "num-examples": num_examples,
    }
    content = RecordDict(
        {"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)}   # costruisco un RecordDict con i pesi aggiornati del modello e le metriche calcolate localmente, che verranno inviate al server 
    )
    return Message(content=content, reply_to=msg)   #non mando dati, ma solo pesi e 3 numeri



# i client allenano e rimandano i pesi
# il server aggrega: nasce il modello globale
# il server manda il modello ai client per la valutazione


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Testa il modello globale (aggregato, non locale) sul blocco IID di test locale di questo client."""
    model = model_from_message(msg, context)
    device = get_device()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    test_loader = load_data(partition_id, num_partitions, context.run_config, split="test")

    _, _, _, _, trues, preds = test_fn(model, test_loader, device)

    # Statistiche additive (somme, non medie) di questo client, invece di mediare le
    # metriche locali (scorretto per rmse/r2/mae, che non sono lineari. Nessun dato
    # grezzo lascia il client, solo 5 numeri.
    sse, sum_abs_error, sum_y, sum_y_sq, num_values = statistiche_additive(trues, preds)

    metrics = {
        "sse": sse,
        "sum_abs_error": sum_abs_error,
        "sum_y": sum_y,
        "sum_y_sq": sum_y_sq,
        "num_values": num_values,
        "num-examples": sum(len(X) for X, _ in test_loader),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})    #non mandoi i pesi ( non c'è ArrayRecord), ma solo le metriche per dare valutazione
    return Message(content=content, reply_to=msg)



# Ogni round, il client scarta completamente lo stato del round precedente e riparte dai pesi
# globali aggregati. È questo che rende la media di FedAvg utile: tutti i
# client hanno parametri a partire dallo stesso punto mandato dal server.