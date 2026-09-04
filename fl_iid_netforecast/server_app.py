"""ServerApp: strategia di aggregazione FL sui modelli LSTM locali dei client (ciascuno su un blocco IID)."""

from copy import deepcopy

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_iid_netforecast.task import build_model, metriche_da_statistiche_additive

app = ServerApp()


# Ultima mse della validazione FEDERATA aggregata: 
# Il flag "fresh" serve da guardia: se un round non producesse una validazione
# federata (es. il round 0, pesi random iniziali, prima ancora che un client valuti), fuone checheckpoint lo
# vede (fresh=False) e non riusa per sbaglio il valore del round precedente.
_validazione_federata: dict = {"mse": None, "fresh": False}


def aggrega_train(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    """Media pesata (per num-examples) della train_mse locale di ogni client. La MSE è additiva
    (media di errori al quadrato), quindi qui la media pesata generica di Flower
    (aggregate_metricrecords) è già esatta: non serve una logica custom come per l'evaluate."""
    print(f" train -> {len(records)} client coinvolti")
    return aggregate_metricrecords(records, weighting_metric_name)  #aggregate_metricrecords è la funzione DI FLOWER


def aggrega_evaluate(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    """Aggrega le metriche di evaluate senza usare la media pesata generica di Flower per
    rmse/r2/mae: non sono lineari, quindi mediare i valori già
    calcolati da ogni client non equivale a calcolarli sull'unione dei dati di tutti i client
    (esempio: due client con rmse locale 1 e 3 sullo stesso numero di finestre non danno un
    rmse globale di 2, ma sqrt(5) ≈ 2.236).

    Ogni client manda invece statistiche additive. Qui vengono sommate su tutti i client coinvolti in questo
    round, e le metriche finali vengono calcolate una sola volta, come se il modello fosse stato
    valutato sull'intero set federato in un unico batch.

    Usata in DUE momenti, con lo stesso identico codice: round-per-round sulla validazione (dove
    la mse aggregata viene anche lasciata in _validazione_federata per la model selection, e una tantum sul test finale (vedi valuta_test_finale)."""
    print(f" evaluate -> {len(records)} client coinvolti")

    total_sse = total_sum_abs_error = total_sum_y = total_sum_y_sq = 0.0
    total_num_values = 0
    for record in records:
        metricrecord = next(iter(record.metric_records.values()))
        total_sse += metricrecord["sse"]
        total_sum_abs_error += metricrecord["sum_abs_error"]
        total_sum_y += metricrecord["sum_y"]
        total_sum_y_sq += metricrecord["sum_y_sq"]
        total_num_values += metricrecord["num_values"]

    mse, rmse, r2, mae = metriche_da_statistiche_additive(
        total_sse, total_sum_abs_error, total_sum_y, total_sum_y_sq, total_num_values
    )

    # Lasciata per l'hook di checkpoint di QUESTO round (che Flower chiama subito dopo, solo
    # durante il ciclo normale di round: il test finale non passa da qui).
    _validazione_federata["mse"] = mse
    _validazione_federata["fresh"] = True

    return MetricRecord(
        {"mse": mse, "rmse": rmse, "r2": r2, "mae": mae, "num_values": total_num_values}
    )


def crea_checkpoint(best: dict):
    """Costruisce evaluate_fn per Flower (parametro di strategy.start). Si limita a leggere la mse della validazione FEDERATA appena
    aggregata da aggrega_evaluate su questo round e, se è la migliore vista finora, salva un
    checkpoint dei pesi correnti in `best` (deepcopy dello state_dict), sovrascrivendo il
    precedente. 

    Flower chiama questo hook ad ogni round con (server_round, arrays): prima del round 1 (pesi
    random iniziali) e poi dopo ogni round successivo. I pesi ricevuti qui sono quindi esattamente quelli
    appena valutati dai client: "pesi del round N" .

    Il round 0 non ha validazione federata (nessun client ha ancora valutato): _validazione_federata
    resta "fresh"=False e viene escluso automaticamente dalla selezione.
    """

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        if not _validazione_federata["fresh"]:
            return None
        _validazione_federata["fresh"] = False  # consumata: vale per questo round e basta
        mse_federata = _validazione_federata["mse"]

        if mse_federata < best["mse"]:
            best["mse"] = mse_federata
            best["round"] = server_round
            best["state_dict"] = deepcopy(arrays.to_torch_state_dict())
            print(f" nuovo miglior checkpoint: round {server_round} (mse validazione federata={mse_federata:.4f})")

        return None

    return evaluate_fn


def valuta_test_finale(grid: Grid, state_dict) -> MetricRecord:
    """Round di valutazione EXTRA, fuori dal ciclo standard di strategy.start(): manda i pesi del
    modello migliore (selezionato sulla validazione federata) a TUTTI i client, chiedendo loro di
    valutarlo sul proprio test set locale.  è un round singolo,
    fuori sequenza, eseguito una sola volta a fine esperimento.

    Il campo "eval-split"="test" nel ConfigRecord del messaggio dice al client di valutare sul
    test invece che sul validation  Le statistiche additive che tornano indietro vengono aggregate con la STESSA
    funzione già usata per la validazione federata round-per-round (aggrega_evaluate): stessa
    logica, solo applicata al test invece che al validation.

    Stampa anche il risultato di OGNI singolo client (identificato dal suo partition-id, che il
    client stesso rimanda indietro nelle metriche), calcolato dalle sue 5 statistiche additive
    con la stessa metriche_da_statistiche_additive usata per l'aggregato: nessun dato grezzo
    aggiuntivo lascia il client, solo la stessa metrica finale già mostrata a livello globale, ma
    scomposta client per client.
    """
    node_ids = list(grid.get_node_ids())  # tutti i nodi client attivi
    record = RecordDict({
        "arrays": ArrayRecord(state_dict),  # i pesi del best model
        "config": ConfigRecord({"eval-split": "test"}),  # dice al client: usa il test
    })
    messages = [
        Message(content=record, dst_node_id=node_id, message_type=MessageType.EVALUATE)  # un messaggio per client
        for node_id in node_ids
    ]

    print(f"\nTest finale federato: invio il modello selezionato a {len(node_ids)} client...")
    replies = list(grid.send_and_receive(messages))  # invia e aspetta le risposte

    valide = [msg for msg in replies if not msg.has_error()]  # scarta le risposte fallite
    n_errori = len(replies) - len(valide)  # quanti client hanno fallito
    print(f" test finale -> {len(valide)} client coinvolti" + (f", {n_errori} errori" if n_errori else ""))

    print(" risultato per client:")
    risultati_client = []  # accumula i risultati prima di stamparli
    for msg in valide:
        metricrecord = next(iter(msg.content.metric_records.values()))  # le 5 statistiche di questo client
        mse, rmse, r2, mae = metriche_da_statistiche_additive(
            metricrecord["sse"], metricrecord["sum_abs_error"],
            metricrecord["sum_y"], metricrecord["sum_y_sq"], metricrecord["num_values"],
        )  # ricalcola le metriche di QUESTO client
        risultati_client.append((metricrecord["partition-id"], mse, rmse, r2, mae, metricrecord["num_values"]))

    for partition_id, mse, rmse, r2, mae, num_values in sorted(risultati_client):  # ordina per numero client
        print(
            f"   client {partition_id}: "
            f"mse={mse:.4f} rmse={rmse:.4f} r2={r2:.4f} mae={mae:.4f} "
            f"({int(num_values)} valori)"
        )

    return aggrega_evaluate([msg.content for msg in valide], "num-examples")  # aggregato finale, non media


@app.main() #grid: Grid è il modo in cui il server vede e raggiunge i nodi disponibili.
def main(grid: Grid, context: Context):
    """Crea il modello globale iniziale, configura la strategia, esegue tutti i round, seleziona
    il modello con la validazione federata migliore, lo valuta una volta sola sul test federato,
    e salva il risultato. Chiamata una volta sola: quando ritorna (nulla), l'esperimento è finito."""

    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    min_available_clients: int = int(context.run_config["min-available-clients"])

    # Il modello globale viene inizializzato UNA SOLA VOLTA, qui. Il seed va fissato solo in
    # questo punto: garantisce che l'inizializzazione dei pesi sia riproducibile tra run, senza
    # azzerare l'apprendimento a ogni round (cosa che accadrebbe se il seed fosse nel client).

    seed = int(context.run_config["random-state"])
    torch.manual_seed(seed) # serve per rendere riproducibile l'inizializzazione dei pesi del modello globale (e quindi anche dei modelli locali, che partono dai pesi globali)
    global_model = build_model(context.run_config)
    arrays = ArrayRecord(global_model.state_dict())

    # "num-examples" (numero di finestre locali) è la chiave con cui FedAvg pesa sia
    # l'aggregazione dei pesi del modello sia quella delle metriche.
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_available_clients,
        min_evaluate_nodes=min_available_clients if fraction_evaluate > 0.0 else 0,
        min_available_nodes=min_available_clients,
        weighted_by_key="num-examples", # serve per pesare l'aggregazione dei pesi e delle metriche in base al numero di finestre locali di ogni istituzione (è di default)

        # train_mse è additiva (media pesata generica di Flower va bene); rmse/r2/mae
        # dell'evaluate no, quindi aggrega_evaluate le ricalcola da statistiche additive
        train_metrics_aggr_fn=aggrega_train,
        evaluate_metrics_aggr_fn=aggrega_evaluate,
    )

    # Model selection: il criterio è la validazione FEDERATA aggregata (aggrega_evaluate), sui
    # dati reali di validazione di ciascun client . L'hook tiene il
    # checkpoint del round migliore visto finora, sovrascrivendo il precedente: in memoria resta
    # un solo checkpoint. Il server non possiede dati propri: nessuna valutazione centralizzata.
    best = {"mse": float("inf"), "round": None, "state_dict": None}
    evaluate_fn = crea_checkpoint(best)

    # esegue l'intero esperimento: tutto il ciclo di training federato, con i round di
    # training e di valutazione, viene gestito da start()
    strategy.start(
        grid=grid,
        initial_arrays=arrays,  # inizializza il modello globale con i pesi random
        train_config=ConfigRecord({"lr": lr}),  #  la configurazione allegata a ogni messaggio di training
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,  # checkpoint (nessuna valutazione centralizzata), prima del round 1 e dopo ogni round
    )

    assert best["state_dict"] is not None, "Nessun round con validazione federata: controlla num-server-rounds e fraction-evaluate"

    print(
        f"\nModel selection (validazione FEDERATA): selezionato il round {best['round']} "
        f"su {num_rounds}, mse validazione={best['mse']:.4f}"
    )

    # Test finale, UNA SOLA VOLTA, federato: l'unico dato mai visto finora,
    metriche_test = valuta_test_finale(grid, best["state_dict"])
    print(
        f"Risultato finale (test federato, round {best['round']}): "
        f"mse={metriche_test['mse']:.4f} rmse={metriche_test['rmse']:.4f} "
        f"r2={metriche_test['r2']:.4f} mae={metriche_test['mae']:.4f}"
    )

    if context.run_config["save-model"]:
        print("\nSalvataggio del modello globale finale (best checkpoint) su disco...")
        torch.save(best["state_dict"], "final_model.pt")
