"""ServerApp: strategia di aggregazione FL sui modelli LSTM locali dei client (ciascuno su un blocco IID)."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_iid_netforecast.task import build_model, get_device, load_test_globale, metriche_da_statistiche_additive
from fl_iid_netforecast.task import test as test_fn

app = ServerApp()


def aggrega_train(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    """Media pesata (per num-examples) della train_mse locale di ogni client. La MSE è additiva
    (media di errori al quadrato), quindi qui la media pesata generica di Flower
    (aggregate_metricrecords) è già esatta: non serve una logica custom come per l'evaluate."""
    print(f" train -> {len(records)} client coinvolti")
    return aggregate_metricrecords(records, weighting_metric_name)  #aggregate_metricrecords è la funzione DI FLOWER


def aggrega_evaluate(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    """Aggrega le metriche di evaluate senza usare la media pesata generica di Flower per
    rmse/r2/harmonic: non sono lineari, quindi mediare i valori già
    calcolati da ogni client non equivale a calcolarli sull'unione dei dati di tutti i client
    (esempio: due client con rmse locale 1 e 3 sullo stesso numero di finestre non danno un
    rmse globale di 2, ma sqrt(5) ≈ 2.236).

    Ogni client manda invece statistiche additive. Qui vengono sommate su tutti i client coinvolti in questo
    round, e le metriche finali vengono calcolate una sola volta, come se il modello fosse stato
    valutato sull'intero test set federato in un unico batch."""
    print(f" evaluate -> {len(records)} client coinvolti")

    total_sse = total_sum_y = total_sum_y_sq = 0.0
    total_num_values = 0
    for record in records:
        metricrecord = next(iter(record.metric_records.values()))
        total_sse += metricrecord["sse"]
        total_sum_y += metricrecord["sum_y"]
        total_sum_y_sq += metricrecord["sum_y_sq"]
        total_num_values += metricrecord["num_values"]

    mse, rmse, r2, h_score = metriche_da_statistiche_additive(
        total_sse, total_sum_y, total_sum_y_sq, total_num_values
    )
    return MetricRecord(
        {"mse": mse, "rmse": rmse, "r2": r2, "harmonic": h_score, "num_values": total_num_values}
    )


def crea_evaluate_centralizzato(run_config: dict, num_partitions: int):
    """Costruisce evaluate_fn per Flower (parametro di strategy.start): valuta il modello
    globale aggregato sul test set globale del server"""
    device = get_device()
    test_loader = load_test_globale(num_partitions, run_config)
    n_finestre = sum(len(X) for X, _ in test_loader)
    print(f"Global evaluation test set (server): {n_finestre} finestre, mai assegnate ai client\n")

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = build_model(run_config)
        model.load_state_dict(arrays.to_torch_state_dict())
        mse, rmse, r2, h_score, _, _ = test_fn(model, test_loader, device)
        print(
            f" global evaluation (server) -> round {server_round}: "
            f"mse={mse:.4f} rmse={rmse:.4f} r2={r2:.4f} harmonic={h_score:.4f}"
        )
        return MetricRecord({"mse": mse, "rmse": rmse, "r2": r2, "harmonic": h_score})

    return evaluate_fn


@app.main() #grid: Grid è il modo in cui il server vede e raggiunge i nodi disponibili.
def main(grid: Grid, context: Context):
    """Crea il modello globale iniziale, configura la strategia, esegue tutti i round e salva
    il risultato. Chiamata una volta sola: quando ritorna (nulla), l'esperimento è finito."""
    

    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    min_available_clients: int = int(context.run_config["min-available-clients"])

    n_client = len(list(grid.get_node_ids()))

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
        
        # train_mse è additiva (media pesata generica di Flower va bene); rmse/r2/harmonic
        # dell'evaluate no, quindi aggrega_evaluate le ricalcola da statistiche additive
        train_metrics_aggr_fn=aggrega_train,
        evaluate_metrics_aggr_fn=aggrega_evaluate,
    )


    # global evaluation, il server valuta da sé il modello globale aggregato su un test set suo.
    evaluate_fn = crea_evaluate_centralizzato(context.run_config, n_client)

    # esegue l'intero esperimento: tutto il ciclo di training federato, con i round di
    # training e di valutazione, viene gestito da start()
    result = strategy.start(    #result: contiene i pesi del modello globale finale e le metriche aggregate di training e valutazione
        grid=grid,
        initial_arrays=arrays,  # inizializza il modello globale con i pesi random
        train_config=ConfigRecord({"lr": lr}),  #  la configurazione allegata a ogni messaggio di training
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,  # valutazione centralizzata lato server, prima del round 1 e dopo ogni round
    )

    if context.run_config["save-model"]:
        print("\nSalvataggio del modello globale finale su disco...")
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
