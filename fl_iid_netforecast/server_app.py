"""ServerApp: strategia di aggregazione FL sui modelli LSTM locali dei client (ciascuno su un blocco IID)."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_iid_netforecast.task import build_model, istituzioni_pool_iid

app = ServerApp()


def aggrega_e_stampa(records: list[RecordDict], weighting_metric_name: str, fase: str) -> MetricRecord:
    """Stampa quanti client hanno partecipato a questa fase del round, poi delega l'aggregazione vera a Flower.
    Con lo split IID ogni client mescola più istituzioni, quindi qui non ha più senso elencarle
    per round: l'elenco delle istituzioni del pool viene stampato una sola volta in main()."""

    print(f" {fase} -> {len(records)} client coinvolti")
    return aggregate_metricrecords(records, weighting_metric_name)  #aggregate_metricrecords è la funzione DI FLOWER


def aggrega_train(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    return aggrega_e_stampa(records, weighting_metric_name, "train")


def aggrega_evaluate(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    return aggrega_e_stampa(records, weighting_metric_name, "evaluate")


@app.main() #grid: Grid è il modo in cui il server vede e raggiunge i nodi disponibili.
def main(grid: Grid, context: Context):
    """Crea il modello globale iniziale, configura la strategia, esegue tutti i round e salva
    il risultato. Chiamata una volta sola: quando ritorna (nulla), l'esperimento è finito."""
    

    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    min_available_clients: int = int(context.run_config["min-available-clients"])

    # Pool di istituzioni usato dallo split IID, stampato una volta a inizio run
    # (fisso per tutta la durata del run: ogni client ne riceve una fetta a ogni round).
    n_client = len(list(grid.get_node_ids()))
    pool = istituzioni_pool_iid(n_client, context.run_config)
    print(f"\nClient: {n_client} | Pool istituzioni IID ({len(pool)}): {pool}\n")

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
        
        # stampano quanti client hanno partecipato, poi delegano l'aggregazione a Flower
        train_metrics_aggr_fn=aggrega_train,
        evaluate_metrics_aggr_fn=aggrega_evaluate,
    )


    # esegue l'intero esperimento: tutto il ciclo di training federato, con i round di
    # training e di valutazione, viene gestito da start()
    result = strategy.start(    #result: contiene i pesi del modello globale finale e le metriche aggregate di training e valutazione
        grid=grid,
        initial_arrays=arrays,  # inizializza il modello globale con i pesi random 
        train_config=ConfigRecord({"lr": lr}),  #  la configurazione allegata a ogni messaggio di training
        num_rounds=num_rounds,
    )

    if context.run_config["save-model"]:
        print("\nSalvataggio del modello globale finale su disco...")
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
