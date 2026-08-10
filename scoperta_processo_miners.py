import pm4py
import pandas as pd
import os

print("1. Caricamento del dataset pulito...")
# Ottiene la cartella dove si trova questo script
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "bpi2012_cleaned.pkl")

if not os.path.exists(dataset_path):
    print(f"Errore: Il file {dataset_path} non esiste in questa cartella.")
    print("Assicurati di eseguire lo script nella cartella dove si trova il dataset.")
    exit()

df = pd.read_pickle(dataset_path)

# pm4py richiede che il dataframe sia formattato con le chiavi standard
print("Formattazione del dataframe per pm4py...")
df = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')


print("\n--- ESECUZIONE ALPHA MINER ---")
print("Calcolo in corso (potrebbe creare il modello a spaghetti/flower model)...")
try:
    net_alpha, im_alpha, fm_alpha = pm4py.discover_petri_net_alpha(df)
except AttributeError:
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    net_alpha, im_alpha, fm_alpha = alpha_miner.apply(df)

print("Salvataggio del grafo Alpha Miner...")
# Salva l'immagine del modello
alpha_path = os.path.join(script_dir, "alpha_miner.png")
pm4py.save_vis_petri_net(net_alpha, im_alpha, fm_alpha, alpha_path)
print(f"-> Immagine Alpha Miner salvata in: {alpha_path}")


print("\n--- ESECUZIONE HEURISTICS MINER ---")
print("Calcolo in corso (estrazione del processo dominante)...")
# Usiamo i filtri di frequenza per avere il grafico pulito per la tesi
try:
    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.9, min_dfg_occurrences=500)
except TypeError:
    heu_net = pm4py.discover_heuristics_net(df, parameters={"dependency_threshold": 0.9, "min_dfg_occurrences": 500})

print("Salvataggio del grafo Heuristics Miner...")
heu_path = os.path.join(script_dir, "heuristics_miner.png")
pm4py.save_vis_heuristics_net(heu_net, heu_path)
print(f"-> Immagine Heuristics Miner salvata in: {heu_path}")

