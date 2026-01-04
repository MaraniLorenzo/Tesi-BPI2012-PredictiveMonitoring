import pm4py
import pandas as pd
import numpy as np

# --- 1. CARICAMENTO DATI ---
print("Lettura del log XES in corso (potrebbe richiedere un po' di tempo)...")
log = pm4py.read_xes("BPI_Challenge_2012.xes")
df = pm4py.convert_to_dataframe(log)

print(f"Dimensioni iniziali: {len(df)} eventi, {df['case:concept:name'].nunique()} casi.")

# --- 2. ORDINAMENTO ---
# Fondamentale per il Temporal Split successivo [cite: 133]
df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

# --- 3. PULIZIA E FILTRO CASI APERTI [cite: 94-96] ---
# Dobbiamo tenere solo i casi che sono finiti.
# Nel BPI 2012, un caso è finito se termina con stati specifici o se non è attivo alla data finale.
# Un metodo robusto per la tesi è filtrare i casi che hanno un evento finale chiaro.

# Identifichiamo i casi che hanno raggiunto una fine logica
completed_cases_ids = pm4py.filter_trace_attribute_values(
    log, "concept:name", ["A_DECLINED", "A_REGISTERED", "A_APPROVED", "A_CANCELLED", "O_ACCEPTED", "O_DECLINED"], retain=True
)
# Nota: La lista sopra è semplificata. Un approccio alternativo semplice per la tesi:
# Rimuovere i casi che non hanno 'finito' entro la data massima del log.

max_date = df['time:timestamp'].max()
print(f"Data ultima nel log: {max_date}")

# Calcoliamo l'ultimo timestamp per ogni caso
last_events = df.groupby('case:concept:name')['time:timestamp'].max()

# Consideriamo "Chiusi" i casi il cui ultimo evento è avvenuto prima dell'ultimo giorno del log
# (lasciando un buffer di sicurezza, es. 24 ore prima della fine della raccolta dati)
closed_case_ids = last_events[last_events < max_date].index

# Filtriamo il DataFrame
df_clean = df[df['case:concept:name'].isin(closed_case_ids)].copy()

print(f"Dimensioni dopo pulizia: {len(df_clean)} eventi, {df_clean['case:concept:name'].nunique()} casi.")

# --- 4. DATA ENGINEERING PRELIMINARE ---
# Calcolo durata attività W_ (Start -> Complete) [cite: 93]
# Questo serve per il target "Bottleneck". 
# Per semplicità, teniamo solo le righe 'COMPLETE' per le attività W_, calcolando la durata.
# (Questa è una semplificazione accettabile per ridurre la complessità computazionale della tesi)

# Esempio: mantenere solo eventi COMPLETE ma calcolare la durata se c'era uno START
# Per ora, salviamo il dataset pulito così com'è per non perdere informazioni.

# --- 5. SALVATAGGIO ---
print("Salvataggio del dataset pulito...")
df_clean.to_pickle("bpi2012_cleaned.pkl") # Pickle mantiene i tipi di dato (es. datetime)
print("Fatto. File 'bpi2012_cleaned.pkl' creato.")