import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 1. CARICAMENTO ---
print("Caricamento dataset pulito...")
dataset = pd.read_pickle("bpi2012_cleaned.pkl")
dataset = dataset.sort_values('time:timestamp') # Fondamentale ordinare per tempo

# --- 2. CALCOLO SOGLIE (Invariato) ---
eventi_w = dataset[dataset['concept:name'].str.startswith('W_')].copy()
eventi_w['durata'] = eventi_w.groupby('case:concept:name')['time:timestamp'].diff().dt.total_seconds().fillna(0)
soglie_ritardo = eventi_w.groupby('concept:name')['durata'].quantile(0.75).to_dict()

# --- 3. NUOVO: CALCOLO WORKLOAD (Pesante ma potente) ---
print("Calcolo del Workload (Casi attivi nel sistema)...")
# Creiamo una serie temporale di +1 (inizio caso) e -1 (fine caso)
casi_start = dataset.groupby('case:concept:name')['time:timestamp'].min()
casi_end = dataset.groupby('case:concept:name')['time:timestamp'].max()

timeline = pd.DataFrame({
    'time': pd.concat([casi_start, casi_end]),
    'change': pd.concat([pd.Series(1, index=casi_start.index), pd.Series(-1, index=casi_end.index)])
}).sort_values('time')

# Cumulative sum ci dice quanti casi sono aperti in ogni istante
timeline['active_cases'] = timeline['change'].cumsum()

# Uniamo questa info al dataset principale (asof merge è molto efficiente)
dataset = pd.merge_asof(dataset, timeline[['time', 'active_cases']], 
                        left_on='time:timestamp', right_on='time', 
                        direction='backward')
# Rimuoviamo la colonna time duplicata
dataset = dataset.drop(columns=['time'])

# --- 4. ESTRAZIONE FEATURE ---
dati_training = []
target_tempo = []
target_bottleneck = []

gruppi_casi = list(dataset.groupby('case:concept:name'))
print(f"Estrazione feature su {len(gruppi_casi)} casi...")

for id_caso, gruppo in tqdm(gruppi_casi):
    gruppo = gruppo.reset_index(drop=True) # Già ordinato prima
    
    lista_eventi = gruppo['concept:name'].tolist()
    lista_tempi = gruppo['time:timestamp'].tolist()
    lista_risorse = gruppo['org:resource'].tolist() if 'org:resource' in gruppo.columns else ['Sconosciuto']*len(gruppo)
    lista_workload = gruppo['active_cases'].tolist() # ### NUOVO ###
    lista_importi = gruppo['AMOUNT_REQ'].tolist() if 'AMOUNT_REQ' in gruppo.columns else [0]*len(gruppo)

    tempo_fine_caso = lista_tempi[-1]
    contatore_attivita = {att: 0 for att in dataset['concept:name'].unique()}
    
    for i in range(len(gruppo) - 1):
        attivita_corrente = lista_eventi[i]
        
        contatore_attivita[attivita_corrente] += 1
        riga_feature = contatore_attivita.copy() 
        
        # Feature complete
        riga_feature['tempo_trascorso'] = (lista_tempi[i] - lista_tempi[0]).total_seconds()
        riga_feature['ora'] = lista_tempi[i].hour
        riga_feature['importo'] = lista_importi[i]
        riga_feature['workload'] = lista_workload[i] # ### NUOVO: QUANTO E' INTASATA LA BANCA ###
        
        # Categoriche
        riga_feature['ultima_attivita'] = attivita_corrente
        riga_feature['risorsa'] = lista_risorse[i]
        
        # Target
        tempo_rimanente = (tempo_fine_caso - lista_tempi[i]).total_seconds() / 86400
        
        # Bottleneck target
        prossima_attivita = lista_eventi[i+1]
        durata_prossima = (lista_tempi[i+1] - lista_tempi[i]).total_seconds()
        e_in_ritardo = 0
        if prossima_attivita in soglie_ritardo:
            if durata_prossima > soglie_ritardo[prossima_attivita]:
                e_in_ritardo = 1
        
        dati_training.append(riga_feature)
        target_tempo.append(tempo_rimanente)
        target_bottleneck.append(e_in_ritardo)

# --- 5. SALVATAGGIO ---
X = pd.DataFrame(dati_training)
X = pd.get_dummies(X, columns=['ultima_attivita', 'risorsa'], prefix=['stato', 'res'])
X = X.fillna(0)
X['target_tempo_rimanente'] = target_tempo
X['target_bottleneck'] = target_bottleneck

print(f"Dataset PRO pronto: {X.shape[0]} righe. Salvataggio...")
X.to_pickle("02_dataset_encoded.pkl")