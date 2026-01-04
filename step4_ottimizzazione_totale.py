import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, mean_absolute_error, f1_score
from sklearn.model_selection import TimeSeriesSplit
import joblib # Per salvare il modello finale

# --- 1. CARICAMENTO ---
print("Caricamento dataset...")
df = pd.read_pickle("02_dataset_encoded.pkl")

# --- 2. PREPARAZIONE DATI ---
# Split Temporale 80/20
split_point = int(len(df) * 0.80)
X = df.drop(columns=['target_tempo_rimanente', 'target_bottleneck'])
y_bott = df['target_bottleneck']
y_time = df['target_tempo_rimanente']

X_train_raw = X.iloc[:split_point]
y_train_raw = y_bott.iloc[:split_point]
X_test = X.iloc[split_point:]
y_test = y_bott.iloc[split_point:]

# --- 3. DATA AUGMENTATION (SMOTE) ---
# Qui "aumentiamo i dati". Creiamo ritardi sintetici per bilanciare le classi.
print(f"Generazione dati sintetici (SMOTE) sul Training Set...")
print(f"Originale: {y_train_raw.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_raw, y_train_raw)

print(f"Dopo SMOTE (Dati Aumentati): {y_train_bal.value_counts().to_dict()}")
print("Ora il modello ha molti più esempi di ritardi su cui imparare.")

# --- 4. OTTIMIZZAZIONE AUTOMATICA (OPTUNA) ---
# Definiamo la funzione che Optuna deve ottimizzare
def objective(trial):
    # Parametri che Optuna proverà a cambiare
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1,
        'random_state': 42,
        # 'tree_method': 'gpu_hist' # Scommenta se hai una GPU NVIDIA configurata
    }
    
    # Addestriamo il modello con questi parametri
    clf = xgb.XGBClassifier(**params)
    
    # Usiamo una validazione temporale (non random) per essere rigorosi
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    # Validazione rapida interna
    for train_index, val_index in tscv.split(X_train_bal):
        X_t, X_v = X_train_bal.iloc[train_index], X_train_bal.iloc[val_index]
        y_t, y_v = y_train_bal.iloc[train_index], y_train_bal.iloc[val_index]
        
        clf.fit(X_t, y_t)
        preds = clf.predict(X_v)
        scores.append(f1_score(y_v, preds))
    
    return np.mean(scores) # Optuna cercherà di massimizzare questo valore

print("\n--- AVVIO RICERCA PARAMETRI OTTIMALI (AI vs AI) ---")
print("Il sistema farà 20 tentativi intelligenti. Può richiedere qualche minuto...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) # Metti 50 o 100 se hai tempo (es. vai a pranzo)

print("\nPARAMETRI VINCENTI:")
best_params = study.best_params
print(best_params)

# --- 5. ADDESTRAMENTO FINALE "THE BEAST" ---
print("\n--- ADDESTRAMENTO MODELLO DEFINITIVO ---")
print("Addestramento su dati AUMENTATI con parametri OTTIMIZZATI...")

# Aggiungiamo i parametri fissi necessari
best_params['n_jobs'] = -1
best_params['random_state'] = 42

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train_bal, y_train_bal) # Addestriamo sui dati aumentati

# Predizione
print("Test sul futuro (Dati reali non toccati)...")
y_pred = final_model.predict(X_test)

# Report
print("\n=== RISULTATI FINALI EXTREME ===")
print(classification_report(y_test, y_pred))

# Salviamo il modello per la gloria
joblib.dump(final_model, "modello_tesi_finale.joblib")
print("Modello salvato.")