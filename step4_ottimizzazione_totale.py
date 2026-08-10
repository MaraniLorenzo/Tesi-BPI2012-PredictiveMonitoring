import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, mean_absolute_error, f1_score
import joblib

# --- 1. CARICAMENTO ---
print("Caricamento dataset...")
df = pd.read_pickle("02_dataset_encoded.pkl")

# --- 2. PREPARAZIONE DATI (60 / 20 / 20) ---
train_split = int(len(df) * 0.60)
val_split = int(len(df) * 0.80)

X = df.drop(columns=['target_tempo_rimanente', 'target_bottleneck'])
y_bott = df['target_bottleneck']

X_train_raw = X.iloc[:train_split]
y_train_raw = y_bott.iloc[:train_split]

X_val = X.iloc[train_split:val_split]
y_val = y_bott.iloc[train_split:val_split]

X_test = X.iloc[val_split:]
y_test = y_bott.iloc[val_split:]

# --- 3. DATA AUGMENTATION (SMOTE solo sul 60%) ---
print(f"Generazione dati sintetici (SMOTE) sul Training Set...")
print(f"Originale: {y_train_raw.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_raw, y_train_raw)

print(f"Dopo SMOTE (Dati Aumentati): {y_train_bal.value_counts().to_dict()}")
print("Ora il modello ha molti più esempi di ritardi su cui imparare.")

# --- 4. OTTIMIZZAZIONE AUTOMATICA (OPTUNA) ---
def objective(trial):
    # Parametri che Optuna proverà a cambiare
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Addestriamo sul 60% bilanciato da SMOTE
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_bal, y_train_bal)
    
    # VALUTIAMO SUL 20% DI VALIDATION (Senza SMOTE)
    y_pred_val = model.predict(X_val)
    score = f1_score(y_val, y_pred_val)
    
    return score  # Optuna cercherà di massimizzare questo valore

print("\n--- AVVIO RICERCA PARAMETRI OTTIMALI (AI vs AI) ---")
print("Il sistema farà 20 tentativi intelligenti. Può richiedere qualche minuto...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) 

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
final_model.fit(X_train_bal, y_train_bal)  # Addestriamo sui dati aumentati

# Predizione
print("Test sul futuro (Test Set non toccato né da SMOTE né da Optuna)...")
y_pred = final_model.predict(X_test)

# Report
print("\n=== RISULTATI FINALI EXTREME ===")
print(classification_report(y_test, y_pred))

# Salviamo il modello per la gloria
joblib.dump(final_model, "modello_tesi_finale.joblib")
print("Modello salvato.")