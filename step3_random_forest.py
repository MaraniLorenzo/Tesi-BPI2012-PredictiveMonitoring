import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Caricamento dataset...")
df = pd.read_pickle("02_dataset_encoded.pkl")

# Split temporale come nello step 3
train_split = int(len(df) * 0.60)
val_split = int(len(df) * 0.80)

# Rimuovo tutte le colonne aggiuntive (Risorse, Workload, Importo, Ora) 
# per avere un dataset di "Solo Control Flow" (Fase 1)
colonne_da_rimuovere = ['target_tempo_rimanente', 'target_bottleneck', 'workload', 'ora', 'importo']
colonne_risorse = [col for col in df.columns if col.startswith('res_')]
colonne_da_rimuovere.extend(colonne_risorse)

X = df.drop(columns=colonne_da_rimuovere)
y_bottleneck = df['target_bottleneck']

# Training
X_train = X.iloc[:train_split]
y_bott_train = y_bottleneck.iloc[:train_split]

# Test
X_test = X.iloc[val_split:]
y_bott_test = y_bottleneck.iloc[val_split:]

print("\n--- Random Forest: Predizione Colli di Bottiglia (Baseline) ---")
# Fase 1: Random Forest senza bilanciamento avanzato (class_weight=None) o con parametri di default
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Addestramento in corso...")
model_rf.fit(X_train, y_bott_train)
y_pred_rf = model_rf.predict(X_test)

print("\nRisultati sul Test Set:")
print(classification_report(y_bott_test, y_pred_rf))
