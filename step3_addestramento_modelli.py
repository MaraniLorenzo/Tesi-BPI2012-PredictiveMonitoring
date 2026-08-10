import pandas as pd
import numpy as np
import xgboost as xgb # ### NUOVO: Libreria avanzata
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, classification_report, confusion_matrix

# --- 1. CARICAMENTO ---
print("Caricamento dataset...")
df = pd.read_pickle("02_dataset_encoded.pkl")

# --- 2. SPLIT TEMPORALE (60 / 20 / 20) ---
train_split = int(len(df) * 0.60)
val_split = int(len(df) * 0.80)

colonne_target = ['target_tempo_rimanente', 'target_bottleneck']
X = df.drop(columns=colonne_target)
y_tempo = df['target_tempo_rimanente']
y_bottleneck = df['target_bottleneck']

# Training Set (60%)
X_train = X.iloc[:train_split]
y_tempo_train = y_tempo.iloc[:train_split]
y_bott_train = y_bottleneck.iloc[:train_split]

# Validation Set (20%) - Spartiacque temporale
X_val = X.iloc[train_split:val_split]
y_tempo_val = y_tempo.iloc[train_split:val_split]
y_bott_val = y_bottleneck.iloc[train_split:val_split]

# Test Set (20%) - Il "Futuro" su cui testiamo
X_test = X.iloc[val_split:]
y_tempo_test = y_tempo.iloc[val_split:]
y_bott_test = y_bottleneck.iloc[val_split:]

print(f"Training su {len(X_train)} campioni | Validation: {len(X_val)} | Test: {len(X_test)} con {len(X.columns)} feature.")

# --- 3. XGBOOST REGRESSOR (Tempo) ---
print("\n--- XGBoost: Predizione Tempo ---")
# n_estimators=500 sfrutta la tua RAM per fare un modello molto profondo
model_reg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42)
model_reg.fit(X_train, y_tempo_train)

y_pred_tempo = model_reg.predict(X_test)
mae = mean_absolute_error(y_tempo_test, y_pred_tempo)
rmse = root_mean_squared_error(y_tempo_test, y_pred_tempo)

print(f"MAE: {mae:.2f} giorni (Prima era ~7.14)")
print(f"RMSE: {rmse:.2f} giorni")

# --- 4. XGBOOST CLASSIFIER (Bottleneck) ---
print("\n--- XGBoost: Predizione Colli di Bottiglia ---")
# scale_pos_weight aiuta con le classi sbilanciate (sostituisce class_weight='balanced')
ratio = float(np.sum(y_bott_train == 0)) / np.sum(y_bott_train == 1)
model_class = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, scale_pos_weight=ratio, n_jobs=-1, random_state=42)

model_class.fit(X_train, y_bott_train)
y_pred_bott = model_class.predict(X_test)

print(classification_report(y_bott_test, y_pred_bott))
print("Matrice Confusione:")
print(confusion_matrix(y_bott_test, y_pred_bott))

# Feature Importance per XGBoost
print("\nTOP FEATURE (Workload è importante?):")
importance = pd.Series(model_class.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print(importance)