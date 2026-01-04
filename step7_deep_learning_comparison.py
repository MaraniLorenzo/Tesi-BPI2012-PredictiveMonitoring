import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
import joblib
import os

# Impostiamo i log per evitare messaggi rossi inutili
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# --- 1. CARICAMENTO E PREPARAZIONE ---
print("Caricamento dataset...")
df = pd.read_pickle("02_dataset_encoded.pkl")

split_point = int(len(df) * 0.80)
X = df.drop(columns=['target_tempo_rimanente', 'target_bottleneck'])
y = df['target_bottleneck']

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

# Scaling (Obbligatorio per DL)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. IL TRUCCO: CALCOLO DEI PESI (Class Weights) ---
print("Calcolo pesi per bilanciare le classi...")
# Calcoliamo quanto sono rari i ritardi
pesi_classi = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(pesi_classi))

print(f"Pesi calcolati: {class_weights_dict}")
print("La rete verrà 'punita' di più se sbaglia i ritardi (Classe 1).")

# --- 3. ARCHITETTURA PIÙ POTENTE ---
model = keras.Sequential([
    # Primo strato più largo
    layers.Dense(256, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(), # Aiuta a stabilizzare l'apprendimento
    layers.Dropout(0.4), # Spegne il 40% dei neuroni per evitare che impari a memoria

    # Secondo strato
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Terzo strato
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),

    # Output
    layers.Dense(1, activation='sigmoid')
])

# Usiamo un Learning Rate più basso per imparare con calma
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# --- 4. ADDESTRAMENTO CON PESI ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

print("\nAvvio addestramento Deep Learning (Revenge Mode)...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100, # Più epoche
    batch_size=128, # Batch più grande
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict, # <--- QUI STA LA MAGIA
    verbose=1
)

# --- 5. VALUTAZIONE ---
print("\n--- RISULTATI FINALI DEEP LEARNING ---")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

model.save("modello_deep_learning_pro.keras")