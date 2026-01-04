import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# --- 1. CARICAMENTO DATI E MODELLO ---
print("Caricamento modello e dati...")
model = joblib.load("modello_tesi_finale.joblib")
df = pd.read_pickle("02_dataset_encoded.pkl")

# Prepariamo i dati come nello step precedente
split_point = int(len(df) * 0.80)
X = df.drop(columns=['target_tempo_rimanente', 'target_bottleneck'])
X_test = X.iloc[split_point:] # Usiamo solo il test set per spiegare il futuro

print(f"Calcolo valori SHAP su {len(X_test)} campioni.")

# --- 2. CALCOLO SHAP (Heavy Computation) ---
# Usiamo TreeExplainer che è ottimizzato per XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("Calcolo completato. Generazione grafici...")

# --- 3. GRAFICO 1: BEESWARM PLOT (Il più famoso) ---
# Mostra l'impatto positivo/negativo di ogni feature
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Impatto delle Feature sui Ritardi (SHAP)", fontsize=16)
plt.tight_layout()
plt.savefig("grafico_shap_summary.png", dpi=300)
print("Salvato: grafico_shap_summary.png")

# --- 4. GRAFICO 2: BAR PLOT (Importanza assoluta) ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP)", fontsize=16)
plt.tight_layout()
plt.savefig("grafico_shap_bar.png", dpi=300)
print("Salvato: grafico_shap_bar.png")

# --- 5. GRAFICO 3: ANALISI CASO SINGOLO (Storytelling) ---
# Prendiamo un caso che il modello ha predetto come RITARDO (Classe 1)
# Cerchiamo un indice nel test set dove la predizione è 1
preds = model.predict(X_test)
risky_indices = [i for i, x in enumerate(preds) if x == 1]

if len(risky_indices) > 0:
    idx = risky_indices[0] # Prendiamo il primo "ritardatario" che troviamo
    print(f"\nAnalisi del caso specifico all'indice {idx}...")
    
    # Force Plot: mostra le forze contrastanti
    shap.force_plot(explainer.expected_value, shap_values[idx,:], X_test.iloc[idx,:], matplotlib=True, show=False)
    plt.savefig("grafico_shap_caso_singolo.png", dpi=300)
    print("Salvato: grafico_shap_caso_singolo.png")
    
    print("\n--- SPIEGAZIONE PER LA TESI ---")
    print(f"Guarda 'grafico_shap_caso_singolo.png'.")
    print("Le barre ROSSE sono le cause del ritardo.")
    print("Le barre BLU sono ciò che ha provato a velocizzare la pratica.")
else:
    print("Nessun ritardo predetto nel test set (improbabile).")

print("\nTutti i grafici sono salvati.")