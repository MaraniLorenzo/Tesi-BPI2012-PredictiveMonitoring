import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# --- 1. CARICAMENTO DATI E MODELLO ---
print("Caricamento modello e dati...")
model = joblib.load("modello_tesi_finale.joblib")
df = pd.read_pickle("02_dataset_encoded.pkl")

# Prepariamo i dati isolando l'ultimo 20% (Test Set) come nel setup 60/20/20
val_split = int(len(df) * 0.80)
X = df.drop(columns=['target_tempo_rimanente', 'target_bottleneck'])
X_test = X.iloc[val_split:] 

print(f"Calcolo valori SHAP su {len(X_test)} campioni.")

# Dizionario di traduzione semantica per i grafici SHAP
def traduci_feature(col):
    translations = {
        'W_Afhandelen leads': 'Gestione Contatti Iniziali',
        'W_Completeren aanvraag': 'Raccolta Dati / Compilazione',
        'W_Valideren aanvraag': 'Valutazione Domanda',
        'W_Nabellen offertes': 'Ricontatto Offerte Inviate',
        'W_Nabellen incomplete dossiers': 'Richiesta Documenti Mancanti',
        'W_Beoordelen fraude': 'Verifica Antifrode',
        'W_Wijzigen contractgegevens': 'Modifica Dati Contratto',
        'A_SUBMITTED': 'Domanda Inviata',
        'A_PARTLYSUBMITTED': 'Domanda Parziale',
        'A_PREACCEPTED': 'Domanda Pre-Approvata',
        'A_ACCEPTED': 'Domanda Accettata',
        'A_FINALIZED': 'Domanda Finalizzata',
        'A_DECLINED': 'Domanda Rifiutata',
        'A_CANCELLED': 'Domanda Annullata',
        'A_APPROVED': 'Domanda Approvata',
        'A_ACTIVATED': 'Domanda Attivata',
        'A_REGISTERED': 'Domanda Registrata',
        'O_SELECTED': 'Offerta Selezionata',
        'O_CREATED': 'Offerta Creata',
        'O_SENT': 'Offerta Inviata',
        'O_SENT_BACK': 'Offerta Restituita',
        'O_ACCEPTED': 'Offerta Accettata',
        'O_CANCELLED': 'Offerta Annullata',
        'O_DECLINED': 'Offerta Rifiutata',
        'tempo_trascorso': 'Tempo Trascorso (Giorni)',
        'workload': 'Carico di Lavoro Globale',
        'importo': 'Importo Richiesto (€)',
        'durata_giorni': 'Giorni Durata Totale',
        'event_count': 'Totale Eventi Registrati'
    }
    if col in translations:
        return translations[col]
    if col.startswith('stato_'):
        base = col.replace('stato_', '')
        return f"Stato: {translations.get(base, base)}"
    if col.startswith('res_'):
        res_id = col.replace('res_', '')
        if res_id == '112':
            return "Sistema Automatico (User 112)"
        return f"Operatore (User {res_id})"
    return col

# Rinominiamo le colonne in italiano per i grafici globali
feature_names_it = [traduci_feature(c) for c in X_test.columns]

# Prepariamo un campione rappresentativo (2.000 casi) per i grafici globali (Beeswarm e Bar Plot)
sample_size = min(2000, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42).copy()
X_sample_it = X_sample.copy()
if 'tempo_trascorso' in X_sample.columns:
    X_sample_it['tempo_trascorso'] = (X_sample_it['tempo_trascorso'] / 86400).round(1)
X_sample_it.columns = feature_names_it

# --- 2. CALCOLO SHAP GLOBALE ---
print(f"Calcolo valori SHAP su campione rappresentativo di {sample_size} casi...")
explainer = shap.TreeExplainer(model)
shap_values_sample = explainer.shap_values(X_sample)

# --- 3. GRAFICO 1: BEESWARM SUMMARY PLOT (Top 20 Feature in Italiano per Tesi) ---
plt.figure(figsize=(9, 10))
shap.summary_plot(shap_values_sample, X_sample_it, max_display=20, show=False)
plt.title("Impatto delle Feature sui Ritardi (SHAP)", fontsize=14, pad=15)
plt.xlabel("Valore SHAP (impatto sull'output del modello)", fontsize=11)
plt.tight_layout()
plt.savefig("grafico_shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("Salvato: grafico_shap_summary.png")

# --- 4. GRAFICO 2: FORCE PLOT (Spiegazione Locale Singolo Caso con Rotazione Testo) ---
preds = model.predict(X_test)
risky_cases = [i for i, x in enumerate(preds) if x == 1]
if len(risky_cases) > 0:
    idx = risky_cases[0]
    X_single = X_test.iloc[[idx]].copy()
    if 'tempo_trascorso' in X_single.columns:
        X_single['tempo_trascorso'] = (X_single['tempo_trascorso'] / 86400).round(1)
    X_single.columns = feature_names_it
    
    shap_val_single = explainer.shap_values(X_single)
    
    plt.figure(figsize=(20, 4))
    shap.force_plot(explainer.expected_value, shap_val_single[0], X_single.iloc[0], matplotlib=True, show=False, text_rotation=30)
    plt.savefig("grafico_shap_force_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Salvato: grafico_shap_force_plot.png")

print("\nI 2 grafici sono stati generati con successo")