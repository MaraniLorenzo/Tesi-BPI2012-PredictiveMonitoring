import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Configurazione pagina
st.set_page_config(page_title="BPI 2012 - Predictive Monitor", layout="wide")

# --- 1. CARICAMENTO RISORSE CON CACHING ---
@st.cache_resource
def carica_modello():
    return joblib.load("modello_tesi_finale.joblib")

@st.cache_data
def carica_dati():
    return pd.read_pickle("02_dataset_encoded.pkl")

try:
    model = carica_modello()
    df = carica_dati()
except Exception as e:
    st.error(f"Errore nel caricamento dati: {e}")
    st.stop()

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
        'ora': 'Ora Evento (0-23)',
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

# --- 2. INTERFACCIA LATERALE ---
st.sidebar.title("🎛️ Control Panel")
st.sidebar.info("Sistema di monitoraggio predittivo per il processo BPI Challenge 2012.")

# Filtri 
val_split = int(len(df) * 0.80)
test_data = df.iloc[val_split:].copy() # Usiamo solo l'ultimo 20% (Test Set futuro)
X_test = test_data.drop(columns=['target_tempo_rimanente', 'target_bottleneck'])
y_test_bott = test_data['target_bottleneck']

st.sidebar.write(f"Casi nel Test Set: {len(X_test)}")

# --- 3. DASHBOARD PRINCIPALE ---
st.title("🏦 BPI 2012: Predictive Process Monitoring")

# KPI Generali
col1, col2, col3 = st.columns(3)
preds = model.predict(X_test)
n_ritardi = sum(preds)
perc_ritardi = (n_ritardi / len(preds)) * 100

col1.metric("Casi Monitorati", len(X_test))
col2.metric("Ritardi Previsti", f"{n_ritardi}", delta_color="inverse")
col3.metric("Rischio Globale", f"{perc_ritardi:.1f}%")

st.divider()

# --- 4. ANALISI CASI A RISCHIO ---
st.subheader("🚨 Casi Critici (Top Priority)")
st.write("Questi sono i casi che il modello XGBoost ha identificato come probabili colli di bottiglia.")

# Creiamo un dataframe sintetico e chiarissimo per l'utente
results = X_test.copy()
results['RITARDO_PREDETTO'] = preds
results['RITARDO_EFFETTIVO'] = y_test_bott
results = results[results['RITARDO_PREDETTO'] == 1] # Mostra solo i ritardi predetti

if not results.empty:
    # Estraiamo la fase attuale e l'operatore direttamente dai dati
    stato_cols = [c for c in X_test.columns if c.startswith('stato_')]
    res_cols = [c for c in X_test.columns if c.startswith('res_')]
    
    tabella_sintetica = pd.DataFrame(index=results.index)
    tabella_sintetica['Allarme'] = '🔴 RITARDO PREVISTO'
    tabella_sintetica['Esito Reale'] = results['RITARDO_EFFETTIVO'].map({1: '🔴 RITARDO CONFERMATO', 0: '🟢 REGOLARE'})
    tabella_sintetica['Importo Richiesto'] = results['importo'].map(lambda x: f"€ {int(x):,}".replace(",", "."))
    tabella_sintetica['Tempo Trascorso'] = (results['tempo_trascorso'] / 86400).map(lambda x: f"{x:.1f} giorni")
    
    # Decodifica fase attuale in italiano
    raw_fasi = results[stato_cols].idxmax(axis=1).str.replace('stato_', '')
    tabella_sintetica['Fase Attuale'] = raw_fasi.map(lambda f: traduci_feature(f))
    
    # Decodifica operatore/risorsa in italiano
    raw_res = results[res_cols].idxmax(axis=1).str.replace('res_', '')
    tabella_sintetica['Operatore / Risorsa'] = raw_res.map(lambda r: "Sistema Automatico (User 112)" if str(r) == '112' else f"Operatore (User {r})")
    
    tabella_sintetica['Workload Sistema'] = results['workload'].map(lambda w: f"{int(w)} pratiche")
    tabella_sintetica['Ora Evento'] = results['ora'].map(lambda h: f"{int(h)}:00")
    
    # Visualizzazione tabella principale ultra-leggibile
    st.dataframe(tabella_sintetica.head(15), use_container_width=True)
    
    with st.expander("🛠️ Visualizza Matrice Tecnica Completa (Tutte le 120 Feature)"):
        results_it = results.rename(columns=traduci_feature)
        if 'Tempo Trascorso (Giorni)' in results_it.columns:
            results_it['Tempo Trascorso (Giorni)'] = (results_it['Tempo Trascorso (Giorni)'] / 86400).map('{:.1f}'.format)
        st.dataframe(results_it.head(15))
else:
    st.success("Nessun ritardo previsto al momento!")

# --- 5. XAI: PERCHÉ QUESTO RITARDO? ---
st.divider()
st.subheader("🔍 Ispezione Dettagliata (Explainable AI)")

# Selectbox per scegliere un caso specifico
if not results.empty:
    selected_index = st.selectbox("Seleziona un caso critico da analizzare:", results.index[:20])
    
    col_sx, col_dx = st.columns([1, 2])
    
    with col_sx:
        st.write(f"**Analisi Caso ID:** {selected_index}")
        record = X_test.loc[selected_index]
        
        # Mostra i dati chiave del caso
        st.write("--- Dati Chiave ---")
        if 'importo' in record:
            st.write(f"💰 **Importo:** € {record['importo']:,.0f}".replace(",", "."))
        if 'tempo_trascorso' in record:
            st.write(f"⏱️ **Tempo Trascorso:** {record['tempo_trascorso']/86400:.1f} giorni")
        if 'workload' in record:
            st.write(f"📉 **Workload Sistema:** {record['workload']:.0f} pratiche concorrenti")
        if 'res_112' in record and record['res_112'] == 1:
            st.error("👤 **Risorsa:** User 112 (Sistema Automatico)")
            
    with col_dx:
        st.write("**Fattori di Rischio (SHAP Waterfall Plot):**")
        st.caption("Rosso (+): Aumenta il rischio ritardo | Blu (-): Riduce il rischio")
        
        # Calcolo spiegazione SHAP
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_test.loc[[selected_index]])
        explanation.feature_names = [traduci_feature(c) for c in X_test.columns]
        
        # Convertiamo il valore grezzo dei secondi in giorni per il rendering nel grafico
        if 'tempo_trascorso' in X_test.columns:
            tempo_idx = list(X_test.columns).index('tempo_trascorso')
            explanation.data[0, tempo_idx] = round(float(explanation.data[0, tempo_idx]) / 86400, 1)
        
        # Rendering Waterfall Plot nitido e leggibile
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(explanation[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
else:
    st.info("Seleziona un caso dalla lista sopra.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Sviluppato con Python, XGBoost & Streamlit per Tesi")