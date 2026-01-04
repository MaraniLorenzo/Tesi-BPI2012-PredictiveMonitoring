import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Configurazione pagina
st.set_page_config(page_title="BPI 2012 - Predictive Monitor", layout="wide")

# --- 1. CARICAMENTO RISORSE (Cachato per velocità) ---
@st.cache_resource
def load_resources():
    model = joblib.load("modello_tesi_finale.joblib")
    data = pd.read_pickle("02_dataset_encoded.pkl")
    return model, data

try:
    model, df = load_resources()
except:
    st.error("Errore: Esegui prima gli step precedenti per generare 'modello_tesi_finale.joblib' e '02_dataset_encoded.pkl'")
    st.stop()

# --- 2. INTERFACCIA LATERALE ---
st.sidebar.title("🎛️ Control Panel")
st.sidebar.info("Sistema di monitoraggio predittivo per il processo BPI Challenge 2012.")

# Filtri
split_point = int(len(df) * 0.80)
test_data = df.iloc[split_point:].copy() # Usiamo solo dati futuri
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

# Creiamo un dataframe leggibile
results = X_test.copy()
results['PREDIZIONE_RITARDO'] = preds
results['RISCHIO_REALE'] = y_test_bott
results = results[results['PREDIZIONE_RITARDO'] == 1] # Mostra solo i ritardi predetti

if not results.empty:
    st.dataframe(results.head(10).style.applymap(lambda x: 'background-color: #ffcccc' if x == 1 else '', subset=['PREDIZIONE_RITARDO']))
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
        if 'workload' in record:
            st.write(f"📉 **Workload Sistema:** {record['workload']:.0f} casi attivi")
        if 'res_112' in record and record['res_112'] == 1:
            st.error("👤 **Risorsa:** User 112 (Critica)")
        st.write(f"💰 **Importo:** {record['importo']}")
        
    with col_dx:
        st.write("**Forze in Gioco (SHAP Force Plot):**")
        st.caption("Rosso = Spinge verso il ritardo | Blu = Riduce il rischio")
        
        # Calcolo SHAP live
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.loc[[selected_index]])
        
        # --- FIX GRAFICO ---
        # Usiamo plt.gcf() (Get Current Figure) per essere sicuri di prendere il grafico giusto
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.loc[selected_index], matplotlib=True, show=False, figsize=(12,4))
        st.pyplot(plt.gcf(), clear_figure=True)
else:
    st.info("Seleziona un caso dalla lista sopra.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Sviluppato con Python, XGBoost & Streamlit per Tesi,")