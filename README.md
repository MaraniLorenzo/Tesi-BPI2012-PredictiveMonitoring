# Predictive Process Monitoring - BPI Challenge 2012

Progetto di Tesi Triennale in Ingegneria Informatica.
Autore: [Il Tuo Nome]

## Descrizione
Questo progetto implementa una pipeline di Machine Learning per predire i tempi di completamento e i colli di bottiglia nel processo di richiesta prestiti (dataset BPI Challenge 2012).

## Tecnologie Utilizzate
- **Linguaggio:** Python
- **Process Mining:** PM4Py
- **Machine Learning:** XGBoost (con ottimizzazione Optuna), Scikit-Learn
- **Deep Learning:** TensorFlow/Keras (con Class Weighting)
- **XAI (Spiegabilità):** SHAP
- **Interfaccia:** Streamlit

## Struttura del Codice
- `01_pulizia_base.py`: Preprocessing iniziale dei log.
- `02_estrazione_feature.py`: Feature Engineering (Workload, Risorse).
- `03_training_xgboost.py`: Addestramento modello base.
- `04_ottimizzazione_totale.py`: Tuning avanzato e Data Augmentation (SMOTE).
- `05_xai_shap.py`: Generazione grafici di spiegabilità.
- `06_dashboard_app.py`: Applicazione web per il monitoraggio.
- `07_deep_learning_comparison.py`: Confronto con Reti Neurali.
