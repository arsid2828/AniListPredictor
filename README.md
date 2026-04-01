# 🎬 AniList AI Predictor 🧠

Un ecosistema di Machine Learning ibrido all-in-one progettato per analizzare, profilare e prevedere matematicamente i voti che un utente darà a **Anime** e **Manga** su [AniList](https://anilist.co/).

Il progetto si è evoluto da un semplice regressore a una suite completa di **BI (Business Intelligence) per Otaku**, integrando modelli di stato dell'arte, spiegabilità (XAI) e analytics profonde.

---

## ✨ Caratteristiche Tecniche d'Elite

### 🔬 Machine Learning Dynamics
*   **Selezione Dinamica del Modello:** Il sistema non usa più un modello fisso. All'avvio del training, confronta **10 algoritmi** (Decision Trees, Random Forest, Gradient Boosting, Ridge, KNN, XGBoost, LightGBM, Ensemble) e seleziona automaticamente quello con il **MAE (Mean Absolute Error)** più basso.
*   **Optuna Tuning:** Integrazione con la libreria Optuna per la ricerca bayesiana degli iperparametri (30 trial per sessione), ottimizzando profondità, learning rate e regolarizzazione.
*   **SHAP Explainability (XAI):** Grazie ai valori di Shapley, ogni singola predizione viene spiegata con un grafico a cascata che mostra esattamente quali feature (es: "Studio Madhouse +0.5", "Tag Seinen +0.3") hanno influenzato il voto finale.
*   **TimeSeriesSplit CV:** Validazione robusta su 5-fold temporali. Testiamo il modello su diverse finestre del passato dell'utente per garantire la massima stabilità nelle previsioni future.
*   **Architettura "Leakage-Free":** Feature engineering iterativo che calcola lo stato storico (frequenze, affinità, mood) senza mai mostrare al modello dati del futuro, simulando un reale processo decisionale umano.

### 📊 Deep Profile Analytics
*   **Contrarian Index:** Misuriamo quanto i tuoi gusti divergono dalla community globale. Sei un "Normie" o un critico severo?
*   **Radar Chart dei Generi:** Visualizzazione poligonale dell'affinità per genere (Action, Romance, Psychological, ecc.).
*   **Mood Timeline:** Grafico cronologico con media mobile (Rolling Mean) per visualizzare come i tuoi standard di voto sono cambiati negli anni.
*   **Studio & Author League:** Ranking degli studi di animazione e degli autori manga basato sulla tua soddisfazione media.

### 🛸 Cold Start & Compatibility
*   **Onboarding Hard-Filtered:** Un flusso a step per nuovi utenti che costruisce uno pseudo-profilo. Include ora il supporto per ogni genere (incluso **Hentai**) e filtri rigorosi sul formato (Movies vs TV).
*   **Compatibilità Utente:** Inserisci due username AniList e ottieni un report di affinità con Gauge Chart, scatter plot dei voti comuni e lista dei "punti di massimo disaccordo".

---

## 🗂️ Struttura del Progetto

```
AnilistProject/
├── app/
│   ├── main.py              # Entry point UI Streamlit
│   ├── shared.py            # Design System (CSS, Badge, Utils)
│   └── pages/               # Multi-page App
│       ├── 1_Analisi_Profilo.py
│       ├── 2_Confronta_Titoli.py
│       └── 3_Compatibilita_Utenti.py
├── src/
│   ├── api.py               # Client GraphQL con Cache TTL (24h)
│   ├── dataset.py           # Pipeline di pulizia dati
│   ├── features.py          # Engineering: source, completion_ratio, repeat
│   ├── models.py            # Motore ML: Optuna, XGBoost, Selection
│   ├── analytics.py         # Calcoli statistici e SHAP
│   └── cold_start.py        # Logica euristica per onboarding
├── cache/                   # Dati JSON temporanei
├── data/                    # Dataset CSV utente
├── models/                  # Modelli .pkl salvati
└── requirements.txt         # Stack tecnologico
```

---

## 🛠️ Tecnologie Utilizzate

| Categoria | Tool |
|---|---|
| **Core** | Python 3.10+, Pandas, Numpy |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Optimization** | Optuna (Bayesian Search) |
| **Explainability** | SHAP (TreeExplainer) |
| **API** | GraphQL (AniList API v2) |
| **UI** | Streamlit (Custom Dark CSS) |
| **Graphics** | Plotly (Interactive), Matplotlib |

---

## 🚀 Guida Rapida

1.  **Requisiti:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Avvio:**
    ```bash
    python -m streamlit run app/main.py
    ```

3.  **Utilizzo:**
    *   **Login:** Inserisci il tuo username AniList o selezionalo dal dropdown dei profili caricati in precedenza.
    *   **Training:** Premi "Fetch & Train" per generare il tuo modello neurale personalizzato.
    *   **Esplora:** Naviga tra le pagine laterali per analizzare il tuo profilo, confrontare titoli o testare la compatibilità con un amico.
    *   **Esporta:** Scarica le tue raccomandazioni in formato **CSV** o **JSON**.

---

## 🛡️ Privacy e Cache
I dati vengono salvati localmente nelle cartelle `cache/` e `data/`. La cache dell'API scade automaticamente ogni **24 ore** per garantirti dati sempre aggiornati senza sovraccaricare i server di AniList.

---
*Sviluppato con ❤️ per la community di AniList.*
