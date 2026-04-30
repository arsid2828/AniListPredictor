# 🎬 AniList Score Predictor UNOFFICIAL 🧠

An unofficial Machine Learning app designed to analyze, profile, and predict the scores a user may give to **Anime** and **Manga** on [AniList](https://anilist.co/).

Il progetto si è evoluto da un semplice regressore a una suite completa di **BI (Business Intelligence) per Otaku**, integrando modelli di stato dell'arte, spiegabilità (XAI) e analytics profonde, con un'interfaccia utente (UI) completamente in lingua inglese.

---

## 🏗️ Struttura del Progetto e File

Il codice è organizzato in due moduli principali: `app/` (che gestisce tutta l'interfaccia utente Streamlit) e `src/` (che contiene il motore logico e predittivo).

### Interfaccia Utente (`app/`)
*   **`main.py`**: È l'**Entry Point** dell'applicazione. Gestisce la dashboard principale, la configurazione iniziale, il login dell'utente (o il flusso *"Cold Start"* per i nuovi utenti), l'addestramento dei modelli e la generazione della "Top 10" delle raccomandazioni utente. Mostra a colpo d'occhio le metriche del modello migliore elaborato.
*   **`shared.py`**: Contiene il **Design System** e le utility condivise dell'app (come l'iniezione del CSS, la configurazione degli stati di sessione e il setup dei badge utente) garantendo coerenza estetica su tutte le pagine.

**Le Pagine (in `app/pages/`)**
*   **`1_Profile_Analysis.py`**: **L'Analisi Profilo** prende lo storico dell'utente e ne ricava grafici dettagliati. Include la "Distribuzione Voti", il radar dei generi, il "Contrarian Index" (per capire quanto ti discosti dalla massa) e analisi su autori e studi preferiti. Serve a capire a fondo le tue preferenze reali rispetto ai voti dati.
*   **`2_Compare_Titles.py`**: **Confronto Titoli** permette di mettere due anime (o manga) testa a testa. Usa *Radar Charts* per confrontare generi, formati e punteggi, e integra un'analisi con *SHAP* per spiegare perché al sistema piace più il Titolo A rispetto al Titolo B per te specifico.
*   **`3_User_Compatibility.py`**: Questa pagina serve a calcolare la **Compatibilità tra Utenti**. Inserendo due username, il sistema confronta i titoli valutati in comune e calcola una % di affinità. Rivela inoltre i titoli su cui gli utenti sono maggiormente in disaccordo o d'accordo (i più grandi match e mismatch).
*   **`4_Activity_History.py`**: Crea una **Cronologia Interattiva** pura della tua attività. Estrae le date esatte in cui hai guardato episodi o letto capitoli, creando timeline che mostrano mesi o anni più attivi e perfino i record di binge-watching in determinati giorni.

### Il Motore Dietro le Quinte (`src/`)
*   **`api.py`**: Gestisce tutte le comunicazioni con il backend di AniList. Costruisce le astruse query **GraphQL** per prelevare dati in batch e gestisce un sistema di cache a 24 ore per non spammare le API ufficiali.
*   **`dataset.py`**: Funziona come una conduttura. Prende l'input grezzo estratto da `api.py` (JSON) e lo manipola, pulisce e aggrega in robusti e strutturati **Dataframe Pandas**, tenendo fuori duplicati ("Completed" vs "Custom Lists") e record invalidi.
*   **`features.py`**: **Il cuore dell'Alchimia dei Dati.** Questo file preleva i dataset originali e inietta centinaia di *features ingegnerizzate* e vettorializzate: rapporto completamento, punteggi derivati da staff e ruoli chiave (es. Madhouse o Hideaki Anno), encoding per label ordinali e tanto altro per renderli digeribili ai modelli.
*   **`models.py`**: **Il Motore Machine Learning.** Costruisce decine di modelli e li fa competere contro una Cross-validation TSVC rigorosa. 
*   **`cold_start.py`**: Entra in gioco quando un utente non ha sufficente storico. Questo set di logiche di **Sistema Esperto ed Euristico** costruisce profili psicografici dell'utente dal nulla, basandosi solo su "3-5 opere preferite" e semplici gusti diretti in input per generare raccomandazioni.
*   **`analytics.py`**: Contiene la logica matematica e statistica pura per plot, XAI, calcolo dello SHAP (Shapley Value Explainations) formattato ed aggregazioni numeriche.

---

## 🧠 Modelli di Machine Learning

Il sistema non affida tutto a una singola rete neurale bloccata, bensì applica un paradigma iterativo in cui fa gareggiare in simultanea multi-modelli e ne valuta le performance sul MAE (Mean Absolute Error). 

I principali modelli che competono ogni volta per essere il "Miglior Modello Utente" sono:

1.  **LightGBM (LGBMRegressor)**: Veloce e potente. Usa il gradient boosting basato su alberi decisionali ed excels nella modellazione di distribuzioni complesse sui pattern di gradimento. (Ideale con abbondanti valutazioni).
2.  **XGBoost (XGBRegressor)**: Un'architettura di boosting rinomata, eccellente a gestire i "missing values" propri dei dati di anime molto di nicchia. Spesso pesantemente regolarizzato ed estremamente affidabile.
3.  **Random Forest**: Robusto verso gli outlier. Crea centinaia di alberi indipendenti che "votano", perfetto se storicamente si distribuiscono voti sia altissimi che molto bassi senza criteri precisi nel genere ("Random").
4.  **Gradient Boosting (standard)**: Modello base del boosting per catturare i gradienti residui su pattern più lineari.
5.  **Ridge Regression (L2)**: Utilizzato per stabilire una base (baseline) matematica lineare e non sovralimentata (overfit).
6.  **KNN (K-Nearest Neighbors)**: Raggruppa i pattern dell'utente basandosi sulla vicinanza in vettori n-dimensionali. In soldoni, se un anime è "vicinissimo" a molti altri anime dati a "10", prende "10". Ottimo per i cluster puristi di genere.
7.  **Ensemble Classico (Voting)**: Una sintesi pura orchestrata per mettere i tre colossi (**Random Forest + Gradient Boosting + XGBoost**) a discutere e sommare pacificamente le loro risposte per creare una singola ed eccellente predizione ibrida super resiliente.

Tutti i modelli vengono instradati tramite **Optuna**, un tuner a ottimizzazione bayesiana. Optuna non sceglie a caso hiperparametri (come la profondità degli alberi) ma capisce la dinamica matematica dei round predittivi passati per perfezionare le successive iterazioni nei suoi *30 trial di test*.

---

## ✨ Explainable AI (XAI) con SHAP

Non ci accontentiamo di capire se l'algoritmo predice '8.5', ma **perché lo fa**. Avvalendoci dei valori SHAP (*SHapley Additive exPlanations*) resi visibili sul frontend dell'APP ("Perché questi voti?"), il Predictor dimostra quanta "spinta", positiva o negativa, una singola caratteristica dell'opera testata ha scaturito sull'incremento finale di voto (es. `Maturity Rating +0.40`).

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
    *   **Dashboard (`app/main.py`)**: Scegli "AniList Profile", inserisci il tuo username e premi il bottone per trainare il modello. Dopodiché genererà direttamente la tua top 10 predittiva.
    *   **Pagine Laterali (da menu)**: Sperimenta con Profilo, Compatibilità tra te e un amico o Analisi dei Titoli a parità di predizione.

---

## 🛡️ Privacy e Cache
I dati vengono salvati localmente nelle cartelle `cache/` (in formato leggero JSON) e `data/` (come pesanti dataset in ML CSV). La cache dell'API scade automaticamente ogni **24 ore** per garantirti dati sempre aggiornati senza sovraccaricare i server di AniList.

---
This project is unofficial and is not affiliated with or endorsed by AniList. Data is sourced from the AniList GraphQL API. Anime and manga titles, descriptions, cover images, and related metadata remain the property of their respective owners.

---

## Private Streamlit Deploy

Production entry point:

```bash
python -m streamlit run app/main.py
```

The app is protected by Streamlit's Google OIDC login plus a server-side allowlist in Streamlit secrets. Copy `.streamlit/secrets.example.toml` into Streamlit Community Cloud secrets, set the deployed `redirect_uri`, and add only trusted Google accounts under `[access]`.

See `DEPLOYMENT_SECURITY.md` for the complete deployment, privacy, and AniList API checklist.
