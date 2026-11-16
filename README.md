# AniList Score Predictor

App Streamlit non ufficiale per analizzare i voti di un profilo AniList, stimare come l'utente potrebbe valutare un titolo e suggerire anime e manga a partire dalle sue preferenze. L'interfaccia è in inglese.

## Funzionalità

| Pagina | Cosa puoi fare |
| --- | --- |
| Prediction Studio | Addestrare un modello sul tuo storico, stimare il voto per un titolo e generare raccomandazioni. In modalità Cold Start, bastano alcuni titoli preferiti e le tue preferenze. |
| Profile Analysis | Esplorare distribuzione dei voti, generi e studi o autori preferiti. |
| Compare Titles | Confrontare due titoli e vedere quali caratteristiche pesano sulle stime del modello, quando sono disponibili le spiegazioni SHAP. |
| User Compatibility | Confrontare i voti di due profili sui titoli valutati da entrambi. |
| Activity History | Consultare la cronologia di visione e lettura registrata su AniList. |
| Preference Arena e Score Positioning | Confrontare titoli a coppie e collocare un titolo tra le opere che hai già valutato. |

## Avvio in locale

Servono Python 3.12 e una connessione a Internet per accedere all'API AniList. Dalla radice del progetto, crea un ambiente virtuale.

```bash
python -m venv .venv
```

Attivalo con `.\.venv\Scripts\Activate.ps1` su Windows PowerShell o con `source .venv/bin/activate` su macOS e Linux, poi installa le dipendenze.

```bash
python -m pip install -r requirements.txt
```

Crea `.streamlit/secrets.toml` con questa configurazione per eseguire l'app sul tuo computer.

```toml
[access]
disable_auth_for_local_dev = true
```

Avvia Streamlit e apri l'indirizzo mostrato nel terminale.

```bash
python -m streamlit run app/main.py --server.address 127.0.0.1
```

In **Prediction Studio**, scegli **AniList Profile (Machine Learning)** e inserisci il nome utente di un profilo pubblico. Usa **New User (Cold Start)** se vuoi partire dai tuoi titoli preferiti senza addestrare un modello.

## Come funziona

L'app legge i dati pubblici tramite l'API GraphQL di AniList. Lo storico dei voti e i metadati dei titoli diventano le variabili del modello. I modelli candidati vengono confrontati su dati ordinati nel tempo usando l'errore assoluto medio (MAE). L'app usa scikit-learn, XGBoost e LightGBM per i modelli, Optuna per la ricerca degli iperparametri e SHAP per le spiegazioni dei modelli compatibili.

`app/` contiene l'interfaccia Streamlit, `src/` contiene il client AniList e la logica di analisi e previsione, `tests/` copre la gestione degli errori dell'API. La [versione base](versione_base/README.md) è una variante separata per la presentazione accademica.

```bash
python -m unittest discover -s tests -v
```

## Dati locali e deploy

L'app conserva la cache dell'API in `cache/`, i dataset in `data/` e i modelli addestrati in `models/`. La pulizia è configurata per rimuovere i file della cache dopo 6 ore e i dataset e i modelli dopo 7 giorni. Le cartelle generate e i segreti locali sono esclusi da Git.

Per ospitare l'app con accesso riservato tramite Google OpenID Connect, segui la [guida al deploy](DEPLOYMENT_SECURITY.md).

## Fonte dei dati

Il progetto è indipendente da AniList. L'app accede in sola lettura ai dati dell'[API AniList](https://docs.anilist.co/) e rimanda alle schede originali. Titoli, descrizioni, copertine e altri metadati appartengono ai rispettivi titolari.
