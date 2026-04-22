# 🔒 Pre-Deploy Security & Privacy Audit — AniList Score Predictor UNOFFICIAL

**Data audit:** 2026-04-22 | **Auditore:** Antigravity (Senior Security + Python/Streamlit + DevOps + Privacy-by-Design)

---

## 1. EXECUTIVE SUMMARY

| | |
|---|---|
| **Deployabile ora?** | ⚠️ **NO — con le correzioni obbligatorie sotto** |
| **Rischio complessivo** | 🟠 **MEDIO-ALTO** |
| **Problemi bloccanti** | 5 |
| **Problemi importanti** | 8 |
| **Ottimizzazioni consigliate** | 9 |

### Problemi più gravi (bloccanti)
1. **`logging.basicConfig(level=logging.INFO)` in produzione** — logga username e user_id di ogni visitatore su stdout (leggibile dal provider cloud)
2. **`X_train_sample` embedded nel `.pkl`** — dati di training reali dell'utente serializzati nell'artifact e caricati in RAM ad ogni sessione
3. **`st.error(f"Failed to load models: {e}")` — exception leak** — stack trace interno mostrato in pagina
4. **Nessun input sanitization/length-limit** — i campi `text_input` accettano stringhe illimitate; possibile DoS su ricerca + training
5. **`username` injettato direttamente in `unsafe_allow_html`** — XSS potenziale dal badge utente

---

## 2. AUDIT DETTAGLIATO PER CATEGORIA

---

### A. STREAMLIT SECURITY REVIEW

**✅ Cosa è fatto bene:**
- `st.session_state` usato correttamente per isolare stato per-sessione
- `init_session_state()` centralizzato con defaults sicuri
- Form usata correttamente in cold start (step 2) per evitare rerun continui
- `st.progress` e `st.spinner` usati appropriatamente

**🔴 PROBLEMA 1 — XSS via `username` in `unsafe_allow_html` (BLOCCANTE)**

File: `app/shared.py` riga 196
```python
# PERICOLOSO: username viene da input utente non sanitizzato
st.sidebar.markdown(f'<div class="user-badge">👤 {username}</div>', unsafe_allow_html=True)
```
**Rischio:** Un utente che inserisce `<script>alert(1)</script>` come username injetta HTML/JS nel sidebar. Sebbene Streamlit filtri parte degli script, il comportamento non è garantito su tutte le versioni.

**Fix:**
```python
import html
def render_user_badge():
    username = st.session_state.get("username", "")
    safe_username = html.escape(str(username))
    if username:
        st.sidebar.markdown(f'<div class="user-badge">👤 {safe_username}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="user-badge-inactive">👤 No active profile</div>', unsafe_allow_html=True)
```

**🔴 PROBLEMA 2 — `title` da API AniList injettato in HTML senza escape**

File: `app/pages/2_Compare_Titles.py` righe 106, 119
```python
st.markdown(f"<p style='text-align:center;'>{title1}</p>", unsafe_allow_html=True)
```
Il titolo viene dall'API AniList, ma viene memorizzato in cache locale. Se il dato cache venisse corrotto, potrebbe contenere HTML.

**Fix:** Usa `st.markdown(f"**{title1}**")` senza `unsafe_allow_html`, oppure `html.escape(title1)`.

**🟡 PROBLEMA 3 — `last_recommendations` in session_state condivide dati tra tab/refresh**

`st.session_state['last_recommendations']` persiste tra rerun nella stessa sessione — comportamento voluto — ma contiene oggetti `cand` (dicts completi dall'API). Non c'è rischio cross-user perché ogni connessione Streamlit ha il proprio session_state, ma il payload è pesante in memoria.

**Raccomandazione:** Salvare solo i campi necessari (id, title, pred, genres) invece dell'intero dict candidato.

**🟡 PROBLEMA 4 — Nessuna configurazione Streamlit per produzione**

Manca il file `.streamlit/config.toml`. In produzione occorre:
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[logger]
level = "warning"
```

---

### B. SECRETS E CHIAVI API

**✅ Nessuna API key nel codice** — AniList GraphQL è pubblico e non richiede autenticazione per read-only. Nessun secret hardcoded trovato.

**🟡 PROBLEMA 5 — `__main__` con username hardcoded in `api.py` riga 643 e `models.py` riga 457**

```python
# api.py riga 643
username = "arsid"  # Test with an example username or yours

# models.py riga 457  
result = train_and_evaluate_all_models("arsid")
```
Questi blocchi `if __name__ == "__main__"` non vengono eseguiti in produzione, ma rivelano un username reale nel codice sorgente (che sarà nel repository pubblico).

**Fix:** Rimuoverli o usare un username fittizio/env variable:
```python
if __name__ == "__main__":
    import os
    test_user = os.environ.get("TEST_USERNAME", "example_user")
    result = train_and_evaluate_all_models(test_user)
```

**🟡 PROBLEMA 6 — `scratch_test.py` e `test_api.py` nel repository**

Contengono username hardcoded (`arsid`) e logica di fetch senza rate limiting. Non vanno deployati.

**Fix:** Aggiungere al `.gitignore`:
```
scratch_test.py
test_api.py
translate3.py
italian_strings.txt
ChatGPT_Prompt_AniListPredictor.txt
```

---

### C. CACHE E PERSISTENZA DATI

**✅ Cosa è fatto bene:**
- TTL di 24h implementato correttamente
- Cache separata per user/media type
- Nessun dato sensibile vero (solo dati pubblici AniList)

**🔴 PROBLEMA 7 — `X_train_sample` salvato nel `.pkl` (BLOCCANTE)**

File: `src/models.py` riga 387, 414, 447
```python
'X_train_sample': X_train_val.sample(min(50, len(X_train_val)), random_state=42),
```
Questo DataFrame (fino a 50 righe dei dati di training dell'utente, con `user_score`, feature numeriche derivate) viene **serializzato nell'artifact `.pkl`**. Chiunque abbia accesso al file system del server può leggere questi dati.

Su Streamlit Cloud, il filesystem è effimero ma condiviso tra rerun. Il rischio principale è che questi dati vengano caricati in RAM e potrebbero essere loggati o esposti da errori.

**Fix:** Rimuovere `X_train_sample` dall'artifact. Salvarlo separatamente solo se necessario per SHAP, e cancellarlo dopo l'uso.

```python
# In _core_train_pipeline, non includerlo nel return dict principale
# In train_and_evaluate_all_models:
model_artifact = {
    'username': username,
    'model_name': result['model_name'],
    'model': result['model'],
    'train_columns': result['train_columns'],
    'metrics': result['metrics'],
    'feature_importance': result['feature_importance'],
    'shap_explanations': result['shap_explanations'],  # già aggregato, ok
    'optuna_result': result['optuna_result'],
    'tscv_scores': result['tscv_scores'],
    'fallback_recommended': result['fallback_recommended'],
    'trained_at': pd.Timestamp.now().isoformat(),
    'dataset_size': len(df_raw),
    # RIMOSSO: 'X_train_sample'
}
```

**🟡 PROBLEMA 8 — File cache/data/models NON vengono mai puliti**

I file `cache/user_list_*.json` (fino a 5MB), `data/*_clean.csv`, `models/*.pkl` si accumulano per ogni username mai cercato. Su Streamlit Cloud (512MB RAM, 1GB storage gratuito) questo può causare problemi.

**Raccomandazione:** Implementare una pulizia periodica o limitare a N profili recenti.

---

### D. COOKIE, TRACKING, ANALYTICS

**✅ Nessun tracking di terze parti trovato.**

L'unico external script è Google Fonts (`@import url('https://fonts.googleapis.com/...')`) in `shared.py`.

**🟡 PROBLEMA 9 — Google Fonts carica da CDN esterna**

Google Fonts registra l'IP del visitatore. In Europa (GDPR), questo può richiedere consenso.

**Fix (opzioni):**
1. Self-host i font (scarica il CSS/WOFF2 e servili localmente)
2. Aggiungere una nota privacy
3. Usare font di sistema: `font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;`

**Configurazione Streamlit:**
```toml
# .streamlit/config.toml
[browser]
gatherUsageStats = false  # Disabilita telemetria Streamlit
```

---

### E. PRIVACY DEI DATI

**Dati raccolti e salvati:**

| Dato | Dove salvato | Rischio |
|------|-------------|---------|
| AniList username | `session_state`, log, cache filename | Basso (dato pubblico) |
| Lista anime/manga con voti | `cache/user_list_*.json`, `data/*_clean.csv` | Medio |
| Cronologia attività | `cache/user_activity_*.json` | Medio |
| Dati di training (feature) | `models/*.pkl` (X_train_sample) | **Alto** |
| Statistiche personali (bias, contrarian) | Solo in RAM/session | Basso |

**🟡 PROBLEMA 10 — Username salvato nel CSV e nel PKL**

`dataset.py` riga 91: `'username': username` — la colonna username viene inclusa nel CSV di training e nel dizionario dell'artifact.

In un'app multi-utente pubblica, questo significa che il modello di `speedy4433` contiene la stringa "speedy4433" nel file .pkl. Non critico ma superfluo.

**Fix:** Rimuovere la colonna `username` dal DataFrame prima di salvare il CSV.

**🟡 PROBLEMA 11 — `save_csv=False` NON rispettato in `3_User_Compatibility.py`**

```python
df1 = build_user_dataframe(user1, force_refresh=False, save_csv=False)
```
Questo è corretto — il CSV non viene salvato per utenti della pagina compatibilità. ✅

Ma il JSON in cache viene comunque scritto (tramite `fetch_user_anime_list` che non ha il flag `save_csv`). Quindi i dati di ogni utente cercato nella pagina compatibilità vengono persistiti su disco.

---

### F. LOGGING E DEBUG

**🔴 PROBLEMA 12 — `logging.basicConfig(level=INFO)` in produzione (BLOCCANTE)**

File: `src/api.py` riga 8
```python
logging.basicConfig(level=logging.INFO)
```

Questo abilita il logging a livello INFO **globalmente** per tutta l'applicazione. In produzione logga:
- Username di ogni utente: `"Fetching anime list for user: arsid"`
- User ID AniList: `"Fetching activity history for user: arsid (ID: 12345)"`
- Ogni pagina di chunk richiesta
- Errori con dettagli interni

Su Streamlit Cloud, questi log sono visibili nella dashboard dell'owner — ma rappresentano comunque una raccolta di dati non necessaria.

**Fix:**
```python
# src/api.py — RIMUOVERE la riga:
# logging.basicConfig(level=logging.INFO)  # <-- RIMOSSA

# Aggiungere invece in .streamlit/config.toml:
# [logger]
# level = "warning"

# Oppure configurare il logger solo per questo modulo:
logger = logging.getLogger(__name__)
# Non chiamare basicConfig qui — lascia che sia l'entry point a configurarlo
```

Tutti i `logger.info(f"Fetching anime list for user: {username}")` andrebbero ridotti a `logger.debug(...)` in produzione.

**🔴 PROBLEMA 13 — Exception leak a pagina (BLOCCANTE)**

File: `app/main.py` riga 135
```python
st.error(f"Failed to load models: {e}. Please retrain.")
```

File: `app/pages/4_Activity_History.py` riga 101
```python
st.error(f"Error generating statistics: {e}")
```

L'eccezione `e` può contenere path locali, nomi di file, dettagli interni del sistema.

**Fix:**
```python
# main.py
try:
    model_artifact = joblib.load(model_path)
    user_history_df = pd.read_csv(csv_path)
except Exception:
    logger.exception("Failed to load model artifact for user %s", username)
    st.error("Failed to load the model. Please retrain.")
    st.stop()

# 4_Activity_History.py
except Exception:
    logger.exception("Error computing activity stats")
    st.error("Error generating statistics. Please try reloading the page.")
```

---

### G. RATE LIMIT, QUOTE E USO API ESTERNE

**✅ Cosa è fatto bene:**
- Retry con backoff esponenziale in `fetch_with_retry`
- Gestione 429 con `Retry-After`
- `time.sleep(1)` tra le pagine di candidati
- Limite a 200 pagine per l'activity history

**🟡 PROBLEMA 14 — Nessun rate limit lato applicazione**

Un utente malevolo può cliccare ripetutamente "Generate Recommendations" con `candidate_limit=1000`, triggering fino a **20 API calls per click** (10 pagine per 2 sort categories). Con 5 click in 10 secondi = 100 request ad AniList.

**Fix:** Aggiungere throttling con session_state:
```python
# Prima del bottone raccomandazioni
import time

last_rec_time = st.session_state.get('last_rec_timestamp', 0)
cooldown_seconds = 30
time_since = time.time() - last_rec_time

if rec_btn:
    if time_since < cooldown_seconds:
        st.warning(f"Please wait {int(cooldown_seconds - time_since)}s before generating again.")
    else:
        st.session_state['last_rec_timestamp'] = time.time()
        # ... proceed with recommendations
```

**🟡 PROBLEMA 15 — Nessun limite al `candidate_limit` lato sicurezza**

Il slider arriva a 1000, ma un utente potrebbe manipolare i parametri. Il valore va sempre validato lato server:
```python
candidate_limit = max(10, min(candidate_limit, 1000))
```

---

### H. SICUREZZA CONTRO UTENTI MALEVOLI

**🔴 PROBLEMA 16 — Nessun input length limit (BLOCCANTE)**

```python
username_input = st.text_input("Enter AniList Username:")
media_query = st.text_input("Search for an anime:")
sel_tag = st.text_input("Tag (e.g., Isekai, Magic):").strip()
```

Un input di 100.000 caratteri viene passato direttamente all'API AniList (che lo rifiuterà, ma il tentativo viene loggato e consuma risorse).

**Fix:**
```python
MAX_USERNAME_LEN = 50
MAX_SEARCH_LEN = 100
MAX_TAG_LEN = 50

username_input = st.text_input("Enter AniList Username:")[:MAX_USERNAME_LEN]
username_lower = username_input.strip().lower()

# Validazione username: solo caratteri alfanumerici e trattini/underscore
import re
if username_lower and not re.match(r'^[a-z0-9_\-]{1,50}$', username_lower):
    st.warning("Username can only contain letters, numbers, underscores and hyphens.")
    st.stop()
```

**🟡 PROBLEMA 17 — Path traversal nella costruzione dei path file**

```python
# main.py riga 76
model_path = MODELS_DIR / f"{username_lower}{model_suffix}"
```

Se `username_lower` contenesse `../../../etc/passwd`, Path() lo normalize, ma su Windows il comportamento è diverso. Il regex fix sopra risolve anche questo.

**🟡 PROBLEMA 18 — Training illimitato (DoS applicativo)**

Il bottone "Train/Retrain" può essere premuto infinite volte da qualsiasi utente. Il training (con Optuna 30 trial, XGBoost, TimeSeriesSplit) è CPU-intensivo.

**Fix:** Aggiungere un cooldown:
```python
last_train_key = f"last_train_{username_lower}"
last_train = st.session_state.get(last_train_key, 0)
if time.time() - last_train < 300:  # 5 minuti di cooldown
    st.warning("Please wait 5 minutes between training runs.")
else:
    # ... proceed with training
    st.session_state[last_train_key] = time.time()
```

---

### I. MODEL SECURITY / ARTIFACT SECURITY

**🔴 PROBLEMA 19 — `joblib.load` su path costruito da input utente (BLOCCANTE)**

```python
# main.py riga 80 e 132
meta = joblib.load(model_path)
model_artifact = joblib.load(model_path)
```

`model_path` è costruito da `username_lower` che viene dall'input utente. Joblib usa pickle internamente. Se un attaccante riuscisse a far caricare un `.pkl` malevolo (es. tramite path traversal), potrebbe eseguire codice arbitrario.

Con il regex fix al problema 16 (solo `[a-z0-9_-]`), il path traversal è mitigato. Ma la fiducia nei file `.pkl` presuppone che nessun attaccante abbia accesso al filesystem.

**Fix aggiuntivi:**
1. Validare sempre che il path risultante sia DENTRO `MODELS_DIR`:
```python
def safe_load_model(username: str, suffix: str):
    safe_path = (MODELS_DIR / f"{username}{suffix}").resolve()
    if not str(safe_path).startswith(str(MODELS_DIR.resolve())):
        raise ValueError("Invalid model path")
    if not safe_path.exists():
        return None
    return joblib.load(safe_path)
```

**🟡 PROBLEMA 20 — `X_train_sample` in pickle (già citato in C)**

Il dato di training serializzato con pickle è un vettore numerico, non codice eseguibile, quindi non è un rischio di arbitrary code execution. Ma è un rischio di data leakage.

---

### J. FILE SYSTEM E REPOSITORY HYGIENE

**`.gitignore` attuale:**
```
cache/
data/
models/
__pycache__/
*.py[cod]
.venv/
venv/
.env
```

**✅ Le directory con dati utente sono correttamente escluse.**

**🟡 PROBLEMA 21 — File di sviluppo non esclusi dal git**

Mancano dal `.gitignore`:
```
scratch_test.py
test_api.py
translate3.py
italian_strings.txt
ChatGPT_Prompt_AniListPredictor.txt
*.log
.streamlit/secrets.toml
```

**Fix — `.gitignore` aggiornato:**
```gitignore
# Data directories (user data - NEVER commit)
cache/
data/
models/

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env

# Streamlit secrets (NEVER commit)
.streamlit/secrets.toml

# Dev/scratch files
scratch_test.py
test_api.py
translate3.py
italian_strings.txt
ChatGPT_Prompt_AniListPredictor.txt

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

**🟡 PROBLEMA 22 — `training_status.json` in cache contiene username reali**

`cache/training_status.json` contiene `"arsid_anime"`. È nella directory `cache/` già ignorata, quindi non finisce nel repo. ✅

---

### K. DEPLOYMENT HARDENING

**🟡 PROBLEMA 23 — `requirements.txt` senza versioni pin**

```
pandas
scikit-learn
requests
streamlit
...
```

Senza versioni pin, ogni deploy può portare breaking changes.

**Fix:** Generare un requirements.txt con versioni fisse:
```
streamlit==1.44.0
pandas==2.2.3
scikit-learn==1.6.1
requests==2.32.3
numpy==2.2.4
xgboost==2.1.4
lightgbm==4.6.0
optuna==4.3.0
shap==0.47.2
plotly==6.0.1
joblib==1.4.2
```

Eseguire: `pip freeze > requirements.txt` nell'ambiente di sviluppo.

**🟡 PROBLEMA 24 — Nessun file `.streamlit/config.toml`**

Creare `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 5

[browser]
gatherUsageStats = false

[logger]
level = "warning"
messageFormat = "%(asctime)s %(levelname)s: %(message)s"

[theme]
base = "dark"
```

**🟡 PROBLEMA 25 — Nessun timeout sulle richieste HTTP**

```python
response = requests.post(API_URL, json=..., headers=...)
```

Nessun timeout — una richiesta bloccata può tenere il thread di Streamlit occupato indefinitamente.

**Fix:**
```python
response = requests.post(
    API_URL,
    json={"query": query, "variables": variables},
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    timeout=30  # 30 secondi max
)
```

---

## 3. PATCH CONSIGLIATE — FILE PER FILE

### `src/api.py`
1. **Rimuovere** `logging.basicConfig(level=logging.INFO)` riga 8
2. **Aggiungere** `timeout=30` a tutti i `requests.post`
3. **Cambiare** tutti i `logger.info` relativi a username in `logger.debug`
4. **Rimuovere/modificare** il blocco `if __name__ == "__main__"` con username hardcoded

### `src/models.py`
1. **Rimuovere** `'X_train_sample'` dall'artifact dict (righe 387, 414, 447)
2. **Modificare** il blocco `if __name__ == "__main__"` (riga 457) usando `os.environ`

### `src/dataset.py`
1. **Rimuovere** la colonna `'username'` dal DataFrame (riga 91 e 216) — non serve come feature

### `app/shared.py`
1. **Aggiungere** `import html` e `html.escape()` nel badge utente (riga 196)

### `app/main.py`
1. **Aggiungere** validazione regex su `username_lower`
2. **Aggiungere** `safe_load_model()` wrapper con path validation
3. **Modificare** `st.error(f"Failed to load models: {e}")` → rimuovere `{e}` dall'output utente
4. **Aggiungere** cooldown rate limiting sui bottoni Train e Generate Recommendations
5. **Limitare** lunghezza dei `text_input`

### `app/pages/2_Compare_Titles.py`
1. **Aggiungere** `html.escape()` sui titoli prima di iniettarli in `unsafe_allow_html`
2. **Aggiungere** gestione eccezione su `joblib.load` senza esporre `{e}`

### `app/pages/4_Activity_History.py`
1. **Modificare** `st.error(f"Error generating statistics: {e}")` — rimuovere `{e}`

### `.gitignore`
1. **Aggiungere** file di sviluppo (vedi sopra)

### Nuovo: `.streamlit/config.toml`
1. **Creare** con la configurazione di produzione

### `requirements.txt`
1. **Aggiornare** con versioni pin (`pip freeze > requirements.txt`)

---

## 4. CHECKLIST PRE-DEPLOY FINALE

### 🔴 OBBLIGATORIE (bloccanti)
- [ ] Rimosso `logging.basicConfig(level=logging.INFO)` da `api.py`
- [ ] Cambiati tutti `logger.info` con username in `logger.debug`
- [ ] Aggiunto `html.escape()` su `username` nel badge (`shared.py`)
- [ ] Rimosso `{e}` dall'output `st.error()` in `main.py` e `4_Activity_History.py`
- [ ] Rimosso `X_train_sample` dall'artifact `.pkl`
- [ ] Aggiunta validazione regex sull'username input (solo `[a-z0-9_-]`)
- [ ] Aggiunto `safe_load_model()` con path boundary check
- [ ] Aggiunto `timeout=30` a tutti i `requests.post`

### 🟡 IMPORTANTI
- [ ] Creato `.streamlit/config.toml` con `gatherUsageStats = false` e `enableXsrfProtection = true`
- [ ] Aggiunto rate limiting/cooldown su Train e Generate Recommendations
- [ ] Rimossi `scratch_test.py`, `test_api.py`, `translate3.py` dal repo (o aggiunti a `.gitignore`)
- [ ] Rimosso username hardcoded `"arsid"` da `api.py:643` e `models.py:457`
- [ ] Aggiornato `requirements.txt` con versioni pin
- [ ] Aggiunto `html.escape()` sui titoli AniList in `2_Compare_Titles.py`
- [ ] Limitata lunghezza dei `text_input` (username max 50, search max 100)
- [ ] Aggiunto `max(10, min(candidate_limit, 1000))` come validazione server-side

### 🟢 CONSIGLIATE
- [ ] Sostituire Google Fonts CDN con font di sistema (GDPR compliance)
- [ ] Aggiungere una Privacy Policy minima se si raccolgono dati utente
- [ ] Rimuovere colonna `username` dai CSV e dai dict artifact
- [ ] Implementare pulizia periodica di cache/data/models (limite per storage)
- [ ] Aggiungere una pagina di errore generica per eccezioni non gestite
- [ ] Ridurre session_state `last_recommendations` a soli i campi necessari
- [ ] Aggiungere `st.cache_data` su `search_anime_by_title` per ridurre API calls su ricerche ripetute
- [ ] Considerare autenticazione (es. `streamlit-authenticator`) se si vuole rendere l'app privata
- [ ] Testare il deploy su Streamlit Community Cloud con account gratuito prima del lancio

---

## 5. RED FLAGS BLOCCANTI

Questi **5 problemi devono essere risolti prima del deploy**:

| # | Problema | File | Fix richiesto |
|---|---------|------|--------------|
| 1 | `logging.basicConfig(INFO)` logga username utenti | `src/api.py:8` | Rimuovere, usare `logger.debug` |
| 2 | `X_train_sample` nei `.pkl` — dati utente in artifact | `src/models.py:387,414,447` | Rimuovere dal dict artifact |
| 3 | `st.error(f"...{e}")` — exception leak in pagina | `app/main.py:135`, `4_Activity.py:101` | Rimuovere `{e}` dall'output |
| 4 | Nessun length/format limit sull'username input | `app/main.py:70` | Aggiungere regex + max length |
| 5 | `username` injettato in `unsafe_allow_html` senza escape | `app/shared.py:196` | Aggiungere `html.escape()` |

---

*Report generato da Antigravity — AniList Score Predictor UNOFFICIAL Pre-Deploy Audit v1.0*
