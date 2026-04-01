# 🎬 AniList AI Predictor 🧠

Un ecosistema di Machine Learning ibrido progettato da zero per analizzare, profilare e prevedere matematicamente i voti che un utente darà a **Anime** e **Manga** su [AniList](https://anilist.co/).

Non si tratta di un semplice sistema di raccomandazione: questo modello studia a fondo i pattern latenti, le variazioni di umore e le preferenze cronologiche dell'utente. L'aspetto rivoluzionario risiede nell'architettura **"Leakage-Free"** (Zero Fuga di Dati), in quanto l'IA calcola i dati storici comportamentali come se viaggiasse nel tempo passo dopo passo.

---

## ✨ Caratteristiche Tecniche Avanzate

### 🔬 Machine Learning — Utenti AniList
*   **Leave-One-Out (LOO) Perturbation Test:** Il modello non si limita a dire "quanto puzza o profuma" la statistica. Clona una versione neutra dell'anime/manga in background per dirti i decimi esatti (`+/- decimi di voto`) di cui ha abbassato o rialzato il voto finale per colpa di quello specifico attributo.
*   **Recency Mood Bias:** Misuriamo la media mobile degli "ultimi 10 anime/manga visti". Se l'utente in quel decennio era depresso o "di manica stretta", l'algoritmo scala proporzionalmente all'umore storico locale.
*   **Contrarian Score (MAF Storico):** Il sistema non guarda solo il segno della devianza ma quantifica letteralmente se l'utente è un puro "Normie" (segue il voto della community) o un "Bastian Contrario", bilanciando quanto delegare l'output alla folla globale vs ai gusti unici dell'utente.
*   **Protezione Out-of-Distribution (OOD):** Con regole euristiche umane, l'UI ha delle protezioni fisiche contro le allucinazioni algoritmiche (es: colpire con penalità pesantissime un tag "Hentai" per utenti che non ne guardano mai, bypassando l'inesperienza latente dell'Albero Decisionale).
*   **Franchise Analysis:** Il modello traccia prequel, sequel, spin-off, adattamenti e character crossover per calcolare un'affinità storica basata sull'intero "universo narrativo" del franchise (es: se hai adorato Naruto, il sistema lo ricorda per Naruto Shippuden).
*   **Creator Intelligence:** L'architettura riconosce registi, autori originali, series composition e character designer. Se adori le opere di un autore, l'IA pesa la sua presenza come feature predittiva (es: affinità verso Hayao Miyazaki = boost per ogni suo film futuro).

### 🛸 Cold Start — Nuovi Utenti (Senza Profilo)
*   **Onboarding Guidato:** Un flusso a step che, partendo da 3-5 anime/manga preferiti e poche domande su gusti (generi, formato, lunghezza, epoca), costruisce un "pseudo-profilo utente" al volo.
*   **Scorer Euristico Content-Based:** Un algoritmo ibrido che incrocia preferenze esplicite, tracce implicite dai preferiti (generi, tematiche, formati), e la qualità globale della community per restituire un punteggio di affinità da 0 a 10, accompagnato da spiegazioni leggibili in linguaggio naturale.
*   **Raccomandazioni Istantanee:** Top 10 consigli personalizzati generati senza bisogno di storico, con hard-filter sui generi sgraditi e ranking per affinità.

---

## 🗂️ Struttura dei File

Il progetto è suddiviso in una pipeline modulare MLOps:

```
AnilistProject/
├── app/
│   └── main.py          # UI Streamlit (dashboard interattiva)
├── src/
│   ├── api.py           # Client GraphQL AniList (anime + manga)
│   ├── dataset.py       # Raffinazione dati in DataFrame pandas
│   ├── features.py      # Feature Engineering storico (anime + manga)
│   ├── models.py        # Training, valutazione e salvataggio modelli
│   └── cold_start.py    # Onboarding e scorer euristico per nuovi utenti
├── cache/               # Cache locale delle risposte API (gitignored)
├── data/                # CSV puliti generati dalla pipeline (gitignored)
├── models/              # Modelli serializzati .pkl (gitignored)
├── requirements.txt     # Dipendenze Python
├── .gitignore
└── README.md
```

---

### 1. `src/api.py` — L'Estrattore GraphQL

L'interfaccia di interrogazione **GraphQL** ufficiale di AniList. Supporta sia **Anime** che **Manga** con query dedicate.

*   Gestione dei rate-limits con retry esponenziale (backoff).
*   Paginazione automatica a Chunk per librerie enormi.
*   Caching locale per non assaltare il server e velocizzare operazioni ripetute.
*   Estrazione di `staff` (registi, autori) e `relations` (franchise, adattamenti).
*   Query dedicate: ricerca per titolo, lista utente, candidati per raccomandazioni (per popolarità e per score).

### 2. `src/dataset.py` — Il Raffinatore

Prende l'infinito dizionario destrutturato JSON di AniList e lo appiattisce in un formato colonnare solido (`pandas`). Gestisce due pipeline parallele:

*   **Anime:** Estrae studi d'animazione, produttori, registi/autori (staff), relazioni franchise, tag con ranking, e deduplicazione.
*   **Manga:** Estrae autori (Story/Art), capitoli, anno di pubblicazione (`startDate`), relazioni e tag.
*   Filtraggio voti nulli (score == 0), ordinamento cronologico, salvataggio CSV.

### 3. `src/features.py` — Il Motore del Feature Engineering

Il cuore pulsante del Machine Learning. Questo file non sbatte i dati così come sono nel modello ma crea il *"Tempo"*.

**Feature Storiche Leakage-Free** (calcolate iterativamente sulla timeline dell'utente):
*   `hist_user_mean`, `hist_user_std`: Media e deviazione standard dei voti passati.
*   `recent_mean_score`: Media mobile degli ultimi 10 titoli visti (Recency Mood Bias).
*   `hist_genre_affinity` / `hist_genre_freq`: Affinità e frequenza per genere.
*   `hist_tag_affinity` / `hist_tag_freq`: Affinità e frequenza per tag con pesi.
*   `hist_studio_affinity` / `hist_studio_freq`: Affinità per studi d'animazione.
*   `hist_producer_affinity` / `hist_producer_freq`: Affinità per produttori.
*   `hist_creator_affinity` / `hist_creator_freq`: Affinità per regista/autore originale.
*   `hist_franchise_count` / `hist_franchise_mean`: Punteggio medio storico del franchise (prequel, sequel, spin-off).
*   `hist_char_count` / `hist_char_mean`: Punteggio medio storico dei crossover di personaggi.
*   `hist_global_diff` / `hist_global_mae`: Deviazione storica dell'utente dalla media globale (Contrarian Score).

**Feature Categoriche Multi-Hot:** Generi, tag (con peso di ranking), formato, paese d'origine — tutti codificati con pesi proporzionali all'impatto reale.

**Inference Builder:** Funzione dedicata (`build_inference_features` / `build_manga_inference_features`) che, dato un anime/manga sconosciuto, ricalcola l'intero stato storico dell'utente e allinea perfettamente le colonne con il modello trainato.

### 4. `src/models.py` — Il Cervello Ibrido

Traina il database e valuta i pesi usando una validazione cronologica (`70/15/15 split` temporale). Supporta sia **Anime** che **Manga** con pipeline identiche ma indipendenti.

**8 modelli confrontati automaticamente:**
| Modello | Ruolo |
|---|---|
| Baseline (Hist Mean) | Linea di base: media storica dell'utente |
| Ridge Regression | Regressore lineare robusto con regolarizzazione |
| KNN Regressor | K-Nearest Neighbors pesato per distanza |
| Decision Tree (Raw) | Albero decisionale senza pruning |
| Decision Tree (Pruned) | Albero con pruning ottimizzato su validation set |
| Random Forest | Foresta di 100 alberi per catturare nicchie ed eccezioni |
| Gradient Boosting | Boosting graduale con learning rate conservativo |
| **Voting Ensemble (Smooth)** | **Modello vincente selezionato di default** |

Il modello vincente è un potentissimo **Voting Ensemble** bilanciato:
*   `60% Random Forest Regressor`: Perfetto per intercettare nicchie, eccezioni umane e incroci astratti ("Amo le commedie Ma SOLO se sono fatte dalla Sunrise, altrimenti odio la regia").
*   `40% Ridge Regression`: Un modello lineare robusto. L'albero di Random Forest calcola per natura a "scalini", ma il Ridge arrotonda e smussa gli sbilanci permettendo voti ultraprecisi come `8.16` abolendo i blocchi interi da 8.00 o 9.00.

### 5. `src/cold_start.py` — L'Onboarding Intelligente

Per utenti **senza profilo AniList**, il sistema costruisce un "pseudo-profilo" tramite un flusso guidato. Supporta sia Anime che Manga con scorer e builder dedicati.

*   **Preferenze Esplicite:** Generi preferiti/da evitare, formato ideale, lunghezza (episodi/capitoli), epoca storica.
*   **Preferenze Implicite:** Derivate automaticamente dai 3-5 preferiti dell'utente — pesi per genere, tag tematici, formato dominante, score medio globale.
*   **Heuristic Scorer:** Algoritmo a punteggio con penalità pesanti per generi evitati, bonus per match espliciti/impliciti, e tie-breaker basato sulla qualità globale.
*   **Spiegazioni in Linguaggio Naturale:** Ogni punteggio è accompagnato da una lista di giustificazioni con emoji e decimi esatti (es: `✅ Ottimo Match Generi (+2.0)`, `❌ Fortemente sconsigliato (-2.5)`).

### 6. `app/main.py` — La Dashboard Streamlit

Il banco di prova visuale. Gestito in `Streamlit` con **due modalità operative** e **media toggle Anime/Manga**.

**Modalità 1: Profilo AniList (Machine Learning)**
*   Fetching e training automatico del modello personalizzato dalla sidebar.
*   Dashboard con dettagli del modello, feature importances e metriche di confronto.
*   **Top 10 Raccomandazioni ML:** Inference su 500 candidati (per popolarità + score), escludendo anime/manga già visti/letti (tranne "Planning").
*   **Predizione Punteggio Singolo:** Ricerca per titolo, selezione esatta, calcolo del voto previsto con cover image.
*   **Alert "Plan to Watch/Read":** Notifica se il titolo è già nella lista "Planning" dell'utente.

**Modalità 2: Nuovo Utente (Cold Start)**
*   Flusso di onboarding a 2 step (preferiti → domande gusti).
*   Top 10 raccomandazioni istantanee basate sul profilo generato.
*   Calcolo affinità singola con spiegazione dettagliata delle ragioni.

---

## 🚀 Come Cominciare

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/arsid2828/AniListPredictor.git
    cd AniListPredictor
    ```

2.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Avvia il server locale:**
    ```bash
    python -m streamlit run app/main.py
    ```

4.  **Usa l'app:**
    *   Seleziona il tipo di media (**Anime** o **Manga**) dalla sidebar.
    *   **Se hai un profilo AniList:** Inserisci il tuo username, traina il modello e ottieni previsioni personalizzate.
    *   **Se sei un nuovo utente:** Scegli la modalità "Cold Start", completa l'onboarding e scopri i tuoi titoli perfetti.

---

## 📦 Dipendenze

| Libreria | Utilizzo |
|---|---|
| `pandas` | Manipolazione dati e DataFrame |
| `numpy` | Calcoli numerici |
| `scikit-learn` | Modelli ML, metriche, preprocessing |
| `requests` | Chiamate HTTP all'API GraphQL |
| `streamlit` | Interfaccia web interattiva |
| `matplotlib` | Visualizzazioni (grafici) |
| `seaborn` | Grafici statistici avanzati |
