# 🎬 AniList AI Predictor 🧠

Un ecosistema di Machine Learning ibrido progettato da zero per analizzare, profilare e prevedere matematicamente i voti che un utente darà a un anime su [AniList](https://anilist.co/).

Non si tratta di un semplice sistema di raccomandazione: questo modello studia a fondo i pattern latenti, le variazioni di umore e le preferenze cronologiche dell'utente. L'aspetto rivoluzionario risiede nell'architettura **"Leakage-Free"** (Zero Fuga di Dati), in quanto l'IA calcola i dati storici comportamentali come se viaggiasse nel tempo passo dopo passo.

## ✨ Caratteristiche Tecniche Avanzate

*   **Leave-One-Out (LOO) Perturbation Test:** Il modello non si limita a dire "quanto puzza o profuma" la statistica. Clona una versione neutra dell'anime in background per dirti i decimi esatti (`+/- decimi di voto`) di cui ha abbassato o rialzato il voto finale per colpa di quello specifico attributo.
*   **Recency Mood Bias:** Misuriamo la media mobile degli "ultimi 10 anime visti". Se l'utente in quel decennio era depresso o "di manica stretta", l'algoritmo scala proporzionalmente all'umore storico locale.
*   **Contrarian Score (MAF Storico):** Il sistema non guarda solo il segno della devianza ma quantifica letteralmente se l'utente è un puro "Normie" (segue il voto della community) o un "Bastian Contrario", bilanciando quanto delegare l'output alla folla globale vs ai gusti unici dell'utente.
*   **Protezione Out-of-Distribution (OOD):** Con regole euristiche umane, l'UI ha delle protezioni fisiche contro le allucinazioni algoritmiche (es: colpire con penalità pesantissime un tag "Hentai" per utenti che non ne guardano mai, bypassando l'inesperienza latente dell'Albero Decisionale).

---

## 🗂️ Struttura dei File
Il progetto è suddiviso in una pipeline modulare MLOps:

### 1. `src/api.py` (L'Estrattore)
L'interfaccia di interrogazione **GraphQL** ufficiale di AniList. Costruita con gestione dei rate-limits, un retry esponenziale (backoff), e la paginazione automatica a Chunk. Ha un sistema di Caching locale per non assaltare il server e velocizzare la pulizia in locale.

### 2. `src/dataset.py` (Il Raffinatore)
Prende l'infinito dizionario destrutturato Json di AniList e lo appiattisce in un formato colonnare solido (`pandas`). È in questa fase che si separano in chirurgico isolamento metriche come gli Studi D'Animazione dai Produttori (es: *Aniplex*). Vengono rimossi i voti nulli troll e applicate le deviazioni standard.

### 3. `src/features.py` (Il Motore del Feature Engineering)
Il cuore pulsante del Machine Learning.
Questo file non sbatte i dati così come sono nel modello ma crea il *"Tempo"*. Attraversa l'istante X della vita dell'utente con indici come `hist_tag_freq` e `hist_user_std`. Produce i dati categorici Multi-Hot di ultima generazione che valutano le percentuali d'impatto dei generi in base al rango specifico della serie.

### 4. `src/models.py` (Il Cervello Ibrido)
Traina il database e valuta i pesi usando una validazione incrociata intelligente nel tempo (`TimeSeriesSplit` personalizzato). 
Il modello vincente è un potentissimo **Voting Ensemble** bilanciato:
*   `60% Random Forest Regressor`: Perfetto per intercettare nicchie, eccezioni umane e incroci astratti ("Amo le commedie Ma SOLO se sono fatte dalla Sunrise, altrimenti odio la regia").
*   `40% Ridge Regression`: Un modello lineare robusto. L'albero di Random Forest calcola per natura a "scalini", ma il Ridge arrotonda e smussa gli sbilanci permettendo voti ultraprecisi come `8.16` abolendo i blocchi interi da 8.00 o 9.00.

### 5. `app/main.py` (La UI di Streamlit)
Il banco di Prova visuale. Gestito in `Streamlit`.
Dalla barra laterale si fetcha e allena dinamicamente il modello utente in base alla libreria AniList. Accoglie l'Inference Predictor visuale che scansiona in parallelo le URL delle Cover Image. Fornisce all'utente finale in chiaro il reverse-logaritmo delle percentuali e la stringa "umana" che giustifica ogni scelta calcolata per un voto.

---

## 🚀 Come cominciare:

1. **Installazione librerie:** `pip install -r requirements.txt`
2. **Avvio Server Locale:** `python -m streamlit run app/main.py`
3. Usare la Sidebar a Sinistra per coniare e trainare dinamicamente la propria storia utente.
