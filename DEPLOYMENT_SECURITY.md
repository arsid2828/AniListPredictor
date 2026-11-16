# Deploy privato con Streamlit

L'app principale usa Google OpenID Connect per autenticare gli utenti e una lista di account autorizzati per controllare l'accesso.

1. Configura Streamlit Community Cloud per eseguire `app/main.py` con Python 3.12.
2. Registra un client OAuth per un'applicazione web in Google Cloud. Imposta come URI di reindirizzamento l'indirizzo del deploy seguito da `/oauth2callback`.
3. Copia [l'esempio di configurazione](.streamlit/secrets.example.toml) nelle impostazioni **Secrets** di Streamlit e imposta `redirect_uri`, `client_id` e `client_secret`. Genera un valore casuale per `cookie_secret`, ad esempio con `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
4. Inserisci gli account autorizzati in `[access].allowed_emails` o i loro hash SHA-256 in `[access].allowed_email_hashes`. Per abilitare la pagina Admin, imposta `[access].admin_email` con uno degli account già autorizzati.
5. Imposta `PUBLIC_CONTACT_EMAIL` con un indirizzo di contatto valido per le pagine Privacy Policy e Data Source & Legal.

Lascia `disable_auth_for_local_dev = false` in produzione. Se il client OAuth di Google è in modalità Testing, aggiungi gli account autorizzati anche alla lista dei tester in Google Cloud.

Verifica l'accesso con un account autorizzato e con uno escluso. Se hai configurato Admin, apri la pagina con l'account indicato in `admin_email` e controlla che gli altri account non vi accedano.

## Dati e credenziali

Conserva le credenziali nelle impostazioni del servizio di hosting o nel file locale `.streamlit/secrets.toml`, escluso da Git. I dati generati a runtime restano nelle cartelle `cache/`, `data/` e `models/`. La pulizia dei file scaduti avviene quando l'app inizializza una sessione. Gli utenti possono rimuovere i file per un profilo dal menu laterale.

L'app utilizza l'API AniList in sola lettura, limita la frequenza delle richieste e gestisce le risposte HTTP 429. Per le condizioni d'uso e i limiti del servizio, consulta la [documentazione di AniList](https://docs.anilist.co/). La configurazione OAuth è descritta nella [guida all'autenticazione di Streamlit](https://docs.streamlit.io/develop/concepts/connections/authentication).
