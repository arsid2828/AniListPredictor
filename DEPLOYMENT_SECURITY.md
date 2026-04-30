# Private Streamlit Deployment Checklist

This app is intended to be deployed as a private Streamlit app protected by Google OpenID Connect and an application-level allowlist.

## Access model

1. Deploy the app from the repository with entry point `app/main.py`.
2. Configure Google OAuth/OIDC for the app URL.
3. Add Streamlit secrets based on `.streamlit/secrets.example.toml`.
4. Add only trusted Google accounts to `[access].allowed_emails` or `[access].allowed_email_hashes`.
5. Keep the Google OAuth consent screen in Testing mode if this app is only for you and a few explicitly added test users.

Important: Google login limits accounts, not physical devices. To limit devices, rely on Google account security controls such as 2-step verification, passkeys, session/device review, and revoking unknown sessions. The app itself enforces an allowed-account list and an app-level session timeout.

## Secrets

Never commit `.streamlit/secrets.toml`. Put production secrets in Streamlit Community Cloud settings:

```toml
[auth]
redirect_uri = "https://YOUR_APP.streamlit.app/oauth2callback"
cookie_secret = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
client_id = "REPLACE_WITH_GOOGLE_OAUTH_CLIENT_ID"
client_secret = "REPLACE_WITH_GOOGLE_OAUTH_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[access]
allowed_emails = ["you@example.com"]
allowed_email_hashes = []
session_max_minutes = 720
disable_auth_for_local_dev = false
```

Use a random `cookie_secret` of at least 32 bytes. Do not set `expose_tokens`; the app does not need Google access tokens.

The local `.streamlit/secrets.toml` may contain `disable_auth_for_local_dev = true` to make development easier. Do not paste that local-only setting as `true` into Streamlit Community Cloud.

## Data protection

- `cache/`, `data/`, and `models/` are gitignored and should not be deployed with personal runtime artifacts.
- AniList API cache is retained for up to 6 hours.
- Derived datasets and local model artifacts are retained for up to 7 days.
- Users can delete local artifacts for an AniList username from the sidebar.
- On Streamlit Community Cloud, runtime storage is ephemeral, but do not rely on that as the only deletion control.

Before deploy, remove local generated artifacts or keep them untracked:

```powershell
git status --short
git ls-files | Select-String -Pattern "^(cache|data|models)/"
```

The second command should print nothing.

## AniList API compliance

- This app is read-only and does not modify AniList accounts or lists.
- It is clearly labeled unofficial and links back to AniList.
- It uses local short-term caching to reduce repeated API requests.
- It handles `429 Too Many Requests` and respects `Retry-After`.
- It sleeps between paginated requests.
- Avoid automated scraping, bulk mirroring, reselling AniList data, or presenting the app as official.
- AniList currently documents 90 requests/minute, with a temporary degraded-state limit of 30 requests/minute. Keep candidate limits moderate and do not run many concurrent sessions.

## Pre-deploy checks

1. Confirm `.streamlit/secrets.toml` is not tracked.
2. Confirm `cache/`, `data/`, and `models/` are not tracked.
3. Set a real `PUBLIC_CONTACT_EMAIL` in hosting settings if you expose the privacy pages.
4. Verify the deployed `redirect_uri` exactly matches the Google OAuth authorized redirect URI.
5. Test login with an allowed Google account.
6. Test login with a non-allowed Google account and confirm it is blocked.
7. Use the delete button for a test AniList username and confirm cache/data/model files disappear from the runtime.
