import os
import sys
from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
ROOT_DIR = APP_DIR.parent
for path in (str(ROOT_DIR), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from shared import (
    API_DOCS_URL,
    APP_NAME,
    APP_SCOPE_NOTE,
    DATA_SOURCE_URL,
    render_app_disclaimer,
    init_session_state,
    inject_css,
    render_user_badge,
)

CONTACT_EMAIL = os.environ.get("PUBLIC_CONTACT_EMAIL", "set-public-contact@example.com")

st.set_page_config(page_title="Data Source & Legal", layout="wide", page_icon="⚖️")
inject_css()
init_session_state()
render_user_badge()

st.title("⚖️ Data Source & Legal")
st.caption(f"Legal and attribution information for {APP_NAME}.")

st.markdown(
    f"""
### Official data source
- Public data is requested from the [AniList website]({DATA_SOURCE_URL}) and the [AniList GraphQL API documentation]({API_DOCS_URL}).
- This project is **unofficial** and is **not affiliated with or endorsed by AniList**.

### What this app does
- Fetches public AniList data requested by the visitor
- Generates analytics, compatibility views, and score predictions
- Links users back to AniList entries for the original source pages

### What this app does not do
- It does not modify AniList accounts or lists
- It is not presented as an official AniList product
- It is not intended to operate as a replacement tracker service

### Intellectual property and content notice
- Anime and manga titles, descriptions, cover images, and related metadata belong to their respective owners and licensors
- AniList-related branding remains the property of AniList
- This app uses source links back to AniList and presents itself as an unofficial companion

### API usage notice
- The app is designed to use AniList data sparingly and with local short-term caching
- Deployment should continue respecting AniList rate limits and terms of use
- If AniList returns `429 Too Many Requests`, the app waits before retrying instead of continuing immediately
- Candidate downloads should be kept moderate, especially while AniList publishes reduced temporary limits
- If the project becomes commercial beyond AniList's stated threshold or needs special approval, contact AniList directly at `contact@anilist.co`

### Private access notice
- App access is restricted through Google OpenID Connect and a server-side allowlist configured in Streamlit secrets
- Google login limits authorized accounts, not physical devices; protect allowed Google accounts with strong device and account security

### Project scope statement
{APP_SCOPE_NOTE}

### Operator contact
For project-specific legal or data questions, publish a real operator contact before production deployment. Current configured contact: `{CONTACT_EMAIL}`
"""
)

render_app_disclaimer()
