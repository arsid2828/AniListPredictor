import os
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import (
    APP_NAME,
    CACHE_RETENTION_HOURS,
    DATA_RETENTION_DAYS,
    render_app_disclaimer,
    init_session_state,
    inject_css,
    render_user_badge,
)

CONTACT_EMAIL = os.environ.get("PUBLIC_CONTACT_EMAIL", "set-public-contact@example.com")

st.set_page_config(page_title="Privacy Policy", layout="wide", page_icon="🔒")
inject_css()
init_session_state()
render_user_badge()

st.title("🔒 Privacy Policy")
st.caption(f"This policy applies to {APP_NAME}.")

if CONTACT_EMAIL == "set-public-contact@example.com":
    st.warning("Set the `PUBLIC_CONTACT_EMAIL` environment variable before production deployment.")

st.markdown(
    f"""
### 1. Who operates this app
This application is operated by the project owner deploying this site. For privacy requests, contact: `{CONTACT_EMAIL}`.

### 2. What data is processed
- Google account identity claims needed for login, especially email address
- AniList usernames entered by visitors
- Public AniList profile, list, title, and activity data requested from the AniList GraphQL API
- Derived analytics needed to generate charts, compatibility views, and score predictions
- Temporary local cache, temporary datasets, and temporary trained models created by the application

### 3. Why the data is processed
- To fetch public AniList data requested by the visitor
- To restrict access to explicitly authorized Google accounts
- To calculate recommendations, compatibility, analytics, and score predictions
- To reduce repeated API calls and respect AniList rate limits through short-lived caching
- To support debugging, reliability, and secure operation of the site

### 4. Legal basis and scope
This app is designed as an unofficial analytics and prediction companion for AniList data. It does not modify AniList accounts and is not intended to replace AniList as a list-tracking service.

### 5. How long data is kept
- API cache files are automatically removed after about {CACHE_RETENTION_HOURS} hours
- Derived datasets and local trained models are automatically removed after about {DATA_RETENTION_DAYS} days
- Runtime retention may be shortened further by manual deletion
- Streamlit authentication cookies may persist in the browser after login; use the logout button on shared devices

### 6. Sharing
This app is not designed to sell personal data. Data may be processed by the hosting provider that runs the deployed site, by Google for authentication, and by AniList when a user asks the app to fetch AniList data.

### 7. User rights
If applicable under your jurisdiction, you may request access, correction, or deletion of locally retained data related to your use of this app by contacting `{CONTACT_EMAIL}`.

### 8. Important note
AniList usernames and related taste/activity data can qualify as personal data depending on context. Deploy this app only with a real public contact channel and hosting disclosures appropriate for your jurisdiction.
"""
)

render_app_disclaimer()
