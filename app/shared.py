import html
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, List

import joblib
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"
APP_NAME = "AniList Score Predictor UNOFFICIAL"
APP_DISCLAIMER = (
    "Data sourced from the AniList GraphQL API. This project is unofficial and is not "
    "affiliated with or endorsed by AniList. Anime and manga titles, cover images, and "
    "descriptions remain the property of their respective owners."
)
APP_SCOPE_NOTE = (
    "This application is an analytics and prediction companion for AniList data. It is not "
    "a replacement list-tracker service and does not modify AniList accounts or lists."
)
DATA_SOURCE_URL = "https://anilist.co"
API_DOCS_URL = "https://anilist.gitbook.io/anilist-apiv2-docs"
CACHE_RETENTION_HOURS = 6
DATA_RETENTION_DAYS = 7
MODEL_RETENTION_DAYS = 7
DEFAULT_SESSION_MAX_MINUTES = 720
DEFAULT_DISPLAY_TIMEZONE = "Europe/Rome"
PROFILE_OWNER_REGISTRY_PATH = CACHE_DIR / "profile_owners.json"
ADMIN_EMAIL = "arsidhocia@gmail.com"

MAX_USERNAME_LEN = 50
MAX_SEARCH_LEN = 100
MAX_TAG_LEN = 50
MAX_STUDIO_LEN = 100
MAX_YEAR_LEN = 4
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
USERNAME_VALIDATION_MESSAGE = "Username can only contain letters, numbers, underscores, and hyphens."

CUSTOM_CSS = """
<style>
    .stApp {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        min-height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    
    .badge-good {
        display: inline-block;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-warning {
        display: inline-block;
        background: linear-gradient(135deg, #f2994a, #f2c94c);
        color: #1a1a2e;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-danger {
        display: inline-block;
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .stExpander {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 112px;
    }

    .profile-metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-left-width: 5px;
        min-height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }

    .profile-metric-label {
        font-size: 0.8rem;
        color: #a0a0b0;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .profile-metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.25;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, rgba(26,26,46,0.8), rgba(22,33,62,0.8));
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.2s ease;
    }
    
    .recommendation-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    .user-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .user-badge-inactive {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.08);
        color: #a0a0b0;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .legal-footer {
        margin-top: 2rem;
        padding: 1rem 1.1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #b9bfd0;
        font-size: 0.92rem;
        line-height: 1.55;
        background: rgba(255,255,255,0.02);
        border-radius: 14px;
    }
</style>
"""

AUTO_TRAIN_DAYS = 7

def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _secret_section(name):
    try:
        section = st.secrets.get(name, {})
    except Exception:
        return {}
    return dict(section) if hasattr(section, "items") else {}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_email(email):
    return str(email or "").strip().lower()


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _viewer_keys_for_email(email: str) -> List[str]:
    normalized = _normalize_email(email)
    if not normalized:
        return ["local-dev"]
    return [normalized, _sha256_text(normalized)]


def _current_viewer_key():
    return _viewer_keys_for_email(_get_current_user_email())[0]


def _load_profile_owner_registry():
    if not PROFILE_OWNER_REGISTRY_PATH.exists():
        return {}
    try:
        with open(PROFILE_OWNER_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_profile_owner_registry(registry):
    try:
        with open(PROFILE_OWNER_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2, sort_keys=True)
    except OSError:
        pass


def register_profile_for_current_viewer(username: str):
    normalized = str(username or "").strip().lower()
    if not normalized:
        return
    registry = _load_profile_owner_registry()
    viewer_key = _current_viewer_key()
    profiles = set(registry.get(viewer_key, []))
    profiles.add(normalized)
    registry[viewer_key] = sorted(profiles)
    _save_profile_owner_registry(registry)


def unregister_profile_for_current_viewer(username: str):
    normalized = str(username or "").strip().lower()
    if not normalized:
        return
    registry = _load_profile_owner_registry()
    viewer_email = _get_current_user_email()
    changed = False
    for viewer_key in _viewer_keys_for_email(viewer_email):
        profiles = set(registry.get(viewer_key, []))
        if normalized not in profiles:
            continue
        profiles.remove(normalized)
        changed = True
        if profiles:
            registry[viewer_key] = sorted(profiles)
        else:
            registry.pop(viewer_key, None)
    if not changed:
        return
    _save_profile_owner_registry(registry)


def _get_current_user_email():
    try:
        return _normalize_email(st.user.get("email") or st.user.email)
    except Exception:
        return ""


def is_admin_user():
    return _get_current_user_email() == ADMIN_EMAIL


def require_admin_user():
    if not is_admin_user():
        st.error("This page is restricted to the site owner.")
        st.stop()


def _get_allowed_emails():
    access = _secret_section("access")
    emails = {_normalize_email(email) for email in _as_list(access.get("allowed_emails"))}
    hashes = {str(item).strip().lower() for item in _as_list(access.get("allowed_email_hashes"))}
    return {email for email in emails if email}, {item for item in hashes if item}


def _is_email_authorized(email):
    allowed_emails, allowed_hashes = _get_allowed_emails()
    normalized = _normalize_email(email)
    if not normalized:
        return False
    if normalized in allowed_emails:
        return True
    return _sha256_text(normalized) in allowed_hashes


def _configured_session_max_minutes():
    access = _secret_section("access")
    try:
        value = int(access.get("session_max_minutes", DEFAULT_SESSION_MAX_MINUTES))
    except (TypeError, ValueError):
        value = DEFAULT_SESSION_MAX_MINUTES
    return max(15, min(value, 24 * 60))


def get_display_timezone_name():
    app_config = _secret_section("app")
    tz_name = str(app_config.get("display_timezone", DEFAULT_DISPLAY_TIMEZONE)).strip()
    return tz_name or DEFAULT_DISPLAY_TIMEZONE


def _get_display_timezone():
    try:
        return ZoneInfo(get_display_timezone_name())
    except Exception:
        return timezone.utc


def format_training_timestamp(trained_at):
    if not trained_at:
        return "Unknown"
    try:
        trained_dt = datetime.fromisoformat(str(trained_at))
        if trained_dt.tzinfo is None:
            trained_dt = trained_dt.replace(tzinfo=timezone.utc)
        localized = trained_dt.astimezone(_get_display_timezone())
        return localized.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(trained_at)


def _render_login_screen():
    st.title(APP_NAME)
    st.caption("Private deployment. Sign in with an authorized Google account.")
    st.button("Sign in with Google", on_click=st.login)


def require_authorized_google_user():
    """
    Stop unauthenticated or unauthorized visitors before any app page can run.

    Configure Streamlit OIDC under [auth] and the app allowlist under [access]
    in .streamlit/secrets.toml or Streamlit Community Cloud secrets.
    """
    auth_config = _secret_section("auth")
    access_config = _secret_section("access")

    if access_config.get("disable_auth_for_local_dev") and not auth_config:
        st.warning("Authentication is disabled for local development. Do not enable this in production.")
        return

    if not auth_config:
        st.error("Authentication is not configured. Add [auth] and [access] to Streamlit secrets before deployment.")
        st.stop()

    if not getattr(st.user, "is_logged_in", False):
        _render_login_screen()
        st.stop()

    email = _get_current_user_email()
    allowed_emails, allowed_hashes = _get_allowed_emails()
    if not allowed_emails and not allowed_hashes:
        st.error("No authorized Google accounts are configured in [access].")
        st.button("Log out", on_click=st.logout)
        st.stop()

    if not _is_email_authorized(email):
        safe_email = html.escape(email or "unknown account")
        st.error(f"{safe_email} is signed in, but is not authorized for this private app.")
        st.button("Log out", on_click=st.logout)
        st.stop()

    max_minutes = _configured_session_max_minutes()
    now_ts = datetime.now().timestamp()
    login_seen_at = st.session_state.setdefault("_login_seen_at", now_ts)
    if now_ts - float(login_seen_at) > max_minutes * 60:
        st.warning("Session expired. Please sign in again.")
        st.logout()
        st.stop()


def render_auth_sidebar():
    email = html.escape(_get_current_user_email())
    if email:
        st.sidebar.caption(f"Signed in as {email}")
    st.sidebar.button("Log out", on_click=st.logout, width="stretch", key="auth_logout")


def purge_expired_runtime_artifacts():
    """Delete stale local artifacts to reduce retention of AniList-derived user data."""
    retention_rules = (
        (CACHE_DIR, CACHE_RETENTION_HOURS * 3600),
        (DATA_DIR, DATA_RETENTION_DAYS * 86400),
        (MODELS_DIR, MODEL_RETENTION_DAYS * 86400),
    )
    now_ts = datetime.now().timestamp()
    for directory, max_age_seconds in retention_rules:
        if not directory.exists():
            continue
        for path in directory.glob("*"):
            if not path.is_file():
                continue
            try:
                age_seconds = now_ts - path.stat().st_mtime
                if age_seconds > max_age_seconds:
                    path.unlink(missing_ok=True)
            except OSError:
                continue


def normalize_limited_text(value, max_len):
    return str(value or "").strip()[:max_len]


def is_valid_username(username):
    return bool(USERNAME_PATTERN.fullmatch(username))


def _resolve_safe_path(base_dir: Path, filename: str) -> Path:
    base_resolved = base_dir.resolve()
    candidate = (base_resolved / filename).resolve()
    if candidate.parent != base_resolved:
        raise ValueError("Invalid path outside the expected directory.")
    return candidate


def get_model_path(username: str, is_manga: bool = False) -> Path:
    suffix = "_manga_best_model.pkl" if is_manga else "_best_model.pkl"
    return _resolve_safe_path(MODELS_DIR, f"{username.lower()}{suffix}")


def get_dataset_path(username: str, is_manga: bool = False) -> Path:
    suffix = "_manga_clean.csv" if is_manga else "_clean.csv"
    return _resolve_safe_path(DATA_DIR, f"{username.lower()}{suffix}")


def safe_load_model_artifact(username: str, is_manga: bool = False):
    model_path = get_model_path(username, is_manga=is_manga)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path.name}")
    return joblib.load(model_path)


def delete_local_user_artifacts(username: str):
    """Remove local cache, dataset, and model files for one username."""
    normalized = str(username or "").strip().lower()
    if not normalized:
        return

    patterns = (
        CACHE_DIR / f"user_list_{normalized}.json",
        CACHE_DIR / f"user_manga_list_{normalized}.json",
        CACHE_DIR / f"user_activity_anime_{normalized}.json",
        CACHE_DIR / f"user_activity_manga_{normalized}.json",
        DATA_DIR / f"{normalized}_clean.csv",
        DATA_DIR / f"{normalized}_manga_clean.csv",
        MODELS_DIR / f"{normalized}_best_model.pkl",
        MODELS_DIR / f"{normalized}_manga_best_model.pkl",
    )

    for path in patterns:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
    unregister_profile_for_all_viewers(normalized)
    unregister_profile_for_current_viewer(normalized)


def _normalize_registry_viewer_email(viewer_email: str) -> str:
    return _normalize_email(viewer_email)


def register_profile_for_viewer(viewer_email: str, username: str):
    normalized_email = _normalize_registry_viewer_email(viewer_email)
    normalized_username = str(username or "").strip().lower()
    if not normalized_email or not normalized_username:
        return False
    registry = _load_profile_owner_registry()
    profiles = set(registry.get(normalized_email, []))
    profiles.add(normalized_username)
    registry[normalized_email] = sorted(profiles)
    _save_profile_owner_registry(registry)
    return True


def unregister_profile_for_viewer(viewer_email: str, username: str):
    normalized_email = _normalize_registry_viewer_email(viewer_email)
    normalized_username = str(username or "").strip().lower()
    if not normalized_email or not normalized_username:
        return False
    registry = _load_profile_owner_registry()
    profiles = set(registry.get(normalized_email, []))
    if normalized_username not in profiles:
        return False
    profiles.remove(normalized_username)
    if profiles:
        registry[normalized_email] = sorted(profiles)
    else:
        registry.pop(normalized_email, None)
    _save_profile_owner_registry(registry)
    return True


def unregister_profile_for_all_viewers(username: str):
    normalized_username = str(username or "").strip().lower()
    if not normalized_username:
        return False
    registry = _load_profile_owner_registry()
    changed = False
    for viewer_key in list(registry.keys()):
        profiles = set(registry.get(viewer_key, []))
        if normalized_username not in profiles:
            continue
        profiles.remove(normalized_username)
        changed = True
        if profiles:
            registry[viewer_key] = sorted(profiles)
        else:
            registry.pop(viewer_key, None)
    if changed:
        _save_profile_owner_registry(registry)
    return changed

def init_session_state():
    if not st.session_state.get("_runtime_cleanup_done"):
        purge_expired_runtime_artifacts()
        st.session_state["_runtime_cleanup_done"] = True
    defaults = {
        "username": "",
        "model_trained_anime": False,
        "model_trained_manga": False,
        "cs_favorites_anime": [],
        "cs_favorites_manga": [],
        "cs_profile_anime": None,
        "cs_profile_manga": None,
        "cs_step_anime": 1,
        "cs_step_manga": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def check_planning_status(username, media_id, media_type="anime"):
    prefix = "user_list_" if media_type == "anime" else "user_manga_list_"
    cache_path = CACHE_DIR / f"{prefix}{username.lower()}.json"
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for lst in data:
                    if lst.get('status') == 'PLANNING':
                        for entry in lst.get('entries', []):
                            if entry.get('mediaId') == media_id:
                                return True
        except:
            pass
    return False

def check_auto_train_needed(model_artifact):
    """Check if model was trained more than AUTO_TRAIN_DAYS ago."""
    trained_at = model_artifact.get('trained_at')
    if not trained_at:
        return True
    try:
        trained_dt = datetime.fromisoformat(str(trained_at))
        if trained_dt.tzinfo is None:
            trained_dt = trained_dt.replace(tzinfo=timezone.utc)
        days_ago = (datetime.now(timezone.utc) - trained_dt.astimezone(timezone.utc)).days
        return days_ago >= AUTO_TRAIN_DAYS
    except:
        return True

def render_metric_card(label, value, delta=None):
    delta_html = ""
    if delta is not None:
        color = "#00b09b" if delta >= 0 else "#eb3349"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div style="color: {color}; font-size: 0.9rem;">{sign}{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_user_badge():
    """Show the currently logged-in username as a badge on any page."""
    render_auth_sidebar()
    username = st.session_state.get("username", "")
    if username:
        safe_username = html.escape(str(username))
        st.sidebar.markdown(f'<div class="user-badge">👤 {safe_username}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="user-badge-inactive">👤 No active profile</div>', unsafe_allow_html=True)


def render_app_disclaimer():
    st.markdown(
        f"""
        <div class="legal-footer">
            <strong>{html.escape(APP_NAME)}</strong><br>
            {html.escape(APP_DISCLAIMER)}<br>
            {html.escape(APP_SCOPE_NOTE)}<br>
            Data retention policy: cache up to {CACHE_RETENTION_HOURS} hours; derived datasets and local models up to {DATA_RETENTION_DAYS} days unless removed earlier.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_cached_profiles():
    """Return only profiles associated with the currently signed-in viewer."""
    registry = _load_profile_owner_registry()
    profiles = set()
    for viewer_key in _viewer_keys_for_email(_get_current_user_email()):
        profiles.update(registry.get(viewer_key, []))
    current_username = normalize_limited_text(st.session_state.get("username", ""), MAX_USERNAME_LEN).lower()
    if current_username:
        profiles.add(current_username)
    return sorted(profile for profile in profiles if profile)


def get_profile_owner_registry():
    return _load_profile_owner_registry()


def get_all_runtime_usernames():
    usernames = set()
    registry = _load_profile_owner_registry()
    for profiles in registry.values():
        usernames.update(str(profile).strip().lower() for profile in profiles if str(profile).strip())

    file_patterns = (
        (MODELS_DIR, "*_best_model.pkl", [r"_manga_best_model$", r"_best_model$"]),
        (DATA_DIR, "*_clean.csv", [r"_manga_clean$", r"_clean$"]),
        (CACHE_DIR, "user_list_*.json", [r"^user_list_"]),
        (CACHE_DIR, "user_manga_list_*.json", [r"^user_manga_list_"]),
        (CACHE_DIR, "user_activity_anime_*.json", [r"^user_activity_anime_"]),
        (CACHE_DIR, "user_activity_manga_*.json", [r"^user_activity_manga_"]),
    )
    for directory, pattern, regexes in file_patterns:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            name = path.stem
            for regex in regexes:
                name = re.sub(regex, "", name)
            name = str(name).strip().lower()
            if name:
                usernames.add(name)
    return sorted(usernames)


def get_user_artifact_paths(username: str) -> Dict[str, Path]:
    normalized = str(username or "").strip().lower()
    return {
        "anime_list_cache": CACHE_DIR / f"user_list_{normalized}.json",
        "manga_list_cache": CACHE_DIR / f"user_manga_list_{normalized}.json",
        "anime_history_cache": CACHE_DIR / f"user_activity_anime_{normalized}.json",
        "manga_history_cache": CACHE_DIR / f"user_activity_manga_{normalized}.json",
        "anime_dataset": DATA_DIR / f"{normalized}_clean.csv",
        "manga_dataset": DATA_DIR / f"{normalized}_manga_clean.csv",
        "anime_model": MODELS_DIR / f"{normalized}_best_model.pkl",
        "manga_model": MODELS_DIR / f"{normalized}_manga_best_model.pkl",
    }


def get_user_artifact_summary(username: str):
    summaries = []
    for artifact_type, path in get_user_artifact_paths(username).items():
        exists = path.exists()
        stat = path.stat() if exists else None
        summaries.append(
            {
                "artifact_type": artifact_type,
                "path": str(path),
                "exists": exists,
                "size_bytes": int(stat.st_size) if stat else 0,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone(_get_display_timezone()).strftime("%d/%m/%Y %H:%M:%S") if stat else "",
            }
        )
    return summaries


def delete_artifact_by_type(username: str, artifact_type: str):
    paths = get_user_artifact_paths(username)
    path = paths.get(str(artifact_type or "").strip())
    if path is None:
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False
