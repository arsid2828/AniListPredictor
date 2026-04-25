import html
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"
APP_NAME = "AniList Score Predictor - Versione Base"
CACHE_RETENTION_HOURS = 6
DATA_RETENTION_DAYS = 7
MODEL_RETENTION_DAYS = 7

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

</style>
"""

AUTO_TRAIN_DAYS = 7

def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
        days_ago = (datetime.now() - trained_dt).days
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
    username = st.session_state.get("username", "")
    if username:
        safe_username = html.escape(str(username))
        st.sidebar.markdown(f'<div class="user-badge">👤 {safe_username}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="user-badge-inactive">👤 No active profile</div>', unsafe_allow_html=True)


def render_app_disclaimer():
    return None


def get_cached_profiles():
    """Scan the models directory for previously trained profiles."""
    profiles = set()
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*_best_model.pkl"):
            name = f.stem
            # Extract username: remove _best_model or _manga_best_model suffix
            name = re.sub(r'_manga_best_model$', '', name)
            name = re.sub(r'_best_model$', '', name)
            if name:
                profiles.add(name.lower())
    # Also check cache for fetched-but-not-yet-trained profiles
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("user_list_*.json"):
            name = f.stem.replace('user_list_', '')
            if name:
                profiles.add(name.lower())
    return sorted(list(profiles))
