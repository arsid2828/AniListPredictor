import streamlit as st
import json
import re
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "cache"

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
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

def init_session_state():
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
        st.sidebar.markdown(f'<div class="user-badge">👤 {username}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="user-badge-inactive">👤 Nessun profilo attivo</div>', unsafe_allow_html=True)


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
                profiles.add(name)
    # Also check cache for fetched-but-not-yet-trained profiles
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("user_list_*.json"):
            name = f.stem.replace('user_list_', '')
            if name:
                profiles.add(name)
    return sorted(profiles)
