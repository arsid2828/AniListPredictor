import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api import search_anime_by_title, get_candidate_anime_for_recommendations, search_manga_by_title, get_candidate_manga_for_recommendations
from src.dataset import build_user_dataframe, build_user_manga_dataframe
from src.features import build_inference_features, build_manga_inference_features
from src.models import train_and_evaluate_all_models, train_and_evaluate_all_manga_models
from src.analytics import get_shap_explanation, shap_to_dataframe
from src.cold_start import (
    build_cold_start_profile, content_based_heuristic_scorer, generate_cold_start_recommendations,
    GENRES_LIST, FORMATS_LIST, LENGTHS_LIST, ERAS_LIST,
    build_manga_cold_start_profile, content_based_heuristic_scorer_manga,
    generate_manga_cold_start_recommendations, MANGA_FORMATS_LIST, MANGA_LENGTHS_LIST
)
from app.shared import (
    APP_NAME,
    MAX_SEARCH_LEN,
    MAX_STUDIO_LEN,
    MAX_TAG_LEN,
    MAX_USERNAME_LEN,
    MAX_YEAR_LEN,
    USERNAME_VALIDATION_MESSAGE,
    check_auto_train_needed,
    check_planning_status,
    delete_local_user_artifacts,
    get_cached_profiles,
    get_dataset_path,
    get_model_path,
    init_session_state,
    inject_css,
    is_valid_username,
    normalize_limited_text,
    render_app_disclaimer,
    render_metric_card,
    render_user_badge,
    safe_load_model_artifact,
)
import plotly.express as px
import plotly.graph_objects as go
import logging

from src.api import CACHE_DIR
logger = logging.getLogger(__name__)

TRAIN_COOLDOWN_SECONDS = 300
RECOMMENDATION_COOLDOWN_SECONDS = 30


def get_remaining_cooldown(session_key, cooldown_seconds):
    remaining = cooldown_seconds - (time.time() - st.session_state.get(session_key, 0))
    return max(0, int(np.ceil(remaining)))

st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🎬")
inject_css()
init_session_state()

st.title("🎬 AniList Score Predictor UNOFFICIAL")
st.markdown("Discover what anime and manga you will love, based on your history or a quick onboarding!")
st.caption(
    "Unofficial AniList companion for analytics and score prediction. "
    "This app does not sync or modify AniList lists."
)

# ========== SIDEBAR ==========
render_user_badge()
media_type = st.sidebar.radio("📌 Media Type:", ["🎬 Anime", "📖 Manga"])
is_manga = "Manga" in media_type
mode = st.sidebar.radio("Select Mode:", ["AniList Profile (Machine Learning)", "New User (Cold Start)"])

# ==========================================
# ANILIST PROFILE (ML) MODE
# ==========================================
if mode == "AniList Profile (Machine Learning)":
    with st.sidebar:
        st.header("1. Setup User Profile")
        st.caption("Data from AniList is cached locally for a limited time to power predictions and analytics.")
        
        # Show previously loaded profiles
        cached_profiles = get_cached_profiles()
        if cached_profiles:
            selected_profile = st.selectbox(
                "📂 Saved Profiles:",
                ["-- New --"] + cached_profiles,
                index=0
            )
            if selected_profile != "-- New --":
                username_default = selected_profile
            else:
                username_default = st.session_state.get("username", "")
        else:
            username_default = st.session_state.get("username", "")
        
        username_input = st.text_input(
            "Enter AniList Username:",
            value=username_default,
            max_chars=MAX_USERNAME_LEN,
        )
        username_lower = normalize_limited_text(username_input, MAX_USERNAME_LEN).lower()
        username_valid = not username_lower or is_valid_username(username_lower)
        model_path = None
        if username_lower and username_valid:
            model_path = get_model_path(username_lower, is_manga=is_manga)
        if username_lower and not username_valid:
            st.warning(USERNAME_VALIDATION_MESSAGE)
        
        model_trained_key = "model_trained_manga" if is_manga else "model_trained_anime"
        
        # Check if model exists for metadata
        if model_path and model_path.exists():
            try:
                meta = safe_load_model_artifact(username_lower, is_manga=is_manga)
                trained_at = meta.get('trained_at', 'Unknown')
                if trained_at != 'Unknown':
                    trained_at = datetime.fromisoformat(str(trained_at)).strftime("%d/%m/%Y %H:%M")
                st.info(f"💾 Model found!\nLast training: {trained_at}")
            except Exception:
                logger.exception("Failed to read model metadata.")
                st.warning("⚠️ Error reading model metadata.")

        col_load, col_train = st.columns(2)
        
        with col_load:
            if st.button("📂 Load", width="stretch", disabled=not (model_path and model_path.exists())):
                st.session_state["username"] = username_lower
                st.session_state[model_trained_key] = True
                st.rerun()
                
        with col_train:
            btn_label = "🔥 Train/Retrain"
            if st.button(btn_label, width="stretch", disabled=not (username_lower and username_valid)):
                last_train_key = f"last_train_{'manga' if is_manga else 'anime'}_{username_lower}"
                remaining = get_remaining_cooldown(last_train_key, TRAIN_COOLDOWN_SECONDS)
                if remaining > 0:
                    st.warning(f"Please wait {remaining}s before training this profile again.")
                    st.stop()
                media_label = "manga" if is_manga else "anime"
                with st.spinner(f"Fetching {media_label} data, training..."):
                    st.session_state["username"] = username_lower
                    st.session_state[last_train_key] = time.time()
                    try:
                        if is_manga:
                            result = train_and_evaluate_all_manga_models(username_lower)
                        else:
                            result = train_and_evaluate_all_models(username_lower)
                    except Exception:
                        logger.exception("Training failed.")
                        st.session_state[model_trained_key] = False
                        st.error("Training failed. Please try again later.")
                        st.stop()
                    
                    if isinstance(result, dict) and result.get("status") in ["error", "fallback"]:
                        if result.get("status") == "fallback":
                            st.warning(result["message"])
                        else:
                            st.error(result["message"])
                        st.session_state[model_trained_key] = False
                    else:
                        st.success("✅ Completed!")
                        st.session_state[model_trained_key] = True
                        st.rerun()

        if username_lower:
            st.markdown("---")
            if st.button("🗑️ Delete Local User Data", width="stretch"):
                delete_local_user_artifacts(username_lower)
                if st.session_state.get("username", "").lower() == username_lower:
                    st.session_state["model_trained_anime"] = False
                    st.session_state["model_trained_manga"] = False
                st.success("Local cache, datasets, activity history, and trained models for this user were deleted.")

    model_trained_key = "model_trained_manga" if is_manga else "model_trained_anime"
    
    if st.session_state[model_trained_key]:
        media_label = "Manga" if is_manga else "Anime"
        username = st.session_state["username"]
        recommendations_key = "last_recommendations_manga" if is_manga else "last_recommendations_anime"
        
        try:
            model_artifact = safe_load_model_artifact(username, is_manga=is_manga)
            csv_path = get_dataset_path(username, is_manga=is_manga)
            user_history_df = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to load model assets.")
            st.error("Failed to load the model. Please retrain.")
            st.stop()
        
        # === AUTO-TRAIN REMINDER ===
        if check_auto_train_needed(model_artifact):
            st.warning("⏰ The model was trained over 7 days ago. Consider retraining it for fresh data!")
        
        # === FALLBACK WARNING ===
        if model_artifact.get('fallback_recommended'):
            st.warning("⚠️ Negative R²: the ML model is not very reliable for this dataset. Consider Cold Start for better results.")
        
        st.header(f"2. {media_label} Dashboard for {username}")
        
        # === Model summary metrics ===
        best_metrics = model_artifact['metrics'][0]  # Already sorted by MAE
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", model_artifact['model_name'][:25])
        with col2:
            st.metric("MAE", f"{best_metrics['MAE']:.3f}")
        with col3:
            st.metric("R²", f"{best_metrics['R2']:.3f}")
        with col4:
            st.metric("Dataset", f"{model_artifact.get('dataset_size', '?')} titles")
        
        # TSCV info
        if model_artifact.get('tscv_scores'):
            scores = model_artifact['tscv_scores']
            st.caption(f"📊 TimeSeriesSplit CV (5-fold): MAE = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        with st.expander("📋 Model Details & Comparison"):
            metrics_df = pd.DataFrame(model_artifact['metrics'])
            
            # Plotly bar chart for model comparison
            fig = px.bar(metrics_df, x='model', y='MAE', color='MAE',
                        color_continuous_scale='RdYlGn_r',
                        title="MAE Comparison between Models (lower = better)")
            fig.update_layout(xaxis_tickangle=-45, height=400, template='plotly_dark')
            st.plotly_chart(fig, width="stretch")
            
            st.dataframe(metrics_df, width="stretch")
            
            if model_artifact.get('feature_importance'):
                fi_df = pd.DataFrame(model_artifact['feature_importance'])
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                               title="Top Feature Importances", color='Importance',
                               color_continuous_scale='Viridis')
                fig_fi.update_layout(height=500, template='plotly_dark', yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_fi, width="stretch")
            
            if model_artifact.get('optuna_result'):
                st.write("**Optuna Tuning:**", model_artifact['optuna_result'])



        # === TOP 10 RECOMMENDATIONS ===
        st.markdown("---")
        st.subheader(f"🌟 Top 10 {media_label} Recommendations")
        
        with st.expander("ℹ️ How does the Candidate search work?"):
            st.markdown("""
            The algorithm sets a global "fishing pool" (the limit below) by fetching half of the results by Popularity and half by Average Score.
            From this pool, it **discards what you have already watched/read** and applies AI predictions.
            **Advanced Filters:** You can force the API to fetch *only* media with a certain Genre, Year, or Tag. If you use very strict filters, we recommend **raising the limit to 1000** to ensure you find enough playable results!
            """)
            
        candidate_limit = st.slider(
            "Number of candidates to download (from the global database):", 
            min_value=10, 
            max_value=1000, 
            value=500, 
            step=10
        )
        
        st.markdown("#### 🎯 Advanced Filters (Optional)")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            genres_opts = ["None", "Action", "Adventure", "Comedy", "Drama", "Ecchi", "Fantasy", "Hentai", "Horror", "Mahou Shoujo", "Mecha", "Music", "Mystery", "Psychological", "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Thriller"]
            sel_genre = st.selectbox("Genre:", genres_opts)
            sel_genre = None if sel_genre == "None" else sel_genre
        with col_f2:
            sel_tag = normalize_limited_text(
                st.text_input("Tag (e.g., Isekai, Magic):", max_chars=MAX_TAG_LEN),
                MAX_TAG_LEN,
            )
            sel_tag = sel_tag if sel_tag else None
        with col_f3:
            if not is_manga:
                sel_year_str = normalize_limited_text(
                    st.text_input("Year (e.g., 2022):", max_chars=MAX_YEAR_LEN),
                    MAX_YEAR_LEN,
                )
                try:
                    sel_year = int(sel_year_str) if sel_year_str else None
                except ValueError:
                    sel_year = None
                    st.warning("Invalid year format.")
            else:
                st.info("Year filter not supported for manga.")
                sel_year = None
        with col_f4:
            sel_studio = normalize_limited_text(
                st.text_input("Studio / Author:", max_chars=MAX_STUDIO_LEN),
                MAX_STUDIO_LEN,
            )
            sel_studio = sel_studio if sel_studio else None
            if sel_studio:
                st.caption("Filtered locally.")
        
        rec_col1, rec_col2 = st.columns([3, 1])
        with rec_col1:
            rec_btn = st.button(f"✨ Generate {media_label} Recommendations", width="stretch")
        
        candidate_limit = max(10, min(int(candidate_limit), 1000))
        if rec_btn:
            last_rec_key = f"last_recommendation_{'manga' if is_manga else 'anime'}_{username.lower()}"
            remaining = get_remaining_cooldown(last_rec_key, RECOMMENDATION_COOLDOWN_SECONDS)
            if remaining > 0:
                st.warning(f"Please wait {remaining}s before generating recommendations again.")
            else:
                with st.spinner(f"Downloading {candidate_limit} candidates and calculating personalized predictions..."):
                    st.session_state[last_rec_key] = time.time()
                    try:
                        if is_manga:
                            candidates = get_candidate_manga_for_recommendations(limit=candidate_limit, genre=sel_genre, tag=sel_tag)
                        else:
                            candidates = get_candidate_anime_for_recommendations(limit=candidate_limit, genre=sel_genre, tag=sel_tag, year=sel_year)
                        
                        if sel_studio:
                            filtered = []
                            for c in candidates:
                                found = False
                                if 'studios' in c and c['studios']:
                                    for e in c['studios'].get('edges', []):
                                        if sel_studio.lower() in e.get('node', {}).get('name', '').lower():
                                            found = True
                                if 'staff' in c and c['staff']:
                                    for e in c['staff'].get('edges', []):
                                        if sel_studio.lower() in e.get('node', {}).get('name', {}).get('full', '').lower():
                                            found = True
                                if found:
                                    filtered.append(c)
                            candidates = filtered
                        
                        cache_prefix = "user_manga_list_" if is_manga else "user_list_"
                        cache_path = CACHE_DIR / f"{cache_prefix}{username.lower()}.json"
                        watched_ids = set()
                        if cache_path.exists():
                            try:
                                with open(cache_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    for lst in data:
                                        for entry in lst.get('entries', []):
                                            m_id = entry.get('mediaId')
                                            if m_id and lst.get('status') != 'PLANNING':
                                                watched_ids.add(m_id)
                            except Exception:
                                logger.exception("Failed to read cached watch history.")
                        
                        valid_candidates = [c for c in candidates if c['id'] not in watched_ids]
                        results_list = []
                        model = model_artifact['model']
                        train_columns = model_artifact['train_columns']
                        
                        if 'sort_date' not in user_history_df.columns:
                            user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                        
                        prog = st.progress(0)
                        for idx, cand in enumerate(valid_candidates):
                            if idx % 10 == 0 and valid_candidates:
                                prog.progress(max(0.01, min((idx + 1) / len(valid_candidates), 1.0)))
                            try:
                                if is_manga:
                                    X_infer = build_manga_inference_features(cand, user_history_df, train_columns)
                                else:
                                    X_infer = build_inference_features(cand, user_history_df, train_columns)
                                pred = float(np.clip(model.predict(X_infer)[0], 0.0, 10.0))
                                results_list.append((pred, cand))
                            except Exception:
                                logger.debug("Skipping one candidate because inference failed.", exc_info=True)
                        prog.empty()
                        
                        results_list.sort(key=lambda x: x[0], reverse=True)
                        st.session_state[recommendations_key] = results_list[:10]
                    except Exception:
                        logger.exception("Failed to generate recommendations.")
                        st.error("Failed to generate recommendations. Please try again.")
            
        if recommendations_key in st.session_state and st.session_state[recommendations_key]:
            top_10 = st.session_state[recommendations_key]
            
            for i, (pred, cand) in enumerate(top_10):
                title = cand['title'].get('english') or cand['title'].get('romaji')
                col_img, col_txt = st.columns([1, 6])
                with col_img:
                    st.image(cand.get('coverImage', {}).get('large') or "", width="stretch")
                with col_txt:
                    st.markdown(f"#### #{i+1} : {title} ⭐️ {pred:.2f} / 10")
                    if is_manga:
                        sd = cand.get('startDate', {}) or {}
                        year = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                        st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Ch: {cand.get('chapters', 'N/A')} | Genres: {', '.join(cand.get('genres', []))}")
                        st.markdown(f"[➡️ Open on AniList](https://anilist.co/manga/{cand.get('id')})")
                    else:
                        st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Ep: {cand.get('episodes', 'N/A')} | Genres: {', '.join(cand.get('genres', []))}")
                        st.markdown(f"[➡️ Open on AniList](https://anilist.co/anime/{cand.get('id')})")
                st.write("---")
            
            # === EXPORT BUTTON ===
            export_data = []
            for pred, cand in top_10:
                title = cand['title'].get('english') or cand['title'].get('romaji')
                export_data.append({
                    'rank': len(export_data) + 1,
                    'title': title,
                    'predicted_score': round(pred, 2),
                    'global_score': (cand.get('averageScore') or 0) / 10.0,
                    'genres': ', '.join(cand.get('genres', [])),
                    'anilist_url': f"https://anilist.co/{'manga' if is_manga else 'anime'}/{cand.get('id')}"
                })
            
            export_df = pd.DataFrame(export_data)
            col_csv, col_json = st.columns(2)
            with col_csv:
                st.download_button(
                    "📥 Export CSV", export_df.to_csv(index=False),
                    f"recommendations_{media_label.lower()}_{username}.csv", "text/csv"
                )
            with col_json:
                st.download_button(
                    "📥 Export JSON", json.dumps(export_data, indent=2, ensure_ascii=False),
                    f"recommendations_{media_label.lower()}_{username}.json", "application/json"
                )

        # === PREDICT SINGLE ===
        st.markdown("---")
        st.subheader(f"🔍 Predict {media_label} Score (Exact Search)")
        media_query = st.text_input(
            f"Search for an {media_label.lower()}:",
            max_chars=MAX_SEARCH_LEN,
        )
        
        if media_query:
            with st.spinner("Searching..."):
                results = search_manga_by_title(media_query) if is_manga else search_anime_by_title(media_query)
            
            if not results:
                st.warning(f"No {media_label.lower()} found.")
            else:
                if is_manga:
                    options = {}
                    for r in results:
                        sd = r.get('startDate', {}) or {}
                        yr = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                        label = f"{r['title'].get('english') or r['title'].get('romaji')} ({yr}) - {r.get('format')}"
                        options[label] = r
                else:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')}) - {r.get('format')}": r for r in results}
                
                selected_option = st.selectbox(f"Select exact {media_label.lower()}:", list(options.keys()))
                
                if st.button("🎯 Calculate Predicted Score"):
                    top_media = options[selected_option]
                    title = top_media['title'].get('english') or top_media['title'].get('romaji')
                    
                    if 'sort_date' not in user_history_df.columns:
                        user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                    
                    if is_manga:
                        X_infer = build_manga_inference_features(top_media, user_history_df, model_artifact['train_columns'])
                    else:
                        X_infer = build_inference_features(top_media, user_history_df, model_artifact['train_columns'])
                    
                    model = model_artifact['model']
                    predicted_score = float(np.clip(model.predict(X_infer)[0], 0.0, 10.0))
                    
                    st.success(f"### Estimated Score for {title}: {predicted_score:.2f} / 10")
                    
                    col_img, col_txt = st.columns([1, 4])
                    with col_img:
                        st.image(top_media.get('coverImage', {}).get('large') or "", width="stretch")
                    with col_txt:
                        hist_mean = user_history_df['user_score'].mean()
                        glob_mean = (top_media.get('averageScore') or 0) / 10.0
                        st.write(f"**Your Average:** {hist_mean:.2f} | **Global Average:** {glob_mean:.2f}")
                        mt = "manga" if is_manga else "anime"
                        if check_planning_status(username, top_media['id'], mt):
                            st.warning("⚠️ It's already in your 'Planning' list on AniList!")
                    
                    # === SHAP EXPLANATION ===
                    shap_vals, expected = get_shap_explanation(model, X_infer)
                    if shap_vals is not None:
                        shap_df = shap_to_dataframe(shap_vals, model_artifact['train_columns'], top_n=12)
                        if shap_df is not None and not shap_df.empty:
                            st.markdown("#### 🧠 SHAP — Why this score?")
                            
                            colors = ['#00b09b' if v > 0 else '#eb3349' for v in shap_df['SHAP']]
                            fig_shap = go.Figure(go.Bar(
                                x=shap_df['SHAP'].values, y=shap_df['Feature'].values,
                                orientation='h', marker_color=colors
                            ))
                            fig_shap.update_layout(
                                title=f"Feature Impact on Score (base: {expected:.2f})" if not isinstance(expected, np.ndarray) else "Feature Impact on Score",
                                height=400, template='plotly_dark',
                                yaxis={'categoryorder': 'total ascending'},
                                xaxis_title="SHAP Impact (+ = raises score, - = lowers score)"
                            )
                            st.plotly_chart(fig_shap, width="stretch")
                    
    else:
        st.info("👈 Enter a username and train models from the sidebar to start!")

# ==========================================
# COLD START MODE
# ==========================================
elif mode == "New User (Cold Start)":
    media_label = "Manga" if is_manga else "Anime"
    icon = "📖" if is_manga else "🛸"
    
    st.header(f"{icon} Onboarding: Find your perfect {media_label}!")
    st.write(f"Answer a few questions to create a temporary profile and get tailored {media_label.lower()} recommendations.")
    
    fav_key = "cs_favorites_manga" if is_manga else "cs_favorites_anime"
    profile_key = "cs_profile_manga" if is_manga else "cs_profile_anime"
    step_key = "cs_step_manga" if is_manga else "cs_step_anime"
    rec_key = "cs_recommendations_manga" if is_manga else "cs_recommendations"
    
    if st.session_state[profile_key] is None:
        if st.session_state[step_key] == 1:
            st.subheader(f"Step 1: Name 3-5 {media_label.lower()} that 'captivated' you")
            search_q = st.text_input(f"Search {media_label.lower()}:", max_chars=MAX_SEARCH_LEN)
            if search_q:
                res = search_manga_by_title(search_q) if is_manga else search_anime_by_title(search_q)
                if res:
                    opts = {f"{r['title'].get('english') or r['title'].get('romaji')}": r for r in res}
                    sel = st.selectbox("Results:", list(opts.keys()))
                    if st.button("➕ Add to Favorites"):
                        media_obj = opts[sel]
                        if not any(a['id'] == media_obj['id'] for a in st.session_state[fav_key]):
                            st.session_state[fav_key].append(media_obj)
                            st.success(f"{sel} added!")
                        else:
                            st.warning("Already inserted!")
            
            if st.session_state[fav_key]:
                st.markdown("---")
                st.write("**Your Favorites:**")
                for fa in st.session_state[fav_key]:
                    st.write(f"- ⭐️ {fa['title'].get('english') or fa['title'].get('romaji')}")
                
                if len(st.session_state[fav_key]) >= 3:
                    if st.button("Proceed to Final Step ➡️"):
                        st.session_state[step_key] = 2
                        st.rerun()
                else:
                    st.info(f"Missing {3 - len(st.session_state[fav_key])} {media_label.lower()}.")
        
        elif st.session_state[step_key] == 2:
            st.subheader("Step 2: Further Preferences")
            with st.form("cold_start_form"):
                fav_g = st.multiselect("Favorite Genres:", GENRES_LIST)
                avoid_g = st.multiselect("Genres to Avoid:", GENRES_LIST)
                col1, col2, col3 = st.columns(3)
                if is_manga:
                    with col1: pref_fmt = st.selectbox("Format:", MANGA_FORMATS_LIST, index=len(MANGA_FORMATS_LIST)-1)
                    with col2: pref_len = st.selectbox("Length (Chap):", MANGA_LENGTHS_LIST, index=len(MANGA_LENGTHS_LIST)-1)
                    with col3: pref_era = st.selectbox("Era:", ERAS_LIST, index=len(ERAS_LIST)-1)
                else:
                    with col1: pref_fmt = st.selectbox("Format:", FORMATS_LIST, index=len(FORMATS_LIST)-1)
                    with col2: pref_len = st.selectbox("Length:", LENGTHS_LIST, index=len(LENGTHS_LIST)-1)
                    with col3: pref_era = st.selectbox("Era:", ERAS_LIST, index=len(ERAS_LIST)-1)
                
                if st.form_submit_button(f"🚀 Generate {media_label} Profile"):
                    if is_manga:
                        profile = build_manga_cold_start_profile(st.session_state[fav_key], fav_g, avoid_g, pref_fmt, pref_len, pref_era)
                    else:
                        profile = build_cold_start_profile(st.session_state[fav_key], fav_g, avoid_g, pref_fmt, pref_len, pref_era)
                    st.session_state[profile_key] = profile
                    st.rerun()
    else:
        st.success("Starting profile loaded!")
        if st.button("🔁 Restart Onboarding"):
            st.session_state[profile_key] = None
            st.session_state[fav_key] = []
            st.session_state[step_key] = 1
            if rec_key in st.session_state: del st.session_state[rec_key]
            st.rerun()
        
        profile = st.session_state[profile_key]
        st.markdown("---")
        st.header(f"🎯 Top 10 Recommended {media_label}")
        
        cs_limit_col, cs_btn_col = st.columns([3, 1])
        with cs_limit_col:
            candidate_limit_cs = st.slider(
                "Number of candidates to analyze (Popular and Top Rated):", 
                min_value=10, 
                max_value=1000, 
                value=500, 
                step=10,
                help="The more candidates you select, the more accurate the search, but it will take longer."
            )
        with cs_btn_col:
            if st.button("🔄 Generate / Update", width="stretch"):
                if rec_key in st.session_state:
                    del st.session_state[rec_key]
                st.rerun()
        
        if rec_key not in st.session_state:
            with st.spinner(f"Calculating {media_label.lower()} recommendations from {candidate_limit_cs} candidates..."):
                candidate_limit_cs = max(10, min(int(candidate_limit_cs), 1000))
                cands = get_candidate_manga_for_recommendations(limit=candidate_limit_cs) if is_manga else get_candidate_anime_for_recommendations(limit=candidate_limit_cs)
                fav_ids = {a['id'] for a in st.session_state[fav_key]}
                cands = [c for c in cands if c['id'] not in fav_ids]
                recs = generate_manga_cold_start_recommendations(profile, cands) if is_manga else generate_cold_start_recommendations(profile, cands)
                st.session_state[rec_key] = recs
        
        for i, (pred, cand) in enumerate(st.session_state[rec_key]):
            title = cand['title'].get('english') or cand['title'].get('romaji')
            col_img, col_txt = st.columns([1, 6])
            with col_img:
                st.image(cand.get('coverImage', {}).get('large') or "", width="stretch")
            with col_txt:
                st.markdown(f"#### #{i+1} : {title}")
                st.markdown(f"**Estimated Affinity**: 🚀 `{pred:.2f} / 10`")
                if is_manga:
                    sd = cand.get('startDate', {}) or {}
                    year = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                    st.write(f"Year: {year} | Ch: {cand.get('chapters', 'N/A')} | Genres: {', '.join(cand.get('genres', []))}")
                else:
                    st.write(f"Year: {cand.get('seasonYear')} | Ep: {cand.get('episodes')} | Genres: {', '.join(cand.get('genres', []))}")
            st.write("---")
        
        # Export cold start recommendations
        if st.session_state.get(rec_key):
            export_cs = [{'rank': i+1, 'title': (c['title'].get('english') or c['title'].get('romaji')), 'score': round(s, 2)} for i, (s, c) in enumerate(st.session_state[rec_key])]
            st.download_button("📥 Export", pd.DataFrame(export_cs).to_csv(index=False), f"cold_start_{media_label.lower()}.csv", "text/csv")
        
        # Single prediction
        st.markdown("---")
        st.subheader(f"🔮 Calculate Single Affinity ({media_label})")
        media_query = st.text_input(
            f"What {media_label.lower()} are you thinking of?",
            max_chars=MAX_SEARCH_LEN,
        )
        
        if media_query:
            with st.spinner("Searching..."):
                results = search_manga_by_title(media_query) if is_manga else search_anime_by_title(media_query)
            if not results:
                st.warning("Not found.")
            else:
                if is_manga:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({(r.get('startDate') or {}).get('year', 'N/A')})": r for r in results}
                else:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')})": r for r in results}
                
                selected_option = st.selectbox("Select:", list(options.keys()), key="cs_sel")
                if st.button("Tell me the Affinity"):
                    top_media = options[selected_option]
                    if is_manga:
                        score, explanations = content_based_heuristic_scorer_manga(top_media, profile)
                    else:
                        score, explanations = content_based_heuristic_scorer(top_media, profile)
                    
                    st.success(f"### Estimated Score: {score:.2f} / 10")
                    colImg, colTxt = st.columns([1, 4])
                    with colImg:
                        st.image(top_media.get('coverImage', {}).get('large') or "", width="stretch")
                    with colTxt:
                        st.subheader("💡 Why?")
                        for expl in explanations:
                            st.write(expl)

render_app_disclaimer()
