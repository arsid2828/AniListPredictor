import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to sys path to allow importing src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api import search_anime_by_title, get_candidate_anime_for_recommendations, search_manga_by_title, get_candidate_manga_for_recommendations
from src.dataset import build_user_dataframe, build_user_manga_dataframe
from src.features import build_inference_features, build_manga_inference_features
from src.models import train_and_evaluate_all_models, train_and_evaluate_all_manga_models
from src.models import MODELS_DIR
from src.dataset import DATA_DIR
from src.cold_start import (
    build_cold_start_profile, 
    content_based_heuristic_scorer, 
    generate_cold_start_recommendations,
    GENRES_LIST, FORMATS_LIST, LENGTHS_LIST, ERAS_LIST,
    build_manga_cold_start_profile,
    content_based_heuristic_scorer_manga,
    generate_manga_cold_start_recommendations,
    MANGA_FORMATS_LIST, MANGA_LENGTHS_LIST
)
import joblib

import logging
logger = logging.getLogger(__name__)

def check_planning_status(username, media_id, media_type="anime"):
    prefix = "user_list_" if media_type == "anime" else "user_manga_list_"
    cache_path = Path(f"c:/Users/arsid/Desktop/AnilistProject/cache/{prefix}{username.lower()}.json")
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

def init_session_state():
    defaults = {
        "username": "",
        # Anime
        "model_trained_anime": False,
        "cs_favorites_anime": [],
        "cs_profile_anime": None,
        "cs_step_anime": 1,
        # Manga
        "model_trained_manga": False,
        "cs_favorites_manga": [],
        "cs_profile_manga": None,
        "cs_step_manga": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(page_title="AniList ML Predictor", layout="wide", page_icon="🎬")
init_session_state()

st.title("🎬 AniList Score Predictor")
st.markdown("Scopri quali anime e manga adorerai, basandoti sul tuo storico o su un rapido onboarding!")

# ========== TOP-LEVEL MEDIA TOGGLE ==========
media_type = st.sidebar.radio("📌 Tipo di Media:", ["🎬 Anime", "📖 Manga"])
is_manga = "Manga" in media_type

mode = st.sidebar.radio("Scegli Modalità:", ["Profilo AniList (Machine Learning)", "Nuovo Utente (Cold Start)"])

# ==========================================
# PROFILO ANILIST (ML) MODE
# ==========================================
if mode == "Profilo AniList (Machine Learning)":
    with st.sidebar:
        st.header("1. Setup User Profile")
        username_input = st.text_input("Enter AniList Username:", value=st.session_state.get("username", "arsid"))
        
        model_trained_key = "model_trained_manga" if is_manga else "model_trained_anime"
        btn_label = "Fetch Manga & Train" if is_manga else "Fetch Anime & Train"
        
        if st.button(btn_label):
            media_label = "manga" if is_manga else "anime"
            with st.spinner(f"Fetching {media_label} data and training models for {username_input}..."):
                st.session_state["username"] = username_input
                if is_manga:
                    result = train_and_evaluate_all_manga_models(username_input)
                else:
                    result = train_and_evaluate_all_models(username_input)
                    
                if isinstance(result, dict) and result.get("status") == "error":
                    st.error(result["message"])
                    st.session_state[model_trained_key] = False
                else:
                    st.success("Models trained successfully!")
                    st.session_state[model_trained_key] = True

    model_trained_key = "model_trained_manga" if is_manga else "model_trained_anime"
    
    if st.session_state[model_trained_key]:
        media_label = "Manga" if is_manga else "Anime"
        st.header(f"2. Dashboard {media_label} per {st.session_state['username']}")
        
        username = st.session_state["username"]
        
        if is_manga:
            model_path = MODELS_DIR / f"{username}_manga_best_model.pkl"
            csv_path = DATA_DIR / f"{username}_manga_clean.csv"
        else:
            model_path = MODELS_DIR / f"{username}_best_model.pkl"
            csv_path = DATA_DIR / f"{username}_clean.csv"
        
        try:
            model_artifact = joblib.load(model_path)
            user_history_df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to load cached models or dataset: {e}. Please retrain.")
            st.stop()
            
        best_model_name = model_artifact['model_name']
        cv_metrics = pd.DataFrame(model_artifact['metrics'])
        
        with st.expander("Visualizza Dettagli Modello"):
            st.dataframe(cv_metrics)
            st.write(f"**Modello Selezionato:** {best_model_name}")
            if model_artifact.get('feature_importance'):
                st.write("**Top Feature Importances:**")
                st.bar_chart(pd.DataFrame(model_artifact['feature_importance']).set_index('Feature'))

        # === 10 Recommendations ===
        st.markdown("---")
        st.subheader(f"🌟 Top 10 Raccomandazioni {media_label} Personali")
        rec_btn_label = f"Genera Raccomandazioni {media_label} tramite ML"
        if st.button(rec_btn_label, icon="✨"):
            with st.spinner("Scarico i candidati ideali..."):
                if is_manga:
                    candidates = get_candidate_manga_for_recommendations(limit=500)
                else:
                    candidates = get_candidate_anime_for_recommendations(limit=500)
                
            with st.spinner("Inferendo i punteggi personalizzati..."):
                cache_prefix = "user_manga_list_" if is_manga else "user_list_"
                cache_path = Path(f"c:/Users/arsid/Desktop/AnilistProject/cache/{cache_prefix}{username.lower()}.json")
                watched_ids = set()
                if cache_path.exists():
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for lst in data:
                                status = lst.get('status')
                                for entry in lst.get('entries', []):
                                    m_id = entry.get('mediaId')
                                    if m_id and status != 'PLANNING':
                                        watched_ids.add(m_id)
                    except: pass
                
                valid_candidates = [c for c in candidates if c['id'] not in watched_ids]
                results_list = []
                model = model_artifact['model']
                train_columns = model_artifact['train_columns']
                
                if 'sort_date' not in user_history_df.columns:
                    user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                    
                total_cands = len(valid_candidates)
                prog = st.progress(0)
                for idx, cand in enumerate(valid_candidates):
                    if idx % 10 == 0: prog.progress(max(0.01, min((idx + 1) / total_cands, 1.0)))
                    try:
                        if is_manga:
                            X_infer = build_manga_inference_features(cand, user_history_df, train_columns)
                        else:
                            X_infer = build_inference_features(cand, user_history_df, train_columns)
                        pred = float(np.clip(model.predict(X_infer)[0], 0.0, 10.0))
                        results_list.append((pred, cand))
                    except: pass
                prog.empty()
                
                results_list.sort(key=lambda x: x[0], reverse=True)
                top_10 = results_list[:10]
                
            for i, (pred, cand) in enumerate(top_10):
                title = cand['title'].get('english') or cand['title'].get('romaji')
                col_img, col_txt = st.columns([1, 6])
                with col_img:
                    st.image(cand.get('coverImage', {}).get('large') or "", use_container_width=True)
                with col_txt:
                    st.markdown(f"#### #{i+1} : {title} ⭐️ {pred:.2f} / 10")
                    if is_manga:
                        start_date = cand.get('startDate', {}) or {}
                        year = start_date.get('year', 'N/A') if isinstance(start_date, dict) else 'N/A'
                        st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Cap: {cand.get('chapters', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                        st.markdown(f"[➡️ Apri su AniList](https://anilist.co/manga/{cand.get('id')})")
                    else:
                        st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Ep: {cand.get('episodes', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                        st.markdown(f"[➡️ Apri su AniList](https://anilist.co/anime/{cand.get('id')})")
                st.write("---")

        # === Predict Score For Specific Media ===
        st.subheader(f"🔍 Prevedi Voto {media_label} (Ricerca Esatta)")
        query_label = "Cerca un Manga:" if is_manga else "Cerca un Anime (es. 'Attack on Titan'):"
        media_query = st.text_input(query_label)
        
        if media_query:
            with st.spinner("Ricerca in corso..."):
                if is_manga:
                    results = search_manga_by_title(media_query)
                else:
                    results = search_anime_by_title(media_query)
                
            if not results:
                st.warning(f"Nessun {media_label.lower()} trovato.")
            else:
                if is_manga:
                    start_dates = {}
                    options = {}
                    for r in results:
                        sd = r.get('startDate', {}) or {}
                        yr = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                        label = f"{r['title'].get('english') or r['title'].get('romaji')} ({yr}) - Formato: {r.get('format')}"
                        options[label] = r
                else:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')}) - Formato: {r.get('format')}": r for r in results}
                    
                selected_option = st.selectbox(f"Seleziona il {media_label.lower()} esatto:", list(options.keys()))
                
                if st.button("Calcola Score Predetto"):
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
                    
                    st.success(f"### Voto Stimato per {title}: {predicted_score:.2f} / 10")
                    
                    col_img, col_txt = st.columns([1, 4])
                    with col_img:
                        st.image(top_media.get('coverImage', {}).get('large') or "", use_container_width=True)
                    with col_txt:
                        hist_mean = user_history_df['user_score'].mean()
                        glob_mean = (top_media.get('averageScore') or 0) / 10.0
                        st.write(f"**Tua Media:** {hist_mean:.2f} | **Media Globale:** {glob_mean:.2f}")
                        mt = "manga" if is_manga else "anime"
                        if check_planning_status(username, top_media['id'], mt):
                            st.warning("E' già in 'Plan to Read/Watch' su AniList!")
                            
    else:
        st.info("Inserisci uno username valido a sinistra e traina i modelli per proseguire!")

# ==========================================
# COLD START MODE
# ==========================================
elif mode == "Nuovo Utente (Cold Start)":
    
    media_label = "Manga" if is_manga else "Anime"
    icon = "📖" if is_manga else "🛸"
    
    st.header(f"{icon} Onboarding: Trova il tuo {media_label} perfetto!")
    st.write(f"Rispondi a poche domande per creare un profilo temporaneo e ottenere raccomandazioni {media_label.lower()} su misura.")
    
    # Session state keys
    fav_key = "cs_favorites_manga" if is_manga else "cs_favorites_anime"
    profile_key = "cs_profile_manga" if is_manga else "cs_profile_anime"
    step_key = "cs_step_manga" if is_manga else "cs_step_anime"
    rec_key = "cs_recommendations_manga" if is_manga else "cs_recommendations"
    
    if st.session_state[profile_key] is None:
        
        if st.session_state[step_key] == 1:
            st.subheader(f"Step 1: Dimmi 3-5 {media_label.lower()} che ti hanno 'Stregato'")
            st.write(f"Cerca {media_label.lower()} e clicca su Aggiungi.")
            search_q = st.text_input(f"Cerca {media_label.lower()}:")
            if search_q:
                if is_manga:
                    res = search_manga_by_title(search_q)
                else:
                    res = search_anime_by_title(search_q)
                if res:
                    opts = {f"{r['title'].get('english') or r['title'].get('romaji')}": r for r in res}
                    sel = st.selectbox("Risultati:", list(opts.keys()))
                    if st.button("➕ Aggiungi a Preferiti"):
                        media_obj = opts[sel]
                        if not any(a['id'] == media_obj['id'] for a in st.session_state[fav_key]):
                            st.session_state[fav_key].append(media_obj)
                            st.success(f"{sel} aggiunto!")
                        else:
                            st.warning(f"Hai già inserito questo {media_label.lower()}!")
                            
            if st.session_state[fav_key]:
                st.markdown("---")
                st.write("**I tuoi Preferiti Finora:**")
                for fa in st.session_state[fav_key]:
                    st.write(f"- ⭐️ {fa['title'].get('english') or fa['title'].get('romaji')}")
                    
                if len(st.session_state[fav_key]) >= 3:
                    if st.button("Prosegui allo Step Finale ➡️"):
                        st.session_state[step_key] = 2
                        st.rerun()
                else:
                    st.info(f"Mancano {3 - len(st.session_state[fav_key])} {media_label.lower()} per sbloccare lo step successivo.")
                        
        elif st.session_state[step_key] == 2:
            st.subheader("Step 2: Ulteriori Preferenze")
            
            with st.form("cold_start_form"):
                fav_g = st.multiselect("Generi Preferiti:", GENRES_LIST)
                avoid_g = st.multiselect("Generi da Evitare ASSOLUTAMENTE:", GENRES_LIST)
                
                col1, col2, col3 = st.columns(3)
                
                if is_manga:
                    with col1:
                        pref_fmt = st.selectbox("Formato Preferito:", MANGA_FORMATS_LIST, index=len(MANGA_FORMATS_LIST)-1)
                    with col2:
                        pref_len = st.selectbox("Lunghezza Ideale (Capitoli):", MANGA_LENGTHS_LIST, index=len(MANGA_LENGTHS_LIST)-1)
                    with col3:
                        pref_era = st.selectbox("Epoca Favorita:", ERAS_LIST, index=len(ERAS_LIST)-1)
                else:
                    with col1:
                        pref_fmt = st.selectbox("Formato Preferito:", FORMATS_LIST, index=len(FORMATS_LIST)-1)
                    with col2:
                        pref_len = st.selectbox("Lunghezza Ideale:", LENGTHS_LIST, index=len(LENGTHS_LIST)-1)
                    with col3:
                        pref_era = st.selectbox("Epoca Favorita:", ERAS_LIST, index=len(ERAS_LIST)-1)
                    
                submit = st.form_submit_button(f"🚀 Genera Profilo {media_label} A.I. Magico")
                
                if submit:
                    if is_manga:
                        profile = build_manga_cold_start_profile(
                            st.session_state[fav_key], fav_g, avoid_g, pref_fmt, pref_len, pref_era
                        )
                    else:
                        profile = build_cold_start_profile(
                            st.session_state[fav_key], fav_g, avoid_g, pref_fmt, pref_len, pref_era
                        )
                    st.session_state[profile_key] = profile
                    st.success("Profilo Generato! Preparati per le raccomandazioni...")
                    st.rerun()
                    
    else:
        # We have a CS Profile!
        st.success("Profilo di partenza caricato correttamente!")
        if st.button("🔁 Rifai Onboarding"):
            st.session_state[profile_key] = None
            st.session_state[fav_key] = []
            st.session_state[step_key] = 1
            if rec_key in st.session_state:
                del st.session_state[rec_key]
            st.rerun()
            
        profile = st.session_state[profile_key]
        
        # Recommendations
        st.markdown("---")
        st.header(f"🎯 Top 10 {media_label} Consigliati per Te")
        
        if rec_key not in st.session_state:
            with st.spinner(f"Calcolo le raccomandazioni {media_label.lower()} basate sui tuoi input..."):
                if is_manga:
                    cands = get_candidate_manga_for_recommendations(limit=500)
                else:
                    cands = get_candidate_anime_for_recommendations(limit=500)
                    
                fav_ids = {a['id'] for a in st.session_state[fav_key]}
                cands = [c for c in cands if c['id'] not in fav_ids]
                
                if is_manga:
                    recs = generate_manga_cold_start_recommendations(profile, cands)
                else:
                    recs = generate_cold_start_recommendations(profile, cands)
                st.session_state[rec_key] = recs
                
        for i, (pred, cand) in enumerate(st.session_state[rec_key]):
            title = cand['title'].get('english') or cand['title'].get('romaji')
            col_img, col_txt = st.columns([1, 6])
            with col_img:
                st.image(cand.get('coverImage', {}).get('large') or "", use_container_width=True)
            with col_txt:
                st.markdown(f"#### #{i+1} : {title}")
                st.markdown(f"**Affinità Stimata**: 🚀 `{pred:.2f} / 10`")
                if is_manga:
                    start_date = cand.get('startDate', {}) or {}
                    year = start_date.get('year', 'N/A') if isinstance(start_date, dict) else 'N/A'
                    st.write(f"Anno: {year} | Cap: {cand.get('chapters', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                else:
                    st.write(f"Anno: {cand.get('seasonYear')} | Ep: {cand.get('episodes')} | Generi: {', '.join(cand.get('genres', []))}")
                
            st.write("---")
            
        # Prediction
        st.markdown("---")
        st.subheader(f"🔮 Calcola Affinità Singola ({media_label})")
        query_label = f"Quale {media_label.lower()} hai in mente? Inseriscilo qui:"
        media_query = st.text_input(query_label)
        
        if media_query:
            with st.spinner("Ricerca..."):
                if is_manga:
                    results = search_manga_by_title(media_query)
                else:
                    results = search_anime_by_title(media_query)
            if not results:
                st.warning("Non trovato.")
            else:
                if is_manga:
                    options = {}
                    for r in results:
                        sd = r.get('startDate', {}) or {}
                        yr = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                        label = f"{r['title'].get('english') or r['title'].get('romaji')} ({yr})"
                        options[label] = r
                else:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')})": r for r in results}
                    
                selected_option = st.selectbox("Seleziona la tua scelta:", list(options.keys()), key="cs_sel")
                
                if st.button("Dimmi l'Affinità"):
                    top_media = options[selected_option]
                    if is_manga:
                        score, explanations = content_based_heuristic_scorer_manga(top_media, profile)
                    else:
                        score, explanations = content_based_heuristic_scorer(top_media, profile)
                    
                    st.success(f"### Score Stimato: {score:.2f} / 10")
                    colImg, colTxt = st.columns([1, 4])
                    with colImg:
                         st.image(top_media.get('coverImage', {}).get('large') or "", use_container_width=True)
                    with colTxt:
                        st.subheader("💡 Perché?")
                        for expl in explanations:
                            st.write(expl)
