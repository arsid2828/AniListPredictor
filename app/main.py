import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to sys path to allow importing src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api import search_anime_by_title, get_candidate_anime_for_recommendations
from src.dataset import build_user_dataframe
from src.features import build_inference_features
from src.models import train_and_evaluate_all_models
from src.models import MODELS_DIR
from src.dataset import DATA_DIR
from src.cold_start import (
    build_cold_start_profile, 
    content_based_heuristic_scorer, 
    generate_cold_start_recommendations,
    GENRES_LIST, FORMATS_LIST, LENGTHS_LIST, ERAS_LIST
)
import joblib

import logging
logger = logging.getLogger(__name__)

def check_planning_status(username, media_id):
    cache_path = Path(f"c:/Users/arsid/Desktop/AnilistProject/cache/user_list_{username.lower()}.json")
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
    if "username" not in st.session_state: st.session_state["username"] = ""
    if "model_trained" not in st.session_state: st.session_state["model_trained"] = False
    if "cs_favorites" not in st.session_state: st.session_state["cs_favorites"] = []
    if "cs_profile" not in st.session_state: st.session_state["cs_profile"] = None
    if "cs_step" not in st.session_state: st.session_state["cs_step"] = 1
    if "recommendation_feedback" not in st.session_state: st.session_state["recommendation_feedback"] = {}

st.set_page_config(page_title="AniList ML Predictor", layout="wide", page_icon="🎬")
init_session_state()

st.title("🎬 AniList User Score Predictor")
st.markdown("Scopri quali anime adorerai, basandoti sul tuo storico o su un rapido onboarding!")

mode = st.sidebar.radio("Scegli Modalità:", ["Profilo AniList (Machine Learning)", "Nuovo Utente (Cold Start)"])

if mode == "Profilo AniList (Machine Learning)":
    # Sidebar for Setup & Training
    with st.sidebar:
        st.header("1. Setup User Profile")
        username_input = st.text_input("Enter AniList Username:", value=st.session_state.get("username", "arsid"))
        if st.button("Fetch Data & Train Models"):
            with st.spinner(f"Fetching data and training models for {username_input}... This might take a minute."):
                st.session_state["username"] = username_input
                result = train_and_evaluate_all_models(username_input)
                if isinstance(result, dict) and result.get("status") == "error":
                    st.error(result["message"])
                    st.session_state["model_trained"] = False
                else:
                    st.success("Models trained successfully!")
                    st.session_state["model_trained"] = True

    # Main block for prediction
    if st.session_state["model_trained"]:
        st.header(f"2. Dashboard per {st.session_state['username']}")
        
        username = st.session_state["username"]
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
        st.subheader("🌟 Top 10 Raccomandazioni Personali")
        if st.button("Genera Raccomandazioni tramite Modello ML", icon="✨"):
            with st.spinner("Scarico i candidati ideali..."):
                candidates = get_candidate_anime_for_recommendations(limit=500)
                
            with st.spinner("Inferendo i punteggi personalizzati..."):
                cache_path = Path(f"c:/Users/arsid/Desktop/AnilistProject/cache/user_list_{username.lower()}.json")
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
                    st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Ep: {cand.get('episodes', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                    st.markdown(f"[➡️ Apri su AniList](https://anilist.co/anime/{cand.get('id')})")
                st.write("---")

        # === Predict Score For Specific Anime ===
        st.subheader("🔍 Prevedi Voto (Ricerca Esatta)")
        anime_query = st.text_input("Cerca un Anime (es. 'Attack on Titan'):")
        
        if anime_query:
            with st.spinner("Ricerca in corso..."):
                results = search_anime_by_title(anime_query)
                
            if not results:
                st.warning("Nessun anime trovato.")
            else:
                options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')}) - Formato: {r.get('format')}": r for r in results}
                selected_option = st.selectbox("Seleziona l'anime esatto:", list(options.keys()))
                
                if st.button("Calcola Score Predetto"):
                    top_anime = options[selected_option]
                    title = top_anime['title'].get('english') or top_anime['title'].get('romaji')
                    
                    if 'sort_date' not in user_history_df.columns:
                        user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                        
                    X_infer = build_inference_features(top_anime, user_history_df, model_artifact['train_columns'])
                    model = model_artifact['model']
                    predicted_score = float(np.clip(model.predict(X_infer)[0], 0.0, 10.0))
                    
                    st.success(f"### Voto Stimato per {title}: {predicted_score:.2f} / 10")
                    
                    col_img, col_txt = st.columns([1, 4])
                    with col_img:
                        st.image(top_anime.get('coverImage', {}).get('large') or "", use_container_width=True)
                    with col_txt:
                        hist_mean = user_history_df['user_score'].mean()
                        glob_mean = (top_anime.get('averageScore') or 0) / 10.0
                        st.write(f"**Tua Media:** {hist_mean:.2f} | **Media Globale:** {glob_mean:.2f}")
                        if check_planning_status(username, top_anime['id']):
                            st.warning("E' già in 'Plan to Watch' su AniList!")
                            
    else:
        st.info("Inserisci uno username valido a sinistra e traina i modelli per proseguire!")

elif mode == "Nuovo Utente (Cold Start)":
    
    st.header("🛸 Onboarding: Trova il tuo Anime perfetto!")
    st.write("Rispondi a poche domande per creare un profilo temporaneo e ottenere raccomandazioni su misura.")
    
    if st.session_state["cs_profile"] is None:
        
        if st.session_state["cs_step"] == 1:
            st.subheader("Step 1: Dimmi 3-5 anime che ti hanno 'Stregato'")
            st.write("Cerca gli anime e clicca su Aggiungi.")
            search_q = st.text_input("Cerca anime:")
            if search_q:
                res = search_anime_by_title(search_q)
                if res:
                    opts = {f"{r['title'].get('english') or r['title'].get('romaji')}": r for r in res}
                    sel = st.selectbox("Risultati:", list(opts.keys()))
                    if st.button("➕ Aggiungi a Preferiti"):
                        anime_obj = opts[sel]
                        if not any(a['id'] == anime_obj['id'] for a in st.session_state["cs_favorites"]):
                            st.session_state["cs_favorites"].append(anime_obj)
                            st.success(f"{sel} aggiunto!")
                        else:
                            st.warning("Hai già inserito questo anime!")
                            
            if st.session_state["cs_favorites"]:
                st.markdown("---")
                st.write("**I tuoi Preferiti Finora:**")
                for fa in st.session_state["cs_favorites"]:
                    st.write(f"- ⭐️ {fa['title'].get('english') or fa['title'].get('romaji')}")
                    
                if len(st.session_state["cs_favorites"]) >= 3:
                    if st.button("Prosegui allo Step Finale ➡️"):
                        st.session_state["cs_step"] = 2
                        st.rerun()
                else:
                    st.info(f"Mancano {3 - len(st.session_state['cs_favorites'])} anime per sbloccare lo step successivo.")
                        
        elif st.session_state["cs_step"] == 2:
            st.subheader("Step 2: Ulteriori Preferenze")
            
            with st.form("cold_start_form"):
                fav_g = st.multiselect("Generi Preferiti:", GENRES_LIST)
                avoid_g = st.multiselect("Generi da Evitare ASSOLUTAMENTE:", GENRES_LIST)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    pref_fmt = st.selectbox("Formato Preferito:", FORMATS_LIST, index=len(FORMATS_LIST)-1)
                with col2:
                    pref_len = st.selectbox("Lunghezza Ideale:", LENGTHS_LIST, index=len(LENGTHS_LIST)-1)
                with col3:
                    pref_era = st.selectbox("Epoca Favorita:", ERAS_LIST, index=len(ERAS_LIST)-1)
                    
                submit = st.form_submit_button("🚀 Genera Profilo A.I. Magico")
                
                if submit:
                    profile = build_cold_start_profile(
                        st.session_state["cs_favorites"], fav_g, avoid_g, pref_fmt, pref_len, pref_era
                    )
                    st.session_state["cs_profile"] = profile
                    st.success("Profilo Generato! Preparati per le raccomandazioni...")
                    st.rerun()
                    
    else:
        # We have a CS Profile!
        st.success("Profilo di partenza caricato correttamente!")
        if st.button("🔁 Rifai Onboarding"):
            st.session_state["cs_profile"] = None
            st.session_state["cs_favorites"] = []
            st.session_state["cs_step"] = 1
            st.rerun()
            
        profile = st.session_state["cs_profile"]
        
        # Recommendations
        st.markdown("---")
        st.header("🎯 Top 10 Consigliati per Te")
        
        if "cs_recommendations" not in st.session_state:
            with st.spinner("Calcolo le raccomandazioni logiche basate sui tuoi input..."):
                cands = get_candidate_anime_for_recommendations(limit=500)
                # Togli quelli gia preferiti originariamente
                fav_ids = {a['id'] for a in st.session_state["cs_favorites"]}
                cands = [c for c in cands if c['id'] not in fav_ids]
                
                recs = generate_cold_start_recommendations(profile, cands)
                st.session_state["cs_recommendations"] = recs
                
        for i, (pred, cand) in enumerate(st.session_state["cs_recommendations"]):
            title = cand['title'].get('english') or cand['title'].get('romaji')
            col_img, col_txt = st.columns([1, 6])
            with col_img:
                st.image(cand.get('coverImage', {}).get('large') or "", use_container_width=True)
            with col_txt:
                st.markdown(f"#### #{i+1} : {title}")
                st.markdown(f"**Affinità Stimata**: 🚀 `{pred:.2f} / 10`")
                st.write(f"Anno: {cand.get('seasonYear')} | Ep: {cand.get('episodes')} | Generi: {', '.join(cand.get('genres', []))}")
                
            st.write("---")
            
        # Prediction
        st.markdown("---")
        st.subheader("🔮 Calcola Affinità Singola")
        anime_query = st.text_input("Quale anime hai in mente? Inseriscilo qui:")
        
        if anime_query:
            with st.spinner("Ricerca..."):
                results = search_anime_by_title(anime_query)
            if not results:
                st.warning("Non trovato.")
            else:
                options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')})": r for r in results}
                selected_option = st.selectbox("Seleziona la tua scelta:", list(options.keys()), key="cs_sel")
                
                if st.button("Dimmi l'Affinità"):
                    top_anime = options[selected_option]
                    score, explanations = content_based_heuristic_scorer(top_anime, profile)
                    
                    st.success(f"### Score Stimato: {score:.2f} / 10")
                    colImg, colTxt = st.columns([1, 4])
                    with colImg:
                         st.image(top_anime.get('coverImage', {}).get('large') or "", use_container_width=True)
                    with colTxt:
                        st.subheader("💡 Perché?")
                        for expl in explanations:
                            st.write(expl)
