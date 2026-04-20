import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import io
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api import search_anime_by_title, get_candidate_anime_for_recommendations, search_manga_by_title, get_candidate_manga_for_recommendations
from src.dataset import build_user_dataframe, build_user_manga_dataframe
from src.features import build_inference_features, build_manga_inference_features
from src.models import train_and_evaluate_all_models, train_and_evaluate_all_manga_models, MODELS_DIR
from src.dataset import DATA_DIR
from src.analytics import get_shap_explanation, shap_to_dataframe
from src.cold_start import (
    build_cold_start_profile, content_based_heuristic_scorer, generate_cold_start_recommendations,
    GENRES_LIST, FORMATS_LIST, LENGTHS_LIST, ERAS_LIST,
    build_manga_cold_start_profile, content_based_heuristic_scorer_manga,
    generate_manga_cold_start_recommendations, MANGA_FORMATS_LIST, MANGA_LENGTHS_LIST
)
from app.shared import inject_css, init_session_state, check_planning_status, check_auto_train_needed, render_metric_card, render_user_badge, get_cached_profiles
import joblib
import plotly.express as px
import plotly.graph_objects as go
import logging

from src.api import CACHE_DIR
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AniList AI Predictor", layout="wide", page_icon="🎬")
inject_css()
init_session_state()

st.title("🎬 AniList Score Predictor")
st.markdown("Scopri quali anime e manga adorerai, basandoti sul tuo storico o su un rapido onboarding!")

# ========== SIDEBAR ==========
render_user_badge()
media_type = st.sidebar.radio("📌 Tipo di Media:", ["🎬 Anime", "📖 Manga"])
is_manga = "Manga" in media_type
mode = st.sidebar.radio("Scegli Modalità:", ["Profilo AniList (Machine Learning)", "Nuovo Utente (Cold Start)"])

# ==========================================
# PROFILO ANILIST (ML) MODE
# ==========================================
if mode == "Profilo AniList (Machine Learning)":
    with st.sidebar:
        st.header("1. Setup User Profile")
        
        # Show previously loaded profiles
        cached_profiles = get_cached_profiles()
        if cached_profiles:
            selected_profile = st.selectbox(
                "📂 Profili già caricati:",
                ["-- Nuovo --"] + cached_profiles,
                index=0
            )
            if selected_profile != "-- Nuovo --":
                username_default = selected_profile
            else:
                username_default = st.session_state.get("username", "")
        else:
            username_default = st.session_state.get("username", "")
        
        username_input = st.text_input("Enter AniList Username:", value=username_default)
        username_lower = username_input.strip().lower()
        
        model_trained_key = "model_trained_manga" if is_manga else "model_trained_anime"
        
        # Check if model exists for metadata
        model_suffix = "_manga_best_model.pkl" if is_manga else "_best_model.pkl"
        model_path = MODELS_DIR / f"{username_lower}{model_suffix}"
        
        if model_path.exists():
            try:
                meta = joblib.load(model_path)
                trained_at = meta.get('trained_at', 'Sconosciuto')
                if trained_at != 'Sconosciuto':
                    trained_at = datetime.fromisoformat(str(trained_at)).strftime("%d/%m/%Y %H:%M")
                st.info(f"💾 Modello trovato!\nUltimo training: {trained_at}")
            except:
                st.warning("⚠️ Errore lettura metadata.")

        col_load, col_train = st.columns(2)
        
        with col_load:
            if st.button("📂 Carica", use_container_width=True, disabled=not model_path.exists()):
                st.session_state["username"] = username_lower
                st.session_state[model_trained_key] = True
                st.rerun()
                
        with col_train:
            btn_label = "🔥 Allena/Retrain"
            if st.button(btn_label, use_container_width=True):
                media_label = "manga" if is_manga else "anime"
                with st.spinner(f"Fetching {media_label} data, training..."):
                    st.session_state["username"] = username_lower
                    if is_manga:
                        result = train_and_evaluate_all_manga_models(username_lower)
                    else:
                        result = train_and_evaluate_all_models(username_lower)
                    
                    if isinstance(result, dict) and result.get("status") in ["error", "fallback"]:
                        if result.get("status") == "fallback":
                            st.warning(result["message"])
                        else:
                            st.error(result["message"])
                        st.session_state[model_trained_key] = False
                    else:
                        st.success(f"✅ Completato!")
                        st.session_state[model_trained_key] = True
                        st.rerun()

    model_trained_key = "model_trained_manga" if is_manga else "model_trained_anime"
    
    if st.session_state[model_trained_key]:
        media_label = "Manga" if is_manga else "Anime"
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
            st.error(f"Failed to load models: {e}. Please retrain.")
            st.stop()
        
        # === AUTO-TRAIN REMINDER ===
        if check_auto_train_needed(model_artifact):
            st.warning("⏰ Il modello è stato trainato più di 7 giorni fa. Considera di ritrainarlo per dati aggiornati!")
        
        # === FALLBACK WARNING ===
        if model_artifact.get('fallback_recommended'):
            st.warning("⚠️ R² negativo: il modello ML è poco affidabile per questo dataset. Considera il Cold Start per risultati migliori.")
        
        st.header(f"2. Dashboard {media_label} per {username}")
        
        # === Model summary metrics ===
        best_metrics = model_artifact['metrics'][0]  # Already sorted by MAE
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modello", model_artifact['model_name'][:25])
        with col2:
            st.metric("MAE", f"{best_metrics['MAE']:.3f}")
        with col3:
            st.metric("R²", f"{best_metrics['R2']:.3f}")
        with col4:
            st.metric("Dataset", f"{model_artifact.get('dataset_size', '?')} titoli")
        
        # TSCV info
        if model_artifact.get('tscv_scores'):
            scores = model_artifact['tscv_scores']
            st.caption(f"📊 TimeSeriesSplit CV (5-fold): MAE = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        with st.expander("📋 Dettagli Modello e Confronto"):
            metrics_df = pd.DataFrame(model_artifact['metrics'])
            
            # Plotly bar chart for model comparison
            fig = px.bar(metrics_df, x='model', y='MAE', color='MAE',
                        color_continuous_scale='RdYlGn_r',
                        title="Confronto MAE tra Modelli (più basso = meglio)")
            fig.update_layout(xaxis_tickangle=-45, height=400, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(metrics_df, use_container_width=True)
            
            if model_artifact.get('feature_importance'):
                fi_df = pd.DataFrame(model_artifact['feature_importance'])
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                               title="Top Feature Importances", color='Importance',
                               color_continuous_scale='Viridis')
                fig_fi.update_layout(height=500, template='plotly_dark', yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)
            
            if model_artifact.get('optuna_result'):
                st.write("**Optuna Tuning:**", model_artifact['optuna_result'])



        # === TOP 10 RECOMMENDATIONS ===
        st.markdown("---")
        st.subheader(f"🌟 Top 10 Raccomandazioni {media_label}")
        
        candidate_limit = st.slider(
            "Numeri di candidati da analizzare (Popolari e Top Rated):", 
            min_value=10, 
            max_value=1000, 
            value=500, 
            step=10,
            help="Più candidati selezioni, più accurata sarà la ricerca, ma impiegherà più tempo per scaricare i dati da AniList."
        )
        
        rec_col1, rec_col2 = st.columns([3, 1])
        with rec_col1:
            rec_btn = st.button(f"✨ Genera Raccomandazioni {media_label}", use_container_width=True)
        
        if rec_btn:
            with st.spinner(f"Scarico {candidate_limit} candidati e calcolo predizioni personalizzate..."):
                if is_manga:
                    candidates = get_candidate_manga_for_recommendations(limit=candidate_limit)
                else:
                    candidates = get_candidate_anime_for_recommendations(limit=candidate_limit)
                
                # Exclude already watched/read
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
                    except: pass
                
                valid_candidates = [c for c in candidates if c['id'] not in watched_ids]
                results_list = []
                model = model_artifact['model']
                train_columns = model_artifact['train_columns']
                
                if 'sort_date' not in user_history_df.columns:
                    user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                
                prog = st.progress(0)
                for idx, cand in enumerate(valid_candidates):
                    if idx % 10 == 0:
                        prog.progress(max(0.01, min((idx + 1) / len(valid_candidates), 1.0)))
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
                st.session_state['last_recommendations'] = results_list[:10]
            
        if 'last_recommendations' in st.session_state and st.session_state['last_recommendations']:
            top_10 = st.session_state['last_recommendations']
            
            for i, (pred, cand) in enumerate(top_10):
                title = cand['title'].get('english') or cand['title'].get('romaji')
                col_img, col_txt = st.columns([1, 6])
                with col_img:
                    st.image(cand.get('coverImage', {}).get('large') or "", use_container_width=True)
                with col_txt:
                    st.markdown(f"#### #{i+1} : {title} ⭐️ {pred:.2f} / 10")
                    if is_manga:
                        sd = cand.get('startDate', {}) or {}
                        year = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                        st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Cap: {cand.get('chapters', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                        st.markdown(f"[➡️ Apri su AniList](https://anilist.co/manga/{cand.get('id')})")
                    else:
                        st.write(f"Global: {(cand.get('averageScore') or 0)/10:.1f} | Ep: {cand.get('episodes', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                        st.markdown(f"[➡️ Apri su AniList](https://anilist.co/anime/{cand.get('id')})")
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
                    "📥 Esporta CSV", export_df.to_csv(index=False),
                    f"raccomandazioni_{media_label.lower()}_{username}.csv", "text/csv"
                )
            with col_json:
                st.download_button(
                    "📥 Esporta JSON", json.dumps(export_data, indent=2, ensure_ascii=False),
                    f"raccomandazioni_{media_label.lower()}_{username}.json", "application/json"
                )

        # === PREDICT SINGLE ===
        st.markdown("---")
        st.subheader(f"🔍 Prevedi Voto {media_label} (Ricerca Esatta)")
        media_query = st.text_input(f"Cerca un {media_label.lower()}:")
        
        if media_query:
            with st.spinner("Ricerca in corso..."):
                results = search_manga_by_title(media_query) if is_manga else search_anime_by_title(media_query)
            
            if not results:
                st.warning(f"Nessun {media_label.lower()} trovato.")
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
                
                selected_option = st.selectbox(f"Seleziona il {media_label.lower()} esatto:", list(options.keys()))
                
                if st.button("🎯 Calcola Score Predetto"):
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
                            st.warning("⚠️ È già nella tua lista 'Planning' su AniList!")
                    
                    # === SHAP EXPLANATION ===
                    shap_vals, expected = get_shap_explanation(model, X_infer)
                    if shap_vals is not None:
                        shap_df = shap_to_dataframe(shap_vals, model_artifact['train_columns'], top_n=12)
                        if shap_df is not None and not shap_df.empty:
                            st.markdown("#### 🧠 SHAP — Perché questo voto?")
                            
                            colors = ['#00b09b' if v > 0 else '#eb3349' for v in shap_df['SHAP']]
                            fig_shap = go.Figure(go.Bar(
                                x=shap_df['SHAP'].values, y=shap_df['Feature'].values,
                                orientation='h', marker_color=colors
                            ))
                            fig_shap.update_layout(
                                title=f"Impatto delle Feature sul Voto (base: {expected:.2f})" if not isinstance(expected, np.ndarray) else "Impatto delle Feature sul Voto",
                                height=400, template='plotly_dark',
                                yaxis={'categoryorder': 'total ascending'},
                                xaxis_title="Impatto SHAP (+ = alza voto, - = abbassa voto)"
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                    
    else:
        st.info("👈 Inserisci uno username e traina i modelli dalla sidebar per iniziare!")

# ==========================================
# COLD START MODE
# ==========================================
elif mode == "Nuovo Utente (Cold Start)":
    media_label = "Manga" if is_manga else "Anime"
    icon = "📖" if is_manga else "🛸"
    
    st.header(f"{icon} Onboarding: Trova il tuo {media_label} perfetto!")
    st.write(f"Rispondi a poche domande per creare un profilo temporaneo e ottenere raccomandazioni {media_label.lower()} su misura.")
    
    fav_key = "cs_favorites_manga" if is_manga else "cs_favorites_anime"
    profile_key = "cs_profile_manga" if is_manga else "cs_profile_anime"
    step_key = "cs_step_manga" if is_manga else "cs_step_anime"
    rec_key = "cs_recommendations_manga" if is_manga else "cs_recommendations"
    
    if st.session_state[profile_key] is None:
        if st.session_state[step_key] == 1:
            st.subheader(f"Step 1: Dimmi 3-5 {media_label.lower()} che ti hanno 'Stregato'")
            search_q = st.text_input(f"Cerca {media_label.lower()}:")
            if search_q:
                res = search_manga_by_title(search_q) if is_manga else search_anime_by_title(search_q)
                if res:
                    opts = {f"{r['title'].get('english') or r['title'].get('romaji')}": r for r in res}
                    sel = st.selectbox("Risultati:", list(opts.keys()))
                    if st.button("➕ Aggiungi a Preferiti"):
                        media_obj = opts[sel]
                        if not any(a['id'] == media_obj['id'] for a in st.session_state[fav_key]):
                            st.session_state[fav_key].append(media_obj)
                            st.success(f"{sel} aggiunto!")
                        else:
                            st.warning("Già inserito!")
            
            if st.session_state[fav_key]:
                st.markdown("---")
                st.write("**I tuoi Preferiti:**")
                for fa in st.session_state[fav_key]:
                    st.write(f"- ⭐️ {fa['title'].get('english') or fa['title'].get('romaji')}")
                
                if len(st.session_state[fav_key]) >= 3:
                    if st.button("Prosegui allo Step Finale ➡️"):
                        st.session_state[step_key] = 2
                        st.rerun()
                else:
                    st.info(f"Mancano {3 - len(st.session_state[fav_key])} {media_label.lower()}.")
        
        elif st.session_state[step_key] == 2:
            st.subheader("Step 2: Ulteriori Preferenze")
            with st.form("cold_start_form"):
                fav_g = st.multiselect("Generi Preferiti:", GENRES_LIST)
                avoid_g = st.multiselect("Generi da Evitare:", GENRES_LIST)
                col1, col2, col3 = st.columns(3)
                if is_manga:
                    with col1: pref_fmt = st.selectbox("Formato:", MANGA_FORMATS_LIST, index=len(MANGA_FORMATS_LIST)-1)
                    with col2: pref_len = st.selectbox("Lunghezza (Cap):", MANGA_LENGTHS_LIST, index=len(MANGA_LENGTHS_LIST)-1)
                    with col3: pref_era = st.selectbox("Epoca:", ERAS_LIST, index=len(ERAS_LIST)-1)
                else:
                    with col1: pref_fmt = st.selectbox("Formato:", FORMATS_LIST, index=len(FORMATS_LIST)-1)
                    with col2: pref_len = st.selectbox("Lunghezza:", LENGTHS_LIST, index=len(LENGTHS_LIST)-1)
                    with col3: pref_era = st.selectbox("Epoca:", ERAS_LIST, index=len(ERAS_LIST)-1)
                
                if st.form_submit_button(f"🚀 Genera Profilo {media_label}"):
                    if is_manga:
                        profile = build_manga_cold_start_profile(st.session_state[fav_key], fav_g, avoid_g, pref_fmt, pref_len, pref_era)
                    else:
                        profile = build_cold_start_profile(st.session_state[fav_key], fav_g, avoid_g, pref_fmt, pref_len, pref_era)
                    st.session_state[profile_key] = profile
                    st.rerun()
    else:
        st.success("Profilo di partenza caricato!")
        if st.button("🔁 Rifai Onboarding"):
            st.session_state[profile_key] = None
            st.session_state[fav_key] = []
            st.session_state[step_key] = 1
            if rec_key in st.session_state: del st.session_state[rec_key]
            st.rerun()
        
        profile = st.session_state[profile_key]
        st.markdown("---")
        st.header(f"🎯 Top 10 {media_label} Consigliati")
        
        cs_limit_col, cs_btn_col = st.columns([3, 1])
        with cs_limit_col:
            candidate_limit_cs = st.slider(
                "Numeri di candidati da analizzare (Popolari e Top Rated):", 
                min_value=10, 
                max_value=1000, 
                value=500, 
                step=10,
                help="Più candidati selezioni, più accurata sarà la ricerca, ma impiegherà più tempo."
            )
        with cs_btn_col:
            if st.button("🔄 Genera / Aggiorna", use_container_width=True):
                if rec_key in st.session_state:
                    del st.session_state[rec_key]
                st.rerun()
        
        if rec_key not in st.session_state:
            with st.spinner(f"Calcolo raccomandazioni {media_label.lower()} su {candidate_limit_cs} candidati..."):
                cands = get_candidate_manga_for_recommendations(limit=candidate_limit_cs) if is_manga else get_candidate_anime_for_recommendations(limit=candidate_limit_cs)
                fav_ids = {a['id'] for a in st.session_state[fav_key]}
                cands = [c for c in cands if c['id'] not in fav_ids]
                recs = generate_manga_cold_start_recommendations(profile, cands) if is_manga else generate_cold_start_recommendations(profile, cands)
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
                    sd = cand.get('startDate', {}) or {}
                    year = sd.get('year', 'N/A') if isinstance(sd, dict) else 'N/A'
                    st.write(f"Anno: {year} | Cap: {cand.get('chapters', 'N/A')} | Generi: {', '.join(cand.get('genres', []))}")
                else:
                    st.write(f"Anno: {cand.get('seasonYear')} | Ep: {cand.get('episodes')} | Generi: {', '.join(cand.get('genres', []))}")
            st.write("---")
        
        # Export cold start recommendations
        if st.session_state.get(rec_key):
            export_cs = [{'rank': i+1, 'title': (c['title'].get('english') or c['title'].get('romaji')), 'score': round(s, 2)} for i, (s, c) in enumerate(st.session_state[rec_key])]
            st.download_button("📥 Esporta", pd.DataFrame(export_cs).to_csv(index=False), f"cold_start_{media_label.lower()}.csv", "text/csv")
        
        # Single prediction
        st.markdown("---")
        st.subheader(f"🔮 Calcola Affinità Singola ({media_label})")
        media_query = st.text_input(f"Quale {media_label.lower()} hai in mente?")
        
        if media_query:
            with st.spinner("Ricerca..."):
                results = search_manga_by_title(media_query) if is_manga else search_anime_by_title(media_query)
            if not results:
                st.warning("Non trovato.")
            else:
                if is_manga:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({(r.get('startDate') or {}).get('year', 'N/A')})": r for r in results}
                else:
                    options = {f"{r['title'].get('english') or r['title'].get('romaji')} ({r.get('seasonYear', 'N/A')})": r for r in results}
                
                selected_option = st.selectbox("Seleziona:", list(options.keys()), key="cs_sel")
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
