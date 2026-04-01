import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to sys path to allow importing src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api import search_anime_by_title
from src.dataset import build_user_dataframe
from src.features import build_inference_features
from src.models import train_and_evaluate_all_models
from src.models import MODELS_DIR
from src.dataset import DATA_DIR
import joblib

import logging
logger = logging.getLogger(__name__)

def check_planning_status(username, media_id):
    from pathlib import Path
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

st.set_page_config(page_title="AniList ML Predictor", layout="wide")

st.title("🎬 AniList User Score Predictor (Machine Learning)")
st.markdown("Predict the score a user will give to a specific anime based on their historical watch data using ML models avoiding data leakage.")

# Stateful variables
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False

# Sidebar for Setup & Training
with st.sidebar:
    st.header("1. Setup User Profile")
    username_input = st.text_input("Enter AniList Username:", value="arsid")
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
    st.header(f"2. Predict Score for {st.session_state['username']}")
    
    # Load user dataset and model artifact
    username = st.session_state["username"]
    model_path = MODELS_DIR / f"{username}_best_model.pkl"
    csv_path = DATA_DIR / f"{username}_clean.csv"
    
    try:
        model_artifact = joblib.load(model_path)
        user_history_df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load cached models or dataset: {e}. Please retrain.")
        st.stop()
        
    # Inform user about which model won
    best_model_name = model_artifact['model_name']
    cv_metrics = pd.DataFrame(model_artifact['metrics'])
    
    with st.expander("View Model Comparison Results"):
        st.dataframe(cv_metrics)
        st.write(f"**Selected Model:** {best_model_name}")
        if model_artifact.get('feature_importance'):
            st.write("**Top Feature Importances:**")
            st.bar_chart(pd.DataFrame(model_artifact['feature_importance']).set_index('Feature'))

    # Anime Search Input
    anime_query = st.text_input("Enter Anime Title (e.g., 'Frieren', 'Attack on Titan'):")
    
    if anime_query and st.button("Search & Predict"):
        with st.spinner("Searching AniList..."):
            results = search_anime_by_title(anime_query)
            
        if not results:
            st.warning("No anime found matching that title.")
        else:
            # For simplicity, we just use the first matching result.
            st.write("### Top Match Found:")
            top_anime = results[0]
            
            title = top_anime['title'].get('english') or top_anime['title'].get('romaji')
            
            # Display image and details side by side
            col_img, col_txt = st.columns([1, 4])
            with col_img:
                cover_url = top_anime.get('coverImage', {}).get('extraLarge') or top_anime.get('coverImage', {}).get('large')
                if cover_url:
                    st.image(cover_url, use_container_width=True)
            
            with col_txt:
                st.subheader(f"{title} ({top_anime.get('seasonYear', 'N/A')})")
                st.write(f"**Format:** {top_anime.get('format')} | **Episodes:** {top_anime.get('episodes')}")
                st.write(f"**Global Avg Score:** {top_anime.get('averageScore', 'N/A')}/100")
                if check_planning_status(username_input, top_anime['id']):
                    st.warning("🟡 **Attenzione:** Hai già questo anime nella tua lista 'Plan to Watch' su AniList!")
            
            with st.spinner("Extracting features and running prediction..."):
                # 1. Feature Engineering (1 row)
                # Ensure parse dates match structure
                if 'sort_date' not in user_history_df.columns:
                    user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                    
                X_infer = build_inference_features(top_anime, user_history_df, model_artifact['train_columns'])
                
                # 2. Predict
                model = model_artifact['model']
                predicted_score = model.predict(X_infer)[0]
                original_pred = predicted_score

                # 3. Applicazione Euristica Umana (Regole Manuali per Casi Estremi)
                try:
                    adult_ratio = user_history_df['isAdult'].mean() if 'isAdult' in user_history_df.columns else 0
                    is_target_adult = top_anime.get('isAdult', False)
                    is_target_hentai = 'Hentai' in top_anime.get('genres', [])
                    
                    if (is_target_adult or is_target_hentai) and adult_ratio < 0.03:
                        if is_target_hentai:
                            predicted_score -= 3.0
                            st.warning("⚠️ **Penalità Anti-Target (-3.0):** Contenuto esplicito (Hentai) fortemente penalizzato perché non presente quasi per nulla nel tuo storico.")
                        else:
                            predicted_score -= 2.0
                            st.warning("⚠️ **Penalità Anti-Target (-2.0):** Contenuto categorizzato per Adulti (18+) o fortemente Ecchi spinto, penalizzato perché fuori dalla tua comfort-zone.")
                except Exception as e:
                    pass

                predicted_score = float(np.clip(predicted_score, 0.0, 10.0))
                
            st.success(f"### Predicted Score for {username}: {predicted_score:.2f} / 10")
            
            col1, col2 = st.columns(2)
            with col1:
                hist_mean = user_history_df['user_score'].mean()
                st.metric(label=f"{username}'s Historical Mean Score", value=f"{hist_mean:.2f}")
            with col2:
                glob_mean = (top_anime.get('averageScore') or 0) / 10.0
                st.metric(label="Global Average Score", value=f"{glob_mean:.2f}")

            # Transparency / Explainability block
            st.markdown("---")
            st.subheader("💡 Perché questo voto?")
            st.markdown("Ecco i valori storici calcolati da zero (fino al momento precedente a questo anime) delle **caratteristiche che hanno pesato maggiormente** in questa singola scelta dell'A.I.:")
            
            top_feat_dict = model_artifact.get('feature_importance')
            if top_feat_dict:
                feature_translations = {
                    'averageScore': 'Voto Globale del Pubblico',
                    'popularity': 'Popolarità / Visualizzazioni',
                    'duration': 'Durata Episodica in minuti',
                    'episodes': 'Numero di Episodi',
                    'hist_user_mean': 'Tua Media Voti Storica',
                    'hist_user_std': 'Tua Deviazione (Volubilità Voti)',
                    'recent_mean_score': 'Tuo Umore (Media Ultimi 10 Visti)',
                    'hist_format_affinity': 'Tua Affinità a questo Formato',
                    'hist_genre_affinity': 'Tua Affinità a questi Generi',
                    'hist_genre_freq': 'Frequenza di Visione (% in questi Generi)',
                    'hist_tag_affinity': 'Tua Affinità a queste Tematiche (Tag)',
                    'hist_tag_freq': 'Frequenza di Visione (% con questi Tag)',
                    'hist_studio_affinity': 'Tuo Storico Voti con questo Studio',
                    'hist_studio_freq': 'Frequenza di Visione (% con questo Studio)',
                    'hist_producer_affinity': 'Tuo Storico Voti con questi Produttori',
                    'hist_producer_freq': 'Frequenza di Visione (% con questi Produttori)',
                    'hist_franchise_count': 'Capitoli di questo Franchise visti in passato',
                    'hist_franchise_mean': 'Tuo Voto Storico espresso su questo Franchise',
                    'hist_char_count': 'Anime con Crossover di questi Personaggi',
                    'hist_char_mean': 'Tuo Voto Storico ai Crossover di questi Personaggi',
                    'hist_global_diff': 'Tuo Scarto dal Pubblico (Hater/Fanboy)',
                    'hist_global_mae': 'Tua Imprevedibilità (Contrarian Score)',
                    'favourites': 'Amore Globale (Favourites)',
                    'isAdult': 'Contenuto per Adulti (18+)'
                }

                def t_feat(f):
                    if f in feature_translations: return feature_translations[f]
                    if f.startswith('genre_'): return f"Presenza Genere {f.replace('genre_', '')}"
                    if f.startswith('tag_'): return f"Tema Centrale {f.replace('tag_', '')}"
                    if f.startswith('format_'): return f"Formato Variante {f.replace('format_', '')}"
                    if f.startswith('countryOfOrigin_'): return f"Paese di Origine {f.replace('countryOfOrigin_', '')}"
                    return f
                    
                for x in top_feat_dict[:6]:
                    c = x['Feature']
                    imp = x['Importance'] * 100.0
                    val = X_infer.iloc[0][c]
                    
                    if imp > 20: imp_msg = "Impatto Severo 🔥"
                    elif imp > 10: imp_msg = "Molto Alto ⭐"
                    elif imp > 5: imp_msg = "Rilevante 📈"
                    else: imp_msg = "Sfumatura 🔹"
                    
                    # Convert log counts back
                    if c in ['popularity', 'favourites'] and val > 0:
                        val_num = int(np.expm1(val))
                        form = f"{val_num:,} utenti".replace(',', '.')
                    elif c.startswith('tag_') or c.startswith('genre_') or c.startswith('countryOfOrigin_') or c.endswith('_freq') or c == 'isAdult':
                        if c.startswith('tag_') and val > 0.0:
                            form = f"Allineato al {int(val*100)}%"
                        elif c.endswith('_freq'):
                            form = f"Rappresenta il {int(val*100)}% del totale"
                        elif c.endswith('_count'):
                            form = f"Ne avevi già visti {int(val)}"
                        else:
                            form = "Sì" if val > 0.0 else "No"
                    elif isinstance(val, (float, np.floating)):
                        form = f"{val:.3f}"
                    else:
                        form = str(val)
                        
                    # Perturbation test to find logical direction
                    X_base = X_infer.copy()
                    if c == 'averageScore':
                        neutral = 7.0
                    elif c.startswith('hist_') and (c.endswith('_freq') or c.endswith('_count')) or c == 'hist_user_std':
                        neutral = 0.0
                    elif c.startswith('hist_') or c == 'recent_mean_score':
                        neutral = hist_mean
                    elif c == 'popularity':
                        neutral = float(np.log1p(1000))
                    elif c == 'favourites':
                        neutral = float(np.log1p(10))
                    elif c == 'duration':
                        neutral = 24.0
                    else:
                        neutral = 0.0
                        
                    X_base.iloc[0, X_base.columns.get_loc(c)] = neutral
                    pred_without = model.predict(X_base)[0]
                    impact = original_pred - pred_without
                    
                    if impact > 0.01:
                        dir_icon = f"⬆️ Ha spinto il voto in SU di +{impact:.2f}"
                    elif impact < -0.01:
                        dir_icon = f"⬇️ Ha affossato il voto di {impact:.2f}"
                    else:
                        dir_icon = "⚖️ Impatto stabile bilanciato"
                        
                    st.write(f"- **{t_feat(c)}**: `{form}`  \n  > *(🧠 Peso decisorio: **{imp:.1f}%** | {dir_icon})*")
            else:
                st.info("Le origini matematiche dettagliate per questo modello di classificazione non sono disponibili.")
else:
    st.info("Please enter a username and train models on their history in the sidebar first.")
    
