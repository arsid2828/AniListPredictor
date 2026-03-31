import streamlit as st
import pandas as pd
import numpy as np
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
            st.subheader(f"{title} ({top_anime.get('seasonYear', 'N/A')})")
            st.write(f"**Format:** {top_anime.get('format')} | **Episodes:** {top_anime.get('episodes')}")
            st.write(f"**Global Avg Score:** {top_anime.get('averageScore', 'N/A')}/100")
            
            with st.spinner("Extracting features and running prediction..."):
                # 1. Feature Engineering (1 row)
                # Ensure parse dates match structure
                if 'sort_date' not in user_history_df.columns:
                    user_history_df['sort_date'] = pd.to_datetime(pd.Timestamp.now())
                    
                X_infer = build_inference_features(top_anime, user_history_df, model_artifact['train_columns'])
                
                # 2. Predict
                model = model_artifact['model']
                predicted_score = model.predict(X_infer)[0]
                
            st.success(f"### Predicted Score for {username}: {predicted_score:.2f} / 10")
            
            col1, col2 = st.columns(2)
            with col1:
                hist_mean = user_history_df['user_score'].mean()
                st.metric(label=f"{username}'s Historical Mean Score", value=f"{hist_mean:.2f}")
            with col2:
                glob_mean = (top_anime.get('averageScore') or 0) / 10.0
                st.metric(label="Global Average Score", value=f"{glob_mean:.2f}")
else:
    st.info("Please enter a username and train models on their history in the sidebar first.")
    
