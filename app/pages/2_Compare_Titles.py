import html
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import (
    MAX_SEARCH_LEN,
    USERNAME_VALIDATION_MESSAGE,
    get_dataset_path,
    init_session_state,
    inject_css,
    is_valid_username,
    render_app_disclaimer,
    render_user_badge,
    safe_load_model_artifact,
)
from src.analytics import get_shap_explanation, shap_to_dataframe
from src.api import search_anime_by_title, search_manga_by_title
from src.features import build_inference_features, build_manga_inference_features

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Compare Titles", layout="wide", page_icon="⚖️")
inject_css()
init_session_state()
render_user_badge()

st.title("⚖️ Compare Two Titles")
st.write("Enter two anime or manga to compare their predicted scores and see the differences.")

username = st.session_state.get("username", "")

media_type = st.radio("Type:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "Manga" if is_manga else "Anime"

if not username:
    st.info("👈 Return to the main page, enter your username and train the model first.")
    st.stop()

if not is_valid_username(username):
    st.error(USERNAME_VALIDATION_MESSAGE)
    st.stop()

try:
    model_artifact = safe_load_model_artifact(username, is_manga=is_manga)
    csv_path = get_dataset_path(username, is_manga=is_manga)
    user_history_df = pd.read_csv(csv_path)
except Exception:
    logger.exception("Failed to load comparison assets.")
    st.error("Failed to load the model. Please retrain from the main page.")
    st.stop()

if "sort_date" not in user_history_df.columns:
    user_history_df["sort_date"] = pd.to_datetime(pd.Timestamp.now())

model = model_artifact["model"]
train_columns = model_artifact["train_columns"]

col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"🅰️ {media_label} 1")
    q1 = st.text_input(f"Search {media_label.lower()} A:", key="cmp_q1", max_chars=MAX_SEARCH_LEN)
    selected_1 = None
    if q1:
        results1 = search_manga_by_title(q1) if is_manga else search_anime_by_title(q1)
        if results1:
            opts1 = {f"{r['title'].get('english') or r['title'].get('romaji')}": r for r in results1}
            sel1 = st.selectbox("Select A:", list(opts1.keys()), key="cmp_sel1")
            selected_1 = opts1[sel1]
            img1 = selected_1.get("coverImage", {}).get("large", "")
            if img1:
                st.image(img1, width=200)

with col_right:
    st.subheader(f"🅱️ {media_label} 2")
    q2 = st.text_input(f"Search {media_label.lower()} B:", key="cmp_q2", max_chars=MAX_SEARCH_LEN)
    selected_2 = None
    if q2:
        results2 = search_manga_by_title(q2) if is_manga else search_anime_by_title(q2)
        if results2:
            opts2 = {f"{r['title'].get('english') or r['title'].get('romaji')}": r for r in results2}
            sel2 = st.selectbox("Select B:", list(opts2.keys()), key="cmp_sel2")
            selected_2 = opts2[sel2]
            img2 = selected_2.get("coverImage", {}).get("large", "")
            if img2:
                st.image(img2, width=200)

if selected_1 and selected_2 and st.button("⚡ Compare!", use_container_width=True):
    with st.spinner("Calculating predictions and SHAP..."):
        if is_manga:
            x1 = build_manga_inference_features(selected_1, user_history_df, train_columns)
            x2 = build_manga_inference_features(selected_2, user_history_df, train_columns)
        else:
            x1 = build_inference_features(selected_1, user_history_df, train_columns)
            x2 = build_inference_features(selected_2, user_history_df, train_columns)

        pred1 = float(np.clip(model.predict(x1)[0], 0.0, 10.0))
        pred2 = float(np.clip(model.predict(x2)[0], 0.0, 10.0))

        title1 = selected_1["title"].get("english") or selected_1["title"].get("romaji")
        title2 = selected_2["title"].get("english") or selected_2["title"].get("romaji")
        safe_title1 = html.escape(title1)
        safe_title2 = html.escape(title2)

        st.markdown("---")
        score_col1, score_vs, score_col2 = st.columns([2, 1, 2])
        with score_col1:
            color1 = "#00b09b" if pred1 >= pred2 else "#a0a0b0"
            st.markdown(f"<h2 style='text-align:center; color:{color1};'>{pred1:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>{safe_title1}</p>", unsafe_allow_html=True)
        with score_vs:
            st.markdown("<h2 style='text-align:center; color:#667eea;'>VS</h2>", unsafe_allow_html=True)
            diff = pred1 - pred2
            if abs(diff) < 0.3:
                st.markdown("<p style='text-align:center;'>🤝 Very similar</p>", unsafe_allow_html=True)
            elif diff > 0:
                st.markdown(f"<p style='text-align:center;'>🅰️ leads by {diff:.2f}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align:center;'>🅱️ leads by {abs(diff):.2f}</p>", unsafe_allow_html=True)
        with score_col2:
            color2 = "#00b09b" if pred2 >= pred1 else "#a0a0b0"
            st.markdown(f"<h2 style='text-align:center; color:{color2};'>{pred2:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>{safe_title2}</p>", unsafe_allow_html=True)

        st.markdown("---")

        glob1 = (selected_1.get("averageScore") or 0) / 10.0
        glob2 = (selected_2.get("averageScore") or 0) / 10.0

        comp_df = pd.DataFrame(
            {
                "Metric": ["Predicted Score", "Global Score", "Genres", "Format", "Popularity"],
                title1: [
                    f"{pred1:.2f}",
                    f"{glob1:.1f}",
                    ", ".join(selected_1.get("genres", [])),
                    selected_1.get("format", "N/A"),
                    selected_1.get("popularity", "N/A"),
                ],
                title2: [
                    f"{pred2:.2f}",
                    f"{glob2:.1f}",
                    ", ".join(selected_2.get("genres", [])),
                    selected_2.get("format", "N/A"),
                    selected_2.get("popularity", "N/A"),
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        shap1, _ = get_shap_explanation(model, x1)
        shap2, _ = get_shap_explanation(model, x2)

        if shap1 is not None and shap2 is not None:
            st.markdown("### 🧠 SHAP - Why these scores?")

            shap_df1 = shap_to_dataframe(shap1, train_columns, top_n=10)
            shap_df2 = shap_to_dataframe(shap2, train_columns, top_n=10)

            shap_col1, shap_col2 = st.columns(2)

            if shap_df1 is not None and not shap_df1.empty:
                with shap_col1:
                    colors1 = ["#00b09b" if v > 0 else "#eb3349" for v in shap_df1["SHAP"]]
                    fig_s1 = go.Figure(
                        go.Bar(
                            x=shap_df1["SHAP"].values,
                            y=shap_df1["Feature"].values,
                            orientation="h",
                            marker_color=colors1,
                        )
                    )
                    fig_s1.update_layout(
                        title=f"SHAP: {title1[:30]}",
                        height=350,
                        template="plotly_dark",
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_s1, use_container_width=True)

            if shap_df2 is not None and not shap_df2.empty:
                with shap_col2:
                    colors2 = ["#00b09b" if v > 0 else "#eb3349" for v in shap_df2["SHAP"]]
                    fig_s2 = go.Figure(
                        go.Bar(
                            x=shap_df2["SHAP"].values,
                            y=shap_df2["Feature"].values,
                            orientation="h",
                            marker_color=colors2,
                        )
                    )
                    fig_s2.update_layout(
                        title=f"SHAP: {title2[:30]}",
                        height=350,
                        template="plotly_dark",
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_s2, use_container_width=True)

render_app_disclaimer()
