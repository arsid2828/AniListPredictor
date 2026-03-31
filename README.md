# AniList Machine Learning Predictor

An end-to-end Machine Learning pipeline that predicts the precise score an AniList user will give to a specific anime.

## 🚀 Features
- **Zero Data Leakage:** Implements strict temporal Train/Validation/Test splitting and historical tracking of user affinity (genres, studios, format) using only previously seen anime.
- **Multiple Models:** Evaluates Baseline, KNN Regressor, Ridge, Decision Trees (Raw & Pruned), Random Forest, and Gradient Boosting.
- **Full API Integration:** Native integration with AniList GraphQL API, with support for chunking, robust nested data gathering, and rate limiting.
- **Interactive Web UI:** Simple Streamlit interface for seamless demonstration.

## 📁 Project Structure
```
AnilistProject/
├── app/
│   └── main.py          # Streamlit UI
├── src/
│   ├── api.py           # AniList GraphQL Fetcher
│   ├── dataset.py       # Data flattening and basic cleaning
│   ├── features.py      # Feature engineering and temporal encoding
│   └── models.py        # ML pipelines, model training and evaluation
├── cache/               # Automatic save dir for raw API queries
├── data/                # Automatic save dir for cleaned CSVs
└── models/              # Automatic save dir for trained ML `.pkl` files
```

## 🛠️ Installation & Execution

1. Ensure you have Python 3.9+ installed.
2. Open a terminal and run the required installations:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Application:
   ```bash
   streamlit run app/main.py
   ```
   *(Note: If the `streamlit` command is not recognized due to PATH issues, use Python directly instead:)*
   ```bash
   python -m streamlit run app/main.py
   ```

## 💡 How it Works
1. Enter an AniList Username in the sidebar (e.g., `arsid`).
2. The application will pull the entire user's rated anime list via GraphQL.
3. It recursively sorts histories by `completedAt` to construct moving features (like `hist_genre_affinity`, `hist_studio_affinity`, and `hist_user_mean`). This guarantees absolutely zero temporal data leakage.
4. Various regression models are built across scikit-learn paths. The winning model based on lowest MAE strictly selected over Test parameters is automatically cached.
5. In the main page, type an anime title. The ML pipeline will extract matching features aligned exactly with the trained dataset, compute your realtime affinites, and output the predicted 0-10 score.

## ⚠️ Limitations & Notes
- If an anime has missing data globally, we fall back to robust defaults (e.g., fixed global mean for NaNs).
- Users with fewer than 10 scores are rejected, as predicting off less than 10 nodes results in garbage predictions.
- The `ccp_alpha` logic is explicitly computed and validated automatically via cost-complexity pruning.
- Deep Learning was explicitly excluded to provide transparent model explanations to support final conclusions on predictions.
