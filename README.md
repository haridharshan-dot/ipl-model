# IPL Match Winner Prediction

This project predicts IPL match winners using a trained CatBoost model on match-level IPL data.

## Project Contents

- `ipl_winner_colab_notebook.ipynb`: End-to-end Google Colab notebook for training, evaluation, and artifact download.
- `streamlit_app.py`: Local web app to test predictions interactively.
- `model_artifacts/ipl_winner_catboost.joblib`: Trained model weights.
- `model_artifacts/ipl_model_meta.json`: Model metadata (feature columns, categorical columns, etc.).
- `matches.csv`: IPL match dataset used locally.
- `requirements_streamlit.txt`: Python dependencies for local app.
- `run_streamlit.bat`: Windows helper to install dependencies and run Streamlit.
- `smoke_test_model.py`: Quick script to verify model files load and predict.

## Model Details

- Base model: `CatBoostClassifier`
- Task: Match winner prediction (classification)
- Key input features include:
  - Team 1, Team 2
  - Toss winner and toss decision
  - Venue and city
  - Season and D/L flag

## 1) Clone This Repository

```bash
git clone https://github.com/haridharshan-dot/ipl-model.git
cd ipl-model
```

## 2) Run Locally (Streamlit App)

### Option A: Windows shortcut

Double-click `run_streamlit.bat`

### Option B: Terminal commands

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## 3) Train / Retrain in Google Colab

1. Open `ipl_winner_colab_notebook.ipynb` in Google Colab.
2. Run all cells in order.
3. If Kaggle auth is needed, upload `kaggle.json` in the auth cell.
4. Train model.
5. Download generated artifacts:
   - `ipl_winner_catboost.joblib`
   - `ipl_model_meta.json`
6. Place downloaded files into `model_artifacts/` or `downloaded/`.

The Streamlit app auto-detects model files from common locations.

## 4) Quick Model Smoke Test

```bash
python smoke_test_model.py
```

Expected output includes:
- `SMOKE_TEST_OK`
- Top predicted teams with probabilities

## 5) How Prediction Works in App

In the app, you choose:
- Team 1 and Team 2
- Toss winner and toss decision
- Season, city, venue, and D/L flag

Then the app shows:
- Predicted winner and confidence
- Top team probabilities
- Optional head-to-head historical results (from `matches.csv`)

## Troubleshooting

- If you ran `python streamlit_app.py` and saw warnings, use:
  - `streamlit run streamlit_app.py`
- If model file not found:
  - Keep files in `model_artifacts/`, `downloaded/`, or set exact paths in app sidebar.
- If dependency errors occur:
  - Reinstall with `pip install -r requirements_streamlit.txt`

## Repository URL

- https://github.com/haridharshan-dot/ipl-model
