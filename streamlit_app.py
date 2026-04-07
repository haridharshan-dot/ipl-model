import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPL Winner Predictor", page_icon="🏏", layout="wide")

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors India",
}


def normalize_team_name(name):
    if name is None:
        return name
    return TEAM_ALIASES.get(name, name)


def predict_match_winner(model, feature_cols, cat_cols, row_dict, top_k=3):
    row = pd.DataFrame([row_dict])

    for c in ["team1", "team2", "winner", "toss_winner"]:
        if c in row.columns:
            row[c] = row[c].map(normalize_team_name)

    if "date" in row.columns:
        row["date"] = pd.to_datetime(row["date"], errors="coerce")
        row["match_year"] = row["date"].dt.year
        row["match_month"] = row["date"].dt.month
        row["match_dayofweek"] = row["date"].dt.dayofweek
        row = row.drop(columns=["date"])

    for c in feature_cols:
        if c not in row.columns:
            row[c] = np.nan

    row = row[feature_cols].copy()

    for c in feature_cols:
        if c in cat_cols:
            row[c] = row[c].astype(str).replace("nan", "Unknown").fillna("Unknown")
        else:
            row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0)

    probs = model.predict_proba(row)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], ranked


def find_default_artifact_paths():
    candidates = [
        Path.cwd(),
        Path.cwd() / "model_artifacts",
        Path.cwd() / "downloaded",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "model_artifacts",
        Path(__file__).resolve().parent / "downloaded",
    ]

    model_name = "ipl_winner_catboost.joblib"
    meta_name = "ipl_model_meta.json"

    found_model = None
    found_meta = None

    for base in candidates:
        model_path = base / model_name
        meta_path = base / meta_name
        if found_model is None and model_path.exists():
            found_model = model_path
        if found_meta is None and meta_path.exists():
            found_meta = meta_path

    default_model = str(found_model) if found_model else model_name
    default_meta = str(found_meta) if found_meta else meta_name
    return default_model, default_meta


def find_default_dataset_path():
    candidates = [
        Path.cwd() / "matches.csv",
        Path.cwd() / "downloaded" / "matches.csv",
        Path(__file__).resolve().parent / "matches.csv",
        Path(__file__).resolve().parent / "downloaded" / "matches.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "matches.csv"


def load_reference_matches(dataset_path):
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        ref = pd.read_csv(dataset_path)
        ref.columns = [c.strip() for c in ref.columns]
        for c in ["team1", "team2", "winner", "toss_winner"]:
            if c in ref.columns:
                ref[c] = ref[c].map(normalize_team_name)
        if "date" in ref.columns:
            ref["date"] = pd.to_datetime(ref["date"], errors="coerce")
        return ref
    except Exception:
        return None


def get_team_options(reference_df, model):
    if reference_df is not None and "team1" in reference_df.columns and "team2" in reference_df.columns:
        teams = sorted(
            list(
                set(reference_df["team1"].dropna().astype(str).tolist())
                | set(reference_df["team2"].dropna().astype(str).tolist())
            )
        )
        if teams:
            return teams
    return sorted([str(c) for c in model.classes_])


def get_head_to_head(reference_df, team1, team2):
    if reference_df is None:
        return None, None
    req = {"team1", "team2", "winner"}
    if not req.issubset(set(reference_df.columns)):
        return None, None

    h2h = reference_df[
        ((reference_df["team1"] == team1) & (reference_df["team2"] == team2))
        | ((reference_df["team1"] == team2) & (reference_df["team2"] == team1))
    ].copy()

    if h2h.empty:
        return pd.DataFrame(columns=["Team", "Wins"]), h2h

    wins = h2h["winner"].value_counts().rename_axis("Team").reset_index(name="Wins")
    return wins, h2h


def main():
    st.markdown(
        """
        <style>
        .big-result {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: linear-gradient(90deg, #f3f7ff, #eefaf4);
            border: 1px solid #d6e6ff;
            margin-bottom: 0.8rem;
        }
        .small-note {
            color: #5f6470;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("IPL Match Winner Predictor")
    st.caption("Chat-style prediction UI with team selection and head-to-head match results.")

    st.sidebar.header("Model Files")
    default_model, default_meta = find_default_artifact_paths()
    model_path = st.sidebar.text_input("Model path", default_model)
    meta_path = st.sidebar.text_input("Metadata path", default_meta)
    dataset_path = st.sidebar.text_input("Matches CSV (optional)", find_default_dataset_path())

    st.sidebar.caption(f"Working directory: {Path.cwd()}")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.warning("Model or metadata file not found.")
        st.info(
            "Place files in model_artifacts, project root, or downloaded folder, or update sidebar paths. "
            "Expected files: ipl_winner_catboost.joblib and ipl_model_meta.json"
        )
        st.stop()

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_cols", [])
    cat_cols = meta.get("cat_cols", [])
    reference_df = load_reference_matches(dataset_path)

    teams = get_team_options(reference_df, model)
    if len(teams) < 2:
        st.error("Need at least two teams for prediction.")
        st.stop()

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [
            {
                "role": "assistant",
                "text": "Select match inputs and click Predict Winner. I will return probabilities and likely result.",
            }
        ]

    c1, c2, c3 = st.columns(3)
    with c1:
        season = st.number_input("Season", min_value=2008, max_value=2100, value=2024, step=1)
        city = st.text_input("City", "Mumbai")
        venue = st.text_input("Venue", "Wankhede Stadium")
    with c2:
        team1 = st.selectbox("Team 1", teams, index=0)
        team2_options = [t for t in teams if t != team1]
        default_idx = 0
        team2 = st.selectbox("Team 2", team2_options, index=default_idx)
        toss_winner = st.selectbox("Toss Winner", [team1, team2], index=0)
    with c3:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"], index=1)
        dl_applied = st.selectbox("D/L Applied", [0, 1], index=0)

    st.subheader("Prediction Chat")
    for msg in st.session_state.chat_log:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    if st.button("Predict Winner", type="primary"):
        if team1 == team2:
            st.error("Team 1 and Team 2 must be different.")
            st.stop()

        row = {
            "season": int(season),
            "city": city,
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "dl_applied": int(dl_applied),
        }

        top_preds, full_preds = predict_match_winner(model, feature_cols, cat_cols, row, top_k=3)
        winner, winner_prob = top_preds[0]

        user_prompt = (
            f"Predict match: {team1} vs {team2} | Toss: {toss_winner} ({toss_decision}) | "
            f"Season: {season} | City: {city}"
        )
        assistant_reply = (
            f"Predicted winner: {winner} ({winner_prob:.2%})\\n\\n"
            f"Top probabilities: "
            + ", ".join([f"{t}: {p:.2%}" for t, p in top_preds])
        )

        st.session_state.chat_log.append({"role": "user", "text": user_prompt})
        st.session_state.chat_log.append({"role": "assistant", "text": assistant_reply})

        st.markdown(
            f"<div class='big-result'><b>Likely Winner:</b> {winner} | "
            f"<b>Confidence:</b> {winner_prob:.2%}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("All Team Probabilities")
        pred_df = pd.DataFrame(full_preds, columns=["Team", "Probability"])
        st.dataframe(pred_df, use_container_width=True)

        st.subheader("Head-to-Head Results")
        wins_df, h2h_df = get_head_to_head(reference_df, team1, team2)
        if wins_df is None:
            st.info("Head-to-head results unavailable. Add matches CSV in sidebar to enable this.")
        elif wins_df.empty:
            st.info("No historical matches found for this team pair in the provided dataset.")
        else:
            st.dataframe(wins_df, use_container_width=True)
            if "date" in h2h_df.columns:
                recent = h2h_df.sort_values("date", ascending=False).head(5)
            else:
                recent = h2h_df.tail(5)
            keep_cols = [c for c in ["date", "season", "team1", "team2", "winner", "venue", "city"] if c in recent.columns]
            if keep_cols:
                st.markdown("<div class='small-note'>Recent matches (latest 5)</div>", unsafe_allow_html=True)
                st.dataframe(recent[keep_cols], use_container_width=True)


if __name__ == "__main__":
    main()
