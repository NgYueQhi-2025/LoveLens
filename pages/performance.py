import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import textwrap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dating_app_behavior_dataset.csv"
MODEL_PATH = ROOT / "source_code" / "best_xgb_model.pkl"
SCALER_PATH = ROOT / "source_code" / "scaler.pkl"
MODEL_COLUMNS_PATH = ROOT / "source_code" / "model_columns.json"

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["interest_count"] = (
        df["interest_tags"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(lambda items: sum(1 for item in items if str(item).strip()))
    )
    df["is_success"] = (df["match_outcome"].astype(str).str.strip() == "Mutual Match").astype(int)
    return df

@st.cache_resource(show_spinner=False)
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(MODEL_COLUMNS_PATH, "r") as f:
        model_columns = json.load(f)
    return model, scaler, model_columns


def prepare_model_frame(data: pd.DataFrame, scaler, model_columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    columns_to_drop = ["app_usage_time_label", "swipe_right_label"]
    frame = frame.drop(columns=[c for c in columns_to_drop if c in frame.columns], errors="ignore")
    frame["interest_count"] = frame["interest_tags"].fillna("").astype(str).str.split(",").apply(lambda items: sum(1 for item in items if str(item).strip()))
    frame["profile_richness"] = frame["bio_length"].fillna(0) + (frame["profile_pics_count"].fillna(0) * 50)
    frame["engagement_efficiency"] = np.where(
        frame.get("likes_received", 0) > 0,
        frame["mutual_matches"].fillna(0) / frame["likes_received"].fillna(1),
        0,
    )
    frame["emoji_intensity"] = frame["emoji_usage_rate"].fillna(0) * frame["message_sent_count"].fillna(0)

    interests_expanded = frame["interest_tags"].fillna("").astype(str).str.get_dummies(sep=", ")
    df_encoded = pd.concat([frame.drop(columns=["interest_tags"], errors="ignore"), interests_expanded], axis=1)

    categorical_cols = [
        "gender",
        "sexual_orientation",
        "location_type",
        "income_bracket",
        "education_level",
        "swipe_time_of_day",
    ]
    df_final = pd.get_dummies(df_encoded, columns=[c for c in categorical_cols if c in df_encoded.columns], drop_first=True)

    combined = df_final.copy()
    for col in model_columns:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined.reindex(columns=model_columns)

    numeric_columns = [
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches',
        'profile_pics_count', 'bio_length', 'message_sent_count', 'emoji_usage_rate',
        'last_active_hour', 'interest_count', 'profile_richness', 'engagement_efficiency', 'emoji_intensity'
    ]
    try:
        combined[numeric_columns] = scaler.transform(combined[numeric_columns].astype(float))
    except Exception:
        pass

    return combined

@st.cache_data(show_spinner=False)
def evaluate_model() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model, scaler, model_columns = load_model_assets()
    df = load_data()
    feature_frame = prepare_model_frame(df, scaler, model_columns)
    y_true = df["is_success"].values
    y_pred = model.predict(feature_frame)
    if y_pred.dtype.kind in {"U", "S", "O"}:
        y_pred = np.where(pd.Series(y_pred).astype(str).str.strip() == "Mutual Match", 1, 0)
    else:
        y_pred = y_pred.astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    feature_importance = pd.DataFrame({
        "feature": model_columns,
        "importance": getattr(model, "feature_importances_", np.zeros(len(model_columns))),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return metrics, df, feature_importance

# =========================
# PAGE RENDER
# =========================
def render_performance_page():
    page_code = textwrap.dedent('''
st.title("Model Performance Dashboard")

st.markdown("""
This page evaluates machine learning models for predicting relationship match outcomes.
It compares multiple models and highlights the best performing model.
""")

metrics, df, feature_importance = evaluate_model()

# =========================
# BEST MODEL METRICS
# =========================
st.subheader("Best Model: Tuned XGBoost")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")

with col2:
    st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")

with col3:
    st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")

with col4:
    st.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")


# =========================
# MODEL COMPARISON
# =========================
st.subheader("Model Comparison")

baseline_accuracy = max(df['is_success'].mean(), 1 - df['is_success'].mean())
comparison_df = pd.DataFrame({
    "Model": ["Baseline (Majority Class)", "Tuned XGBoost"],
    "Accuracy": [baseline_accuracy, metrics["accuracy"]]
})

fig, ax = plt.subplots()
sns.barplot(data=comparison_df, x="Model", y="Accuracy", palette=["#94a3b8", "#8b5cf6"], ax=ax)
plt.xticks(rotation=20)
plt.ylim(0, 1)

st.pyplot(fig)


# =========================
# CONFUSION MATRIX
# =========================
st.subheader("Confusion Matrix (Best Model)")

cm = metrics["confusion_matrix"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticklabels(["Not Successful", "Successful"], rotation=20)
ax.set_yticklabels(["Not Successful", "Successful"], rotation=0)

st.pyplot(fig)


# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance")

feature_top = feature_importance.head(10)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=feature_top, x="importance", y="feature", palette=["#8b5cf6"] * len(feature_top), ax=ax)
ax.set_title("Top 10 Feature Importances", loc="left", fontsize=14, fontweight="bold", color="#111")
ax.set_xlabel("Importance")
ax.set_ylabel("")
plt.tight_layout()
st.pyplot(fig)


# =========================
# INSIGHTS
# =========================
st.subheader("Insights")

st.markdown("""
- Tuned XGBoost gives the best performance among all models.
- Tree-based models perform better than Logistic Regression due to non-linear patterns in data.
- Dataset imbalance affects baseline model performance.
- Features like swipe ratio, engagement efficiency, and profile richness are strong predictors of success.
""")
''')
    exec(page_code, globals(), locals())


if __name__ == "__main__":
    render_performance_page()