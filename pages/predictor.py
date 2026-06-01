"""
Member 2 — Match Outcome Predictor Page
Tying the (Data) Knot: Love, Life & Likes
WIA1006/WID3006 Machine Learning, Sem 2, 2025/2026

This file is the standalone Streamlit page for the Match Outcome Predictor.
It can be run standalone OR imported as a page in a multi-page app.

USAGE (standalone):
    streamlit run predictor_page.py

USAGE (multi-page, in main app.py):
    import predictor_page
    predictor_page.show()
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG (only when run standalone)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Match Predictor — Love, Life & Likes",
        page_icon="💘",
        layout="wide",
    )


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        /* ── HERO ── */
        .hero-banner {
            background: linear-gradient(135deg, #1a0a2e 0%, #3b0764 50%, #6d1e9e 100%);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: "💘";
            position: absolute;
            right: 2rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 6rem;
            opacity: 0.15;
        }
        .hero-banner h1 {
            font-family: 'DM Serif Display', serif;
            color: #fff;
            font-size: 2.2rem;
            margin: 0 0 0.4rem 0;
        }
        .hero-banner p {
            color: #e9d5ff;
            margin: 0;
            font-size: 1rem;
            font-weight: 300;
        }

        /* ── SECTION HEADERS ── */
        .section-title {
            font-family: 'DM Serif Display', serif;
            color: #6d1e9e;
            font-size: 1.25rem;
            border-left: 4px solid #a855f7;
            padding-left: 0.75rem;
            margin-bottom: 1rem;
        }

        /* ── RESULT CARDS ── */
        .result-card {
            border-radius: 14px;
            padding: 2rem;
            text-align: center;
            animation: fadeUp 0.5s ease;
        }
        .result-success {
            background: linear-gradient(135deg, #064e3b, #065f46);
            border: 1px solid #10b981;
        }
        .result-fail {
            background: linear-gradient(135deg, #4c0519, #881337);
            border: 1px solid #f43f5e;
        }
        .result-card .emoji { font-size: 3.5rem; }
        .result-card .verdict {
            font-family: 'DM Serif Display', serif;
            font-size: 1.8rem;
            color: #fff;
            margin: 0.5rem 0 0.25rem;
        }
        .result-card .sub {
            color: rgba(255,255,255,0.75);
            font-size: 0.95rem;
        }

        /* ── CONFIDENCE BAR ── */
        .conf-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }
        .conf-bar-bg {
            background: #e5e7eb;
            border-radius: 999px;
            height: 10px;
            width: 100%;
        }
        .conf-bar-fill {
            height: 10px;
            border-radius: 999px;
            transition: width 0.8s ease;
        }

        /* ── INSIGHT CHIPS ── */
        .insight-chip {
            display: inline-block;
            background: #f3e8ff;
            color: #6d28d9;
            border-radius: 999px;
            padding: 0.3rem 0.85rem;
            font-size: 0.82rem;
            margin: 0.25rem;
            font-weight: 500;
        }

        /* ── INPUT PANELS ── */
        .input-panel {
            background: #fafafa;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }

        /* ── FALLBACK NOTE ── */
        .fallback-note {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            font-size: 0.88rem;
            color: #78350f;
            margin-bottom: 1.5rem;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MODEL LOADING (with built-in fallback)
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "source_code" / "best_xgb_model.pkl"
SCALER_PATH = ROOT / "source_code" / "scaler.pkl"
COLUMNS_PATH = ROOT / "source_code" / "model_columns.json"


@st.cache_resource
def load_artifacts():
    """Try to load saved model artifacts; return None if not found."""
    model, scaler, columns = None, None, None
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
        if COLUMNS_PATH.exists():
            with open(COLUMNS_PATH) as f:
                columns = json.load(f)
    except Exception:
        pass
    return model, scaler, columns


# ─────────────────────────────────────────────
# PREPROCESSING — mirrors the notebook exactly
# ─────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "app_usage_time_min", "swipe_right_ratio", "likes_received",
    "mutual_matches", "profile_pics_count", "bio_length",
    "message_sent_count", "emoji_usage_rate", "last_active_hour",
    "interest_count", "profile_richness", "engagement_efficiency",
    "emoji_intensity",
]

CATEGORICAL_COLS = [
    "gender", "sexual_orientation", "location_type",
    "income_bracket", "education_level", "swipe_time_of_day",
]

ALL_INTERESTS = [
    "Cooking", "Fitness", "Gaming", "Hiking", "Movies",
    "Music", "Pets", "Photography", "Reading", "Sports",
    "Travel", "Yoga",
]

SUCCESSFUL_OUTCOMES = {"Date Happened", "Mutual Match", "Instant Match", "Relationship Formed"}


def engineer_features(raw: dict) -> pd.DataFrame:
    """Replicate the notebook's feature engineering on a single user dict."""
    df = pd.DataFrame([raw])

    # Engineered features
    df["interest_count"] = len(raw.get("selected_interests", []))
    df["profile_richness"] = df["bio_length"] + (df["profile_pics_count"] * 50)
    df["engagement_efficiency"] = np.where(
        df["likes_received"] > 0,
        df["mutual_matches"] / df["likes_received"],
        0,
    )
    df["emoji_intensity"] = df["emoji_usage_rate"] * df["message_sent_count"]

    # Interest tags → binary columns
    for interest in ALL_INTERESTS:
        df[interest] = 1 if interest in raw.get("selected_interests", []) else 0

    # One-hot encode categoricals (drop_first=True — mimics the notebook)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Drop helper / raw columns not used as features
    for col in ["selected_interests", "interest_tags", "match_outcome",
                "binary_match_outcome", "app_usage_time_label", "swipe_right_label"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def align_to_model_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Add missing dummy columns (= 0) and drop extras."""
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    return df[columns]


def fallback_predict(raw: dict) -> tuple[int, float, float]:
    """
    Rule-based heuristic used when the real model file is absent.
    Returns (prediction, prob_success, prob_fail).
    """
    score = 0.0
    score += min(raw["swipe_right_ratio"] * 0.3, 0.3)
    score += min(raw["engagement_efficiency"] * 2, 0.25)
    score += min(raw["app_usage_time_min"] / 300, 0.15)
    score += min(raw["interest_count"] / 12, 0.10)
    score += min((raw["profile_pics_count"] * 50 + raw["bio_length"]) / 600, 0.10)
    score += min(raw["message_sent_count"] / 200, 0.10)
    score = max(0.01, min(0.99, score))
    prediction = 1 if score >= 0.50 else 0
    return prediction, round(score, 4), round(1 - score, 4)


# ─────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────
def show():
    inject_css()

    # ── Hero ──
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Match Outcome Predictor</h1>
            <p>
                Fill in your dating-app profile below and our trained XGBoost model
                will predict whether you'll land a <strong>Successful Match</strong> or
                end up <strong>Ghosted / Unmatched</strong> — with a confidence score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load artifacts ──
    model, scaler, columns = load_artifacts()
    model_loaded = model is not None

    if not model_loaded:
        st.markdown(
            """
            <div class="fallback-note">
                ⚠️ <strong>Model files not found</strong> — running in <em>demo mode</em> using a
                rule-based heuristic. To use the real XGBoost model, place
                <code>best_xgb_model.pkl</code>, <code>scaler.pkl</code>, and
                <code>model_columns.json</code> in the same directory as this script.
                These are exported at the end of the Colab notebook.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════
    # INPUT FORM
    # ═══════════════════════════════════════════
    with st.form("predictor_form"):

        # ── Section 1: Demographics ──
        st.markdown('<p class="section-title">👤 Demographic Profile</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Non-binary", "Other"],
                help="Your gender identity",
            )
        with c2:
            sexual_orientation = st.selectbox(
                "Sexual Orientation",
                ["Straight", "Gay", "Lesbian", "Bisexual", "Other"],
            )
        with c3:
            location_type = st.selectbox(
                "Location Type",
                ["Urban", "Suburban", "Small Town", "Remote Area"],
            )

        c4, c5 = st.columns(2)
        with c4:
            income_bracket = st.selectbox(
                "Income Bracket",
                ["Low", "Medium", "High", "Very High"],
            )
        with c5:
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Bachelor's", "Master's", "MBA", "PhD", "Other"],
            )

        st.markdown("<hr style='margin:1.25rem 0;border-color:#f3e8ff'>", unsafe_allow_html=True)

        # ── Section 2: App Behaviour ──
        st.markdown('<p class="section-title">📱 App Usage & Behaviour</p>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            app_usage_time_min = st.slider(
                "Daily App Usage (min)", 0, 480, 90,
                help="How many minutes per day do you spend on the dating app?",
            )
        with b2:
            swipe_right_ratio = st.slider(
                "Swipe-Right Ratio", 0.0, 1.0, 0.45, step=0.01,
                help="Proportion of right swipes out of total swipes",
            )
        with b3:
            swipe_time_of_day = st.selectbox(
                "Peak Swipe Time",
                ["Morning", "Afternoon", "Evening", "Night"],
            )

        b4, b5, b6 = st.columns(3)
        with b4:
            likes_received = st.number_input("Likes Received", 0, 500, 45)
        with b5:
            mutual_matches = st.number_input("Mutual Matches", 0, 200, 12)
        with b6:
            last_active_hour = st.slider("Last Active Hour (0–23)", 0, 23, 21)

        st.markdown("<hr style='margin:1.25rem 0;border-color:#f3e8ff'>", unsafe_allow_html=True)

        # ── Section 3: Profile & Messaging ──
        st.markdown('<p class="section-title">✍️ Profile & Messaging</p>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            profile_pics_count = st.slider("Profile Photos", 1, 10, 4)
        with p2:
            bio_length = st.slider("Bio Length (chars)", 0, 500, 120)
        with p3:
            message_sent_count = st.slider("Messages Sent", 0, 500, 60)

        p4, p5 = st.columns(2)
        with p4:
            emoji_usage_rate = st.slider(
                "Emoji Usage Rate", 0.0, 1.0, 0.30, step=0.01,
                help="How often you use emojis in messages (0 = never, 1 = every message)",
            )
        with p5:
            selected_interests = st.multiselect(
                "Your Interests (select all that apply)",
                options=ALL_INTERESTS,
                default=["Music", "Travel", "Fitness"],
            )

        st.markdown("<hr style='margin:1.25rem 0;border-color:#f3e8ff'>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "💘  Predict My Match Outcome",
            use_container_width=True,
            type="primary",
        )

    # ═══════════════════════════════════════════
    # PREDICTION
    # ═══════════════════════════════════════════
    if submitted:
        raw = {
            "gender": gender,
            "sexual_orientation": sexual_orientation,
            "location_type": location_type,
            "income_bracket": income_bracket,
            "education_level": education_level,
            "app_usage_time_min": app_usage_time_min,
            "swipe_right_ratio": swipe_right_ratio,
            "swipe_time_of_day": swipe_time_of_day,
            "likes_received": likes_received,
            "mutual_matches": mutual_matches,
            "last_active_hour": last_active_hour,
            "profile_pics_count": profile_pics_count,
            "bio_length": bio_length,
            "message_sent_count": message_sent_count,
            "emoji_usage_rate": emoji_usage_rate,
            "selected_interests": selected_interests,
        }

        # Derived values needed for heuristic & display
        interest_count = len(selected_interests)
        engagement_eff = mutual_matches / likes_received if likes_received > 0 else 0.0
        raw["interest_count"] = interest_count
        raw["engagement_efficiency"] = engagement_eff

        with st.spinner("Running prediction model…"):
            if model_loaded:
                try:
                    df_input = engineer_features(raw)
                    if scaler is not None:
                        num_cols_present = [c for c in NUMERICAL_FEATURES if c in df_input.columns]
                        df_input[num_cols_present] = scaler.transform(df_input[num_cols_present])
                    if columns is not None:
                        df_input = align_to_model_columns(df_input, columns)
                    prediction = int(model.predict(df_input)[0])
                    proba = model.predict_proba(df_input)[0]
                    prob_success = round(float(proba[1]), 4)
                    prob_fail = round(float(proba[0]), 4)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}. Falling back to demo mode.")
                    prediction, prob_success, prob_fail = fallback_predict(raw)
            else:
                prediction, prob_success, prob_fail = fallback_predict(raw)

        # ── Result Card ──
        st.markdown("---")
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card result-success">
                    <div class="emoji">💚</div>
                    <div class="verdict">Successful Match Predicted!</div>
                    <div class="sub">Your profile signals high compatibility — keep swiping!</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-fail">
                    <div class="emoji">👻</div>
                    <div class="verdict">Ghosted / No Match Predicted</div>
                    <div class="sub">Some tweaks to your profile or behaviour might help — see tips below.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence Scores ──
        st.markdown('<p class="section-title">📊 Confidence Scores</p>', unsafe_allow_html=True)
        col_s, col_f = st.columns(2)

        with col_s:
            pct_s = int(prob_success * 100)
            bar_color = "#10b981" if prediction == 1 else "#6b7280"
            st.markdown(
                f"""
                <div class="conf-label">✅ Probability of Success</div>
                <div style="font-size:1.6rem;font-weight:600;color:{bar_color}">{pct_s}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct_s}%;background:{bar_color}"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_f:
            pct_f = int(prob_fail * 100)
            bar_color_f = "#f43f5e" if prediction == 0 else "#6b7280"
            st.markdown(
                f"""
                <div class="conf-label">👻 Probability of Ghost / No Match</div>
                <div style="font-size:1.6rem;font-weight:600;color:{bar_color_f}">{pct_f}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct_f}%;background:{bar_color_f}"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Feature Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">🔍 Your Profile at a Glance</p>', unsafe_allow_html=True)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("App Usage", f"{app_usage_time_min} min/day")
        g2.metric("Swipe-Right Ratio", f"{swipe_right_ratio:.0%}")
        g3.metric("Engagement Efficiency", f"{engagement_eff:.2%}")
        g4.metric("Interests Listed", interest_count)

        # ── Personalised Tips ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">💡 Personalised Insights</p>', unsafe_allow_html=True)

        tips = []
        if swipe_right_ratio < 0.3:
            tips.append("📌 Your swipe-right ratio is low — being a bit more open could increase match chances.")
        elif swipe_right_ratio > 0.85:
            tips.append("📌 Very high swipe-right ratio may dilute your matches — be more selective.")
        if engagement_eff < 0.05:
            tips.append("📌 Low engagement efficiency — try sending the first message more often.")
        if profile_pics_count < 3:
            tips.append("📌 Adding more profile photos (at least 4) significantly improves visibility.")
        if bio_length < 50:
            tips.append("📌 A longer, descriptive bio helps potential matches understand you better.")
        if interest_count < 3:
            tips.append("📌 Listing more interests broadens your appeal to potential matches.")
        if message_sent_count < 20:
            tips.append("📌 Initiating more conversations generally leads to more matches.")
        if app_usage_time_min < 15:
            tips.append("📌 Very low daily usage — more time on the app means more opportunities.")
        if not tips:
            tips.append("✨ Your profile looks well-optimised! Keep up the great engagement.")

        chips_html = "".join(f'<span class="insight-chip">{t}</span>' for t in tips)
        st.markdown(chips_html, unsafe_allow_html=True)

        # ── Export Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Export Prediction Summary (JSON)"):
            summary = {
                "prediction": "Successful Match" if prediction == 1 else "Ghosted / No Match",
                "probability_success": prob_success,
                "probability_fail": prob_fail,
                "model_used": "XGBoost (Tuned)" if model_loaded else "Demo Heuristic",
                "inputs": {
                    "gender": gender,
                    "sexual_orientation": sexual_orientation,
                    "location_type": location_type,
                    "income_bracket": income_bracket,
                    "education_level": education_level,
                    "app_usage_time_min": app_usage_time_min,
                    "swipe_right_ratio": swipe_right_ratio,
                    "likes_received": likes_received,
                    "mutual_matches": mutual_matches,
                    "profile_pics_count": profile_pics_count,
                    "bio_length": bio_length,
                    "message_sent_count": message_sent_count,
                    "emoji_usage_rate": emoji_usage_rate,
                    "interests": selected_interests,
                },
            }
            st.json(summary)
            st.download_button(
                "⬇️ Download Summary",
                data=json.dumps(summary, indent=2),
                file_name="match_prediction_summary.json",
                mime="application/json",
            )


# ─────────────────────────────────────────────
# MODULE API
# ─────────────────────────────────────────────
def render_predictor_page():
    show()


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    render_predictor_page()