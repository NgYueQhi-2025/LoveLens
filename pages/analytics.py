from pathlib import Path

import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dating_app_behavior_dataset.csv"


def resolve_asset_path(relative_path: Path | str) -> Path:
	candidate = ROOT / Path(relative_path)
	if candidate.exists():
		return candidate
	if Path(relative_path).exists():
		return Path(relative_path)
	raise FileNotFoundError(f"Asset not found: {relative_path}")

PALETTE = {
	"pink": "#f05cc4",
	"violet": "#6f6fe9",
	"cyan": "#14b8c4",
	"gold": "#e1b11a",
	"red": "#ef4c3c",
	"dark": "#0b1020",
	"panel": "#0f1629",
	"text": "#e5e7eb",
	"muted": "#8b93a7",
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
	"""Convert hex color like '#rrggbb' to 'rgba(r,g,b,a)'."""
	h = hex_color.lstrip('#')
	if len(h) == 3:
		h = ''.join([c*2 for c in h])
	r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
	return f"rgba({r}, {g}, {b}, {alpha})"


def apply_theme() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
		.stApp {
			background:
				radial-gradient(circle at top left, rgba(240,92,196,0.14), transparent 28%),
				radial-gradient(circle at top right, rgba(111,111,233,0.12), transparent 25%),
				linear-gradient(180deg, #050816 0%, #070d19 38%, #08111e 100%);
			color: #e5e7eb;
			font-family: 'Manrope', 'Segoe UI', 'Aptos', sans-serif;
			font-size: 16px;
			line-height: 1.55;
			-webkit-font-smoothing: antialiased;
			text-rendering: optimizeLegibility;
		}

		.stApp::before {
			content: "";
			position: fixed;
			inset: 0;
			pointer-events: none;
			background-image:
				linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
				linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
			background-size: 96px 96px;
			mask-image: linear-gradient(180deg, rgba(0,0,0,0.6), transparent 92%);
		}

		.block-container {
			padding-top: 1.15rem;
			padding-bottom: 1.4rem;
			max-width: 1320px;
		}

		.hero-title {
			font-family: 'Space Grotesk', 'Manrope', sans-serif;
			font-size: clamp(2rem, 3vw, 3.1rem);
			font-weight: 700;
			letter-spacing: -0.05em;
			line-height: 0.98;
			margin: 0.2rem 0 0.7rem 0;
			color: #f8fafc;
		}

		.hero-copy {
			max-width: 940px;
			color: #b3bccf;
			font-size: 0.98rem;
			line-height: 1.7;
			font-weight: 500;
			margin-bottom: 0.9rem;
		}

		.eyebrow {
			display: inline-flex;
			gap: 0.45rem;
			align-items: center;
			padding: 0.42rem 0.8rem;
			border-radius: 999px;
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.78rem;
			font-weight: 800;
			letter-spacing: 0.12em;
			text-transform: uppercase;
			color: #ff7ed7;
			background: rgba(240,92,196,0.08);
			border: 1px solid rgba(240,92,196,0.16);
		}

		.feature-chip {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.78rem;
			font-weight: 700;
			letter-spacing: 0.01em;
			color: #dce2f0;
			background: rgba(255,255,255,0.05);
			border: 1px solid rgba(255,255,255,0.07);
		}

		.tab-note {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.92rem;
			font-weight: 500;
			color: #a7b0c4;
			line-height: 1.6;
		}

		.panel-title {
			font-family: 'Space Grotesk', 'Manrope', sans-serif;
			font-size: 1rem;
			font-weight: 700;
			letter-spacing: -0.02em;
			color: #f2f4f8;
			margin-bottom: 0.25rem;
		}

		.panel-subtitle {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.86rem;
			font-weight: 500;
			color: #8f98ad;
			margin-bottom: 0.85rem;
			line-height: 1.55;
		}

		.stat-label {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.75rem;
			font-weight: 800;
			letter-spacing: 0.08em;
			text-transform: uppercase;
			color: rgba(255,255,255,0.85);
		}

		.stat-value {
			font-family: 'Space Grotesk', 'Manrope', sans-serif;
			font-size: 1.7rem;
			font-weight: 700;
			letter-spacing: -0.04em;
			margin-top: 0.35rem;
		}

		.stat-note {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.76rem;
			font-weight: 600;
			color: rgba(255,255,255,0.78);
			margin-top: 0.35rem;
		}

		[data-testid="stMarkdownContainer"] p,
		[data-testid="stMarkdownContainer"] li {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.94rem;
			line-height: 1.65;
			color: #d9e0ef;
		}

		[data-testid="stMetricLabel"] {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			font-size: 0.75rem !important;
			font-weight: 700 !important;
			letter-spacing: 0.08em;
			text-transform: uppercase;
			color: #aeb6c9 !important;
		}

		[data-testid="stMetricValue"] {
			font-family: 'Space Grotesk', 'Manrope', sans-serif;
			font-size: 1.9rem !important;
			font-weight: 700 !important;
			letter-spacing: -0.05em;
			color: #f8fafc !important;
		}

		.stTabs [data-baseweb="tab-list"] {
			border-radius: 999px;
			padding: 0.2rem;
			gap: 0.25rem;
			background: rgba(8,12,22,0.76);
			border: 1px solid rgba(255,255,255,0.05);
		}

		.stTabs [data-baseweb="tab"] {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
			height: 2.5rem;
			padding: 0 1rem;
			border-radius: 999px;
			font-size: 0.82rem;
			font-weight: 700;
			letter-spacing: 0.01em;
			color: #97a0b4;
		}

		.stTabs [aria-selected="true"] {
			background: linear-gradient(135deg, rgba(240,92,196,0.18), rgba(111,111,233,0.20));
			color: #ffffff;
		}

		[data-testid="stDataFrame"] {
			border-radius: 18px;
			overflow: hidden;
			border: 1px solid rgba(255,255,255,0.06);
		}

		[data-testid="stDataFrame"] * {
			font-family: 'Manrope', 'Segoe UI', sans-serif;
		}

		[data-testid="stSidebar"] {
			background: linear-gradient(180deg, rgba(8,12,24,0.96), rgba(12,18,34,0.96));
			border-right: 1px solid rgba(255,255,255,0.05);
		}

		.hero-shell {
			position: relative;
			overflow: hidden;
			background: linear-gradient(135deg, rgba(16,24,43,0.9), rgba(8,12,24,0.92));
			border: 1px solid rgba(255,255,255,0.06);
			border-radius: 26px;
			padding: 20px 22px 18px 22px;
			box-shadow: 0 18px 48px rgba(0,0,0,0.30);
			margin-bottom: 1rem;
		}

		.hero-shell::after {
			content: "";
			position: absolute;
			inset: auto -80px -110px auto;
			width: 200px;
			height: 200px;
			background: radial-gradient(circle, rgba(240,92,196,0.34), transparent 70%);
			filter: blur(10px);
		}

                .bar-track {
                        background: rgba(255,255,255,0.06);
                        overflow: hidden;
                        margin-top: 0.55rem;
                }

                .bar-fill {
                        height: 100%;
                        border-radius: 999px;
                }

                .bar-head {
                        display: flex;
                        justify-content: space-between;
                        align-items: baseline;
                        gap: 1rem;
                        color: #dbe2f1;
                        font-weight: 700;
                        font-size: 0.92rem;
                }

                [data-testid="stMetric"] {
                        background: rgba(15,22,38,0.88);
                        border: 1px solid rgba(255,255,255,0.06);
                        border-radius: 16px;
                        padding: 0.65rem 0.8rem;
                        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
                }

                [data-testid="stMetricLabel"] {
                        color: #9ca5b9;
                        font-size: 0.75rem;
                }

                [data-testid="stMetricValue"] {
                        color: #f7fafc;
                        font-weight: 900;
                        letter-spacing: -0.03em;
                }

                </style>
		""",
		unsafe_allow_html=True,
	)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
	df = pd.read_csv(DATA_PATH)
	df["interest_count"] = (
		df["interest_tags"]
		.fillna("")
		.astype(str)
		.str.split(",")
		.apply(lambda items: sum(1 for item in items if item.strip()))
	)
	df["is_mutual_match"] = (df["match_outcome"].astype(str).str.strip().str.lower() == "mutual match").astype(int)
	return df


@st.cache_resource(show_spinner=False)
def load_model_assets():
	model = joblib.load(resolve_asset_path(Path("source_code") / "best_xgb_model.pkl"))
	scaler = joblib.load(resolve_asset_path(Path("source_code") / "scaler.pkl"))
	with open(resolve_asset_path(Path("source_code") / "model_columns.json"), "r") as f:
		model_columns = json.load(f)
	return model, scaler, model_columns


def prepare_model_frame(data: pd.DataFrame, scaler, model_columns: list[str]) -> pd.DataFrame:
	# Follow the original Notebook preprocessing so the produced feature matrix
	# matches the saved `model_columns.json` and `scaler.pkl` used at training time.
	frame = data.copy()

	# Drop redundant label columns if present
	columns_to_drop = ["app_usage_time_label", "swipe_right_label"]
	frame = frame.drop(columns=[c for c in columns_to_drop if c in frame.columns], errors="ignore")

	# Feature engineering (match Notebook.ipynb)
	frame["interest_count"] = frame["interest_tags"].fillna("").astype(str).str.split(',').apply(lambda items: sum(1 for item in items if str(item).strip()))
	frame["profile_richness"] = frame["bio_length"].fillna(0) + (frame["profile_pics_count"].fillna(0) * 50)
	frame["engagement_efficiency"] = np.where(frame.get("likes_received", 0) > 0, frame["mutual_matches"].fillna(0) / frame["likes_received"].fillna(1), 0)
	frame["emoji_intensity"] = frame["emoji_usage_rate"].fillna(0) * frame["message_sent_count"].fillna(0)

	# Expand interest tags into binary indicators (sep=', ' as in notebook when creating dummies)
	interests_expanded = frame["interest_tags"].fillna("").astype(str).str.get_dummies(sep=", ")
	df_encoded = pd.concat([frame.drop(columns=["interest_tags"], errors="ignore"), interests_expanded], axis=1)

	# One-hot encode categorical features exactly as the notebook
	categorical_cols = [
		"gender",
		"sexual_orientation",
		"location_type",
		"income_bracket",
		"education_level",
		"swipe_time_of_day",
	]
	df_final = pd.get_dummies(df_encoded, columns=[c for c in categorical_cols if c in df_encoded.columns], drop_first=True)

	# Ensure all model columns exist and in the right order
	combined = df_final.copy()
	for col in model_columns:
		if col not in combined.columns:
			combined[col] = 0
	combined = combined.reindex(columns=model_columns)

	# Numeric features list (same as notebook)
	numeric_columns = [
		'app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches',
		'profile_pics_count', 'bio_length', 'message_sent_count', 'emoji_usage_rate',
		'last_active_hour', 'interest_count', 'profile_richness', 'engagement_efficiency', 'emoji_intensity'
	]

	# Apply scaler (expecting provided scaler fitted during training)
	try:
		combined[numeric_columns] = scaler.transform(combined[numeric_columns].astype(float))
	except Exception:
		# If scaler fails (missing columns or None), skip scaling but preserve columns
		pass

	return combined


@st.cache_data(show_spinner=False)
def build_model_snapshot(data: pd.DataFrame) -> dict:
	model, scaler, model_columns = load_model_assets()
	feature_frame = prepare_model_frame(data, scaler, model_columns)
	y_true = (data["match_outcome"].astype(str).str.strip() == "Mutual Match").astype(int)
	y_pred = model.predict(feature_frame)

	feature_importances = pd.Series(model.feature_importances_, index=model_columns)
	grouped: dict[str, float] = {}
	for column_name, importance in feature_importances.items():
		base_name = column_name
		for prefix in ["gender_", "sexual_orientation_", "location_type_", "income_bracket_", "education_level_", "swipe_time_of_day_"]:
			if column_name.startswith(prefix):
				base_name = prefix[:-1]
				break
		if column_name in {"app_usage_time_min", "swipe_right_ratio", "likes_received", "mutual_matches", "profile_pics_count", "bio_length", "message_sent_count", "emoji_usage_rate", "last_active_hour", "interest_count", "profile_richness", "engagement_efficiency", "emoji_intensity"}:
			base_name = column_name
		elif base_name == column_name:
			base_name = "interest_tags"
		grouped[base_name] = grouped.get(base_name, 0.0) + float(importance)

	feature_df = (
		pd.DataFrame({"feature": list(grouped.keys()), "importance": list(grouped.values())})
		.sort_values("importance", ascending=False)
		.reset_index(drop=True)
	)

	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
		"f1": f1_score(y_true, y_pred, zero_division=0),
		"features": feature_df,
		"classes": model.classes_.tolist(),
		"target_positive_label": "Mutual Match",
	}


def figure_style(fig: plt.Figure) -> plt.Figure:
	fig.patch.set_facecolor(PALETTE["panel"])
	for axis in fig.axes:
		axis.set_facecolor(PALETTE["panel"])
		axis.tick_params(colors="#9ca5b9")
		for spine in axis.spines.values():
			spine.set_color("#1c2740")
	return fig


def build_donut_chart(title: str, series: pd.Series, colors: list[str]) -> go.Figure:
	fig = go.Figure(
		data=[
			go.Pie(
				labels=series.index.tolist(),
				values=series.values.tolist(),
				hole=0.62,
				textinfo="none",
				hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
				marker=dict(colors=colors[: len(series)], line=dict(color=PALETTE["dark"], width=2)),
				rotation=90,
			),
		]
	)
	fig.update_layout(
		title=dict(text=title, x=0.03, xanchor="left", font=dict(size=18, color="#eef2f7")),
		paper_bgcolor=PALETTE["panel"],
		plot_bgcolor=PALETTE["panel"],
		height=255,
		margin=dict(l=8, r=8, t=34, b=8),
		showlegend=False,
	)
	return fig


def render_donut(ax: plt.Axes, values: pd.Series, colors: list[str]) -> None:
	ax.pie(
		values.values,
		startangle=95,
		colors=colors[: len(values)],
		wedgeprops=dict(width=0.38, edgecolor=PALETTE["dark"], linewidth=2),
	)
	ax.set(aspect="equal")


def render_horizontal_bars(ax: plt.Axes, series: pd.Series, color: str) -> None:
	ordered = series.sort_values(ascending=True)
	ax.barh(ordered.index, ordered.values, color=color, alpha=0.92)
	ax.grid(axis="x", color="white", alpha=0.07, linewidth=0.8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)


def render_score_bar(label: str, value: float, color: str) -> None:
	st.markdown(
		f"""
		<div class="bar-head">
			<span>{label}</span>
			<span style="color: {color};">{value * 100:.1f}%</span>
		</div>
		<div class="bar-track">
			<div class="bar-fill" style="width: {value * 100:.1f}%; background: linear-gradient(90deg, {color}, rgba(255,255,255,0.15));"></div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_metric_cards(df: pd.DataFrame, snapshot: dict) -> None:
	unique_outcomes = df["match_outcome"].nunique()
	dominant_outcome = df["match_outcome"].value_counts().idxmax()
	dominant_share = df["match_outcome"].value_counts(normalize=True).max() * 100

	# Build a list of visually rich KPI cards (label, value, note, icon, gradient)
	metrics = [
		{
			"label": "Total Records",
			"value": f"{len(df):,}",
			"note": "Dataset samples",
			"gradient": ("#5b21b6", "#ec4899"),
		},
		{
			"label": "Features",
			"value": f"{len(df.columns)}",
			"note": "Input variables",
			"gradient": ("#0ea5a4", "#7c3aed"),
		},
		{
			"label": "Model Accuracy",
			"value": f"{snapshot['accuracy'] * 100:.1f}%",
			"note": "XGBoost snapshot",
			"gradient": ("#0f172a", "#0ea5a4"),
		},
		{
			"label": "Avg Swipe Ratio",
			"value": f"{df['swipe_right_ratio'].mean():.2f}",
			"note": "Behavioral engagement",
			"gradient": ("#f97316", "#ef4444"),
		},
		{
			"label": "Match Classes",
			"value": f"{unique_outcomes}",
			"note": "Outcome types",
			"gradient": ("#7c3aed", "#06b6d4"),
		},
		{
			"label": "Top Outcome",
			"value": f"{dominant_outcome}",
			"note": f"{dominant_share:.1f}% of records",
			"gradient": ("#111827", "#4f46e5"),
		},
	]

	cols = st.columns(len(metrics), gap="small")
	for col, metric in zip(cols, metrics):
		grad_a, grad_b = metric["gradient"]
		rgba_a = hex_to_rgba(grad_a, 0.4)
		rgba_b = hex_to_rgba(grad_b, 0.4)
		label = metric["label"]
		value = metric["value"]
		note = metric["note"]
		html = f"""
		<div style="border-radius:14px;padding:14px;background:linear-gradient(135deg,{rgba_a}, {rgba_b});color:rgba(255,255,255,0.98);box-shadow:0 10px 30px rgba(2,6,23,0.6);">
			<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
				<div style="font-size:0.84rem;font-weight:700;opacity:0.95;">{label}</div>
			</div>
			<div style="font-size:1.65rem;font-weight:900;margin-top:8px;line-height:1;">{value}</div>
			<div style="font-size:0.78rem;opacity:0.92;margin-top:6px;color:rgba(255,255,255,0.92);">{note}</div>
		</div>
		"""
		with col:
			st.markdown(html, unsafe_allow_html=True)


def render_analytics_page() -> None:
	apply_theme()
	df = load_data()
	snapshot = build_model_snapshot(df)

	highlights = [
		"EDA charts",
		"Graphs",
		"Correlation heatmap",
		"Dataset insights",
		"Trend analysis",
	]

	st.markdown(
		"""
		<div class="hero-shell">
			<div class="hero-title">Visualization Analytics</div>
			<div class="hero-copy">
				Comprehensive exploratory data analysis and machine learning insights for the dating app behavior dataset
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	st.markdown(
		"<div class='chip-row'>" + "".join(
			f"<span class='feature-chip'>• {item}</span>" for item in highlights
		) + "</div>",
		unsafe_allow_html=True,
	)

	st.write("")
	render_metric_cards(df, snapshot)
	st.write("")

	tabs = st.tabs(["Overview", "EDA", "Trends", "Correlation", "Dataset"])

	gender_counts = df["gender"].value_counts()
	match_binary = np.where(df["match_outcome"].astype(str).str.strip().eq("Mutual Match"), "Successful", "Not Successful")
	binary_outcome_counts = pd.Series(match_binary).value_counts()
	education_order = list(df["education_level"].dropna().value_counts().index)
	swipe_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Late Night", "After Midnight"]

	with tabs[0]:
		st.markdown("<div class='tab-note'>A compact overview of who is using the app, what outcomes are most common, and how the model performs on a realistic training snapshot.</div>", unsafe_allow_html=True)
		col_a, col_b = st.columns([0.92, 0.92])
		with col_a:
			st.markdown("<div class='panel-title'>Gender Distribution</div><div class='panel-subtitle'>Breakdown of user gender identities</div>", unsafe_allow_html=True)
			fig = build_donut_chart(
				"",
				gender_counts,
				[PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"], PALETTE["gold"], PALETTE["red"], "#1f2937"],
			)
			st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
		with col_b:
			st.markdown("<div class='panel-title'>Match Outcome Distribution</div><div class='panel-subtitle'>Results of user matching interactions across outcome categories</div>", unsafe_allow_html=True)
			outcome_counts = df["match_outcome"].value_counts().head(10)
			fig = build_donut_chart(
				"",
				outcome_counts,
				[PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"], PALETTE["gold"], PALETTE["red"], "#93c5fd", "#1f2937", "#7c2d12", "#4ade80", "#a855f7"],
			)
			st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

		st.write("")
		col_c, col_d = st.columns([1, 1])
		with col_c:
			st.markdown("<div class='panel-title'>Best Model Metrics</div><div class='panel-subtitle'>Random Forest classifier performance breakdown</div>", unsafe_allow_html=True)
			render_score_bar("Accuracy", snapshot["accuracy"], PALETTE["pink"])
			st.write("")
			render_score_bar("Precision", snapshot["precision"], PALETTE["violet"])
			st.write("")
			render_score_bar("Recall", snapshot["recall"], PALETTE["cyan"])
			st.write("")
			render_score_bar("F1 Score", snapshot["f1"], PALETTE["gold"])
			st.write("")
			inner_a, inner_b = st.columns(2)
			with inner_a:
				st.metric("Overall Accuracy", f"{snapshot['accuracy'] * 100:.1f}%")
			with inner_b:
				st.metric("Macro F1", f"{snapshot['f1']:.2f}")
		with col_d:
			st.markdown("<div class='panel-title'>Dataset Snapshot</div><div class='panel-subtitle'>High-level insights extracted from the raw records</div>", unsafe_allow_html=True)
			summary_rows = [
				("Dominant gender", gender_counts.idxmax(), f"{gender_counts.max() / len(df) * 100:.1f}%"),
				("Most common outcome", outcome_counts.idxmax(), f"{outcome_counts.max() / len(df) * 100:.1f}%"),
				("Average app usage", f"{df['app_usage_time_min'].mean():.0f} min", "per user"),
				("Average messages", f"{df['message_sent_count'].mean():.1f}", "per user"),
				("Average active hour", f"{df['last_active_hour'].mean():.1f}", "24-hour clock"),
			]
			for label, value, note in summary_rows:
				st.markdown(
					f"""
					<div style="display:flex;justify-content:space-between;align-items:baseline;padding:0.55rem 0;border-bottom:1px solid rgba(255,255,255,0.06);gap:1rem;">
						<span style="color:#b2bbcd;font-weight:600;">{label}</span>
						<span style="color:#f8fafc;font-weight:800;">{value}</span>
					</div>
					<div style="color:#7f889b;font-size:0.82rem;margin-top:-0.2rem;margin-bottom:0.25rem;">{note}</div>
					""",
					unsafe_allow_html=True,
				)

	with tabs[1]:
		st.markdown("<div class='tab-note'>EDA charts focus on behavior and demographic slices to expose how engagement varies across user groups.</div>", unsafe_allow_html=True)
		left, right = st.columns(2)
		with left:
				# Build stacked percentage bars of swipe behavior by education
				# Determine the top 3 swipe behavior labels and group others as 'Other'
				sr = df["swipe_right_label"].fillna("(unknown)")
				top_labels = sr.value_counts().nlargest(3).index.tolist()
				labels_order = top_labels + [l for l in ["Other"] if l not in top_labels]

				edu = df.copy()
				edu["swipe_group"] = edu["swipe_right_label"].fillna("(unknown)").apply(lambda v: v if v in top_labels else "Other")
				edu["education_level"] = pd.Categorical(edu["education_level"], categories=education_order, ordered=True)
				pivot = (
					edu.groupby(["education_level", "swipe_group"]).size().unstack(fill_value=0).reindex(education_order).fillna(0)
				)
				# convert to percentage per education level
				pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100

				fig, ax = plt.subplots(figsize=(7.6, 4.6))
				stack_colors = [PALETTE["pink"], PALETTE["cyan"], PALETTE["red"], "#374151"]
				bottom = np.zeros(len(pivot_pct.index))
				for i, col in enumerate(pivot_pct.columns):
					vals = pivot_pct[col].values
					ax.bar(pivot_pct.index, vals, bottom=bottom, color=stack_colors[i % len(stack_colors)], label=col, width=0.72)
					bottom += vals
				ax.set_title("Swipe Behavior by Education", loc="left", fontsize=16, fontweight="800", color="#eef2f7", pad=14)
				ax.set_ylabel("% of users", color="#9ca5b9")
				ax.set_ylim(0, 60)
				ax.tick_params(axis="x", rotation=28, labelcolor="#9ca5b9")
				ax.grid(axis="y", color="white", alpha=0.05)
				ax.legend(frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.18))
				figure_style(fig)
				st.pyplot(fig, use_container_width=True)
				plt.close(fig)
		with right:
			location_summary = (
				df.groupby("location_type")["mutual_matches"]
				.agg(["count", "sum"]) 
				.rename(columns={"count": "users", "sum": "matches"})
				.sort_values("users", ascending=False)
			)
			fig, ax = plt.subplots(figsize=(7.6, 4.6))
			x = np.arange(len(location_summary.index))
			width = 0.38
			ax.bar(x - width / 2, location_summary["users"], width, color=PALETTE["violet"], label="users")
			ax.bar(x + width / 2, location_summary["matches"], width, color=PALETTE["pink"], label="matches")
			ax.set_xticks(x)
			ax.set_xticklabels(location_summary.index, color="#9ca5b9")
			ax.set_title("Location Type Analysis", loc="left", fontsize=16, fontweight="800", color="#eef2f7", pad=14)
			ax.set_ylabel("Count", color="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.05)
			ax.legend(frameon=False, labelcolor="#cdd6e6")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

		st.write("")
		lower_left, lower_right = st.columns(2)
		with lower_left:
			interest_tags = (
				df["interest_tags"]
				.dropna()
				.astype(str)
				.str.split(",")
				.explode()
				.str.strip()
				.replace("", np.nan)
				.dropna()
			)
			top_interests = interest_tags.value_counts().head(8)
			fig, ax = plt.subplots(figsize=(6.4, 4.2))
			render_horizontal_bars(ax, top_interests, PALETTE["cyan"])
			ax.set_title("Top Interest Tags", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_xlabel("Frequency", color="#9ca5b9")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with lower_right:
			match_time = (
				df.groupby("swipe_time_of_day")["match_outcome"]
				.count()
				.reindex(swipe_order)
				.dropna()
			)
			fig, ax = plt.subplots(figsize=(6.4, 4.2))
			ax.bar(match_time.index, match_time.values, color=PALETTE["gold"], alpha=0.9)
			ax.set_title("Match Activity by Swipe Time", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_ylabel("Records", color="#9ca5b9")
			ax.tick_params(axis="x", rotation=25, labelcolor="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.08)
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

	with tabs[2]:
		st.markdown("<div class='tab-note'>Trend analysis highlights how engagement intensifies during different hours and interaction phases.</div>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)
		with col1:
			hour_summary = (
				df.groupby("last_active_hour")
				.agg({"swipe_right_ratio": "mean", "mutual_matches": "mean", "message_sent_count": "mean"})
				.sort_index()
			)
			fig, ax = plt.subplots(figsize=(6.8, 4.2))
			ax.plot(hour_summary.index, hour_summary["swipe_right_ratio"] * 100, color=PALETTE["pink"], linewidth=2.6, marker="o", label="Swipe ratio")
			ax.plot(hour_summary.index, hour_summary["mutual_matches"], color=PALETTE["violet"], linewidth=2.6, marker="s", label="Mutual matches")
			ax.set_title("Engagement Trend by Hour", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_xlabel("Last active hour", color="#9ca5b9")
			ax.set_ylabel("Normalised values", color="#9ca5b9")
			ax.grid(color="white", alpha=0.08)
			ax.legend(frameon=False, labelcolor="#cdd6e6")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with col2:
			time_map = (
				df.groupby("swipe_time_of_day")
				.agg({"app_usage_time_min": "mean", "likes_received": "mean"})
				.reindex(swipe_order)
				.dropna()
			)
			fig, ax = plt.subplots(figsize=(6.8, 4.2))
			x = np.arange(len(time_map.index))
			ax.bar(x - 0.18, time_map["app_usage_time_min"], width=0.36, color=PALETTE["cyan"], label="App usage")
			ax.bar(x + 0.18, time_map["likes_received"], width=0.36, color=PALETTE["gold"], label="Likes received")
			ax.set_xticks(x)
			ax.set_xticklabels(time_map.index, color="#9ca5b9", rotation=25)
			ax.set_title("Time-of-Day Interaction Mix", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.set_ylabel("Average value", color="#9ca5b9")
			ax.grid(axis="y", color="white", alpha=0.08)
			ax.legend(frameon=False, labelcolor="#cdd6e6")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)

	with tabs[3]:
		st.markdown("<div class='tab-note'>Correlation explores how strongly behavior, activity, and profile richness move together.</div>", unsafe_allow_html=True)
		corr_cols = [
			"app_usage_time_min",
			"swipe_right_ratio",
			"likes_received",
			"mutual_matches",
			"profile_pics_count",
			"bio_length",
			"message_sent_count",
			"emoji_usage_rate",
			"last_active_hour",
			"interest_count",
		]
		corr = df[corr_cols].corr(numeric_only=True)
		col_left, col_right = st.columns([1.05, 0.95])
		with col_left:
			fig, ax = plt.subplots(figsize=(8.0, 5.4))
			sns.heatmap(
				corr,
				ax=ax,
				cmap=sns.color_palette([PALETTE["dark"], PALETTE["pink"], PALETTE["violet"], PALETTE["cyan"]], as_cmap=True),
				vmin=-1,
				vmax=1,
				center=0,
				square=True,
				linewidths=0.6,
				linecolor="#111826",
				cbar_kws={"shrink": 0.75},
			)
			ax.set_title("Feature Correlation Heatmap", loc="left", fontsize=14, fontweight="bold", color="#eef2f7", pad=16)
			ax.tick_params(axis="x", labelrotation=35, colors="#9ca5b9")
			ax.tick_params(axis="y", colors="#9ca5b9")
			figure_style(fig)
			st.pyplot(fig, use_container_width=True)
			plt.close(fig)
		with col_right:
			corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
			strongest = corr_abs.stack().sort_values(ascending=False).head(6)
			st.markdown("<div class='panel-title'>Strongest Relationships</div><div class='panel-subtitle'>Most connected feature pairs in the dataset</div>", unsafe_allow_html=True)
			for (left, right), value in strongest.items():
				st.markdown(
					f"""
					<div style="margin-bottom:0.95rem;">
						<div class="bar-head"><span>{left} • {right}</span><span>{value:.2f}</span></div>
						<div class="bar-track"><div class="bar-fill" style="width:{value * 100:.1f}%; background: linear-gradient(90deg, {PALETTE['violet']}, {PALETTE['pink']});"></div></div>
					</div>
					""",
					unsafe_allow_html=True,
				)

	with tabs[4]:
		st.markdown("<div class='tab-note'>Dataset insights combine schema checks, descriptive statistics, and a quick preview of the raw rows.</div>", unsafe_allow_html=True)
		col1, col2 = st.columns([1.05, 0.95])
		with col1:
			st.markdown("<div class='panel-title'>Data Quality Snapshot</div><div class='panel-subtitle'>A quick view of completeness and structure</div>", unsafe_allow_html=True)
			missing = df.isna().sum()
			quality = pd.DataFrame({
				"column": missing.index,
				"missing": missing.values,
				"dtype": df.dtypes.astype(str).values,
			}).sort_values(["missing", "column"], ascending=[False, True])
			st.dataframe(quality, use_container_width=True, height=350)
		with col2:
			st.markdown("<div class='panel-title'>Descriptive Statistics</div><div class='panel-subtitle'>Central tendency for the core behavior variables</div>", unsafe_allow_html=True)
			numeric_frame = df[[
				"app_usage_time_min",
				"swipe_right_ratio",
				"likes_received",
				"mutual_matches",
				"profile_pics_count",
				"bio_length",
				"message_sent_count",
				"emoji_usage_rate",
				"last_active_hour",
				"interest_count",
			]]
			st.dataframe(numeric_frame.describe().T, use_container_width=True, height=350)

		st.write("")
		st.markdown("<div class='panel-title'>Dataset Preview</div><div class='panel-subtitle'>A slice of the raw records used across the dashboard</div>", unsafe_allow_html=True)
		st.dataframe(df.head(12), use_container_width=True, height=330)


if __name__ == "__main__":
	st.set_page_config(
		page_title="Data Insights Dashboard",
		page_icon="📊",
		layout="wide",
		initial_sidebar_state="expanded",
	)
	render_analytics_page()
