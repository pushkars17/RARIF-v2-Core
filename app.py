from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PipelineConfig, PredictionInput
from data_generator import (
    RISK_LABELS,
    engineer_features,
    load_or_build_processed_df,
    month_to_season,
)
from db_store import RarifDB
from model import load_saved_bundle

CITY_PROFILES = {
    "Pune": {"state": "Maharashtra", "lat": 18.5204, "lon": 73.8567},
    "Mumbai": {"state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    "Chandigarh": {"state": "Punjab", "lat": 30.7333, "lon": 76.7794},
    "Chennai": {"state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    "Delhi": {"state": "Delhi", "lat": 28.6139, "lon": 77.2090},
    "Bangalore": {"state": "Karnataka", "lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"state": "Telangana", "lat": 17.3850, "lon": 78.4867},
    "Kolkata": {"state": "West Bengal", "lat": 22.5726, "lon": 88.3639},
}

TIER_COLORS = {
    "Low": "#16a34a",
    "Medium": "#d97706",
    "High": "#dc2626",
}


def inject_custom_css():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.35rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }
            .hero-banner {
                background: linear-gradient(120deg, #090d16 0%, #172554 55%, #0f766e 100%);
                color: #f8fafc;
                padding: 1.35rem 1.7rem;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.14);
                margin-bottom: 1.1rem;
            }
            .hero-title {
                font-size: 1.65rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0;
            }
            .hero-sub {
                color: #cbd5e1;
                font-size: 0.92rem;
                margin-top: 0.35rem;
            }
            .kpi-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(128, 128, 128, 0.18);
                border-radius: 12px;
                padding: 0.92rem 1.08rem;
            }
            .kpi-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                opacity: 0.72;
                font-weight: 700;
            }
            .kpi-value {
                font-size: 1.42rem;
                font-weight: 700;
                margin-top: 0.18rem;
            }
            .kpi-delta {
                font-size: 0.8rem;
                opacity: 0.78;
                margin-top: 0.14rem;
            }
            .risk-pill {
                display: inline-block;
                padding: 0.32rem 0.88rem;
                border-radius: 999px;
                font-weight: 700;
                font-size: 0.9rem;
                color: #ffffff;
                letter-spacing: 0.02em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_cached_model_bundle(artifact_path_str: str):
    return load_saved_bundle(Path(artifact_path_str))


@st.cache_data
def get_cached_processed_df(csv_path_str: str, cache_path_str: str, use_cache: bool = True) -> pd.DataFrame:
    cfg = PipelineConfig(
        data_csv_path=Path(csv_path_str),
        parquet_cache_path=Path(cache_path_str),
    )
    return load_or_build_processed_df(cfg, use_cache=use_cache)


def sync_duckdb_once(db: RarifDB, df_proc: pd.DataFrame, force: bool = False):
    """Syncs hotspot tables cleanly without causing DuckDB tuple deletion conflicts on rerun."""
    if force or not st.session_state.get("rarif_db_synced", False):
        try:
            db.sync_hotspots(df_proc)
            st.session_state["rarif_db_synced"] = True
            st.session_state.pop("rarif_db_warning", None)
        except Exception as exc:
            st.session_state["rarif_db_warning"] = str(exc)


def score_single_input(payload: PredictionInput, preprocessor, estimator) -> dict:
    row = pd.DataFrame([payload.model_dump()])
    row["date"] = f"2024-{payload.month:02d}-15"
    row["risk_score"] = 0.0

    feat_df = engineer_features(row)
    feat_df["season"] = month_to_season(payload.month)

    X_proc = preprocessor.transform(feat_df)
    pred_idx = int(estimator.predict(X_proc)[0])
    pred_label = RISK_LABELS[pred_idx]

    if hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(X_proc)[0], dtype=float)
    else:
        probs = np.zeros(3, dtype=float)
        probs[pred_idx] = 1.0

    expected_risk_index = float(
        np.clip((probs[0] * 17.0 + probs[1] * 50.0 + probs[2] * 85.0), 0.0, 100.0)
    )
    confidence = float(np.max(probs))

    return {
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "probs": probs,
        "confidence": confidence,
        "expected_risk_index": expected_risk_index,
    }


def compute_factor_sensitivity(
    base_payload: PredictionInput, base_index: float, preprocessor, estimator
) -> pd.DataFrame:
    shifts = [
        ("Visibility -> High", {"visibility": "high"}),
        ("Weather -> Clear", {"weather": "clear"}),
        ("Traffic Density -> Low", {"traffic_density": "low"}),
        ("Peak Hour -> Off-Peak (0)", {"is_peak_hour": 0}),
        ("Road Type -> Urban", {"road_type": "urban"}),
    ]
    rows = []
    for label, updates in shifts:
        alt = base_payload.model_copy(update=updates)
        alt_res = score_single_input(alt, preprocessor, estimator)
        rows.append(
            {
                "Factor Shift": label,
                "Index Reduction (pts)": round(base_index - alt_res["expected_risk_index"], 1),
            }
        )
    return pd.DataFrame(rows).sort_values("Index Reduction (pts)", ascending=True)


def build_safety_advisories(payload: PredictionInput, pred_label: str) -> list[str]:
    advisories = []
    if pred_label == "High":
        advisories.append(
            "High-risk corridor state: reduce speed below posted limits, avoid sudden lane shifts, and increase braking buffer."
        )
    elif pred_label == "Medium":
        advisories.append(
            "Moderate-risk state: keep steady lane discipline and watch for abrupt speed drops near merges or junctions."
        )
    else:
        advisories.append(
            "Low-risk baseline: maintain consistent speed, wear safety gear/seatbelts, and stay alert for local crossings."
        )

    if payload.visibility == "low" or payload.weather in ("fog", "rain"):
        advisories.append(
            f"Low visibility / {payload.weather} condition: use low-beam headlights or fog lamps (avoid high beam in fog) and widen following distance."
        )
    if payload.traffic_density == "high" or payload.is_peak_hour == 1:
        advisories.append(
            "Heavy traffic / peak-hour window: expect stop-and-go waves, two-wheeler filtering, and intersection congestion."
        )
    if payload.hour in (0, 1, 2, 3, 4, 5, 22, 23):
        advisories.append(
            "Late-night / early-morning hours: watch for driver fatigue, reduced lighting, and heavier freight/highway movement."
        )
    if payload.festival != "None":
        advisories.append(
            f"Festival period ({payload.festival}): anticipate crowds, local diversions, and irregular stopping."
        )
    return advisories


def render_risk_gauge(risk_index: float, pred_label: str) -> go.Figure:
    color = TIER_COLORS.get(pred_label, "#2563eb")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_index,
            number={"suffix": " / 100", "font": {"size": 28}},
            title={"text": f"Composite Risk Index ({pred_label})", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#f8fafc",
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.8,
                    "value": risk_index,
                },
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=45, b=15))
    return fig


def render_kpi_strip(df_proc: pd.DataFrame, model_name: str, macro_f1: float):
    total_records = len(df_proc)
    avg_risk = float(df_proc["risk_score"].mean())
    high_share = float((df_proc["risk_tier"] == 2).mean() * 100.0)
    top_city = (
        df_proc.groupby("city")["risk_score"].mean().sort_values(ascending=False).index[0]
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Filtered Records", f"{total_records:,}", "8 Indian Cities · 2022–2025"),
        (c2, "Mean Risk Score", f"{avg_risk:.3f}", f"High-Risk Share: {high_share:.1f}%"),
        (c3, "Highest Risk City", top_city, "By mean composite risk score"),
        (c4, "Active Model", model_name, f"Test Macro F1: {macro_f1:.4f}"),
    ]
    for col, label, val, sub in cards:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-delta">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(
        page_title="RARIF v2 | Road Accident Risk Intelligence",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🚦 RARIF Framework v2 — Road Accident Risk Intelligence</div>
            <div class="hero-sub">
                Pre-accident risk telemetry, what-if stress testing, DuckDB hotspot analytics, and SHAP explainability across Indian corridors
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = PipelineConfig()
    artifact_path = Path(cfg.model_artifact_path)

    if not artifact_path.exists():
        st.error("Model artifact `artifacts/best_model.joblib` nahi mila.")
        st.code("python main.py --model xgboost --trials 10", language="powershell")
        st.info("Upar wali command run karne ke baad is page ko refresh karein.")
        return

    saved = get_cached_model_bundle(str(artifact_path))
    estimator = saved["estimator"]
    preprocessor = saved["preprocessor"]
    model_name = saved["model_name"]
    macro_f1 = float(saved["metrics"]["macro_f1"])

    df_proc = get_cached_processed_df(
        str(cfg.data_csv_path), str(cfg.parquet_cache_path), use_cache=True
    )
    db = RarifDB(cfg.duckdb_path)
    sync_duckdb_once(db, df_proc, force=False)

    with st.sidebar:
        st.header("🎛️ Telemetry Filters")
        selected_cities = st.multiselect(
            "Cities",
            options=sorted(df_proc["city"].unique().tolist()),
            default=sorted(df_proc["city"].unique().tolist()),
        )
        selected_roads = st.multiselect(
            "Road Type",
            options=sorted(df_proc["road_type"].unique().tolist()),
            default=sorted(df_proc["road_type"].unique().tolist()),
        )
        selected_weather = st.multiselect(
            "Weather",
            options=sorted(df_proc["weather"].unique().tolist()),
            default=sorted(df_proc["weather"].unique().tolist()),
        )
        hour_range = st.slider("Hour Window", min_value=0, max_value=23, value=(0, 23))

        st.divider()
        st.caption("Quick Preset Scenarios")
        preset = st.radio(
            "Load Scenario",
            [
                "Custom Input",
                "🌧️ Monsoon Rush Hour (High Risk)",
                "🌫️ Late-Night Highway Fog",
                "☀️ Clear Off-Peak Urban Drive",
            ],
        )

        st.divider()
        if st.button("🔄 Re-Sync DuckDB Hotspots"):
            sync_duckdb_once(db, df_proc, force=True)
            st.success("DuckDB hotspots synced.")

        if st.session_state.get("rarif_db_warning"):
            st.caption(f"DB Note: {st.session_state['rarif_db_warning']}")

    filtered_df = df_proc[
        df_proc["city"].isin(selected_cities if selected_cities else df_proc["city"].unique())
        & df_proc["road_type"].isin(
            selected_roads if selected_roads else df_proc["road_type"].unique()
        )
        & df_proc["weather"].isin(
            selected_weather if selected_weather else df_proc["weather"].unique()
        )
        & (df_proc["hour"] >= hour_range[0])
        & (df_proc["hour"] <= hour_range[1])
    ].copy()

    if filtered_df.empty:
        filtered_df = df_proc.copy()

    render_kpi_strip(filtered_df, model_name, macro_f1)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_sim, tab_hotspots, tab_geo, tab_model = st.tabs(
        [
            "🎯 Risk Simulator & What-If",
            "📊 Corridor & Hourly Hotspots",
            "🗺️ Geospatial View",
            "🧠 Model Telemetry & SHAP",
        ]
    )

    # TAB 1: Live Predictor, What-If Comparison, Sensitivity & Safety Advisories
    with tab_sim:
        defaults = {
            "city": "Pune",
            "hour": 18,
            "month": 7,
            "day_of_week": "Friday",
            "road_type": "highway",
            "lanes": 3,
            "traffic_signal": 0,
            "traffic_density": "high",
            "is_peak_hour": 1,
            "weather": "rain",
            "visibility": "low",
            "temperature": 26,
            "cause": "overspeeding",
            "festival": "None",
        }
        if preset == "🌧️ Monsoon Rush Hour (High Risk)":
            defaults.update(
                {
                    "city": "Mumbai",
                    "hour": 19,
                    "month": 7,
                    "road_type": "highway",
                    "weather": "rain",
                    "visibility": "low",
                    "traffic_density": "high",
                    "is_peak_hour": 1,
                    "cause": "weather",
                }
            )
        elif preset == "🌫️ Late-Night Highway Fog":
            defaults.update(
                {
                    "city": "Delhi",
                    "hour": 3,
                    "month": 1,
                    "road_type": "highway",
                    "weather": "fog",
                    "visibility": "low",
                    "traffic_density": "medium",
                    "is_peak_hour": 0,
                    "cause": "overspeeding",
                }
            )
        elif preset == "☀️ Clear Off-Peak Urban Drive":
            defaults.update(
                {
                    "city": "Chandigarh",
                    "hour": 11,
                    "month": 4,
                    "road_type": "urban",
                    "weather": "clear",
                    "visibility": "high",
                    "traffic_density": "low",
                    "is_peak_hour": 0,
                    "cause": "distraction",
                }
            )

        left_col, right_col = st.columns([1.15, 1.0], gap="large")

        with left_col:
            st.markdown("#### Road & Environmental Parameters")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                city = st.selectbox(
                    "City",
                    list(CITY_PROFILES.keys()),
                    index=list(CITY_PROFILES.keys()).index(defaults["city"]),
                )
                hour = st.slider("Hour of Day (0–23)", 0, 23, defaults["hour"])
                month = st.slider("Month (1–12)", 1, 12, defaults["month"])
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_of_week = st.selectbox(
                    "Day of Week", days, index=days.index(defaults["day_of_week"])
                )
                road_types = ["highway", "urban", "rural"]
                road_type = st.selectbox(
                    "Road Type", road_types, index=road_types.index(defaults["road_type"])
                )
                lanes = st.slider("Lanes", 1, 8, defaults["lanes"])

            with r1c2:
                weathers = ["clear", "rain", "fog"]
                weather = st.selectbox("Weather", weathers, index=weathers.index(defaults["weather"]))
                vis_opts = ["high", "medium", "low"]
                visibility = st.selectbox(
                    "Visibility", vis_opts, index=vis_opts.index(defaults["visibility"])
                )
                dens_opts = ["low", "medium", "high"]
                traffic_density = st.selectbox(
                    "Traffic Density",
                    dens_opts,
                    index=dens_opts.index(defaults["traffic_density"]),
                )
                is_peak_hour = st.selectbox(
                    "Peak Hour Flag",
                    [0, 1],
                    index=defaults["is_peak_hour"],
                    format_func=lambda x: "Peak Hour (1)" if x == 1 else "Non-Peak (0)",
                )
                traffic_signal = st.selectbox(
                    "Traffic Signal",
                    [0, 1],
                    index=defaults["traffic_signal"],
                    format_func=lambda x: "Signal Present (1)" if x == 1 else "No Signal (0)",
                )
                temperature = st.slider("Temperature (°C)", 5, 48, defaults["temperature"])

            c_col1, c_col2 = st.columns(2)
            with c_col1:
                causes = ["overspeeding", "weather", "distraction", "drunk driving", "poor road"]
                cause = st.selectbox("Primary Risk Cause", causes, index=causes.index(defaults["cause"]))
            with c_col2:
                festivals = ["None", "Diwali", "Holi", "Eid", "New Year"]
                festival = st.selectbox(
                    "Festival Context", festivals, index=festivals.index(defaults["festival"])
                )

        with right_col:
            st.markdown("#### Live Risk Assessment")
            profile = CITY_PROFILES[city]
            is_weekend = 1 if day_of_week in ("Saturday", "Sunday") else 0

            payload = PredictionInput(
                city=city,
                state=profile["state"],
                latitude=profile["lat"],
                longitude=profile["lon"],
                month=month,
                hour=hour,
                day_of_week=day_of_week,
                is_weekend=is_weekend,
                is_peak_hour=is_peak_hour,
                lanes=lanes,
                traffic_signal=traffic_signal,
                temperature=temperature,
                road_type=road_type,
                weather=weather,
                visibility=visibility,
                traffic_density=traffic_density,
                cause=cause,
                festival=festival,
            )

            res = score_single_input(payload, preprocessor, estimator)
            badge_color = TIER_COLORS[res["pred_label"]]

            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.45rem;">
                    <span class="risk-pill" style="background:{badge_color};">
                        {res["pred_label"].upper()} RISK TIER
                    </span>
                    <span style="font-size:0.88rem; opacity:0.85;">Confidence: <strong>{res["confidence"]:.1%}</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                render_risk_gauge(res["expected_risk_index"], res["pred_label"]),
                use_container_width=True,
            )

            prob_df = pd.DataFrame(
                {
                    "Risk Tier": ["Low", "Medium", "High"],
                    "Probability": res["probs"],
                }
            )
            fig_prob = px.bar(
                prob_df,
                x="Risk Tier",
                y="Probability",
                color="Risk Tier",
                color_discrete_map=TIER_COLORS,
                range_y=[0, 1.08],
                text=prob_df["Probability"].map(lambda p: f"{p:.1%}"),
                title="Class Probability Distribution",
            )
            fig_prob.update_traces(textposition="outside")
            fig_prob.update_layout(height=240, showlegend=False, margin=dict(t=38, b=10))
            st.plotly_chart(fig_prob, use_container_width=True)

            safer_payload = payload.model_copy(
                update={
                    "weather": "clear",
                    "visibility": "high",
                    "traffic_density": "low",
                    "is_peak_hour": 0,
                }
            )
            safer_res = score_single_input(safer_payload, preprocessor, estimator)
            delta_idx = res["expected_risk_index"] - safer_res["expected_risk_index"]
            st.caption(
                f"What-If Baseline (Clear + High Visibility + Low Traffic + Off-Peak): "
                f"**{safer_res['pred_label']}** ({safer_res['expected_risk_index']:.1f}/100) · "
                f"Delta: **{delta_idx:+.1f} pts**"
            )

        st.divider()
        adv_col, sens_col = st.columns([1.05, 1.0], gap="large")
        with adv_col:
            st.markdown("#### Actionable Safety Advisories")
            for tip in build_safety_advisories(payload, res["pred_label"]):
                st.info(tip)

            export_doc = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "inputs": payload.model_dump(),
                "prediction": {
                    "risk_tier": res["pred_label"],
                    "composite_risk_index": round(res["expected_risk_index"], 2),
                    "confidence": round(res["confidence"], 4),
                    "probabilities": {
                        "Low": round(float(res["probs"][0]), 4),
                        "Medium": round(float(res["probs"][1]), 4),
                        "High": round(float(res["probs"][2]), 4),
                    },
                },
            }
            st.download_button(
                "⬇️ Download Scenario Report (JSON)",
                data=json.dumps(export_doc, indent=2),
                file_name=f"rarif_risk_{city.lower()}_{hour:02d}h.json",
                mime="application/json",
            )

        with sens_col:
            st.markdown("#### Factor Sensitivity (Shift to Calmer Baseline)")
            sens_df = compute_factor_sensitivity(
                payload, res["expected_risk_index"], preprocessor, estimator
            )
            fig_sens = px.bar(
                sens_df,
                x="Index Reduction (pts)",
                y="Factor Shift",
                orientation="h",
                color="Index Reduction (pts)",
                color_continuous_scale="Tealgrn",
            )
            fig_sens.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_sens, use_container_width=True)

    # TAB 2: Corridor & Hourly Hotspots
    with tab_hotspots:
        h1, h2 = st.columns(2)

        city_agg = (
            filtered_df.groupby(["city", "state"], as_index=False)
            .agg(
                accidents=("risk_score", "count"),
                avg_risk_score=("risk_score", "mean"),
                high_risk_pct=("risk_tier", lambda s: float((s == 2).mean() * 100.0)),
            )
            .sort_values("avg_risk_score", ascending=False)
        )

        hour_agg = (
            filtered_df.groupby("hour", as_index=False)
            .agg(
                accidents=("risk_score", "count"),
                avg_risk_score=("risk_score", "mean"),
                high_risk_pct=("risk_tier", lambda s: float((s == 2).mean() * 100.0)),
            )
            .sort_values("hour")
        )

        with h1:
            fig_city = px.bar(
                city_agg,
                x="city",
                y="avg_risk_score",
                color="high_risk_pct",
                color_continuous_scale="YlOrRd",
                title="Mean Risk Score by City",
                labels={"avg_risk_score": "Mean Risk Score", "high_risk_pct": "High Risk %"},
            )
            fig_city.update_layout(height=360)
            st.plotly_chart(fig_city, use_container_width=True)

        with h2:
            fig_hour = px.area(
                hour_agg,
                x="hour",
                y="avg_risk_score",
                markers=True,
                title="Hourly Risk Curve (0–23h)",
                labels={"avg_risk_score": "Mean Risk Score", "hour": "Hour of Day"},
            )
            fig_hour.update_layout(height=360)
            st.plotly_chart(fig_hour, use_container_width=True)

        c_low1, c_low2 = st.columns(2)
        with c_low1:
            road_weather = (
                filtered_df.groupby(["road_type", "weather"], as_index=False)["risk_score"]
                .mean()
                .pivot(index="road_type", columns="weather", values="risk_score")
            )
            fig_heat = px.imshow(
                road_weather,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Reds",
                title="Risk Intensity Heatmap: Road Type vs. Weather",
            )
            fig_heat.update_layout(height=320)
            st.plotly_chart(fig_heat, use_container_width=True)

        with c_low2:
            cause_agg = (
                filtered_df.groupby("cause", as_index=False)
                .agg(count=("risk_score", "count"), avg_risk=("risk_score", "mean"))
                .sort_values("avg_risk", ascending=True)
            )
            fig_cause = px.bar(
                cause_agg,
                x="avg_risk",
                y="cause",
                orientation="h",
                color="avg_risk",
                color_continuous_scale="Blues",
                title="Mean Risk Score by Primary Cause",
                labels={"avg_risk": "Mean Risk Score", "cause": "Cause"},
            )
            fig_cause.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_cause, use_container_width=True)

        st.markdown("#### City Hotspot Summary Table")
        st.dataframe(city_agg, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download City Summary (CSV)",
            data=city_agg.to_csv(index=False),
            file_name="rarif_city_hotspots.csv",
            mime="text/csv",
        )

    # TAB 3: Geospatial View
    with tab_geo:
        st.markdown("#### Spatial Distribution of Recorded Incidents Across Indian Cities")
        map_sample = filtered_df.sample(min(len(filtered_df), 2500), random_state=42).copy()
        map_sample["Risk Tier"] = map_sample["risk_tier"].map(RISK_LABELS)

        fig_map = px.scatter_geo(
            map_sample,
            lat="latitude",
            lon="longitude",
            color="Risk Tier",
            color_discrete_map=TIER_COLORS,
            hover_name="city",
            hover_data=["road_type", "weather", "visibility", "hour", "risk_score"],
            scope="asia",
            center={"lat": 21.5, "lon": 78.5},
            projection="natural earth",
            opacity=0.65,
            title="Incident Spatial Scatter (Sampled up to 2,500 filtered points)",
        )
        fig_map.update_geos(
            lataxis_range=[6.5, 35.5],
            lonaxis_range=[68.0, 92.0],
            showcountries=True,
            countrycolor="#94a3b8",
            showland=True,
            landcolor="#f1f5f9",
        )
        fig_map.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_map, use_container_width=True)

    # TAB 4: Model Telemetry & SHAP
    with tab_model:
        try:
            runs_df = db.get_model_runs()
        except Exception:
            runs_df = pd.DataFrame()

        if not runs_df.empty:
            st.markdown("#### Experiment & Run History (`rarif.duckdb`)")
            st.dataframe(runs_df, use_container_width=True, hide_index=True)

        shap_csv = Path(cfg.outputs_dir) / "top_shap_features.csv"
        if shap_csv.exists():
            shap_df = pd.read_csv(shap_csv).head(12)
            fig_shap = px.bar(
                shap_df.iloc[::-1],
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Top 12 SHAP Risk Drivers (Mean |SHAP|)",
                labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"},
            )
            fig_shap.update_layout(height=380)
            st.plotly_chart(fig_shap, use_container_width=True)

        img_c1, img_c2 = st.columns(2)
        shap_img = Path(cfg.outputs_dir) / "shap_summary.png"
        cm_img = Path(cfg.outputs_dir) / "confusion_matrix.png"

        with img_c1:
            if shap_img.exists():
                st.image(str(shap_img), caption="Saved SHAP Summary Chart", use_container_width=True)
        with img_c2:
            if cm_img.exists():
                st.image(str(cm_img), caption="Saved Confusion Matrix", use_container_width=True)


if __name__ == "__main__":
    main()