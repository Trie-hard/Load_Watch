"""LoadWatch Streamlit dashboard — workload-based soft-tissue injury risk scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import FRIENDLY_NAMES, factors_for_row
from src.utils import DATA_PROCESSED, MODELS_DIR, PRIMARY_SEASON

st.set_page_config(
    page_title="LoadWatch — NBA Injury Risk",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEASON = PRIMARY_SEASON


@st.cache_data(show_spinner=False)
def load_scored(season: str = SEASON) -> pd.DataFrame:
    path = DATA_PROCESSED / f"scored_{season}.parquet"
    csv_path = DATA_PROCESSED / f"scored_{season}.csv"
    if path.exists():
        df = pd.read_parquet(path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return pd.DataFrame()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE")


@st.cache_data(show_spinner=False)
def load_injuries(season: str = SEASON) -> pd.DataFrame:
    path = DATA_PROCESSED / f"injuries_clean_{season}.parquet"
    csv_path = DATA_PROCESSED / f"injuries_clean_{season}.csv"
    if path.exists():
        df = pd.read_parquet(path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return pd.DataFrame()
    df["injury_date"] = pd.to_datetime(df["injury_date"])
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .block-container { padding-top: 1.2rem; max-width: 1200px; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
        .lw-hero {
            background: linear-gradient(135deg, #0b1f33 0%, #143d5c 55%, #1a5f7a 100%);
            color: #f4f7fb;
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .lw-hero h1 { margin: 0; font-size: 1.9rem; color: #fff !important; }
        .lw-hero p { margin: 0.35rem 0 0; opacity: 0.85; }
        .risk-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .risk-High { background: #ffe1e1; color: #9b1c1c; }
        .risk-Moderate { background: #fff3d6; color: #8a5a00; }
        .risk-Low { background: #e3f6ea; color: #146c2e; }
        div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_gauge(score: float, band: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "", "font": {"size": 36}},
            title={"text": f"Current risk · {band}", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1a5f7a"},
                "steps": [
                    {"range": [0, 60], "color": "#d9f2e3"},
                    {"range": [60, 85], "color": "#ffe8b8"},
                    {"range": [85, 100], "color": "#ffc9c9"},
                ],
                "threshold": {
                    "line": {"color": "#9b1c1c", "width": 3},
                    "thickness": 0.75,
                    "value": 85,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def risk_trend_chart(player_df: pd.DataFrame, injuries: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=player_df["GAME_DATE"],
            y=player_df["risk_score"],
            mode="lines+markers",
            name="Risk score",
            line=dict(color="#1a5f7a", width=2.5),
            marker=dict(size=6, color="#1a5f7a"),
            hovertemplate="%{x|%b %d}: %{y:.1f}<extra>Risk</extra>",
        )
    )
    # Mark games
    fig.add_trace(
        go.Scatter(
            x=player_df["GAME_DATE"],
            y=player_df["MIN_FLOAT"],
            name="Minutes",
            yaxis="y2",
            mode="markers",
            marker=dict(size=7, color="rgba(20,60,90,0.25)"),
            hovertemplate="%{x|%b %d}: %{y:.1f} min<extra></extra>",
        )
    )

    if not injuries.empty:
        inj = injuries.copy()
        # Overlay injury dates near the risk curve
        for _, row in inj.iterrows():
            fig.add_vline(x=row["injury_date"], line_width=1, line_dash="dot", line_color="#c0392b", opacity=0.7)
        fig.add_trace(
            go.Scatter(
                x=inj["injury_date"],
                y=[player_df["risk_score"].max() * 0.95] * len(inj),
                mode="markers",
                name="Soft-tissue injury",
                marker=dict(symbol="x", size=12, color="#c0392b", line=dict(width=2)),
                hovertemplate="%{x|%b %d}<br>%{text}<extra>Injury</extra>",
                text=inj.get("injury_type", pd.Series([""] * len(inj))).astype(str).str.slice(0, 80),
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title=None,
        yaxis=dict(title="Risk score (0–100)", range=[0, 100]),
        yaxis2=dict(title="Minutes", overlaying="y", side="right", showgrid=False, range=[0, 50]),
        hovermode="x unified",
        plot_bgcolor="#f7fafc",
    )
    return fig


def shap_bar(factors: list[dict]) -> go.Figure:
    if not factors:
        return go.Figure()
    labels = [f["label"] for f in factors][::-1]
    values = [f["shap"] * 10 for f in factors][::-1]
    colors = ["#c0392b" if v > 0 else "#1a7a4c" for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=40, t=20, b=10),
        xaxis_title="Contribution to risk (display pts)",
        yaxis_title=None,
        plot_bgcolor="#f7fafc",
    )
    return fig


def workload_heatmap(player_df: pd.DataFrame, days: int = 30) -> go.Figure:
    if player_df.empty:
        return go.Figure()
    end = player_df["GAME_DATE"].max()
    start = end - pd.Timedelta(days=days - 1)
    window = player_df[(player_df["GAME_DATE"] >= start) & (player_df["GAME_DATE"] <= end)][
        ["GAME_DATE", "MIN_FLOAT"]
    ].copy()
    all_days = pd.DataFrame({"GAME_DATE": pd.date_range(start, end, freq="D")})
    cal = all_days.merge(window, on="GAME_DATE", how="left").fillna({"MIN_FLOAT": 0})
    cal["week"] = ((cal["GAME_DATE"] - start).dt.days // 7).astype(int)
    cal["dow"] = cal["GAME_DATE"].dt.dayofweek  # Mon=0
    cal["label"] = cal["GAME_DATE"].dt.strftime("%b %d")

    pivot = cal.pivot(index="dow", columns="week", values="MIN_FLOAT")
    text = cal.pivot(index="dow", columns="week", values="label")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"W{c+1}" for c in pivot.columns],
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=[[0, "#eef3f7"], [0.5, "#7eb6c9"], [1, "#0b1f33"]],
            hovertemplate="%{customdata}<br>%{z:.1f} min<extra></extra>",
            customdata=text.values,
            colorbar=dict(title="Min"),
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), title=f"Minutes · last {days} days")
    return fig


def team_overview(df: pd.DataFrame, team: str) -> pd.DataFrame:
    team_df = df[df["TEAM_NAME"] == team].copy() if "TEAM_NAME" in df.columns else df[df["TEAM_ABBREVIATION"] == team].copy()
    latest = (
        team_df.sort_values("GAME_DATE")
        .groupby("PLAYER_ID", as_index=False)
        .tail(1)
        .sort_values("risk_score", ascending=False)
    )
    cols = [c for c in ["PLAYER_NAME", "risk_score", "risk_band", "MIN_FLOAT", "minutes_last_7d", "rest_days", "top_factor_1"] if c in latest.columns]
    return latest[cols]


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="lw-hero">
          <h1>LoadWatch</h1>
          <p>Workload-based soft-tissue injury <b>risk scores</b> for NBA players — not binary injury predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_scored()
    injuries = load_injuries()
    metrics = load_metrics()

    if df.empty:
        st.error(
            "No scored data found. From the project root run:\n\n"
            "`python scripts/run_pipeline.py`\n\n"
            "Then restart this dashboard."
        )
        st.stop()

    team_col = "TEAM_NAME" if "TEAM_NAME" in df.columns and df["TEAM_NAME"].notna().any() else "TEAM_ABBREVIATION"
    teams = sorted(df[team_col].dropna().unique().tolist())

    with st.sidebar:
        st.header("Roster filter")
        page = st.radio("View", ["Player detail", "Team risk overview"], label_visibility="collapsed")
        team = st.selectbox("Team", teams, index=0)
        team_players = (
            df[df[team_col] == team][["PLAYER_ID", "PLAYER_NAME"]]
            .drop_duplicates()
            .sort_values("PLAYER_NAME")
        )
        player_name = st.selectbox("Player", team_players["PLAYER_NAME"].tolist())
        player_id = int(team_players.loc[team_players["PLAYER_NAME"] == player_name, "PLAYER_ID"].iloc[0])
        st.caption(f"Season {SEASON} · {df['PLAYER_ID'].nunique()} players scored")
        st.markdown("---")
        st.markdown(
            "Risk bands use season percentile ranks:  \n"
            "**High** ≥ 85th · **Moderate** 60–85 · **Low** < 60"
        )

    player_df = df[df["PLAYER_ID"] == player_id].sort_values("GAME_DATE")
    latest = player_df.iloc[-1]
    player_inj = injuries[injuries["player_id"] == player_id] if not injuries.empty else injuries

    if page == "Team risk overview":
        st.subheader(f"{team} — current risk ranking")
        overview = team_overview(df, team)
        st.dataframe(
            overview.rename(
                columns={
                    "PLAYER_NAME": "Player",
                    "risk_score": "Risk",
                    "risk_band": "Band",
                    "MIN_FLOAT": "Last min",
                    "minutes_last_7d": "Min (7d)",
                    "rest_days": "Rest days",
                    "top_factor_1": "Top factor",
                }
            ),
            width='stretch',
            hide_index=True,
        )
        fig = px.bar(
            overview.head(12),
            x="risk_score",
            y="PLAYER_NAME",
            color="risk_band",
            color_discrete_map={"High": "#c0392b", "Moderate": "#d4a017", "Low": "#1a7a4c"},
            orientation="h",
            labels={"risk_score": "Risk score", "PLAYER_NAME": ""},
        )
        fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')
    else:
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            st.markdown(
                f"<span class='risk-badge risk-{latest['risk_band']}'>{latest['risk_band']} risk</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(risk_gauge(latest["risk_score"], latest["risk_band"]), width='stretch')
        with c2:
            st.metric("Risk score", f"{latest['risk_score']:.1f}")
            st.metric("Minutes (last game)", f"{latest['MIN_FLOAT']:.1f}")
        with c3:
            st.metric("Minutes last 7d", f"{latest['minutes_last_7d']:.1f}")
            st.metric("Games last 7d", f"{int(latest['games_last_7d'])}")
        with c4:
            st.metric("Rest days", f"{int(latest['rest_days'])}")
            b2b = "Yes" if int(latest["is_back_to_back"]) == 1 else "No"
            st.metric("Back-to-back", b2b)

        st.plotly_chart(risk_trend_chart(player_df, player_inj), width='stretch')

        left, right = st.columns(2)
        with left:
            st.subheader("What's driving this score?")
            factors = factors_for_row(latest.get("top_factors_json", ""))
            if not factors and latest.get("top_factor_1"):
                factors = [{"label": latest["top_factor_1"], "shap": 0.1}]
            if factors:
                for f in factors[:5]:
                    st.write(f"• {f.get('display', f.get('label', ''))}")
                st.plotly_chart(shap_bar(factors[:5]), width='stretch')
            else:
                st.info("SHAP factors not found — re-run `python scripts/run_pipeline.py`.")
        with right:
            st.subheader("Workload calendar")
            st.plotly_chart(workload_heatmap(player_df, days=30), width='stretch')

    with st.expander("Model transparency — what this does and does not claim"):
        st.markdown(
            f"""
**What LoadWatch estimates**
- A **relative soft-tissue injury risk score (0–100)** from workload patterns
  (minutes, density, rest, age, injury recency).
- Scores are **percentile-ranked** within the {SEASON} modeled population so
  “85” means higher estimated risk than ~85% of player-games — not an 85% chance of injury.

**What it does *not* claim**
- It is **not** a medical diagnosis, clearance tool, or binary injury predictor.
- It does **not** prove causation; features are correlational patterns from historical data.
- Contact injuries (fractures, lacerations) are intentionally out of scope.

**Validation note**
- Models use a **chronological train/test split** (no random k-fold leakage).
- Report **ROC-AUC** and **PR-AUC**; raw accuracy is misleading under ~95%+ negative labels.
"""
        )
        if metrics:
            rows = metrics.get("metrics", [])
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption(metrics.get("notes", ""))


if __name__ == "__main__":
    main()
