from __future__ import annotations

import io
import os
import re
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

from excel_report import build_excel
from model_engine import (
    analyze_card,
    discover_card,
    discover_event_options,
    fetch_market_rows,
    load_assets,
    realized_metrics,
    update_market_history,
    update_prediction_log,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
STATE_DIR = Path(os.getenv("UFC_EDGE_STATE_DIR", str(ROOT / "state")))

st.set_page_config(page_title="UFC Edge Model", page_icon="🥊", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def assets():
    return load_assets(DATA_DIR)


@st.cache_data(ttl=300, show_spinner=False)
def cached_card(event_search, refresh_results=False):
    return discover_card(event_search, refresh_results=refresh_results)


@st.cache_data(ttl=300, show_spinner=False)
def cached_event_options():
    return discover_event_options()


@st.cache_data(ttl=5, show_spinner=False)
def cached_markets(card_json, manual_odds):
    return fetch_market_rows(pd.read_json(io.StringIO(card_json)).to_dict("records"), manual_odds)


def uploaded_texts(files):
    texts = []
    for uploaded in files or []:
        try:
            content = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower()
            if suffix == ".pdf":
                texts.append("\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages))
            elif suffix == ".docx":
                texts.append("\n".join(paragraph.text for paragraph in Document(io.BytesIO(content)).paragraphs))
            elif suffix in {".txt", ".md", ".csv"}:
                texts.append(content.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return [text for text in texts if text.strip()]


def pct(value, digits=1):
    return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}%}"


def money(value):
    return f"${float(value):,.0f}"


def dashboard_table(analyses):
    body = []
    for row in analyses:
        decision_class = "bet" if row["action"] == "BET" else "no-bet"
        edge_class = "edge-positive" if np.isfinite(row["net_edge"]) and row["net_edge"] >= 0.03 else ""
        body.append(f"""
        <tr>
          <td><b>{escape(row['fighter_a'])}</b><span>vs {escape(row['fighter_b'])}</span></td>
          <td><b>{escape(row['likely_winner'])}</b><span>{pct(row['likely_probability'])} win probability</span></td>
          <td><b>{escape(row['trade_side'])}</b><span>{'underdog value' if row['model_probability'] < .5 else 'favored outcome'}</span></td>
          <td class="num strong">{pct(row['model_probability'])}</td>
          <td class="num">{pct(row['live_ask'])}</td>
          <td class="num">{pct(row['exit_target'])}</td>
          <td class="num {edge_class}">{pct(row['net_edge'])}</td>
          <td><i class="decision {decision_class}">{row['action']}</i></td>
          <td class="num">{money(row['position_dollars'])}</td>
        </tr>""")
    return f"""
    <div class="pricing-table-wrap"><table class="pricing-table">
      <thead><tr><th>Fight</th><th>Likely winner</th><th>Trade side</th><th>Fair P</th><th>Live ask</th><th>Exit target*</th><th>Net edge</th><th>Decision</th><th>Position</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table></div>"""


st.markdown("""
<style>
  .stApp { background:#f3f6f9; color:#0a192d; }
  .block-container { max-width:1480px; padding:1.6rem 2.2rem 3rem; }
  [data-testid="stHeader"] { background:transparent; }
  [data-testid="stToolbar"] { visibility:hidden; }
  .brandbar { margin:-1.6rem -2.2rem 2.3rem; padding:1.05rem 2.2rem; background:#071526; color:#fff; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #29425d; }
  .brand { display:flex; align-items:center; gap:.8rem; }
  .brand-mark { width:38px; height:38px; border:1px solid #55708b; display:grid; place-items:center; font-size:.7rem; font-weight:800; letter-spacing:.12em; }
  .brand b { display:block; font-size:.88rem; letter-spacing:.16em; }
  .brand small { display:block; margin-top:.18rem; color:#95a9bd; font-size:.52rem; letter-spacing:.16em; }
  .model-pill { border:1px solid #344d66; border-radius:999px; padding:.48rem .76rem; color:#bdcada; font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; }
  .model-pill i { width:7px; height:7px; display:inline-block; border-radius:50%; background:#26b68b; margin-right:.42rem; }
  .hero { min-height:190px; display:grid; grid-template-columns:minmax(0,1fr) 175px; align-items:center; gap:3rem; margin-bottom:1.35rem; }
  .hero-copy { min-width:0; }
  .hero .eyebrow { display:block; margin:0 0 .75rem; color:#1c4d77; font-size:.62rem; font-weight:800; letter-spacing:.17em; }
  .hero h1 { position:static; display:block; margin:0; max-width:850px; color:#0a192d; font-size:clamp(2.15rem,4.1vw,3.55rem); line-height:1.02; letter-spacing:-.045em; }
  .hero .hero-sub { display:block; margin-top:1rem; color:#667587; font-size:.9rem; line-height:1.45; }
  .holdout { min-width:155px; border-left:3px solid #08745a; padding:.3rem 0 .3rem 1rem; }
  .holdout span { display:block; margin:0; font-size:.55rem; font-weight:800; letter-spacing:.12em; }
  .holdout b { display:block; margin:.15rem 0; font-size:1.7rem; }
  .holdout small { color:#667587; font-size:.58rem; }
  div[data-testid="stTextInput"] label p, div[data-testid="stNumberInput"] label p, div[data-testid="stSelectbox"] label p { color:#667587; font-size:.58rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
  .stTextInput input, .stNumberInput input { background:#fff; border-color:#cbd5df; }
  .stButton button { height:2.7rem; border:0; border-radius:0; background:#0a192d; color:#fff; font-size:.66rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
  .stButton button:hover { background:#1c4d77; color:#fff; }
  .stDownloadButton button { border-radius:0; border:1px solid #0a192d; color:#0a192d; font-weight:800; }
  .statusline { margin:.2rem 0 1rem; color:#667587; font-size:.68rem; }
  .statusline b { color:#08745a; letter-spacing:.08em; }
  [data-testid="stMetric"] { padding:1rem 1.1rem; background:#fff; border:1px solid #d6dee7; }
  [data-testid="stMetricLabel"] p { color:#667587; font-size:.58rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
  [data-testid="stMetricValue"] { color:#0a192d; font-size:1.65rem; font-weight:800; }
  .board-title { margin:1.1rem 0 0; padding:1rem 1.15rem; background:#fff; border:1px solid #d6dee7; border-bottom:0; }
  .board-title small { color:#1c4d77; font-size:.58rem; font-weight:800; letter-spacing:.15em; }
  .board-title h2 { margin:.25rem 0 0; font-size:1.1rem; }
  .pricing-table-wrap { overflow-x:auto; background:#fff; border:1px solid #d6dee7; }
  .pricing-table { width:100%; min-width:1180px; border-collapse:collapse; }
  .pricing-table th { padding:.65rem .72rem; text-align:left; background:#eaf0f5; color:#647588; border-bottom:1px solid #ccd5df; font-size:.54rem; letter-spacing:.1em; text-transform:uppercase; }
  .pricing-table td { padding:.76rem .72rem; border-bottom:1px solid #e0e6ed; font-size:.72rem; vertical-align:middle; }
  .pricing-table tr:last-child td { border-bottom:0; }
  .pricing-table td b,.pricing-table td span { display:block; }
  .pricing-table td span { margin-top:.18rem; color:#667587; font-size:.61rem; }
  .pricing-table .num { text-align:right; font-variant-numeric:tabular-nums; }
  .pricing-table .strong { font-weight:800; }
  .pricing-table .edge-positive { color:#08745a; font-weight:800; }
  .decision { min-width:68px; display:inline-block; padding:.38rem .45rem; text-align:center; color:white; font-style:normal; font-size:.55rem; font-weight:900; letter-spacing:.08em; }
  .decision.bet { background:#08745a; }.decision.no-bet { background:#b42318; }
  .pricing-table .why { max-width:270px; color:#536476; line-height:1.4; }
  .math-card { min-height:345px; padding:1.35rem; background:#fff; border:1px solid #d6dee7; }
  .math-card>small { color:#1c4d77; font-size:.58rem; font-weight:800; letter-spacing:.15em; }
  .math-card h3 { margin:.35rem 0 1rem; font-size:1.25rem; }
  .formula-row { display:grid; grid-template-columns:135px 1fr; gap:1rem; padding:.65rem 0; border-top:1px solid #d6dee7; font-size:.7rem; }
  .formula-row span { color:#667587; }
  .fineprint { margin-top:1.6rem; color:#7b8998; font-size:.62rem; }
  @media(max-width:700px){.block-container{padding:1rem}.brandbar{margin:-1rem -1rem 1.5rem;padding:1rem}.model-pill,.holdout{display:none}.hero{display:block;min-height:0}.hero h1{font-size:2.4rem}}
</style>
""", unsafe_allow_html=True)

bundle, fighters = assets()
st.markdown("""
<div class="brandbar"><div class="brand"><span class="brand-mark">UE</span><div><b>UFC EDGE</b><small>CALIBRATED FIGHT PRICING</small></div></div><div class="model-pill"><i></i> Gradient boosting · cached</div></div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="hero"><div class="hero-copy"><span class="eyebrow">UFC TRADING ENGINE</span><h1>Fair value versus the live market.</h1><span class="hero-sub">Separate the likely winner from the best-priced trade, then define the exit before entering.</span></div><div class="holdout"><span>UNSEEN HOLDOUT</span><b>{bundle['metrics']['accuracy']:.1%}</b><small>accuracy</small></div></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Model controls")
    min_edge = st.number_input("Minimum net edge", min_value=0.00, max_value=0.20, value=0.03, step=0.005, format="%.3f")
    cost_buffer = st.number_input("Cost buffer", min_value=0.00, max_value=0.10, value=0.02, step=0.005, format="%.3f")
    min_fights = st.number_input("Minimum prior UFC fights", min_value=0, max_value=25, value=3, step=1)
    convergence = st.slider("Pre-fight convergence scenario", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    max_card_exposure = st.slider("Maximum card exposure", min_value=0.02, max_value=0.25, value=0.10, step=0.01)
    market_mode = st.selectbox("Market source", ["Polymarket CLOB only", "Best available / manual"])
    refresh_results = st.checkbox("Check completed results", value=False)
    manual_odds = st.text_area("Manual odds override", placeholder="Islam Makhachev, -320\nIan Machado Garry, +250", height=90)
    use_research = st.checkbox("Use capped research overlay", value=False)
    reports = st.file_uploader("Research reports", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True, disabled=not use_research)

event_options = cached_event_options()
option_labels = [option["label"] for option in event_options] + ["CUSTOM  •  Fighter A vs Fighter B"]
option_values = {option["label"]: option["value"] for option in event_options}
event_col, bankroll_col, button_col = st.columns([6, 2, 2], vertical_alignment="bottom")
selected_event = event_col.selectbox("Live & upcoming UFC events", option_labels)
bankroll = bankroll_col.number_input("Bankroll", min_value=100, max_value=10_000_000, value=10_000, step=100)
run = button_col.button("Refresh live prices", use_container_width=True, type="primary")
if selected_event.startswith("CUSTOM"):
    event_search = st.text_input("Custom matchup", placeholder="Islam Makhachev vs Ian Machado Garry").strip()
else:
    event_search = option_values[selected_event]

if run:
    started = time.perf_counter()
    try:
        if not event_search:
            raise RuntimeError("Type both fighter names as Fighter A vs Fighter B.")
        with st.spinner("Refreshing card and market prices…"):
            card = cached_card(event_search, refresh_results)
            card_json = pd.DataFrame(card).to_json(orient="records")
            markets = cached_markets(card_json, manual_odds)
            research = uploaded_texts(reports) if use_research else []
            analyses = analyze_card(
                card, markets, bundle, fighters, bankroll=bankroll, research_texts=research,
                min_edge=min_edge, cost_buffer=cost_buffer, min_prior_fights=int(min_fights),
                convergence=convergence, max_card_exposure=max_card_exposure,
                polymarket_only=market_mode == "Polymarket CLOB only",
            )
            log = update_prediction_log(STATE_DIR, analyses)
            market_history = update_market_history(STATE_DIR, analyses)
        st.session_state["analyses"] = analyses
        st.session_state["log"] = log
        st.session_state["market_history"] = market_history
        st.session_state["event_search"] = event_search
        st.session_state["bankroll"] = bankroll
        st.session_state["convergence"] = convergence
        st.session_state["market_mode"] = market_mode
        st.session_state["elapsed"] = time.perf_counter() - started
        st.session_state["excel"] = build_excel(
            analyses, bundle, log, bankroll, event_search,
            cost_buffer=cost_buffer, min_edge=min_edge, min_prior_fights=int(min_fights),
        )
    except Exception as exc:
        st.error(str(exc))

analyses = st.session_state.get("analyses")
if not analyses:
    st.markdown("<div class='math-card' style='text-align:center;min-height:240px;display:grid;place-items:center'><div><small>READY</small><h3>Choose a current UFC card above.</h3><span style='color:#667587;font-size:.75rem'>Then refresh to load its fights and live Polymarket order books.</span></div></div>", unsafe_allow_html=True)
else:
    log = st.session_state["log"]
    market_history = st.session_state.get("market_history", pd.DataFrame())
    bets = [row for row in analyses if row["action"] == "BET"]
    valid_edges = [row["net_edge"] for row in analyses if np.isfinite(row["net_edge"])]
    top_edge = max(valid_edges) if valid_edges else np.nan
    total_risk = sum(row["position_dollars"] for row in bets)
    poly_count = sum(row["market_source"] == "Polymarket CLOB" for row in analyses)
    st.markdown(f"<div class='statusline'><b>LIVE REFRESH</b> in {st.session_state['elapsed']:.2f}s · {len(analyses)} fights · {poly_count} Polymarket order books · source mode: {escape(st.session_state['market_mode'])}</div>", unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Bet signals", len(bets), f"of {len(analyses)} fights")
    metric_columns[1].metric("Top net edge", pct(top_edge), "after cost buffer")
    metric_columns[2].metric("Capital at risk", money(total_risk), f"{total_risk / bankroll:.1%} of bankroll")
    metric_columns[3].metric("Holdout accuracy", f"{bundle['metrics']['accuracy']:.1%}", f"{bundle['metrics']['holdout_fights']:,} unseen fights")

    dashboard_tab, model_tab, raw_tab, performance_tab, history_tab = st.tabs(["Dashboard", "Trade Math", "Raw Inputs", "Performance", "History"])
    with dashboard_tab:
        st.markdown(f"<div class='board-title'><small>PRICING BOARD</small><h2>{escape(analyses[0]['event'])}</h2></div>{dashboard_table(analyses)}", unsafe_allow_html=True)
        st.caption("*Exit target is a scenario, not a forecast: live ask + convergence assumption × (model fair value − live ask). A trade side below 50% can still be a BET when its price is cheaper than its estimated chance.")
        filename = re.sub(r"[^a-z0-9]+", "_", event_search.lower()).strip("_") or "ufc"
        st.download_button("Download Excel report", st.session_state["excel"], file_name=f"{filename}_edge_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with model_tab:
        selected_math = st.selectbox("Fight example", range(len(analyses)), format_func=lambda index: f"{analyses[index]['fighter_a']} vs {analyses[index]['fighter_b']}", key="math_fight")
        example = analyses[selected_math]
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"""
            <div class="math-card"><small>HOW THE DECISION WORKS</small><h3>Four numbers, in order</h3>
            <div class="formula-row"><b>Likely winner</b><span>The higher of the two calibrated win probabilities: {escape(example['likely_winner'])} at {pct(example['likely_probability'])}.</span></div>
            <div class="formula-row"><b>Trade side</b><span>The outcome with the larger price discount: {escape(example['trade_side'])}. It does not have to be above 50%.</span></div>
            <div class="formula-row"><b>Net edge</b><span>{pct(example['model_probability'])} fair value − {pct(example['live_ask'])} live ask − {pct(cost_buffer)} costs = {pct(example['net_edge'])}.</span></div>
            <div class="formula-row"><b>Exit scenario</b><span>{pct(example['live_ask'])} + {pct(st.session_state['convergence'])} × (fair value − ask) = {pct(example['exit_target'])} target.</span></div>
            <div class="formula-row"><b>Decision</b><span>BET only when net edge clears {pct(min_edge)}, experience clears {int(min_fights)} fights, and an executable price exists.</span></div>
            <div class="formula-row"><b>Position</b><span>Quarter Kelly, capped per fight and scaled so total card exposure stays below the portfolio limit.</span></div></div>
            """, unsafe_allow_html=True)
        with right:
            importance = pd.DataFrame({"Factor": [name.replace(" diff", "") for name in bundle["importance"]], "Importance": list(bundle["importance"].values())}).sort_values("Importance")
            st.markdown("<div class='math-card'><small>MODEL STRUCTURE</small><h3>What the 200 trees use</h3>", unsafe_allow_html=True)
            st.bar_chart(importance.set_index("Factor"), horizontal=True, color="#173B5E", height=270)
            st.markdown("</div>", unsafe_allow_html=True)
    with raw_tab:
        selected_raw = st.selectbox("Fight", range(len(analyses)), format_func=lambda index: f"{analyses[index]['fighter_a']} vs {analyses[index]['fighter_b']}", key="raw_fight")
        raw = analyses[selected_raw]
        market_cols = st.columns(5)
        market_cols[0].metric("Fighter A P", pct(raw["probability_a"]))
        market_cols[1].metric("Fighter B P", pct(raw["probability_b"]))
        market_cols[2].metric("Live bid", pct(raw["live_bid"]))
        market_cols[3].metric("Live ask", pct(raw["live_ask"]))
        market_cols[4].metric("Source", raw["market_source"] or "No market")
        raw_frame = pd.DataFrame(raw["raw_inputs"]).rename(columns={"Fighter A": raw["fighter_a"], "Fighter B": raw["fighter_b"]})
        st.dataframe(raw_frame, use_container_width=True, hide_index=True)
        st.caption(f"Model factors are calculated before the fight. Market snapshot refreshed {raw['as_of_utc']}. Primary read: {raw['why']}")
        if raw["market_url"]:
            st.link_button("Open matched Polymarket market", raw["market_url"])
        if len(market_history):
            pair_history = market_history[
                (market_history["event"] == raw["event"])
                & (market_history["fighter_a"] == raw["fighter_a"])
                & (market_history["fighter_b"] == raw["fighter_b"])
                & (market_history["trade_side"] == raw["trade_side"])
            ].copy()
            if len(pair_history) > 1:
                pair_history["timestamp_utc"] = pd.to_datetime(pair_history["timestamp_utc"], errors="coerce")
                pair_history = pair_history.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc")
                st.markdown("#### Recorded price movement")
                st.line_chart(pair_history[["live_bid", "live_ask", "model_probability", "exit_target"]])
                st.caption("This chart tests the convergence thesis using snapshots recorded by this app. It does not assume convergence occurred.")
    with performance_tab:
        realized = realized_metrics(log)
        columns = st.columns(4)
        columns[0].metric("Unseen holdout", f"{bundle['metrics']['holdout_fights']:,}")
        columns[1].metric("Accuracy", f"{bundle['metrics']['accuracy']:.1%}")
        columns[2].metric("Brier score", f"{bundle['metrics']['brier']:.3f}")
        columns[3].metric("ROC AUC", f"{bundle['metrics']['auc']:.3f}")
        live = st.columns(4)
        live[0].metric("Recorded bets", realized["graded"])
        live[1].metric("Wins", realized["wins"])
        live[2].metric("Losses", realized["losses"])
        live[3].metric("Recorded P&L", money(realized["pnl"]))
        st.caption("The 2025–2026 holdout was not used to train or calibrate the model.")
    with history_tab:
        prediction_history, price_history = st.tabs(["Predictions", "Market snapshots"])
        with prediction_history:
            if len(log):
                display_columns = ["event_date", "fighter_a", "fighter_b", "pick", "model_probability", "market_probability", "net_edge", "action", "position_dollars", "status", "outcome", "pnl"]
                ordered = log.sort_values("timestamp_utc", ascending=False) if "timestamp_utc" in log.columns else log
                st.dataframe(ordered[[column for column in display_columns if column in ordered.columns]], use_container_width=True, hide_index=True)
            else:
                st.info("No recorded predictions yet.")
        with price_history:
            if len(market_history):
                st.dataframe(market_history.sort_values("timestamp_utc", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("No market snapshots recorded yet.")

st.markdown(f"<div class='fineprint'>{bundle['version']} · For research use · Probabilities are estimates, not guarantees</div>", unsafe_allow_html=True)
