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
    fetch_market_rows,
    load_assets,
    realized_metrics,
    update_prediction_log,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
STATE_DIR = Path(os.getenv("UFC_EDGE_STATE_DIR", str(ROOT / "state")))

st.set_page_config(page_title="UFC Edge Model", page_icon="🥊", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def assets():
    return load_assets(DATA_DIR)


@st.cache_data(ttl=300, show_spinner=False)
def cached_card(event_search, refresh_results=False):
    return discover_card(event_search, refresh_results=refresh_results)


@st.cache_data(ttl=20, show_spinner=False)
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
          <td>{escape(row['pick'])}</td>
          <td class="num strong">{pct(row['model_probability'])}</td>
          <td class="num">{pct(row['market_probability'])}</td>
          <td class="num {edge_class}">{pct(row['net_edge'])}</td>
          <td><i class="decision {decision_class}">{row['action']}</i></td>
          <td class="num">{money(row['position_dollars'])}</td>
          <td class="why">{escape(row['why'])}</td>
        </tr>""")
    return f"""
    <div class="pricing-table-wrap"><table class="pricing-table">
      <thead><tr><th>Fight</th><th>Model pick</th><th>Model P</th><th>Market P</th><th>Net edge</th><th>Decision</th><th>Position</th><th>Primary drivers</th></tr></thead>
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
  .hero { display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; margin-bottom:1.7rem; }
  .hero p { margin:0 0 .48rem; color:#1c4d77; font-size:.62rem; font-weight:800; letter-spacing:.17em; }
  .hero h1 { margin:0; max-width:800px; font-size:clamp(2.2rem,4.4vw,3.7rem); line-height:1; letter-spacing:-.05em; }
  .hero span { display:block; margin-top:.85rem; color:#667587; font-size:.9rem; }
  .holdout { min-width:155px; border-left:3px solid #08745a; padding:.3rem 0 .3rem 1rem; }
  .holdout span { margin:0; font-size:.55rem; font-weight:800; letter-spacing:.12em; }
  .holdout b { display:block; margin:.15rem 0; font-size:1.7rem; }
  .holdout small { color:#667587; font-size:.58rem; }
  div[data-testid="stTextInput"] label p, div[data-testid="stNumberInput"] label p { color:#667587; font-size:.58rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
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
  .pricing-table { width:100%; min-width:1080px; border-collapse:collapse; }
  .pricing-table th { padding:.65rem .72rem; text-align:left; background:#eaf0f5; color:#647588; border-bottom:1px solid #ccd5df; font-size:.54rem; letter-spacing:.1em; text-transform:uppercase; }
  .pricing-table td { padding:.76rem .72rem; border-bottom:1px solid #e0e6ed; font-size:.72rem; vertical-align:middle; }
  .pricing-table tr:last-child td { border-bottom:0; }
  .pricing-table td:first-child b,.pricing-table td:first-child span { display:block; }
  .pricing-table td:first-child span { margin-top:.18rem; color:#667587; font-size:.66rem; }
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
  @media(max-width:700px){.block-container{padding:1rem}.brandbar{margin:-1rem -1rem 1.5rem;padding:1rem}.model-pill,.holdout{display:none}.hero h1{font-size:2.4rem}}
</style>
""", unsafe_allow_html=True)

bundle, fighters = assets()
st.markdown("""
<div class="brandbar"><div class="brand"><span class="brand-mark">UE</span><div><b>UFC EDGE</b><small>CALIBRATED FIGHT PRICING</small></div></div><div class="model-pill"><i></i> Gradient boosting · cached</div></div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="hero"><div><p>DECISION ENGINE</p><h1>One screen. One price. One decision.</h1><span>Independent UFC probabilities compared with executable market prices.</span></div><div class="holdout"><span>UNSEEN HOLDOUT</span><b>{bundle['metrics']['accuracy']:.1%}</b><small>accuracy</small></div></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Model controls")
    min_edge = st.number_input("Minimum net edge", min_value=0.00, max_value=0.20, value=0.03, step=0.005, format="%.3f")
    cost_buffer = st.number_input("Cost buffer", min_value=0.00, max_value=0.10, value=0.02, step=0.005, format="%.3f")
    min_fights = st.number_input("Minimum prior UFC fights", min_value=0, max_value=25, value=3, step=1)
    refresh_results = st.checkbox("Check completed results", value=False)
    manual_odds = st.text_area("Manual odds override", placeholder="Islam Makhachev, -320\nIan Machado Garry, +250", height=90)
    use_research = st.checkbox("Use capped research overlay", value=False)
    reports = st.file_uploader("Research reports", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True, disabled=not use_research)

event_col, bankroll_col, button_col = st.columns([6, 2, 2], vertical_alignment="bottom")
event_search = event_col.text_input("Event or fight", value="UFC 330", placeholder="UFC 330 or Fighter A vs Fighter B")
bankroll = bankroll_col.number_input("Bankroll", min_value=100, max_value=10_000_000, value=10_000, step=100)
run = button_col.button("Run model", use_container_width=True, type="primary")

if run:
    started = time.perf_counter()
    try:
        with st.spinner("Refreshing card and market prices…"):
            card = cached_card(event_search, refresh_results)
            card_json = pd.DataFrame(card).to_json(orient="records")
            markets = cached_markets(card_json, manual_odds)
            research = uploaded_texts(reports) if use_research else []
            analyses = analyze_card(
                card, markets, bundle, fighters, bankroll=bankroll, research_texts=research,
                min_edge=min_edge, cost_buffer=cost_buffer, min_prior_fights=int(min_fights),
            )
            log = update_prediction_log(STATE_DIR, analyses)
        st.session_state["analyses"] = analyses
        st.session_state["log"] = log
        st.session_state["event_search"] = event_search
        st.session_state["bankroll"] = bankroll
        st.session_state["elapsed"] = time.perf_counter() - started
        st.session_state["excel"] = build_excel(
            analyses, bundle, log, bankroll, event_search,
            cost_buffer=cost_buffer, min_edge=min_edge, min_prior_fights=int(min_fights),
        )
    except Exception as exc:
        st.error(str(exc))

analyses = st.session_state.get("analyses")
if not analyses:
    st.markdown("<div class='math-card' style='text-align:center;min-height:260px;display:grid;place-items:center'><div><small>READY</small><h3>Enter tonight’s event and run the model.</h3><span style='color:#667587;font-size:.75rem'>The trained model stays loaded. Only the card and odds refresh.</span></div></div>", unsafe_allow_html=True)
else:
    log = st.session_state["log"]
    bets = [row for row in analyses if row["action"] == "BET"]
    valid_edges = [row["net_edge"] for row in analyses if np.isfinite(row["net_edge"])]
    top_edge = max(valid_edges) if valid_edges else np.nan
    total_risk = sum(row["position_dollars"] for row in bets)
    st.markdown(f"<div class='statusline'><b>REFRESHED</b> in {st.session_state['elapsed']:.2f}s · {len(analyses)} fights · {sum(bool(row['market_source']) for row in analyses)} matched market prices</div>", unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Bet signals", len(bets), f"of {len(analyses)} fights")
    metric_columns[1].metric("Top net edge", pct(top_edge), "after cost buffer")
    metric_columns[2].metric("Capital at risk", money(total_risk), f"{total_risk / bankroll:.1%} of bankroll")
    metric_columns[3].metric("Holdout accuracy", f"{bundle['metrics']['accuracy']:.1%}", f"{bundle['metrics']['holdout_fights']:,} unseen fights")

    dashboard_tab, model_tab, performance_tab, history_tab = st.tabs(["Dashboard", "Model", "Performance", "History"])
    with dashboard_tab:
        st.markdown(f"<div class='board-title'><small>PRICING BOARD</small><h2>{escape(analyses[0]['event'])}</h2></div>{dashboard_table(analyses)}", unsafe_allow_html=True)
        filename = re.sub(r"[^a-z0-9]+", "_", event_search.lower()).strip("_") or "ufc"
        st.download_button("Download Excel report", st.session_state["excel"], file_name=f"{filename}_edge_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with model_tab:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("""
            <div class="math-card"><small>MODEL MATH</small><h3>Transparent decision stack</h3>
            <div class="formula-row"><b>Feature vector</b><span>Eight pre-fight A-minus-B differences</span></div>
            <div class="formula-row"><b>Gradient boosting</b><span>Base score + 0.04 × 200 shallow trees</span></div>
            <div class="formula-row"><b>Order symmetry</b><span>[GB(x) + 1 − GB(−x)] ÷ 2</span></div>
            <div class="formula-row"><b>Calibration</b><span>logistic(1.143 × logit(raw P))</span></div>
            <div class="formula-row"><b>Net edge</b><span>Model P − market P − cost buffer</span></div>
            <div class="formula-row"><b>Position</b><span>Quarter Kelly, capped at 2.0% of bankroll</span></div></div>
            """, unsafe_allow_html=True)
        with right:
            importance = pd.DataFrame({"Factor": [name.replace(" diff", "") for name in bundle["importance"]], "Importance": list(bundle["importance"].values())}).sort_values("Importance")
            st.markdown("<div class='math-card'><small>GLOBAL FEATURE IMPORTANCE</small><h3>What the trees use most</h3>", unsafe_allow_html=True)
            st.bar_chart(importance.set_index("Factor"), horizontal=True, color="#173B5E", height=270)
            st.markdown("</div>", unsafe_allow_html=True)
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
        if len(log):
            display_columns = ["event_date", "fighter_a", "fighter_b", "pick", "model_probability", "market_probability", "net_edge", "action", "position_dollars", "status", "outcome", "pnl"]
            st.dataframe(log[display_columns].sort_values("timestamp_utc", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No recorded predictions yet.")

st.markdown(f"<div class='fineprint'>{bundle['version']} · For research use · Probabilities are estimates, not guarantees</div>", unsafe_allow_html=True)
