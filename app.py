from __future__ import annotations

import io
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from docx import Document
from pypdf import PdfReader

from excel_report import build_excel
from model_engine import (
    analyze_card,
    discover_card as engine_discover_card,
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


def _canonical(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _json_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def active_event_cards():
    """Build selectable full cards from active Polymarket UFC moneylines."""
    response = requests.get("https://gamma-api.polymarket.com/events", params={
        "tag_slug": "ufc", "active": "true", "closed": "false",
        "order": "endDate", "ascending": "true", "limit": 200,
    }, timeout=8)
    response.raise_for_status()
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    groups = {}
    for event in response.json():
        title = str(event.get("title") or "")
        numbered = re.match(r"^(UFC\s+\d+)\s*:", title, re.I)
        fight_night = re.match(r"^(UFC\s+Fight\s+Night)\s*:", title, re.I)
        if not numbered and not fight_night:
            continue
        for market in event.get("markets", []):
            if str(market.get("sportsMarketType") or "").lower() != "moneyline":
                continue
            outcomes = [str(value).strip() for value in _json_list(market.get("outcomes"))]
            token_ids = [str(value).strip() for value in _json_list(market.get("clobTokenIds"))]
            if len(outcomes) != 2 or set(map(_canonical, outcomes)) == {"yes", "no"}:
                continue
            start = pd.to_datetime(market.get("gameStartTime") or event.get("endDate"), utc=True, errors="coerce")
            if pd.isna(start):
                continue
            local_start = start.to_pydatetime().astimezone(eastern)
            card_name = numbered.group(1).upper() if numbered else "UFC Fight Night"
            key = f"{_canonical(card_name)}|{local_start.date().isoformat()}"
            group = groups.setdefault(key, {
                "name": card_name, "date": local_start.date(), "start": start.to_pydatetime(),
                "end": pd.to_datetime(event.get("endDate"), utc=True, errors="coerce"), "bouts": [],
            })
            group["start"] = min(group["start"], start.to_pydatetime())
            group["bouts"].append({
                "event": card_name, "event_date": local_start.date().isoformat(),
                "fighter_a": outcomes[0], "fighter_b": outcomes[1],
                "winner": "", "fight_url": "",
                "market_token_ids": {
                    _canonical(outcomes[index]): token_ids[index]
                    for index in range(2) if len(token_ids) == 2
                },
                "event_start_utc": start.to_pydatetime().isoformat(),
            })
            break
    cards = []
    today = now.astimezone(eastern).date()
    for key, group in groups.items():
        unique = {}
        for bout in group["bouts"]:
            pair = tuple(sorted((_canonical(bout["fighter_a"]), _canonical(bout["fighter_b"]))))
            unique[pair] = bout
        bouts = list(unique.values())
        end = group["end"].to_pydatetime() if pd.notna(group["end"]) else None
        if group["start"] <= now and (end is None or now <= end):
            status = "LIVE"
        elif group["date"] == today:
            status = "TODAY"
        else:
            status = "UPCOMING"
        cards.append({
            "value": f"POLY_CARD::{key}", "bouts": bouts, "status": status, "start": group["start"],
            "label": f"{status}  •  {group['name']}  •  {group['date'].strftime('%b %d').replace(' 0', ' ')}  •  {len(bouts)} fights",
        })
    priority = {"LIVE": 0, "TODAY": 1, "UPCOMING": 2}
    return sorted(cards, key=lambda card: (priority[card["status"]], card["start"]))


@st.cache_resource(show_spinner=False)
def assets():
    return load_assets(DATA_DIR)


@st.cache_data(ttl=300, show_spinner=False)
def cached_card(event_search, refresh_results=False):
    if str(event_search).startswith("POLY_CARD::"):
        selected = next((card for card in active_event_cards() if card["value"] == event_search), None)
        if selected and selected["bouts"]:
            return selected["bouts"]
        raise RuntimeError("That card is no longer active. Refresh the page and choose another UFC card.")
    return engine_discover_card(event_search, refresh_results=refresh_results)


@st.cache_data(ttl=300, show_spinner=False)
def cached_event_options():
    try:
        cards = active_event_cards()
        if cards:
            return [{"label": card["label"], "value": card["value"]} for card in cards]
    except Exception:
        pass
    return [{"label": "UFC 330  •  Aug 15  •  12 fights", "value": "UFC 330"}]


@st.cache_data(ttl=30, show_spinner=False)
def cached_price_histories(token_ids):
    """Public Polymarket price history for the selected outcome tokens."""
    def fetch(token_id):
        response = requests.get("https://clob.polymarket.com/prices-history", params={
            "market": token_id, "interval": "1d", "fidelity": 5,
        }, timeout=8)
        response.raise_for_status()
        return token_id, response.json().get("history", [])

    histories = {}
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(token_ids)))) as executor:
        futures = [executor.submit(fetch, token_id) for token_id in token_ids]
        for future in as_completed(futures):
            try:
                token_id, history = future.result()
                histories[token_id] = history
            except Exception:
                continue
    return histories


def convergence_monitor(analyses, prediction_log, price_histories, capture_target=.65,
                        minimum_exit_return=.03, stop_loss=.15, cost_buffer=.02):
    """Measure observed price discovery and create pre-fight HOLD/SELL signals."""
    open_bets = {}
    if len(prediction_log):
        for record in prediction_log.to_dict("records"):
            if str(record.get("status")) == "OPEN" and str(record.get("action")) == "BET":
                open_bets[str(record.get("prediction_id"))] = record
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    local_now = now.astimezone(eastern)
    morning = local_now.replace(hour=8, minute=0, second=0, microsecond=0)
    if local_now < morning:
        morning = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for analysis in analyses:
        prediction_id = f"{_canonical(analysis['event'])}|{'|'.join(sorted((_canonical(analysis['fighter_a']), _canonical(analysis['fighter_b']))))}"
        position = open_bets.get(prediction_id)
        token_map = analysis.get("market_token_ids") or {}
        token_id = token_map.get(_canonical(analysis["trade_side"])) if isinstance(token_map, dict) else None
        points = []
        for point in price_histories.get(str(token_id), []):
            try:
                points.append({"timestamp": datetime.fromtimestamp(float(point["t"]), timezone.utc), "price": float(point["p"])})
            except (KeyError, TypeError, ValueError, OSError):
                continue
        points.sort(key=lambda point: point["timestamp"])
        since_morning = [point for point in points if morning.astimezone(timezone.utc) <= point["timestamp"] <= now]
        morning_price = (since_morning[0]["price"] if since_morning else (points[0]["price"] if points else np.nan))
        bid, ask = float(analysis.get("live_bid", np.nan)), float(analysis.get("live_ask", np.nan))
        current_mid = np.nanmean([value for value in (bid, ask) if np.isfinite(value)]) if np.isfinite(bid) or np.isfinite(ask) else np.nan
        fair = float(analysis["model_probability"])
        morning_gap = fair - morning_price if np.isfinite(morning_price) else np.nan
        gap_closed = ((current_mid - morning_price) / morning_gap) if np.isfinite(morning_gap) and abs(morning_gap) >= .005 else np.nan
        gap_closed = float(np.clip(gap_closed, -2, 2)) if np.isfinite(gap_closed) else np.nan
        event_start = pd.to_datetime(analysis.get("event_start_utc"), utc=True, errors="coerce")
        hours_to_event = max(0, (event_start.to_pydatetime() - now).total_seconds() / 3600) if pd.notna(event_start) else np.nan
        required_capture = capture_target
        if np.isfinite(hours_to_event) and hours_to_event <= 6:
            required_capture = max(.40, capture_target - .15)
        elif np.isfinite(hours_to_event) and hours_to_event <= 24:
            required_capture = max(.50, capture_target - .05)
        entry = float(position.get("market_probability", np.nan)) if position else np.nan
        dollars = float(position.get("position_dollars", 0)) if position else 0
        entry_gap = fair - entry if np.isfinite(entry) else np.nan
        capture = ((bid - entry) / entry_gap) if np.isfinite(bid) and np.isfinite(entry_gap) and abs(entry_gap) >= .005 else np.nan
        exit_return = (bid / entry - 1) if np.isfinite(bid) and 0 < entry < 1 else np.nan
        unrealized_pnl = dollars * exit_return if np.isfinite(exit_return) else np.nan
        target_price = entry + required_capture * entry_gap if np.isfinite(entry_gap) else np.nan
        remaining_edge = fair - ask - cost_buffer if np.isfinite(ask) else np.nan
        if not position:
            signal = "WATCH"
            reason = "No recorded model position"
        elif not np.isfinite(bid):
            signal = "NO EXIT PRICE"
            reason = "No executable bid is available"
        elif np.isfinite(exit_return) and exit_return <= -stop_loss:
            signal = "REVIEW"
            reason = f"Sellable value is {abs(exit_return):.1%} below entry"
        elif exit_return >= minimum_exit_return and (
            (np.isfinite(capture) and capture >= required_capture)
            or (np.isfinite(target_price) and bid >= target_price)
            or (np.isfinite(remaining_edge) and remaining_edge <= 0)
        ):
            signal = "SELL"
            reason = f"Captured {capture:.0%} of the entry-to-fair-value gap" if np.isfinite(capture) else "The remaining model edge has closed"
        else:
            signal = "HOLD"
            reason = f"Target bid {target_price:.1%}; current bid {bid:.1%}" if np.isfinite(target_price) else "Waiting for measurable convergence"
        rows.append({
            "fight": f"{analysis['fighter_a']} vs {analysis['fighter_b']}",
            "trade_side": analysis["trade_side"], "entry_price": entry,
            "morning_price": morning_price, "current_bid": bid, "current_mid": current_mid,
            "fair_value": fair, "gap_closed": gap_closed, "position_capture": capture,
            "unrealized_return": exit_return, "unrealized_pnl": unrealized_pnl,
            "position_dollars": dollars, "target_price": target_price,
            "hours_to_event": hours_to_event, "signal": signal, "reason": reason,
            "token_id": token_id, "history": points,
        })
    return rows


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


def convergence_table(rows):
    body = []
    signal_class = {"SELL": "bet", "HOLD": "hold", "REVIEW": "review", "WATCH": "watch", "NO EXIT PRICE": "watch"}
    for row in rows:
        body.append(f"""
        <tr>
          <td><b>{escape(row['trade_side'])}</b><span>{escape(row['fight'])}</span></td>
          <td class="num">{pct(row['entry_price'])}</td>
          <td class="num">{pct(row['morning_price'])}</td>
          <td class="num strong">{pct(row['current_bid'])}</td>
          <td class="num">{pct(row['fair_value'])}</td>
          <td class="num">{pct(row['gap_closed'])}</td>
          <td class="num">{pct(row['unrealized_return'])}</td>
          <td class="num">{money(row['unrealized_pnl']) if np.isfinite(row['unrealized_pnl']) else '—'}</td>
          <td><i class="decision {signal_class.get(row['signal'], 'watch')}">{escape(row['signal'])}</i><span>{escape(row['reason'])}</span></td>
        </tr>""")
    return f"""
    <div class="pricing-table-wrap"><table class="pricing-table">
      <thead><tr><th>Position</th><th>Entry</th><th>8 AM price</th><th>Sellable bid</th><th>Model fair</th><th>Gap closed today</th><th>Return if sold</th><th>Unrealized P&amp;L</th><th>Signal</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table></div>"""


st.markdown("""
<style>
  :root { --red:#e10600; --black:#090a0c; --panel:#131416; --line:#2a2c30; --white:#f4f4f1; --muted:#96999f; --amber:#f2a900; }
  .stApp, .stApp * { font-family:"Arial Narrow","Helvetica Neue",Arial,sans-serif; }
  .stApp { background:var(--black); color:var(--white); }
  .block-container { max-width:1540px; padding:1.35rem 2.35rem 3rem; }
  [data-testid="stHeader"] { background:transparent; }
  [data-testid="stToolbar"] { visibility:hidden; }
  section[data-testid="stSidebar"] { background:#0d0e10; border-right:1px solid var(--line); }
  section[data-testid="stSidebar"] * { color:var(--white); }
  .brandbar { margin:-1.35rem -2.35rem 0; padding:.78rem 2.35rem; background:#050506; color:#fff; display:flex; align-items:center; justify-content:space-between; border-bottom:4px solid var(--red); }
  .brand { display:flex; align-items:center; gap:.85rem; }
  .brand-mark { width:48px; height:31px; background:var(--red); display:grid; place-items:center; font-size:.72rem; font-weight:950; font-style:italic; letter-spacing:.04em; clip-path:polygon(8% 0,100% 0,92% 100%,0 100%); }
  .brand b { display:block; font-size:.92rem; font-weight:950; font-style:italic; letter-spacing:.08em; }
  .brand small { display:block; margin-top:.1rem; color:#85888e; font-size:.5rem; font-weight:800; letter-spacing:.18em; }
  .model-pill { border-left:3px solid var(--red); padding:.32rem 0 .32rem .75rem; color:#c5c7cb; font-size:.57rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
  .model-pill i { width:6px; height:6px; display:inline-block; border-radius:50%; background:var(--red); margin-right:.45rem; box-shadow:0 0 0 3px rgba(225,6,0,.18); }
  .hero { min-height:250px; display:grid; grid-template-columns:minmax(0,1fr) 190px; align-items:center; gap:3rem; margin:0 -2.35rem 1.2rem; padding:2.35rem; background:#0b0c0e; border-bottom:1px solid var(--line); position:relative; overflow:hidden; }
  .hero:after { content:""; position:absolute; right:220px; top:-70px; width:9px; height:390px; background:var(--red); transform:rotate(22deg); opacity:.85; }
  .hero-copy { min-width:0; position:relative; z-index:1; }
  .hero .eyebrow { display:block; margin:0 0 .7rem; color:var(--red); font-size:.62rem; font-weight:950; font-style:italic; letter-spacing:.22em; }
  .hero h1 { position:static; display:block; margin:0; max-width:960px; color:var(--white); font-size:clamp(2.8rem,5.6vw,5.4rem); font-weight:950; font-style:italic; text-transform:uppercase; line-height:.88; letter-spacing:-.055em; }
  .hero h1 em { color:var(--red); font-style:inherit; }
  .hero .hero-sub { display:block; margin-top:1.25rem; color:#b0b2b7; font-size:.68rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
  .holdout { min-width:160px; position:relative; z-index:1; border-top:4px solid var(--red); border-bottom:1px solid #45474c; padding:.85rem .15rem .75rem; }
  .holdout span { display:block; margin:0; color:#a0a3a9; font-size:.52rem; font-weight:900; letter-spacing:.18em; }
  .holdout b { display:block; margin:.1rem 0; color:#fff; font-size:2.35rem; font-style:italic; line-height:1; }
  .holdout small { color:#74777d; font-size:.54rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
  div[data-testid="stTextInput"] label p, div[data-testid="stNumberInput"] label p, div[data-testid="stSelectbox"] label p { color:#8c8f95; font-size:.55rem; font-weight:900; letter-spacing:.16em; text-transform:uppercase; }
  .stTextInput input, .stNumberInput input { background:#141517!important; border:1px solid #3a3c41!important; border-radius:0!important; color:#fff!important; }
  div[data-baseweb="select"]>div { background:#141517!important; border:1px solid #3a3c41!important; border-radius:0!important; color:#fff!important; }
  .stButton button { height:2.8rem; border:0; border-radius:0; background:var(--red); color:#fff; font-size:.62rem; font-weight:950; font-style:italic; letter-spacing:.13em; text-transform:uppercase; clip-path:polygon(0 0,96% 0,100% 50%,96% 100%,0 100%); }
  .stButton button:hover { background:#ff1a13; color:#fff; }
  .stDownloadButton button { border-radius:0; border:1px solid #66696f; background:transparent; color:#fff; font-weight:900; text-transform:uppercase; }
  .stDownloadButton button:hover { border-color:var(--red); color:#fff; }
  .statusline { margin:.35rem 0 1rem; padding:.48rem 0; color:#85888e; font-size:.59rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; border-bottom:1px solid var(--line); }
  .statusline b { color:var(--red); letter-spacing:.15em; }
  [data-testid="stMetric"] { padding:.9rem 1rem; background:#111214; border:1px solid var(--line); border-top:3px solid var(--red); border-radius:0; }
  [data-testid="stMetricLabel"] p { color:#8b8e94; font-size:.52rem; font-weight:900; letter-spacing:.15em; text-transform:uppercase; }
  [data-testid="stMetricValue"] { color:#fff; font-size:1.75rem; font-weight:950; font-style:italic; }
  [data-testid="stMetricDelta"] { color:#888b91; font-size:.6rem; }
  .stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:1px solid var(--line); }
  .stTabs [data-baseweb="tab"] { height:3rem; padding:0 1rem; border-radius:0; color:#7f8288; font-size:.58rem; font-weight:950; letter-spacing:.12em; text-transform:uppercase; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] { color:#fff; border-bottom:3px solid var(--red); }
  .board-title { margin:1rem 0 0; padding:.9rem 1.05rem; background:#0e0f11; border:1px solid var(--line); border-left:5px solid var(--red); border-bottom:0; }
  .board-title small { color:var(--red); font-size:.52rem; font-weight:950; font-style:italic; letter-spacing:.18em; }
  .board-title h2 { margin:.2rem 0 0; color:#fff; font-size:1.12rem; font-weight:950; font-style:italic; text-transform:uppercase; }
  .pricing-table-wrap { overflow-x:auto; background:#111214; border:1px solid var(--line); }
  .pricing-table { width:100%; min-width:1180px; border-collapse:collapse; }
  .pricing-table th { padding:.66rem .72rem; text-align:left; background:var(--red); color:#fff; border-right:1px solid rgba(255,255,255,.16); font-size:.5rem; font-weight:950; letter-spacing:.12em; text-transform:uppercase; }
  .pricing-table td { padding:.78rem .72rem; color:#e9e9e6; border-bottom:1px solid var(--line); font-size:.7rem; vertical-align:middle; }
  .pricing-table tr:nth-child(even) td { background:#151619; }
  .pricing-table tr:last-child td { border-bottom:0; }
  .pricing-table td b,.pricing-table td span { display:block; }
  .pricing-table td b { color:#fff; font-weight:950; text-transform:uppercase; }
  .pricing-table td span { margin-top:.18rem; color:#81848a; font-size:.58rem; }
  .pricing-table .num { text-align:right; font-variant-numeric:tabular-nums; }
  .pricing-table .strong { color:#fff; font-weight:950; }
  .pricing-table .edge-positive { color:#ff3932; font-weight:950; }
  .decision { min-width:70px; display:inline-block; padding:.37rem .46rem; text-align:center; color:white; border:1px solid transparent; font-style:italic; font-size:.53rem; font-weight:950; letter-spacing:.09em; }
  .decision.bet { background:var(--red); }.decision.no-bet { background:#25272b; color:#a8abb0; border-color:#404247; }
  .decision.hold { background:transparent; border-color:#e5e5e2; }.decision.review { background:var(--amber); color:#090a0c; }.decision.watch { background:#2a2c30; color:#b8bac0; }
  .pricing-table .why { max-width:270px; color:#999ca2; line-height:1.4; }
  .math-card { min-height:345px; padding:1.3rem; background:#111214; color:#ecece9; border:1px solid var(--line); border-top:4px solid var(--red); }
  .math-card>small { color:var(--red); font-size:.52rem; font-weight:950; font-style:italic; letter-spacing:.18em; }
  .math-card h3 { margin:.32rem 0 1rem; color:#fff; font-size:1.3rem; font-weight:950; font-style:italic; text-transform:uppercase; }
  .formula-row { display:grid; grid-template-columns:145px 1fr; gap:1rem; padding:.68rem 0; border-top:1px solid var(--line); font-size:.68rem; }
  .formula-row b { color:#fff; text-transform:uppercase; letter-spacing:.04em; }
  .formula-row span { color:#999ca2; line-height:1.45; }
  [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] { border:1px solid var(--line); }
  .stCaptionContainer, [data-testid="stCaptionContainer"] { color:#85888e!important; }
  .fineprint { margin-top:1.6rem; padding-top:.8rem; border-top:1px solid var(--line); color:#5f6268; font-size:.55rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
  @media(max-width:700px){.block-container{padding:1rem}.brandbar{margin:-1rem -1rem 0;padding:.85rem 1rem}.model-pill,.holdout,.hero:after{display:none}.hero{display:block;min-height:0;margin:0 -1rem 1rem;padding:2rem 1rem}.hero h1{font-size:3rem}.hero .hero-sub{line-height:1.55}.stTabs [data-baseweb="tab"]{padding:0 .55rem}.formula-row{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)

bundle, fighters = assets()
st.markdown("""
<div class="brandbar"><div class="brand"><span class="brand-mark">UE</span><div><b>UFC EDGE</b><small>MODEL // MARKET // EXECUTION</small></div></div><div class="model-pill"><i></i> LIVE CLOB // MODEL ONLINE</div></div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="hero"><div class="hero-copy"><span class="eyebrow">UFC FIGHT PRICING // LIVE</span><h1>Price the fight.<br><em>Trade the gap.</em></h1><span class="hero-sub">STATISTICAL FAIR VALUE // LIVE POLYMARKET // CONTROLLED EXPOSURE</span></div><div class="holdout"><span>UNSEEN BACKTEST</span><b>{bundle['metrics']['accuracy']:.1%}</b><small>{bundle['metrics']['holdout_fights']:,} fights</small></div></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Model controls")
    min_edge = st.number_input("Minimum net edge", min_value=0.00, max_value=0.20, value=0.03, step=0.005, format="%.3f")
    cost_buffer = st.number_input("Cost buffer", min_value=0.00, max_value=0.10, value=0.02, step=0.005, format="%.3f")
    min_fights = st.number_input("Minimum prior UFC fights", min_value=0, max_value=25, value=3, step=1)
    convergence = st.slider("Pre-fight convergence scenario", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    max_card_exposure = st.slider("Maximum card exposure", min_value=0.02, max_value=0.25, value=0.10, step=0.01)
    capture_target = st.slider("Gap capture before sell", min_value=0.40, max_value=0.90, value=0.65, step=0.05)
    minimum_exit_return = st.slider("Minimum return before sell", min_value=0.00, max_value=0.25, value=0.03, step=0.01)
    stop_loss = st.slider("Loss review threshold", min_value=0.05, max_value=0.40, value=0.15, step=0.05)
    market_mode = st.selectbox("Market source", ["Polymarket CLOB only", "Best available / manual"])
    refresh_results = st.checkbox("Check completed results", value=False)
    manual_odds = st.text_area("Manual odds override", placeholder="Islam Makhachev, -320\nIan Machado Garry, +250", height=90)
    use_research = st.checkbox("Use capped research overlay", value=False)
    reports = st.file_uploader("Research reports", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True, disabled=not use_research)

event_options = cached_event_options()
option_labels = [option["label"] for option in event_options] + ["CUSTOM MATCHUP  •  Fighter A vs Fighter B"]
option_values = {option["label"]: option["value"] for option in event_options}
event_col, bankroll_col, button_col = st.columns([6, 2, 2], vertical_alignment="bottom")
selected_event = event_col.selectbox("Fight card", option_labels)
bankroll = bankroll_col.number_input("Bankroll", min_value=100, max_value=10_000_000, value=10_000, step=100)
run = button_col.button("Run live card", use_container_width=True, type="primary")
if selected_event.startswith("CUSTOM"):
    event_search = st.text_input("Custom matchup", placeholder="Islam Makhachev vs Ian Machado Garry").strip()
else:
    event_search = option_values[selected_event]

if run:
    started = time.perf_counter()
    try:
        if not event_search:
            raise RuntimeError("Type both fighter names as Fighter A vs Fighter B.")
        with st.spinner("SYNCING LIVE MARKET…"):
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
            token_ids = tuple(sorted({
                str(token)
                for row in analyses for token in (row.get("market_token_ids") or {}).values()
                if token
            }))
            price_histories = cached_price_histories(token_ids) if token_ids else {}
            convergence_rows = convergence_monitor(
                analyses, log, price_histories, capture_target=capture_target,
                minimum_exit_return=minimum_exit_return, stop_loss=stop_loss,
                cost_buffer=cost_buffer,
            )
        st.session_state["analyses"] = analyses
        st.session_state["log"] = log
        st.session_state["market_history"] = market_history
        st.session_state["price_histories"] = price_histories
        st.session_state["convergence_rows"] = convergence_rows
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
    st.markdown("<div class='math-card' style='text-align:center;min-height:210px;display:grid;place-items:center'><div><small>LIVE ENGINE READY</small><h3>SELECT CARD // RUN MARKET</h3></div></div>", unsafe_allow_html=True)
else:
    log = st.session_state["log"]
    market_history = st.session_state.get("market_history", pd.DataFrame())
    bets = [row for row in analyses if row["action"] == "BET"]
    valid_edges = [row["net_edge"] for row in analyses if np.isfinite(row["net_edge"])]
    top_edge = max(valid_edges) if valid_edges else np.nan
    total_risk = sum(row["position_dollars"] for row in bets)
    poly_count = sum(row["market_source"] == "Polymarket CLOB" for row in analyses)
    st.markdown(f"<div class='statusline'><b>MARKET SYNC</b> // {st.session_state['elapsed']:.2f}s // {len(analyses)} fights // {poly_count} live order books // {escape(st.session_state['market_mode'])}</div>", unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Bets", len(bets), f"{len(analyses)} fights screened")
    metric_columns[1].metric("Top edge", pct(top_edge), "net of buffer")
    metric_columns[2].metric("Card risk", money(total_risk), f"{total_risk / bankroll:.1%} of bankroll")
    metric_columns[3].metric("Backtest", f"{bundle['metrics']['accuracy']:.1%}", f"{bundle['metrics']['holdout_fights']:,} unseen fights")

    dashboard_tab, convergence_tab, model_tab, raw_tab, performance_tab, history_tab = st.tabs(["Fight Board", "Positions", "Model Math", "Raw Data", "Track Record", "Log"])
    with dashboard_tab:
        st.markdown(f"<div class='board-title'><small>LIVE FIGHT BOARD</small><h2>{escape(analyses[0]['event'])}</h2></div>{dashboard_table(analyses)}", unsafe_allow_html=True)
        st.caption("*EXIT TARGET = LIVE ASK + ASSUMED CONVERGENCE TOWARD MODEL FAIR VALUE.")
        filename = re.sub(r"[^a-z0-9]+", "_", event_search.lower()).strip("_") or "ufc"
        st.download_button("Download Excel report", st.session_state["excel"], file_name=f"{filename}_edge_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with convergence_tab:
        convergence_rows = st.session_state.get("convergence_rows", [])
        positions = [row for row in convergence_rows if row["position_dollars"] > 0]
        sell_signals = [row for row in positions if row["signal"] == "SELL"]
        gaps = [row["gap_closed"] for row in convergence_rows if np.isfinite(row["gap_closed"])]
        unrealized = sum(row["unrealized_pnl"] for row in positions if np.isfinite(row["unrealized_pnl"]))
        monitor_metrics = st.columns(4)
        monitor_metrics[0].metric("Open positions", len(positions))
        monitor_metrics[1].metric("Sell signals", len(sell_signals))
        monitor_metrics[2].metric("Gap closed today", pct(float(np.mean(gaps))) if gaps else "—")
        monitor_metrics[3].metric("P&L at live bid", money(unrealized))
        st.markdown("<div class='board-title'><small>PRICE CONVERGENCE</small><h2>ENTRY // LIVE BID // FAIR VALUE</h2></div>" + convergence_table(convergence_rows), unsafe_allow_html=True)
        st.caption("MODEL SIGNAL ONLY // EXIT VALUE USES THE EXECUTABLE BID.")
        selected_monitor = st.selectbox("Price path", range(len(convergence_rows)), format_func=lambda index: convergence_rows[index]["fight"], key="convergence_fight")
        monitor = convergence_rows[selected_monitor]
        chart_rows = monitor["history"]
        if chart_rows:
            chart = pd.DataFrame(chart_rows).rename(columns={"timestamp": "Time", "price": "Polymarket price"}).set_index("Time")
            chart["Model fair value"] = monitor["fair_value"]
            if np.isfinite(monitor["entry_price"]):
                chart["Recorded entry"] = monitor["entry_price"]
            if np.isfinite(monitor["target_price"]):
                chart["Sell target"] = monitor["target_price"]
            st.line_chart(chart, height=310, color=["#E10600", "#F4F4F1", "#7C7F85", "#F2A900"][:len(chart.columns)])
        else:
            st.info("Price history is not yet available for this outcome.")
        st.markdown(f"""
        <div class="math-card"><small>BEHAVIORAL PRICE CONVERGENCE</small><h3>MEASURE THE MOVE. NEVER ASSUME IT.</h3>
        <div class="formula-row"><b>Today’s gap closed</b><span>(Current midpoint − 8 AM price) ÷ (model fair value − 8 AM price). This shows how much of this morning’s mispricing has disappeared.</span></div>
        <div class="formula-row"><b>Position capture</b><span>(Current sellable bid − recorded entry) ÷ (model fair value − recorded entry). This measures the part of the original opportunity that can actually be realized now.</span></div>
        <div class="formula-row"><b>SELL</b><span>Triggered only after the bid produces at least {pct(minimum_exit_return)} return and captures the required share of the original gap. The threshold becomes slightly more conservative about holding as the event approaches.</span></div>
        <div class="formula-row"><b>HOLD</b><span>The model still sees enough unclosed value between the executable bid and statistical fair value.</span></div>
        <div class="formula-row"><b>REVIEW</b><span>The live bid is at least {pct(stop_loss)} below the recorded entry, so the position requires a risk decision rather than an automatic convergence assumption.</span></div></div>
        """, unsafe_allow_html=True)
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
            st.bar_chart(importance.set_index("Factor"), horizontal=True, color="#E10600", height=270)
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

st.markdown(f"<div class='fineprint'>{bundle['version']} // RESEARCH MODEL // PROBABILITIES ARE ESTIMATES, NOT GUARANTEES</div>", unsafe_allow_html=True)
