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
    fetch_completed_results_for_log,
    fetch_market_rows,
    grade_prediction_log,
    load_assets,
    load_prediction_log,
    merge_prediction_log,
    realized_metrics,
    safe_float,
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


@st.cache_data(ttl=600, show_spinner=False)
def cached_completed_results(open_log_json):
    open_log = pd.read_json(io.StringIO(open_log_json), orient="records")
    return fetch_completed_results_for_log(open_log)


def reconcile_paper_results(log):
    """Grade due paper bets from completed UFCStats results when the app is opened."""
    if not isinstance(log, pd.DataFrame) or log.empty:
        return load_prediction_log(STATE_DIR)
    open_bets = log[(log["status"] == "OPEN") & (log["action"] == "BET")]
    if open_bets.empty:
        return log
    signature_columns = ["prediction_id", "event_date", "event", "fighter_a", "fighter_b", "action", "status"]
    try:
        completed = cached_completed_results(open_bets[signature_columns].to_json(orient="records"))
        return grade_prediction_log(STATE_DIR, completed) if completed else log
    except Exception:
        return log


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
        entry = safe_float(position.get("entry_price"), safe_float(position.get("market_probability"))) if position else np.nan
        dollars = safe_float(position.get("position_dollars"), 0) if position else 0
        entry_timestamp = (position.get("entry_timestamp_utc") or position.get("timestamp_utc") or "") if position else ""
        entry_gap = fair - entry if np.isfinite(entry) else np.nan
        capture = ((bid - entry) / entry_gap) if np.isfinite(bid) and np.isfinite(entry_gap) and abs(entry_gap) >= .005 else np.nan
        exit_return = (bid / entry - 1) if np.isfinite(bid) and 0 < entry < 1 else np.nan
        unrealized_pnl = dollars * exit_return if np.isfinite(exit_return) else np.nan
        target_price = entry + required_capture * entry_gap if np.isfinite(entry_gap) else np.nan
        target_move = target_price - entry if np.isfinite(target_price) and np.isfinite(entry) else np.nan
        target_progress = ((bid - entry) / target_move) if np.isfinite(bid) and np.isfinite(target_move) and abs(target_move) >= .005 else np.nan
        distance_to_target = target_price - bid if np.isfinite(target_price) and np.isfinite(bid) else np.nan
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
            "target_progress": target_progress, "distance_to_target": distance_to_target,
            "entry_timestamp_utc": entry_timestamp,
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


def signed_money(value):
    if value is None or not np.isfinite(float(value)):
        return "—"
    amount = float(value)
    return f"{'+' if amount >= 0 else '-'}${abs(amount):,.0f}"


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


def fight_board(analyses, convergence_rows=None):
    """Render a compact bout-by-bout board with the decision as the visual endpoint."""
    convergence_rows = convergence_rows or []
    rows = []
    for number, row in enumerate(analyses, start=1):
        monitor = convergence_rows[number - 1] if number <= len(convergence_rows) else {}
        position_open = safe_float(monitor.get("position_dollars"), 0) > 0
        is_bet = position_open or row["action"] == "BET"
        action_class = "bet-card" if is_bet else "pass-card"
        action_label = "BET TAKEN" if position_open else row["action"]
        if position_open:
            action_note = f"{money(monitor['position_dollars'])} @ {pct(monitor.get('entry_price'))}"
        else:
            action_note = f"{money(row['position_dollars'])} // READY" if is_bet else "EDGE / DATA FILTER"
        reason = row.get("why") or "No model explanation available"
        rows.append(f"""
        <article class="fight-row {action_class}">
          <div class="bout-id"><span>BOUT</span><b>{number:02d}</b></div>
          <div class="matchup">
            <strong>{escape(row['fighter_a'])}</strong>
            <i>VS</i>
            <strong>{escape(row['fighter_b'])}</strong>
            <small>MODEL LEAN&nbsp;&nbsp; {escape(row['likely_winner'])} &nbsp;{pct(row['likely_probability'])}</small>
          </div>
          <div class="trade-side"><span>TRADE</span><b>{escape(row['trade_side'])}</b><small>{escape(reason)}</small></div>
          <div class="price-cell"><span>MODEL</span><b>{pct(row['model_probability'])}</b></div>
          <div class="price-cell live"><span><i></i>LIVE ASK</span><b>{pct(row['live_ask'])}</b><small>bid {pct(row['live_bid'])}</small></div>
          <div class="price-cell edge"><span>NET EDGE</span><b>{pct(row['net_edge'])}</b><small>after buffer</small></div>
          <div class="fight-action"><span>{action_label}</span><b>{action_note}</b></div>
        </article>""")
    return f"<div class='fight-board'>{''.join(rows)}</div>"


def event_strip(analyses):
    bets = [row for row in analyses if row["action"] == "BET"]
    ranked = [row for row in bets if np.isfinite(safe_float(row.get("net_edge")))]
    top = max(ranked, key=lambda row: row["net_edge"]) if ranked else None
    top_trade = escape(top["trade_side"]) if top else "NO QUALIFYING TRADE"
    top_edge = pct(top["net_edge"]) if top else "—"
    live_books = sum(row.get("market_source") == "Polymarket CLOB" for row in analyses)
    refreshed_at = pd.to_datetime(analyses[0].get("as_of_utc"), utc=True, errors="coerce")
    refreshed = refreshed_at.strftime("%H:%M:%S UTC") if pd.notna(refreshed_at) else "—"
    return f"""
    <section class="event-strip">
      <div class="event-name"><span>LIVE PRICING BOARD</span><h2>{escape(analyses[0]['event'])}</h2><small>{len(analyses)} fights // {live_books} executable order books</small></div>
      <div class="top-trade"><span>TOP TRADE</span><b>{top_trade}</b><small>{top_edge} NET EDGE</small></div>
      <div class="tape-status"><span class="live-dot"></span><b>MARKET LIVE</b><small>AUTO 30S // {refreshed}</small></div>
    </section>"""


def active_positions_strip(rows):
    """Show frozen paper entries and live progress toward the executable sell target."""
    positions = [row for row in rows if safe_float(row.get("position_dollars"), 0) > 0]
    if not positions:
        return ""
    cards = []
    for row in positions:
        progress = safe_float(row.get("target_progress"))
        bar_width = max(0, min(100, progress * 100)) if np.isfinite(progress) else 0
        progress_label = pct(progress, 0) if np.isfinite(progress) else "—"
        distance = safe_float(row.get("distance_to_target"))
        if np.isfinite(distance) and distance <= 0:
            target_note = f"TARGET CLEARED BY {abs(distance):.1%}"
        elif np.isfinite(distance):
            target_note = f"{distance:.1%} TO TARGET"
        else:
            target_note = "WAITING FOR LIVE BID"
        entry_time = pd.to_datetime(row.get("entry_timestamp_utc"), utc=True, errors="coerce")
        timestamp = entry_time.strftime("%b %d // %H:%M UTC") if pd.notna(entry_time) else "ENTRY RECORDED"
        signal_class = {"SELL": "sell", "HOLD": "hold", "REVIEW": "review"}.get(row.get("signal"), "watch")
        cards.append(f"""
        <article class="active-position">
          <div class="active-position-top"><span class="taken-badge">BET TAKEN</span><i class="position-signal {signal_class}">{escape(row.get('signal', 'WATCH'))}</i></div>
          <h3>{escape(row['trade_side'])}</h3><small>{escape(row['fight'])}</small>
          <div class="position-prices">
            <div><span>ENTRY</span><b>{pct(row['entry_price'])}</b></div>
            <div><span>LIVE BID</span><b>{pct(row['current_bid'])}</b></div>
            <div><span>SELL TARGET</span><b>{pct(row['target_price'])}</b></div>
          </div>
          <div class="progress-label"><span>TARGET PROGRESS</span><b>{progress_label}</b></div>
          <div class="target-track"><i style="width:{bar_width:.0f}%"></i></div>
          <div class="position-foot"><span>{money(row['position_dollars'])} POSITION // {signed_money(row['unrealized_pnl'])} LIVE P&amp;L</span><b>{target_note}</b></div>
          <div class="entry-stamp">{timestamp}</div>
        </article>""")
    return f"""
    <section class="active-book">
      <header><div><span>OPEN PAPER POSITIONS</span><h2>RECORDED BETS // LIVE TARGET MONITOR</h2></div><b>{len(positions)} ACTIVE</b></header>
      <div class="active-grid">{''.join(cards)}</div>
    </section>"""


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
  .hero { min-height:176px; display:grid; grid-template-columns:minmax(0,1fr) 190px; align-items:center; gap:3rem; margin:0 -2.35rem 1.1rem; padding:1.75rem 2.35rem; background:#0b0c0e; border-bottom:1px solid var(--line); position:relative; overflow:hidden; }
  .hero:after { content:""; position:absolute; right:220px; top:-90px; width:9px; height:360px; background:var(--red); transform:rotate(22deg); opacity:.85; }
  .hero-copy { min-width:0; position:relative; z-index:1; }
  .hero .eyebrow { display:block; margin:0 0 .7rem; color:var(--red); font-size:.62rem; font-weight:950; font-style:italic; letter-spacing:.22em; }
  .hero h1 { position:static; display:block; margin:0; max-width:960px; color:var(--white); font-size:clamp(2.25rem,4.4vw,4.35rem); font-weight:950; font-style:italic; text-transform:uppercase; line-height:.88; letter-spacing:-.055em; }
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
  .event-strip { margin:1rem 0 0; display:grid; grid-template-columns:minmax(0,1.7fr) minmax(220px,.85fr) 230px; background:#0e0f11; border:1px solid var(--line); border-left:7px solid var(--red); }
  .event-strip>div { min-height:94px; padding:1rem 1.1rem; display:flex; flex-direction:column; justify-content:center; border-right:1px solid var(--line); }
  .event-strip>div:last-child { border-right:0; }
  .event-strip span,.event-strip small { color:#7f8288; font-size:.5rem; font-weight:900; letter-spacing:.14em; text-transform:uppercase; }
  .event-strip h2 { margin:.18rem 0; color:#fff; font-size:1.45rem; font-weight:950; font-style:italic; text-transform:uppercase; line-height:1; }
  .event-strip b { margin:.2rem 0; color:#fff; font-size:.92rem; font-weight:950; text-transform:uppercase; }
  .top-trade { background:#151618; }
  .top-trade small { color:#ff3932; }
  .tape-status { align-items:flex-start; }
  .tape-status b { color:#fff; font-size:.75rem; }
  .live-dot { width:8px; height:8px; margin-bottom:.35rem; background:var(--red); border-radius:50%; box-shadow:0 0 0 4px rgba(225,6,0,.14); }
  .active-book { margin:1rem 0 1.2rem; border:1px solid var(--line); background:#0d0e10; }
  .active-book>header { min-height:70px; padding:.9rem 1.1rem; display:flex; align-items:center; justify-content:space-between; gap:1rem; border-left:7px solid var(--red); border-bottom:1px solid var(--line); }
  .active-book>header span { color:var(--red); font-size:.5rem; font-weight:950; font-style:italic; letter-spacing:.18em; }
  .active-book>header h2 { margin:.2rem 0 0; color:#fff; font-size:1.08rem; font-weight:950; font-style:italic; letter-spacing:.02em; text-transform:uppercase; }
  .active-book>header>b { padding:.5rem .7rem; color:#fff; background:var(--red); font-size:.58rem; font-weight:950; letter-spacing:.12em; white-space:nowrap; }
  .active-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); }
  .active-position { min-width:0; padding:1rem 1.05rem; border-right:1px solid var(--line); border-bottom:1px solid var(--line); background:#111214; }
  .active-position:nth-child(3n) { border-right:0; }
  .active-position-top { display:flex; align-items:center; justify-content:space-between; gap:.75rem; }
  .taken-badge { padding:.3rem .48rem; background:var(--red); color:#fff; font-size:.48rem; font-weight:950; font-style:italic; letter-spacing:.12em; }
  .position-signal { padding:.28rem .45rem; border:1px solid #4d5056; color:#fff; font-size:.48rem; font-weight:950; font-style:normal; letter-spacing:.1em; }
  .position-signal.sell { background:#fff; border-color:#fff; color:#090a0c; }
  .position-signal.review { background:var(--amber); border-color:var(--amber); color:#090a0c; }
  .position-signal.watch { color:#8d9096; }
  .active-position h3 { margin:.8rem 0 .1rem; color:#fff; font-size:1rem; font-weight:950; font-style:italic; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .active-position>small { display:block; min-height:1.8rem; color:#777a80; font-size:.5rem; font-weight:800; text-transform:uppercase; }
  .position-prices { margin:.7rem 0; display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
  .position-prices>div { padding:.58rem .45rem; border-right:1px solid var(--line); }
  .position-prices>div:first-child { padding-left:0; }
  .position-prices>div:last-child { padding-right:0; border-right:0; }
  .position-prices span,.progress-label span { display:block; color:#777a80; font-size:.45rem; font-weight:950; letter-spacing:.1em; }
  .position-prices b { display:block; margin-top:.12rem; color:#fff; font-size:1rem; font-weight:950; font-style:italic; font-variant-numeric:tabular-nums; }
  .position-prices>div:nth-child(2) b { color:#ff3932; }
  .progress-label { display:flex; align-items:center; justify-content:space-between; }
  .progress-label b { color:#fff; font-size:.68rem; font-weight:950; font-style:italic; }
  .target-track { height:7px; margin:.35rem 0 .55rem; overflow:hidden; background:#292b2f; }
  .target-track i { display:block; height:100%; background:var(--red); }
  .position-foot { display:flex; justify-content:space-between; gap:.8rem; color:#9b9ea4; font-size:.48rem; font-weight:900; letter-spacing:.04em; text-transform:uppercase; }
  .position-foot b { color:#fff; text-align:right; }
  .entry-stamp { margin-top:.45rem; color:#5e6167; font-size:.44rem; font-weight:900; letter-spacing:.08em; }
  .fight-board { border-left:1px solid var(--line); border-right:1px solid var(--line); }
  .fight-row { min-height:112px; display:grid; grid-template-columns:58px minmax(270px,1.55fr) minmax(210px,1.15fr) 105px 112px 112px 150px; align-items:stretch; background:#111214; border-bottom:1px solid var(--line); }
  .fight-row:nth-child(even) { background:#151619; }
  .fight-row>div { min-width:0; padding:.9rem .85rem; display:flex; flex-direction:column; justify-content:center; border-right:1px solid var(--line); }
  .fight-row>div:last-child { border-right:0; }
  .bout-id { align-items:center; background:#0b0c0e; }
  .bout-id span,.trade-side span,.price-cell span { color:#777a80; font-size:.48rem; font-weight:950; letter-spacing:.13em; text-transform:uppercase; }
  .bout-id b { color:#5e6167; font-size:1.05rem; font-style:italic; }
  .matchup strong { color:#fff; font-size:.82rem; font-weight:950; text-transform:uppercase; line-height:1.1; }
  .matchup i { margin:.18rem 0; color:var(--red); font-size:.5rem; font-weight:950; font-style:italic; }
  .matchup small,.trade-side small,.price-cell small { margin-top:.45rem; color:#777a80; font-size:.5rem; font-weight:800; line-height:1.3; text-transform:uppercase; }
  .trade-side b { margin-top:.28rem; color:#f1f1ee; font-size:.72rem; font-weight:950; text-transform:uppercase; }
  .trade-side small { max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .price-cell { text-align:right; align-items:flex-end; }
  .price-cell b { margin-top:.16rem; color:#fff; font-size:1.25rem; font-weight:950; font-style:italic; font-variant-numeric:tabular-nums; }
  .price-cell.live span i { width:6px; height:6px; display:inline-block; margin-right:.35rem; border-radius:50%; background:var(--red); }
  .price-cell.edge b { color:#ff3932; }
  .fight-action { align-items:center; text-align:center; }
  .fight-action span { width:100%; padding:.55rem .4rem; color:#fff; font-size:.78rem; font-weight:950; font-style:italic; letter-spacing:.12em; text-transform:uppercase; }
  .fight-action b { margin-top:.45rem; color:#85888e; font-size:.48rem; font-weight:900; letter-spacing:.08em; }
  .bet-card .fight-action { background:var(--red); }
  .bet-card .fight-action span,.bet-card .fight-action b { color:#fff; }
  .pass-card .fight-action { background:#202226; }
  .pass-card .fight-action span { color:#a0a3a9; }
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
  @media(max-width:1100px){.active-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.active-position:nth-child(3n){border-right:1px solid var(--line)}.active-position:nth-child(2n){border-right:0}.fight-row{grid-template-columns:48px minmax(230px,1.4fr) minmax(190px,1fr) 90px 100px 105px 125px}.event-strip{grid-template-columns:1fr 1fr}.tape-status{grid-column:1/-1;border-top:1px solid var(--line)}}
  @media(max-width:700px){.block-container{padding:1rem}.brandbar{margin:-1rem -1rem 0;padding:.85rem 1rem}.model-pill,.holdout,.hero:after{display:none}.hero{display:block;min-height:0;margin:0 -1rem 1rem;padding:1.6rem 1rem}.hero h1{font-size:2.65rem}.hero .hero-sub{line-height:1.55}.stTabs [data-baseweb="tab"]{padding:0 .55rem}.formula-row{grid-template-columns:1fr}.event-strip{display:block}.event-strip>div{min-height:76px;border-right:0;border-bottom:1px solid var(--line)}.active-grid{display:block}.active-position{border-right:0}.active-book>header h2{font-size:.9rem}.fight-row{display:grid;grid-template-columns:44px 1fr 104px}.fight-row .bout-id{grid-row:1/3}.fight-row .matchup{grid-column:2}.fight-row .trade-side{grid-column:2;grid-row:2}.fight-row .price-cell{display:none}.fight-row .fight-action{grid-column:3;grid-row:1/3}}
</style>
""", unsafe_allow_html=True)

bundle, fighters = assets()
if "log" not in st.session_state:
    st.session_state["log"] = reconcile_paper_results(load_prediction_log(STATE_DIR))
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
    auto_refresh = st.checkbox("Auto-refresh live prices every 30 seconds", value=True)
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

auto_requested = bool(st.session_state.pop("_auto_refresh_requested", False))
if auto_requested:
    event_search = st.session_state.get("active_event_search", event_search)

if run or auto_requested:
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
            log = reconcile_paper_results(log)
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
        st.session_state["active_event_search"] = event_search
        st.session_state["live_active"] = True
        st.session_state["_last_auto_refresh_tick"] = time.monotonic()
        st.session_state["excel"] = build_excel(
            analyses, bundle, log, bankroll, event_search,
            cost_buffer=cost_buffer, min_edge=min_edge, min_prior_fights=int(min_fights),
            convergence=convergence, max_card_exposure=max_card_exposure,
            capture_target=capture_target, minimum_exit_return=minimum_exit_return,
            stop_loss=stop_loss,
            convergence_rows=convergence_rows, market_history=market_history,
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
    convergence_rows = st.session_state.get("convergence_rows", [])
    st.markdown(event_strip(analyses), unsafe_allow_html=True)
    st.markdown(f"<div class='statusline'><b>SYNCED</b> // {st.session_state['elapsed']:.2f}s // EXECUTABLE ASK DRIVES ENTRY // EXECUTABLE BID DRIVES EXIT // {escape(st.session_state['market_mode'])}</div>", unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Bets", len(bets), f"{len(analyses)} fights screened")
    metric_columns[1].metric("Top edge", pct(top_edge), "net of buffer")
    metric_columns[2].metric("Card risk", money(total_risk), f"{total_risk / bankroll:.1%} of bankroll")
    metric_columns[3].metric("Backtest", f"{bundle['metrics']['accuracy']:.1%}", f"{bundle['metrics']['holdout_fights']:,} unseen fights")
    active_strip = active_positions_strip(convergence_rows)
    if active_strip:
        st.markdown(active_strip, unsafe_allow_html=True)

    dashboard_tab, convergence_tab, performance_tab, model_tab, raw_tab, history_tab = st.tabs(["Fight Board", "Positions", "Backtester", "Model Math", "Raw Data", "Audit Log"])
    with dashboard_tab:
        st.markdown(fight_board(analyses, convergence_rows), unsafe_allow_html=True)
        with st.expander("Detailed pricing table"):
            st.markdown(dashboard_table(analyses), unsafe_allow_html=True)
        filename = re.sub(r"[^a-z0-9]+", "_", event_search.lower()).strip("_") or "ufc"
        st.download_button("Download detailed Excel", st.session_state["excel"], file_name=f"{filename}_edge_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with convergence_tab:
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
        paper = log[log["action"] == "BET"].copy() if len(log) else pd.DataFrame()
        open_paper = paper[paper["status"] == "OPEN"].copy() if len(paper) else pd.DataFrame()
        completed_paper = paper[paper["status"] == "COMPLETED"].copy() if len(paper) else pd.DataFrame()
        paper_metrics = st.columns(6)
        paper_metrics[0].metric("Open bets", realized["open"])
        paper_metrics[1].metric("Settled", realized["graded"])
        paper_metrics[2].metric("Wins", realized["wins"])
        paper_metrics[3].metric("Losses", realized["losses"])
        paper_metrics[4].metric("Win rate", pct(realized["win_rate"]))
        paper_metrics[5].metric("Realized P&L", money(realized["pnl"]))
        st.markdown("<div class='board-title'><small>PAPER LEDGER</small><h2>EVERY BET SIGNAL IS RECORDED AT THE FIRST LIVE ASK</h2></div>", unsafe_allow_html=True)
        if len(open_paper):
            st.markdown("#### Open paper bets")
            open_paper["fight"] = open_paper["fighter_a"] + " vs " + open_paper["fighter_b"]
            open_columns = ["entry_timestamp_utc", "event", "fight", "pick", "model_probability", "entry_price", "net_edge", "position_dollars", "potential_profit", "status"]
            st.dataframe(open_paper[[column for column in open_columns if column in open_paper.columns]], width="stretch", hide_index=True, column_config={
                "model_probability": st.column_config.NumberColumn("Model P", format="percent"),
                "entry_price": st.column_config.NumberColumn("Entry Price", format="percent"),
                "net_edge": st.column_config.NumberColumn("Net Edge", format="percent"),
                "position_dollars": st.column_config.NumberColumn("Stake", format="dollar"),
                "potential_profit": st.column_config.NumberColumn("Profit If Win", format="dollar"),
            })
        else:
            st.info("No open paper bets. A row is added automatically the first time a fight qualifies as BET.")
        if len(completed_paper):
            st.markdown("#### Completed paper bets")
            completed_paper["fight"] = completed_paper["fighter_a"] + " vs " + completed_paper["fighter_b"]
            completed_paper = completed_paper.sort_values("settled_timestamp_utc", ascending=False)
            completed_columns = ["event_date", "fight", "pick", "winner", "entry_price", "position_dollars", "outcome", "return_pct", "pnl", "exit_type"]
            st.dataframe(completed_paper[[column for column in completed_columns if column in completed_paper.columns]], width="stretch", hide_index=True, column_config={
                "entry_price": st.column_config.NumberColumn("Entry Price", format="percent"),
                "position_dollars": st.column_config.NumberColumn("Stake", format="dollar"),
                "return_pct": st.column_config.NumberColumn("Return", format="percent"),
                "pnl": st.column_config.NumberColumn("P&L", format="dollar"),
            })
            curve = completed_paper.copy()
            curve["settled_timestamp_utc"] = pd.to_datetime(curve["settled_timestamp_utc"], utc=True, errors="coerce")
            curve["pnl"] = pd.to_numeric(curve["pnl"], errors="coerce").fillna(0)
            curve = curve.dropna(subset=["settled_timestamp_utc"]).sort_values("settled_timestamp_utc")
            if len(curve):
                curve["Cumulative P&L"] = curve["pnl"].cumsum()
                st.line_chart(curve.set_index("settled_timestamp_utc")[["Cumulative P&L"]], color="#E10600", height=270)
        else:
            st.caption("Completed bets will appear here after the official UFC result is published and the app refreshes.")
        st.download_button(
            "Download paper ledger CSV", log.to_csv(index=False).encode("utf-8"),
            file_name="ufc_paper_trade_ledger.csv", mime="text/csv",
        )
        with st.expander("Restore a prior paper ledger"):
            ledger_file = st.file_uploader("Paper ledger CSV", type=["csv"], key="paper_ledger_restore")
            if ledger_file is not None and st.button("Restore ledger", key="restore_ledger_button"):
                try:
                    restored = merge_prediction_log(STATE_DIR, pd.read_csv(io.BytesIO(ledger_file.getvalue())))
                    st.session_state["log"] = reconcile_paper_results(restored)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ledger could not be restored: {exc}")
        st.markdown("<div class='board-title'><small>FROZEN MODEL TEST</small><h2>UNSEEN HISTORICAL FIGHTS</h2></div>", unsafe_allow_html=True)
        columns = st.columns(4)
        columns[0].metric("Unseen holdout", f"{bundle['metrics']['holdout_fights']:,}")
        columns[1].metric("Accuracy", f"{bundle['metrics']['accuracy']:.1%}")
        columns[2].metric("Brier score", f"{bundle['metrics']['brier']:.3f}")
        columns[3].metric("ROC AUC", f"{bundle['metrics']['auc']:.3f}")
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


@st.fragment(run_every=30)
def live_refresh_clock():
    """Trigger a full market reprice every 30 seconds while a card is active."""
    if not auto_refresh or not st.session_state.get("live_active"):
        return
    now = time.monotonic()
    last = float(st.session_state.get("_last_auto_refresh_tick", now))
    if now - last >= 25:
        st.session_state["_last_auto_refresh_tick"] = now
        st.session_state["_auto_refresh_requested"] = True
        st.rerun()


live_refresh_clock()
st.markdown(f"<div class='fineprint'>{bundle['version']} // RESEARCH MODEL // PROBABILITIES ARE ESTIMATES, NOT GUARANTEES</div>", unsafe_allow_html=True)
