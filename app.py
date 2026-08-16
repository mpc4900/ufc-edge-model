from __future__ import annotations

import io
import os
import re
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

from excel_report import build_excel
import model_engine as engine


# Attribute binding avoids a hard startup failure when Streamlit reloads
# app.py a few seconds before model_engine.py during a GitHub redeploy.
analyze_card = engine.analyze_card
canonical_name = engine.canonical_name
discover_card = engine.discover_card
discover_event_options = engine.discover_event_options
feature_vector = engine.feature_vector
fetch_completed_results_for_log = engine.fetch_completed_results_for_log
fetch_market_rows = engine.fetch_market_rows
grade_prediction_log = engine.grade_prediction_log
load_assets = engine.load_assets
load_prediction_log = engine.load_prediction_log
local_drivers = engine.local_drivers
mark_prediction_log = getattr(engine, "mark_prediction_log", lambda state_dir, rows: load_prediction_log(state_dir))
merge_prediction_log = engine.merge_prediction_log
normalize_prediction_log = engine.normalize_prediction_log
pair_key = engine.pair_key
price_to_american = engine.price_to_american
realized_metrics = engine.realized_metrics
safe_float = engine.safe_float
update_market_history = engine.update_market_history
update_prediction_log = engine.update_prediction_log


def repair_prediction_log(log):
    """Use the current ledger repair, with a compatibility path for rolling deploys."""
    if hasattr(engine, "repair_prediction_log"):
        return engine.repair_prediction_log(log)
    work = normalize_prediction_log(log)
    if work.empty:
        return work
    keys, labels = [], []
    for record in work.to_dict("records"):
        raw = re.sub(r"\s+", " ", str(record.get("event") or "")).strip()
        normalized = canonical_name(raw)
        date = pd.to_datetime(record.get("event_date"), errors="coerce")
        date_key = date.strftime("%Y-%m-%d") if pd.notna(date) else "undated"
        numbered = re.search(r"\bufc\s+(\d+)\b", normalized)
        if numbered:
            number = numbered.group(1)
            keys.append(f"ufc-{number}|{date_key}")
            labels.append("UFC 330: Makhachev vs. Machado Garry" if number == "330" else f"UFC {number}")
        elif normalized.startswith("ufc fight night"):
            keys.append(f"ufc-fight-night|{date_key}")
            labels.append("UFC Fight Night")
        else:
            keys.append("")
            labels.append("")
    work["_event_key"], work["_event_label"] = keys, labels
    work["_bout_key"] = ["|".join(pair_key(a, b)) for a, b in zip(work["fighter_a"], work["fighter_b"])]
    work = work[
        work["_event_key"].ne("")
        & work["fighter_a"].astype(str).str.strip().ne("")
        & work["fighter_b"].astype(str).str.strip().ne("")
    ].copy()
    selected = []
    for (_, _), group in work.groupby(["_event_key", "_bout_key"], sort=False):
        ranked = group.copy()
        ranked["_quality"] = (
            ranked["entry_source"].astype(str).str.contains("recovered pre-fight snapshot", case=False, regex=False).astype(int) * 1000
            + ranked["status"].eq("COMPLETED").astype(int) * 200
            + ranked["action"].eq("BET").astype(int) * 40
            + ranked["entry_price"].notna().astype(int) * 10
        )
        row = ranked.sort_values("_quality").iloc[-1].copy()
        row["event"] = row["_event_label"]
        row["prediction_id"] = f"{canonical_name(row['event'])}|{row['_bout_key']}"
        selected.append(row)
    repaired = pd.DataFrame(selected).drop(
        columns=["_event_key", "_event_label", "_bout_key", "_quality"], errors="ignore"
    )
    return normalize_prediction_log(repaired).reset_index(drop=True)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
STATE_DIR = Path(os.getenv("UFC_EDGE_STATE_DIR", str(ROOT / "state")))
SEED_LOG = ROOT / "seed_prediction_log.csv"

st.set_page_config(page_title="UFC Edge Ledger", page_icon="🥊", layout="wide", initial_sidebar_state="collapsed")


def pct(value, digits=1):
    value = safe_float(value)
    return "—" if not np.isfinite(value) else f"{value:.{digits}%}"


def money(value, signed=False):
    value = safe_float(value)
    if not np.isfinite(value):
        return "—"
    if signed:
        return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"
    return f"${value:,.0f}"


def odds(value):
    price = safe_float(value)
    if not 0 < price < 1:
        return "—"
    return f"{price:.1%} / {price_to_american(price):+d}"


@st.cache_resource(show_spinner=False)
def assets():
    return load_assets(DATA_DIR)


@st.cache_data(ttl=60, show_spinner=False)
def active_event_options():
    # UFC 330 remains an offline engine example but is now a settled event.
    return [item for item in discover_event_options() if item.get("value") != "UFC 330"]


def market_rows(card_json, manual_odds):
    """Always request a fresh executable order-book snapshot."""
    card = pd.read_json(io.StringIO(card_json)).to_dict("records")
    return fetch_market_rows(card, manual_odds)


@st.cache_data(ttl=600, show_spinner=False)
def completed_for_open_log(open_json):
    frame = pd.read_json(io.StringIO(open_json), orient="records")
    return fetch_completed_results_for_log(frame)


def seed_and_grade_ledger():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = load_prediction_log(STATE_DIR)
    if SEED_LOG.exists():
        log = merge_prediction_log(STATE_DIR, pd.read_csv(SEED_LOG))
    log = repair_prediction_log(log)
    log.to_csv(STATE_DIR / "prediction_log.csv", index=False)
    open_rows = log[(log["action"] == "BET") & (log["status"] == "OPEN")]
    if len(open_rows):
        columns = ["prediction_id", "event_date", "event", "fighter_a", "fighter_b", "action", "status"]
        try:
            completed = completed_for_open_log(open_rows[columns].to_json(orient="records"))
            if completed:
                log = grade_prediction_log(STATE_DIR, completed)
        except Exception:
            pass
    log = repair_prediction_log(log)
    log.to_csv(STATE_DIR / "prediction_log.csv", index=False)
    return log


def uploaded_texts(files):
    texts = []
    for uploaded in files or []:
        try:
            content = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower()
            if suffix == ".pdf":
                text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
            elif suffix == ".docx":
                text = "\n".join(paragraph.text for paragraph in Document(io.BytesIO(content)).paragraphs)
            else:
                text = content.decode("utf-8", errors="ignore")
            if text.strip():
                texts.append(text)
        except Exception:
            continue
    return texts


def event_groups(log):
    groups = []
    if not len(log):
        return groups
    for event, frame in log.groupby("event", sort=False, dropna=False):
        frame = frame.copy()
        bets = frame[frame["action"] == "BET"]
        settled = bets[bets["status"] == "COMPLETED"]
        wins = int((settled["outcome"] == "WIN").sum())
        losses = int((settled["outcome"] == "LOSS").sum())
        stake = pd.to_numeric(bets["position_dollars"], errors="coerce").fillna(0).sum()
        pnl = pd.to_numeric(bets["pnl"], errors="coerce").fillna(0).sum()
        event_date = pd.to_datetime(frame["event_date"], errors="coerce").max()
        groups.append({
            "event": str(event), "date": event_date, "frame": frame, "bets": len(bets),
            "wins": wins, "losses": losses, "stake": float(stake), "pnl": float(pnl),
            "roi": float(pnl / stake) if stake else np.nan,
            "win_rate": float(wins / len(settled)) if len(settled) else np.nan,
            "status": "SETTLED" if len(bets) and len(settled) == len(bets) else "OPEN",
        })
    return sorted(groups, key=lambda row: row["date"] if pd.notna(row["date"]) else pd.Timestamp.min, reverse=True)


def ui_event_key(name, date_value):
    date = pd.to_datetime(date_value, errors="coerce")
    date_key = date.strftime("%Y-%m-%d") if pd.notna(date) else "undated"
    normalized = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    numbered = re.search(r"\bufc\s+(\d+)\b", normalized)
    if numbered:
        return f"ufc-{numbered.group(1)}|{date_key}"
    if normalized.startswith("ufc fight night"):
        return f"ufc-fight-night|{date_key}"
    return f"{normalized}|{date_key}"


def option_event_key(option):
    bouts = option.get("bouts") or []
    if bouts:
        return ui_event_key(bouts[0].get("event"), bouts[0].get("event_date"))
    value = str(option.get("value") or "")
    if value.startswith("POLY_CARD::"):
        raw = value.split("::", 1)[1]
        name, _, date = raw.partition("|")
        return ui_event_key(name, date)
    return ui_event_key(value, "")


def event_menu(events, active_options):
    """One menu entry per event, with an accurate status label."""
    records = {}
    for event in events:
        key = ui_event_key(event["event"], event["date"])
        date_label = event["date"].strftime("%b %-d") if pd.notna(event["date"]) else "Date pending"
        records[key] = {
            "id": f"recorded::{key}", "kind": "recorded", "event": event,
            "label": f"{event['status']}  /  {event['event']}  /  {date_label}",
        }
    for option in active_options:
        key = option_event_key(option)
        records[key] = {
            "id": f"live::{key}", "kind": "live", "option": option,
            "label": option["label"],
        }
    menu = list(records.values())
    menu.sort(key=lambda item: (
        0 if item["kind"] == "recorded" and item["event"]["status"] == "SETTLED" else 1,
        -(item["event"]["date"].value if item["kind"] == "recorded" and pd.notna(item["event"]["date"]) else 0),
    ))
    return menu


def record_raw_inputs(record, bundle, fighters):
    vector, _, experience, raw_inputs = feature_vector(
        record["fighter_a"], record["fighter_b"], fighters, bundle["features"]
    )
    pick_is_a = str(record.get("pick")) == str(record.get("fighter_a"))
    drivers = local_drivers(bundle, vector)
    for driver in drivers:
        driver["impact_pick"] = driver["impact_a"] if pick_is_a else -driver["impact_a"]
    return experience, raw_inputs, sorted(drivers, key=lambda row: abs(row["impact_pick"]), reverse=True)


def analyses_from_log(frame, bundle, fighters):
    rows = []
    for record in frame.to_dict("records"):
        experience, raw_inputs, drivers = record_raw_inputs(record, bundle, fighters)
        price = safe_float(record.get("market_probability"))
        model_p = safe_float(record.get("model_probability"))
        rows.append({
            "event": record.get("event", ""), "event_date": record.get("event_date", ""),
            "fighter_a": record.get("fighter_a", ""), "fighter_b": record.get("fighter_b", ""),
            "winner": record.get("winner", ""), "fight_url": "", "pick": record.get("pick", ""),
            "trade_side": record.get("pick", ""), "likely_winner": record.get("pick", ""),
            "likely_probability": model_p, "model_probability": model_p,
            "probability_a": model_p if record.get("pick") == record.get("fighter_a") else 1 - model_p,
            "probability_b": model_p if record.get("pick") == record.get("fighter_b") else 1 - model_p,
            "market_probability": price, "live_ask": price, "live_bid": safe_float(record.get("exit_price")),
            "exit_target": safe_float(record.get("target_price")), "scenario_move": np.nan,
            "scenario_return": np.nan, "american_odds": price_to_american(price),
            "net_edge": safe_float(record.get("net_edge")), "action": record.get("action", "NO BET"),
            "position_dollars": safe_float(record.get("position_dollars"), 0), "experience": experience,
            "why": record.get("decision_reason", ""), "drivers": drivers[:3], "raw_inputs": raw_inputs,
            "research_shift": 0, "market_source": record.get("entry_source", ""),
            "market_url": record.get("entry_market_url", ""), "market_timestamp": record.get("entry_timestamp_utc", ""),
            "as_of_utc": record.get("timestamp_utc", ""), "model_version": record.get("model_version", bundle["version"]),
        })
    return rows


def kpi_strip(items):
    cells = [f"<div class='kpi {tone}'><span>{escape(label)}</span><b>{escape(value)}</b><small>{escape(note)}</small></div>" for label, value, note, tone in items]
    st.markdown(f"<section class='kpi-strip'>{''.join(cells)}</section>", unsafe_allow_html=True)


def event_header(event):
    date_label = event["date"].strftime("%B %-d, %Y") if pd.notna(event["date"]) else "Date unavailable"
    record_value = str(event.get("record_value", f"{event['wins']}-{event['losses']}"))
    record_label = str(event.get("record_label", "MODEL RECORD"))
    st.markdown(
        f"<section class='event-head'><div><span>{event['status']} / {date_label}</span><h1>{escape(event['event'])}</h1></div>"
        f"<div class='event-record'><b>{escape(record_value)}</b><span>{escape(record_label)}</span></div></section>",
        unsafe_allow_html=True,
    )


def bets_table(frame):
    bets = frame[frame["action"] == "BET"].copy()
    if not len(bets):
        st.info("No recorded positions for this event.")
        return
    bets["Fight"] = bets["fighter_a"] + " vs " + bets["fighter_b"]
    bets["Bet"] = bets["pick"]
    bets["Entry"] = bets["entry_price"]
    bets["Odds"] = bets["entry_price"].map(odds)
    bets["Stake"] = bets["position_dollars"]
    bets["Result"] = bets["outcome"].where(bets["outcome"].astype(str).str.strip().ne(""), bets["status"])
    bets["P&L"] = bets["pnl"]
    bets["Why"] = bets["decision_reason"]
    bets["Price source"] = bets["entry_source"]
    st.dataframe(
        bets[["Fight", "Bet", "Entry", "Odds", "Stake", "Result", "P&L", "Price source", "Why"]],
        width="stretch", hide_index=True,
        column_config={
            "Entry": st.column_config.NumberColumn("Polymarket entry", format="percent"),
            "Stake": st.column_config.NumberColumn("Capital", format="dollar"),
            "P&L": st.column_config.NumberColumn("Net P&L", format="dollar"),
            "Price source": st.column_config.TextColumn(width="medium"),
            "Why": st.column_config.TextColumn("Primary reason", width="large"),
        },
    )


def math_detail(record, bundle, fighters):
    experience, raw_inputs, drivers = record_raw_inputs(record, bundle, fighters)
    model_p = safe_float(record.get("model_probability"))
    entry = safe_float(record.get("market_probability"))
    edge = safe_float(record.get("net_edge"))
    pnl_value = safe_float(record.get("pnl"), 0)
    stake = safe_float(record.get("position_dollars"), 0)
    result_formula = (
        f"{money(stake)} × (1 ÷ {entry:.1%} − 1) = {money(pnl_value, signed=True)}"
        if record.get("outcome") == "WIN" and 0 < entry < 1
        else f"Binary contract settled at $0; loss = {money(stake)}" if record.get("outcome") == "LOSS"
        else "No capital was deployed."
    )
    st.markdown(
        f"<section class='math-panel'><div><span>MODEL FAIR VALUE</span><b>{pct(model_p)}</b></div>"
        f"<div><span>POLYMARKET ENTRY</span><b>{pct(entry)}</b></div><div><span>COST BUFFER</span><b>2.0%</b></div>"
        f"<div><span>NET EDGE</span><b>{pct(edge)}</b></div></section>", unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='equation'>{pct(model_p)} fair value − {pct(entry)} entry − 2.0% cost buffer = "
        f"<strong>{pct(edge)} net edge</strong><br>{escape(result_formula)}</div>", unsafe_allow_html=True,
    )
    driver_frame = pd.DataFrame([{"Factor": item["factor"], "Raw difference": item["value"], "Probability impact": item["impact_pick"]} for item in drivers])
    left, right = st.columns([1.05, 1.35])
    with left:
        st.markdown("#### Probability drivers")
        st.dataframe(driver_frame, width="stretch", hide_index=True, column_config={
            "Raw difference": st.column_config.NumberColumn(format="%.3f"),
            "Probability impact": st.column_config.NumberColumn(format="percent"),
        })
    with right:
        st.markdown(f"#### Raw inputs / {experience} prior fights for the lower-experience fighter")
        raw = pd.DataFrame(raw_inputs).rename(columns={"Fighter A": record["fighter_a"], "Fighter B": record["fighter_b"]})
        st.dataframe(raw, width="stretch", hide_index=True)


@st.fragment(run_every=30)
def live_event_board(option, bankroll, min_edge, cost_buffer, min_fights, max_card_exposure, manual_odds, research_texts):
    """Refresh the public Polymarket CLOB snapshot every 30 seconds."""
    key_suffix = re.sub(r"[^a-z0-9]+", "-", str(option.get("value") or "live").lower()).strip("-")[-48:]
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with left:
        custom_fight = st.text_input(
            "Price one fight instead",
            placeholder="Fighter A vs Fighter B",
            key=f"custom-fight-{key_suffix}",
        )
    with right:
        st.button("Refresh now", width="stretch", key=f"refresh-{key_suffix}")

    search = custom_fight.strip() if re.search(r"\s+vs\.?\s+", custom_fight, re.I) else option["value"]
    try:
        started = time.perf_counter()
        card = discover_card(search)
        rows = market_rows(pd.DataFrame(card).to_json(), manual_odds)
        current = analyze_card(
            card, rows, bundle, fighters, bankroll=bankroll,
            research_texts=research_texts, min_edge=min_edge,
            cost_buffer=cost_buffer, min_prior_fights=int(min_fights),
            max_card_exposure=max_card_exposure,
            polymarket_only=not bool(manual_odds.strip()),
        )
        update_prediction_log(STATE_DIR, current)
        live_log = load_prediction_log(STATE_DIR)
        marks = []
        for row in current:
            prediction_id = f"{canonical_name(row['event'])}|{'|'.join(pair_key(row['fighter_a'], row['fighter_b']))}"
            match = live_log[
                (live_log["prediction_id"] == prediction_id)
                & (live_log["status"] == "OPEN")
                & (live_log["action"] == "BET")
            ]
            if match.empty:
                continue
            entry = safe_float(match.iloc[-1].get("entry_price"))
            stake = safe_float(match.iloc[-1].get("position_dollars"), 0)
            bid = safe_float(row.get("live_bid"))
            target = safe_float(row.get("exit_target"))
            unrealized_return = bid / entry - 1 if 0 < entry < 1 and 0 <= bid <= 1 else np.nan
            progress = (bid - entry) / (target - entry) if np.isfinite(bid) and target > entry else np.nan
            signal = "TARGET REACHED" if np.isfinite(bid) and bid >= target else ("MARKED PROFIT" if np.isfinite(unrealized_return) and unrealized_return > 0 else "OPEN")
            marks.append({
                "prediction_id": prediction_id, "current_bid": bid,
                "current_ask": row.get("live_ask"), "target_price": target,
                "target_progress": float(np.clip(progress, 0, 1)) if np.isfinite(progress) else np.nan,
                "unrealized_return": unrealized_return,
                "unrealized_pnl": stake * unrealized_return if np.isfinite(unrealized_return) else np.nan,
                "signal": signal,
            })
        if marks:
            mark_prediction_log(STATE_DIR, marks)
        update_market_history(STATE_DIR, current)
        elapsed = time.perf_counter() - started
        refreshed = datetime.now(ZoneInfo("America/New_York")).strftime("%b %-d, %-I:%M:%S %p ET")
        matched = sum(np.isfinite(safe_float(row.get("live_ask"))) for row in current)
        st.markdown(
            f"<div class='live-status'><i></i>POLYMARKET CLOB &nbsp; / &nbsp; UPDATED {escape(refreshed)}"
            f" &nbsp; / &nbsp; {matched} OF {len(card)} FIGHTS PRICED &nbsp; / &nbsp; {elapsed:.2f}s"
            f" &nbsp; / &nbsp; AUTO-REFRESHES EVERY 30 SECONDS</div>",
            unsafe_allow_html=True,
        )
        live = pd.DataFrame([{
            "Fight": f"{row['fighter_a']} vs {row['fighter_b']}",
            "Position": row["trade_side"], "Model fair value": row["model_probability"],
            "Polymarket ask": row["live_ask"], "Net edge": row["net_edge"],
            "Decision": row["action"], "Capital": row["position_dollars"],
            "Reason": row["why"], "Market": row.get("market_url", ""),
        } for row in current])
        st.dataframe(
            live, width="stretch", hide_index=True,
            column_config={
                "Model fair value": st.column_config.NumberColumn(format="percent"),
                "Polymarket ask": st.column_config.NumberColumn(format="percent"),
                "Net edge": st.column_config.NumberColumn(format="percent"),
                "Capital": st.column_config.NumberColumn(format="dollar"),
                "Reason": st.column_config.TextColumn(width="large"),
                "Market": st.column_config.LinkColumn("Polymarket"),
            },
        )
        if matched == 0:
            st.warning("No executable Polymarket order books matched this card. No position was recorded from substitute odds.")
    except Exception as exc:
        st.error(f"Live price refresh failed: {exc}")


st.markdown("""
<style>
  :root{--red:#d20a0a;--ink:#070707;--line:#2b2b2b;--paper:#f2f2ef;--green:#36b56b}
  .stApp{background:var(--ink);color:var(--paper);font-family:"Arial Narrow","Roboto Condensed","Helvetica Neue",Arial,sans-serif}
  [data-testid="stIconMaterial"],.material-symbols-rounded{font-family:"Material Symbols Rounded"!important}
  .block-container{max-width:1500px;padding:0 2.5rem 4rem}[data-testid="stHeader"]{background:transparent}[data-testid="stToolbar"]{visibility:hidden}
  .mast{height:66px;margin:0 -2.5rem;display:flex;align-items:center;justify-content:space-between;padding:0 2.5rem;border-bottom:6px solid var(--red);background:#000}
  .mast .wordmark{font-size:1.32rem;font-weight:950;font-style:italic;letter-spacing:-.04em}.mast .wordmark i{color:var(--red);font-style:inherit}
  .mast .descriptor{color:#8f8f8f;font-size:.64rem;font-weight:800;letter-spacing:.18em}.tape{margin:0 -2.5rem 1.35rem;padding:.55rem 2.5rem;background:#181818;border-bottom:1px solid #333;color:#a6a6a6;font-size:.6rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase}.tape b{color:#fff}.tape i{display:inline-block;width:7px;height:7px;background:var(--red);margin-right:.5rem}
  div[data-testid="stRadio"]>label{display:none}div[data-testid="stRadio"] div[role="radiogroup"]{display:flex;border-top:1px solid #333;border-bottom:1px solid #333;margin:0 0 1.25rem}div[data-testid="stRadio"] label{flex:0 0 auto;margin:0;padding:.9rem 1.6rem;border-right:1px solid #333;background:#0d0d0d;color:#888;font-size:.68rem;font-weight:900;letter-spacing:.16em}div[data-testid="stRadio"] label:has(input:checked){background:var(--red);color:#fff}
  .event-head{display:flex;justify-content:space-between;align-items:flex-end;min-height:118px;padding:1.2rem 0 1.05rem;border-bottom:1px solid #3a3a3a}.event-head span{color:var(--red);font-size:.62rem;font-weight:900;letter-spacing:.18em;text-transform:uppercase}.event-head h1{margin:.25rem 0 0;color:#fff;font-size:clamp(1.8rem,3vw,3.2rem);font-weight:950;font-style:italic;text-transform:uppercase;letter-spacing:-.045em;line-height:.95}.event-record{text-align:right;border-left:4px solid var(--red);padding-left:1.2rem}.event-record b{display:block;font-size:2.6rem;line-height:.9}.event-record span{color:#999}
  .kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid #333;margin-bottom:1.15rem}.kpi{min-height:105px;padding:1.15rem 1rem 1rem 0;border-right:1px solid #333}.kpi+.kpi{padding-left:1rem}.kpi:last-child{border-right:0}.kpi span,.kpi small{display:block;color:#858585;font-size:.58rem;font-weight:900;letter-spacing:.14em;text-transform:uppercase}.kpi b{display:block;margin:.18rem 0;color:#fff;font-size:2rem;font-weight:950;font-style:italic}.kpi.positive b{color:var(--green)}.kpi.negative b{color:#ff4c4c}
  h3{color:#fff!important;font-size:1rem!important;text-transform:uppercase;letter-spacing:.08em;border-left:4px solid var(--red);padding-left:.65rem}h4{color:#fff!important}div[data-testid="stDataFrame"]{border:1px solid #303030}
  div[data-testid="stSelectbox"] label p,div[data-testid="stTextInput"] label p,div[data-testid="stNumberInput"] label p,div[data-testid="stFileUploader"] label p{color:#8e8e8e;font-size:.58rem;font-weight:900;letter-spacing:.14em;text-transform:uppercase}div[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input{background:#111!important;color:#fff!important;border:1px solid #444!important;border-radius:0!important}.stButton button,.stDownloadButton button{border-radius:0!important;min-height:2.7rem;background:var(--red);color:#fff;border:0;font-size:.64rem;font-weight:950;letter-spacing:.12em;text-transform:uppercase}.stDownloadButton button{background:#151515;border:1px solid #555}
  .section-line{margin:1.3rem 0 .65rem;padding:.55rem 0;border-top:3px solid var(--red);border-bottom:1px solid #333;color:#fff;font-size:.72rem;font-weight:950;letter-spacing:.14em;text-transform:uppercase}.math-panel{display:grid;grid-template-columns:repeat(4,1fr);margin:.7rem 0 0;border:1px solid #333}.math-panel div{padding:.85rem 1rem;border-right:1px solid #333}.math-panel div:last-child{border-right:0}.math-panel span{display:block;color:#888;font-size:.56rem;font-weight:900;letter-spacing:.13em}.math-panel b{display:block;margin-top:.2rem;font-size:1.45rem;color:#fff}.equation{padding:1rem;border:1px solid #333;border-top:0;background:#101010;color:#b5b5b5;font:600 .82rem/1.7 "Courier New",monospace}.equation strong{color:#fff}
  .event-index{display:grid;grid-template-columns:2fr .7fr .7fr .9fr .9fr;border-top:1px solid #333;border-left:5px solid var(--red);background:#0f0f0f}.event-index>div{padding:.9rem 1rem;border-right:1px solid #333}.event-index span{display:block;color:#777;font-size:.55rem;font-weight:900;letter-spacing:.13em}.event-index b{display:block;color:#fff;margin-top:.2rem}.live-status{display:flex;align-items:center;gap:.55rem;margin:.55rem 0 1rem;color:#9a9a9a;font-size:.58rem;font-weight:900;letter-spacing:.13em;text-transform:uppercase}.live-status i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#35b86b;box-shadow:0 0 0 4px rgba(53,184,107,.12)}.history-intro{display:flex;justify-content:space-between;align-items:end;margin:1.2rem 0 .8rem;padding-bottom:.65rem;border-bottom:3px solid var(--red)}.history-intro b{color:#fff;font-size:.9rem;letter-spacing:.1em}.history-intro span{color:#777;font-size:.58rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.stExpander{border:1px solid #333!important;border-radius:0!important;background:#0d0d0d}section[data-testid="stSidebar"]{background:#0b0b0b;border-right:1px solid #333}section[data-testid="stSidebar"] *{color:#eee}.fineprint{margin-top:2.5rem;padding-top:.7rem;border-top:1px solid #333;color:#666;font-size:.55rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
  @media(max-width:900px){.block-container{padding:0 1rem 3rem}.mast,.tape{margin-left:-1rem;margin-right:-1rem;padding-left:1rem;padding-right:1rem}.kpi-strip{grid-template-columns:repeat(2,1fr)}.event-head{align-items:flex-start}.math-panel{grid-template-columns:repeat(2,1fr)}}
</style>
""", unsafe_allow_html=True)


bundle, fighters = assets()
log = seed_and_grade_ledger()
events = event_groups(log)
performance = realized_metrics(log)

st.markdown("<header class='mast'><div class='wordmark'><i>UFC</i> EDGE LEDGER</div>" f"<div class='descriptor'>{escape(bundle['version'])} / GRADIENT BOOSTING</div></header>", unsafe_allow_html=True)
st.markdown(f"<div class='tape'><i></i><b>MODEL TRACK RECORD</b> &nbsp; / &nbsp; {performance['graded']} SETTLED BETS &nbsp; / &nbsp; {money(performance['total_pnl'], signed=True)} NET P&amp;L &nbsp; / &nbsp; {bundle['metrics']['holdout_fights']:,} UNSEEN HOLDOUT FIGHTS</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### MODEL CONTROLS")
    bankroll = st.number_input("Bankroll", min_value=100.0, value=10_100.0, step=100.0)
    min_edge = st.number_input("Minimum net edge", min_value=0.0, max_value=0.25, value=0.03, step=0.005, format="%.3f")
    cost_buffer = st.number_input("Cost buffer", min_value=0.0, max_value=0.10, value=0.02, step=0.005, format="%.3f")
    min_fights = st.number_input("Minimum prior UFC fights", min_value=0, max_value=20, value=3, step=1)
    max_card_exposure = st.number_input("Maximum card exposure", min_value=0.01, max_value=0.50, value=0.20, step=0.01, format="%.2f")
    research_files = st.file_uploader("Optional research reports", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True)
    manual_odds = st.text_area("Optional manual American odds", placeholder="Fighter Name, +150")
    restore = st.file_uploader("Restore model ledger", type=["csv"])
    if restore is not None and st.button("Restore ledger"):
        merge_prediction_log(STATE_DIR, pd.read_csv(io.BytesIO(restore.getvalue())))
        st.rerun()

view = st.radio("Navigation", ["EVENT", "POSITIONS + P&L", "HISTORY"], horizontal=True, label_visibility="collapsed")

if view == "EVENT":
    active_options = active_event_options()
    menu = event_menu(events, active_options)
    if not menu:
        st.info("No tracked or upcoming UFC cards are available.")
    else:
        menu_by_id = {item["id"]: item for item in menu}
        selected_id = st.selectbox(
            "Event", list(menu_by_id), index=0,
            format_func=lambda item_id: menu_by_id[item_id]["label"],
        )
        selected = menu_by_id[selected_id]
        if selected["kind"] == "recorded":
            selected_event = selected["event"]
            event_header(selected_event)
            kpi_strip([
                ("Net P&L", money(selected_event["pnl"], signed=True), f"{pct(selected_event['roi'])} return on capital", "positive" if selected_event["pnl"] >= 0 else "negative"),
                ("Win rate", pct(selected_event["win_rate"]), f"{selected_event['wins']} wins / {selected_event['losses']} losses", ""),
                ("Capital used", money(selected_event["stake"]), f"across {selected_event['bets']} recorded bets", ""),
                ("Holdout accuracy", pct(bundle["metrics"]["accuracy"]), f"{bundle['metrics']['holdout_fights']:,} unseen fights", ""),
            ])
            st.markdown("<div class='section-line'>Positions taken at the original Polymarket ask</div>", unsafe_allow_html=True)
            bets_table(selected_event["frame"])
            bets = selected_event["frame"][selected_event["frame"]["action"] == "BET"]
            if len(bets):
                selected_pick = st.selectbox("Review a position", list(bets.index), format_func=lambda idx: f"{bets.loc[idx, 'pick']} / {bets.loc[idx, 'fighter_a']} vs {bets.loc[idx, 'fighter_b']}")
                math_detail(bets.loc[selected_pick].to_dict(), bundle, fighters)
            event_analyses = analyses_from_log(selected_event["frame"], bundle, fighters)
            excel_bytes = build_excel(event_analyses, bundle, log, bankroll, selected_event["event"], cost_buffer=cost_buffer, min_edge=min_edge, min_prior_fights=int(min_fights), max_card_exposure=max_card_exposure, fighters=fighters)
            st.download_button("Download full Excel audit", excel_bytes, file_name="UFC_Edge_Model_Audit.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            option = selected["option"]
            bouts = option.get("bouts") or []
            live_name = bouts[0].get("event") if bouts else option["label"].split("•")[1].strip()
            live_date = pd.to_datetime(bouts[0].get("event_date"), errors="coerce") if bouts else pd.NaT
            event_header({
                "event": live_name, "date": live_date,
                "status": option.get("status", "LIVE MARKET"), "wins": 0, "losses": 0,
                "record_value": len(bouts), "record_label": "FIGHTS ON CARD",
            })
            live_event_board(
                option, bankroll, min_edge, cost_buffer, int(min_fights),
                max_card_exposure, manual_odds, uploaded_texts(research_files),
            )

elif view == "POSITIONS + P&L":
    st.markdown("<section class='event-head'><div><span>MODEL TRACK RECORD</span><h1>POSITIONS + TOTAL P&L</h1></div>" f"<div class='event-record'><b>{performance['wins']}-{performance['losses']}</b><span>SETTLED RECORD</span></div></section>", unsafe_allow_html=True)
    kpi_strip([
        ("Total model P&L", money(performance["total_pnl"], signed=True), f"{pct(performance['net_return'])} on all recorded capital", "positive" if performance["total_pnl"] >= 0 else "negative"),
        ("Realized P&L", money(performance["pnl"], signed=True), f"{pct(performance['roi'])} settled ROI", "positive" if performance["pnl"] >= 0 else "negative"),
        ("Win rate", pct(performance["win_rate"]), f"{performance['graded']} settled bets", ""),
        ("Open risk", money(performance["open_risk"]), f"{performance['open']} open positions", ""),
    ])
    bets_table(log)
    settled = log[(log["action"] == "BET") & (log["status"] == "COMPLETED")].copy()
    if len(settled):
        settled["Fight"] = settled["pick"]
        settled["Net P&L"] = settled["pnl"]
        st.bar_chart(settled.sort_values("pnl").set_index("Fight")[["Net P&L"]], horizontal=True, color="#d20a0a", height=350)
    st.download_button("Download ledger CSV", log.to_csv(index=False).encode("utf-8"), file_name="UFC_Model_Ledger.csv", mime="text/csv")

else:
    settled_events = [event for event in events if event["status"] == "SETTLED" and event["bets"] > 0]
    st.markdown("<section class='event-head'><div><span>VERIFIED EVENT ARCHIVE</span><h1>BET HISTORY</h1></div>" f"<div class='event-record'><b>{len(settled_events)}</b><span>SETTLED EVENTS</span></div></section>", unsafe_allow_html=True)
    kpi_strip([
        ("Net P&L", money(performance["pnl"], signed=True), f"{pct(performance['roi'])} realized return", "positive" if performance["pnl"] >= 0 else "negative"),
        ("Win rate", pct(performance["win_rate"]), f"{performance['wins']} wins / {performance['losses']} losses", ""),
        ("Settled bets", f"{performance['graded']}", "recorded before each fight", ""),
        ("Capital used", money(performance["staked"]), "across settled positions", ""),
    ])
    if not settled_events:
        st.info("No settled events have been recorded yet.")
    else:
        history_ids = list(range(len(settled_events)))
        selected_history_id = st.selectbox(
            "Historical event", history_ids,
            format_func=lambda idx: f"{settled_events[idx]['event']}  /  {settled_events[idx]['date'].strftime('%b %-d, %Y') if pd.notna(settled_events[idx]['date']) else 'Date unavailable'}",
        )
        event = settled_events[selected_history_id]
        date_label = event["date"].strftime("%b %-d, %Y") if pd.notna(event["date"]) else "Date unavailable"
        st.markdown(f"<div class='event-index'><div><span>SETTLED / {date_label}</span><b>{escape(event['event'])}</b></div><div><span>RECORD</span><b>{event['wins']}-{event['losses']}</b></div><div><span>WIN RATE</span><b>{pct(event['win_rate'])}</b></div><div><span>CAPITAL</span><b>{money(event['stake'])}</b></div><div><span>NET P&L</span><b>{money(event['pnl'], signed=True)}</b></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='history-intro'><b>RECORDED POSITIONS</b><span>Original entry price · stake · result · reason</span></div>", unsafe_allow_html=True)
        bets_table(event["frame"])
        bets = event["frame"][event["frame"]["action"] == "BET"]
        if len(bets):
            selected_pick = st.selectbox(
                "Review the math behind a position", list(bets.index),
                format_func=lambda idx: f"{bets.loc[idx, 'pick']} / {bets.loc[idx, 'fighter_a']} vs {bets.loc[idx, 'fighter_b']}",
            )
            math_detail(bets.loc[selected_pick].to_dict(), bundle, fighters)

st.markdown(f"<div class='fineprint'>{bundle['metrics']['training_fights']:,} TRAINING FIGHTS / {bundle['metrics']['holdout_fights']:,} UNSEEN HOLDOUT / {len(fighters):,} FIGHTER SNAPSHOTS / PAPER MODEL — NOT AN EXECUTION SERVICE</div>", unsafe_allow_html=True)
