from __future__ import annotations

import io
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process


FEATURE_LABELS = {
    "Elo diff": "Opponent-adjusted Elo",
    "Record diff": "Smoothed UFC record",
    "Recent diff": "Recent form",
    "Strike diff": "Striking differential",
    "TD diff": "Takedown differential",
    "Control diff": "Control differential",
    "Reach diff": "Reach",
    "Age advantage": "Age advantage",
}
POSITIVE_TERMS = {
    "healthy", "ready", "sharp", "improved", "advantage", "strong", "cardio",
    "prepared", "momentum", "elite", "durable",
}
NEGATIVE_TERMS = {
    "injury", "injured", "illness", "surgery", "compromised", "missed weight",
    "bad cut", "short notice", "fatigue", "withdrawal", "decline", "hospital",
}
UFC_330_PAIRS = [
    ("Islam Makhachev", "Ian Machado Garry"),
    ("Mackenzie Dern", "Gillian Robertson"),
    ("Jalin Turner", "Kaue Fernandes"),
    ("Mansur Abdul-Malik", "Dustin Stoltzfus"),
    ("Edson Barboza", "Esteban Ribovics"),
    ("Chidi Njokuani", "Joel Alvarez"),
    ("Charles Johnson", "Eduardo Chapolin"),
    ("Donte Johnson", "Eric McConico"),
    ("Vicente Luque", "Tresean Gore"),
    ("Rafael Tobias", "Lucas Fernando"),
    ("Neil Magny", "Ramiz Brahimaj"),
    ("Jeremiah Wells", "Myktybek Orolbai"),
]

HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; UFC-Edge-Model/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
})


def canonical_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    aliases = {
        "ian garry": "ian machado garry",
        "myktybek orolbay": "myktybek orolbai",
        "myktybek orolbay uulu": "myktybek orolbai",
        "eduardo henrique da silva dos santos": "eduardo chapolin",
    }
    return aliases.get(text, text)


def pair_key(fighter_a: str, fighter_b: str):
    return tuple(sorted((canonical_name(fighter_a), canonical_name(fighter_b))))


def safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def sigmoid(value):
    value = np.clip(value, -35, 35)
    return 1 / (1 + np.exp(-value))


def american_to_price(odds):
    odds = float(odds)
    return (-odds) / ((-odds) + 100) if odds < 0 else 100 / (odds + 100)


def price_to_american(price):
    price = safe_float(price)
    if not 0 < price < 1:
        return None
    return int(round(-100 * price / (1 - price))) if price >= 0.5 else int(round(100 * (1 - price) / price))


def load_assets(data_dir: Path):
    bundle = joblib.load(data_dir / "model_bundle.joblib")
    fighters = pd.read_csv(data_dir / "fighter_snapshot.csv.gz", compression="gzip", low_memory=False)
    fighters["Canonical"] = fighters["Fighter"].map(canonical_name)
    return bundle, fighters


def best_fighter_match(name: str, fighters: pd.DataFrame):
    key = canonical_name(name)
    exact = fighters.loc[fighters["Canonical"] == key]
    if len(exact):
        return exact.iloc[0]
    match = process.extractOne(key, fighters["Canonical"].dropna().tolist(), scorer=fuzz.WRatio, score_cutoff=82)
    if match:
        return fighters.loc[fighters["Canonical"] == match[0]].iloc[0]
    return pd.Series({
        "Fighter": name, "UFC fights": 0, "Wins": 0, "Losses": 0, "Draws": 0,
        "Elo": 1500, "Smoothed win %": 0.5, "Recent 5": 0.5,
        "Adj strike diff/min": 0, "Adj TD diff/15": 0,
        "Adj control min/15": 0, "Reach": np.nan, "Age": np.nan,
    })


def feature_vector(fighter_a: str, fighter_b: str, fighters: pd.DataFrame, features):
    a = best_fighter_match(fighter_a, fighters)
    b = best_fighter_match(fighter_b, fighters)
    a_reach, b_reach = safe_float(a.get("Reach")), safe_float(b.get("Reach"))
    a_age, b_age = safe_float(a.get("Age")), safe_float(b.get("Age"))
    values = {
        "Elo diff": safe_float(a.get("Elo"), 1500) - safe_float(b.get("Elo"), 1500),
        "Record diff": safe_float(a.get("Smoothed win %"), 0.5) - safe_float(b.get("Smoothed win %"), 0.5),
        "Recent diff": safe_float(a.get("Recent 5"), 0.5) - safe_float(b.get("Recent 5"), 0.5),
        "Strike diff": safe_float(a.get("Adj strike diff/min"), 0) - safe_float(b.get("Adj strike diff/min"), 0),
        "TD diff": safe_float(a.get("Adj TD diff/15"), 0) - safe_float(b.get("Adj TD diff/15"), 0),
        "Control diff": safe_float(a.get("Adj control min/15"), 0) - safe_float(b.get("Adj control min/15"), 0),
        "Reach diff": a_reach - b_reach if np.isfinite(a_reach) and np.isfinite(b_reach) else 0,
        "Age advantage": b_age - a_age if np.isfinite(a_age) and np.isfinite(b_age) else 0,
    }
    vector = np.array([values[feature] for feature in features], dtype=float)
    experience = min(int(safe_float(a.get("UFC fights"), 0)), int(safe_float(b.get("UFC fights"), 0)))
    return vector, values, experience


def calibrated_probability(bundle, matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    model = bundle["model"]
    direct = model.predict_proba(matrix)[:, 1]
    reverse = 1 - model.predict_proba(-matrix)[:, 1]
    raw = np.clip((direct + reverse) / 2, 1e-6, 1 - 1e-6)
    return sigmoid(bundle["calibration_slope"] * np.log(raw / (1 - raw)))


def local_drivers(bundle, vector):
    full = float(calibrated_probability(bundle, vector)[0])
    rows = []
    for index, feature in enumerate(bundle["features"]):
        neutral = vector.copy()
        neutral[index] = 0
        without = float(calibrated_probability(bundle, neutral)[0])
        rows.append({
            "factor": FEATURE_LABELS[feature], "value": float(vector[index]),
            "impact_a": full - without,
        })
    return sorted(rows, key=lambda row: abs(row["impact_a"]), reverse=True)


def http_json(url, params=None):
    response = HTTP.get(url, params=params, timeout=7)
    response.raise_for_status()
    return response.json()


def http_soup(url):
    response = HTTP.get(url, timeout=7)
    if response.status_code >= 400 and url.startswith("http://"):
        response = HTTP.get("https://" + url[len("http://"):], timeout=7)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def list_ufcstats_events(kind="upcoming"):
    soup = http_soup(f"http://ufcstats.com/statistics/events/{kind}?page=all")
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        if not link:
            continue
        date = row.select_one("span.b-statistics__date")
        events.append({
            "event": link.get_text(" ", strip=True), "url": link.get("href", "").strip(),
            "date_text": date.get_text(" ", strip=True) if date else "", "kind": kind,
        })
    return events


def parse_event_date(soup):
    match = re.search(r"DATE:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", soup.get_text(" ", strip=True))
    return pd.to_datetime(match.group(1), errors="coerce") if match else pd.NaT


def parse_ufcstats_card(event):
    soup = http_soup(event["url"])
    event_date = parse_event_date(soup)
    bouts = []
    for row in soup.select("tr.b-fight-details__table-row[data-link]"):
        names = list(dict.fromkeys([
            node.get_text(" ", strip=True) for node in row.select("a.b-link_style_black")
            if node.get_text(" ", strip=True)
        ]))
        if len(names) < 2:
            continue
        flags = [node.get_text(" ", strip=True).upper() for node in row.select("i.b-flag__text")]
        winner = names[0] if flags and flags[0] == "W" else (names[1] if len(flags) > 1 and flags[1] == "W" else "")
        bouts.append({
            "event": event["event"], "event_date": str(event_date.date()) if pd.notna(event_date) else "",
            "fighter_a": names[0], "fighter_b": names[1], "winner": winner,
            "fight_url": row.get("data-link", ""),
        })
    return bouts


def discover_card(event_search: str, refresh_results: bool = False):
    single = re.match(r"^\s*(.+?)\s+vs\.?\s+(.+?)\s*$", event_search, re.I)
    if single:
        return [{
            "event": f"{single.group(1).strip()} vs {single.group(2).strip()}",
            "event_date": str(datetime.now().date()), "fighter_a": single.group(1).strip(),
            "fighter_b": single.group(2).strip(), "winner": "", "fight_url": "",
        }]
    query = canonical_name(re.sub(r"\blive\b", "", event_search, flags=re.I))
    if "ufc 330" in query and not refresh_results:
        return [{
            "event": "UFC 330: Makhachev vs. Machado Garry", "event_date": "2026-08-15",
            "fighter_a": a, "fighter_b": b, "winner": "", "fight_url": "",
        } for a, b in UFC_330_PAIRS]
    candidates = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(list_ufcstats_events, kind) for kind in ("upcoming", "completed")]
        for future in as_completed(futures):
            try:
                candidates.extend(future.result())
            except Exception:
                pass
    if candidates:
        score, selected = max(
            ((fuzz.WRatio(query, canonical_name(event["event"])), event) for event in candidates),
            key=lambda item: item[0],
        )
        if score >= 55:
            card = parse_ufcstats_card(selected)
            if card:
                return card
    raise RuntimeError("Event not found. Enter a UFC event number or type Fighter A vs Fighter B.")


def parse_jsonish(value, default=None):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def fetch_polymarket_prices():
    rows = []
    data = http_json("https://gamma-api.polymarket.com/events", params={
        "tag_id": 279, "active": "true", "closed": "false", "limit": 200,
    })
    for event in data:
        for market in event.get("markets", []):
            question = str(market.get("question") or event.get("title") or "")
            outcomes = parse_jsonish(market.get("outcomes"), []) or []
            prices = parse_jsonish(market.get("outcomePrices"), []) or []
            if len(outcomes) >= 2 and len(prices) >= 2 and canonical_name(outcomes[0]) not in {"yes", "no"}:
                rows.append({
                    "fighter_a": str(outcomes[0]), "fighter_b": str(outcomes[1]),
                    "price_a": safe_float(prices[0]), "price_b": safe_float(prices[1]),
                    "source": "Polymarket",
                })
                continue
            match = re.search(r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)", question, re.I)
            ask, bid = safe_float(market.get("bestAsk")), safe_float(market.get("bestBid"))
            if match and 0 < ask < 1:
                rows.append({
                    "fighter_a": match.group(1).strip(), "fighter_b": match.group(2).strip(),
                    "price_a": ask, "price_b": 1 - bid if 0 < bid < 1 else np.nan,
                    "source": "Polymarket",
                })
    return rows


def fetch_kalshi_prices():
    rows = []
    data = http_json("https://api.elections.kalshi.com/trade-api/v2/events", params={
        "series_ticker": "KXUFCFIGHT", "status": "open", "with_nested_markets": "true", "limit": 200,
    })
    for event in data.get("events", []):
        named = []
        for market in event.get("markets", []):
            name = str(market.get("yes_sub_title") or "").strip()
            ask = safe_float(market.get("yes_ask_dollars"))
            if not 0 < ask < 1:
                cents = safe_float(market.get("yes_ask"))
                ask = cents / 100 if 0 < cents <= 100 else np.nan
            if name and 0 < ask < 1:
                named.append((name, ask))
        if len(named) >= 2:
            rows.append({
                "fighter_a": named[0][0], "fighter_b": named[1][0],
                "price_a": named[0][1], "price_b": named[1][1], "source": "Kalshi",
            })
    return rows


def fetch_espn_prices():
    rows = []
    data = http_json("https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard")
    for event in data.get("events", []):
        for competition in event.get("competitions", []):
            competitors = competition.get("competitors", [])
            odds_list = competition.get("odds") or []
            if len(competitors) != 2 or not odds_list:
                continue
            names = [str(item.get("athlete", {}).get("displayName") or "") for item in competitors]
            odds = odds_list[0]
            values = []
            for side in ("homeTeamOdds", "awayTeamOdds"):
                item = odds.get(side, {})
                american = item.get("moneyLine") or item.get("value")
                values.append(american_to_price(american) if american not in (None, "") else np.nan)
            if all(0 < value < 1 for value in values):
                rows.append({
                    "fighter_a": names[0], "fighter_b": names[1],
                    "price_a": values[0], "price_b": values[1], "source": "ESPN listed odds",
                })
    return rows


def parse_manual_odds(text: str):
    odds = {}
    for line in text.splitlines():
        match = re.match(r"\s*(.+?)\s*[,=:]\s*([+-]?\d{3,4})\s*$", line)
        if match:
            odds[canonical_name(match.group(1))] = float(match.group(2))
    return odds


def fetch_market_rows(card, manual_odds_text=""):
    rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetcher) for fetcher in (fetch_polymarket_prices, fetch_kalshi_prices, fetch_espn_prices)]
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                pass
    manual = parse_manual_odds(manual_odds_text)
    for bout in card:
        a, b = canonical_name(bout["fighter_a"]), canonical_name(bout["fighter_b"])
        if a in manual and b in manual:
            rows.append({
                "fighter_a": bout["fighter_a"], "fighter_b": bout["fighter_b"],
                "price_a": american_to_price(manual[a]), "price_b": american_to_price(manual[b]),
                "source": "Manual current odds",
            })
    return rows


def best_market_for_bout(bout, market_rows):
    aligned = []
    for row in market_rows:
        if pair_key(row["fighter_a"], row["fighter_b"]) != pair_key(bout["fighter_a"], bout["fighter_b"]):
            continue
        same = canonical_name(row["fighter_a"]) == canonical_name(bout["fighter_a"])
        price_a = safe_float(row["price_a"] if same else row["price_b"])
        price_b = safe_float(row["price_b"] if same else row["price_a"])
        if 0 < price_a < 1 and 0 < price_b < 1:
            aligned.append({"price_a": price_a, "price_b": price_b, "source": row["source"]})
    if not aligned:
        return None
    best_a = min(aligned, key=lambda row: row["price_a"])
    best_b = min(aligned, key=lambda row: row["price_b"])
    return {
        "price_a": best_a["price_a"], "price_b": best_b["price_b"],
        "source_a": best_a["source"], "source_b": best_b["source"],
    }


def research_shift(texts, fighter_a, fighter_b):
    if not texts:
        return 0.0
    scores = {}
    combined = [canonical_name(text) for text in texts]
    for fighter in (fighter_a, fighter_b):
        key = canonical_name(fighter)
        last = key.split()[-1]
        relevant = [text for text in combined if key in text or last in text.split()]
        raw = [sum(term in text for term in POSITIVE_TERMS) - sum(term in text for term in NEGATIVE_TERMS) for text in relevant]
        scores[fighter] = float(np.clip(np.mean(raw) / 4, -1, 1)) if raw else 0
    return float(np.clip((scores[fighter_a] - scores[fighter_b]) * 0.04, -0.04, 0.04))


def analyze_card(card, market_rows, bundle, fighters, bankroll=10_000, research_texts=None,
                 min_edge=0.03, cost_buffer=0.02, kelly_fraction=0.25,
                 max_position_pct=0.02, min_prior_fights=3):
    research_texts = research_texts or []
    analyses = []
    for bout in card:
        vector, values, experience = feature_vector(bout["fighter_a"], bout["fighter_b"], fighters, bundle["features"])
        core_a = float(calibrated_probability(bundle, vector)[0])
        shift = research_shift(research_texts, bout["fighter_a"], bout["fighter_b"])
        probability_a = float(sigmoid(math.log(core_a / (1 - core_a)) + shift))
        probability_b = 1 - probability_a
        market = best_market_for_bout(bout, market_rows)
        if market:
            edge_a = probability_a - market["price_a"] - cost_buffer
            edge_b = probability_b - market["price_b"] - cost_buffer
            if edge_a >= edge_b:
                pick, probability, price, edge = bout["fighter_a"], probability_a, market["price_a"], edge_a
                source, pick_is_a = market["source_a"], True
            else:
                pick, probability, price, edge = bout["fighter_b"], probability_b, market["price_b"], edge_b
                source, pick_is_a = market["source_b"], False
            all_in = min(0.999, price + cost_buffer)
            full_kelly = max(0, (probability - all_in) / max(1e-9, 1 - all_in))
            action = "BET" if edge >= min_edge and experience >= min_prior_fights else "NO BET"
            position = round(bankroll * min(max_position_pct, kelly_fraction * full_kelly), 2) if action == "BET" else 0
        else:
            pick_is_a = probability_a >= probability_b
            pick = bout["fighter_a"] if pick_is_a else bout["fighter_b"]
            probability = max(probability_a, probability_b)
            price, edge, source, action, position = np.nan, np.nan, "", "NO BET", 0
        drivers = local_drivers(bundle, vector)
        for driver in drivers:
            driver["impact_pick"] = driver["impact_a"] if pick_is_a else -driver["impact_a"]
        top = sorted(drivers, key=lambda row: abs(row["impact_pick"]), reverse=True)[:3]
        if market is None:
            why = "No current market price"
        elif experience < min_prior_fights:
            why = f"Only {experience} prior UFC fights on the less-experienced side"
        else:
            why = "; ".join(f"{row['factor']} {row['impact_pick']:+.1%}" for row in top[:2])
        analyses.append({
            **bout, "pick": pick, "model_probability": probability,
            "probability_a": probability_a, "probability_b": probability_b,
            "market_probability": price, "american_odds": price_to_american(price),
            "net_edge": edge, "action": action, "position_dollars": position,
            "experience": experience, "why": why, "drivers": top,
            "research_shift": shift, "market_source": source,
            "as_of_utc": datetime.now(timezone.utc).isoformat(), "model_version": bundle["version"],
        })
    return analyses


LOG_COLUMNS = [
    "prediction_id", "timestamp_utc", "event_date", "event", "fighter_a", "fighter_b",
    "pick", "model_probability", "market_probability", "net_edge", "action",
    "position_dollars", "status", "winner", "outcome", "pnl", "model_version",
]


def update_prediction_log(state_dir: Path, analyses):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "prediction_log.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    for column in [
        "prediction_id", "timestamp_utc", "event_date", "event", "fighter_a", "fighter_b",
        "pick", "action", "status", "winner", "outcome", "model_version",
    ]:
        log[column] = log[column].fillna("").astype("object")
    for column in ["model_probability", "market_probability", "net_edge", "position_dollars", "pnl"]:
        log[column] = pd.to_numeric(log[column], errors="coerce").astype(float)
    result_map = {pair_key(row["fighter_a"], row["fighter_b"]): row["winner"] for row in analyses if row.get("winner")}
    for index, row in log.loc[log["status"] == "OPEN"].iterrows():
        winner = result_map.get(pair_key(row["fighter_a"], row["fighter_b"]))
        if not winner:
            continue
        if row["action"] == "BET":
            won = canonical_name(row["pick"]) == canonical_name(winner)
            price, stake = safe_float(row["market_probability"]), safe_float(row["position_dollars"], 0)
            pnl = stake * (1 / price - 1) if won and 0 < price < 1 else -stake
            outcome = "WIN" if won else "LOSS"
        else:
            pnl, outcome = 0, "NO BET"
        log.at[index, "status"] = "COMPLETED"
        log.at[index, "winner"] = winner
        log.at[index, "outcome"] = outcome
        log.at[index, "pnl"] = round(pnl, 2)
    for row in analyses:
        prediction_id = f"{canonical_name(row['event'])}|{'|'.join(pair_key(row['fighter_a'], row['fighter_b']))}"
        record = {
            "prediction_id": prediction_id, "timestamp_utc": row["as_of_utc"],
            "event_date": row["event_date"], "event": row["event"],
            "fighter_a": row["fighter_a"], "fighter_b": row["fighter_b"], "pick": row["pick"],
            "model_probability": row["model_probability"], "market_probability": row["market_probability"],
            "net_edge": row["net_edge"], "action": row["action"], "position_dollars": row["position_dollars"],
            "status": "COMPLETED" if row.get("winner") else "OPEN", "winner": row.get("winner", ""),
            "outcome": "", "pnl": 0, "model_version": row["model_version"],
        }
        completed = ((log.get("prediction_id", pd.Series(dtype=str)) == prediction_id) & (log.get("status", pd.Series(dtype=str)) == "COMPLETED")).any()
        if completed:
            continue
        existing = log.index[(log.get("prediction_id", pd.Series(dtype=str)) == prediction_id) & (log.get("status", pd.Series(dtype=str)) == "OPEN")].tolist()
        # A completed bout is only graded when it was recorded pre-fight as OPEN.
        # This prevents hindsight results from being added as fresh predictions.
        if row.get("winner") and not existing:
            continue
        if existing:
            for key, value in record.items():
                log.loc[existing[-1], key] = value
        else:
            incoming = pd.DataFrame([record], columns=LOG_COLUMNS)
            log = incoming if log.empty else pd.concat([log, incoming], ignore_index=True)
    log.to_csv(path, index=False)
    return log


def realized_metrics(log):
    completed = log[log["status"] == "COMPLETED"] if len(log) else pd.DataFrame()
    bets = completed[completed["action"] == "BET"] if len(completed) else pd.DataFrame()
    wins = int((bets["outcome"] == "WIN").sum()) if len(bets) else 0
    losses = int((bets["outcome"] == "LOSS").sum()) if len(bets) else 0
    staked = float(pd.to_numeric(bets.get("position_dollars", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(bets) else 0
    pnl = float(pd.to_numeric(bets.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(bets) else 0
    return {"wins": wins, "losses": losses, "graded": len(bets), "pnl": pnl, "roi": pnl / staked if staked else np.nan}
