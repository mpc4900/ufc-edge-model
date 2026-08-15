from __future__ import annotations

import io
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

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
    source_columns = {
        "Elo diff": "Elo",
        "Record diff": "Smoothed win %",
        "Recent diff": "Recent 5",
        "Strike diff": "Adj strike diff/min",
        "TD diff": "Adj TD diff/15",
        "Control diff": "Adj control min/15",
        "Reach diff": "Reach",
        "Age advantage": "Age",
    }
    raw_inputs = []
    for feature in features:
        column = source_columns[feature]
        raw_inputs.append({
            "Factor": FEATURE_LABELS[feature],
            "Fighter A": safe_float(a.get(column)),
            "Fighter B": safe_float(b.get(column)),
            "Model difference": float(values[feature]),
        })
    return vector, values, experience, raw_inputs


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


def _active_polymarket_ufc_cards():
    """Return full UFC cards assembled from active Polymarket moneyline events."""
    data = http_json("https://gamma-api.polymarket.com/events", params={
        "tag_slug": "ufc", "active": "true", "closed": "false",
        "order": "endDate", "ascending": "true", "limit": 200,
    })
    grouped = {}
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    for event in data:
        title = str(event.get("title") or "").strip()
        numbered = re.match(r"^(UFC\s+\d+)\s*:", title, re.I)
        fight_night = re.match(r"^(UFC\s+Fight\s+Night)\s*:", title, re.I)
        if not numbered and not fight_night:
            continue
        for market in event.get("markets", []):
            if str(market.get("sportsMarketType") or "").lower() != "moneyline":
                continue
            outcomes = [str(value).strip() for value in (parse_jsonish(market.get("outcomes"), []) or [])]
            if len(outcomes) != 2 or set(map(canonical_name, outcomes)) == {"yes", "no"}:
                continue
            start = pd.to_datetime(market.get("gameStartTime") or event.get("endDate"), utc=True, errors="coerce")
            if pd.isna(start):
                continue
            local_start = start.to_pydatetime().astimezone(eastern)
            card_name = numbered.group(1).upper() if numbered else "UFC Fight Night"
            key = f"{canonical_name(card_name)}|{local_start.date().isoformat()}"
            group = grouped.setdefault(key, {
                "card_name": card_name,
                "event_date": local_start.date().isoformat(),
                "start_utc": start.to_pydatetime(),
                "end_utc": pd.to_datetime(event.get("endDate"), utc=True, errors="coerce"),
                "bouts": [],
            })
            group["start_utc"] = min(group["start_utc"], start.to_pydatetime())
            group["bouts"].append({
                "event": card_name,
                "event_date": local_start.date().isoformat(),
                "fighter_a": outcomes[0],
                "fighter_b": outcomes[1],
                "winner": "",
                "fight_url": "",
            })
            break
    cards = []
    today = now.astimezone(eastern).date()
    for key, group in grouped.items():
        unique = {}
        for bout in group["bouts"]:
            unique[pair_key(bout["fighter_a"], bout["fighter_b"])] = bout
        group["bouts"] = list(unique.values())
        local_date = datetime.fromisoformat(group["event_date"]).date()
        end_value = group["end_utc"]
        end_utc = end_value.to_pydatetime() if pd.notna(end_value) else None
        if group["start_utc"] <= now and (end_utc is None or now <= end_utc):
            status = "LIVE"
        elif local_date == today:
            status = "TODAY"
        else:
            status = "UPCOMING"
        date_label = local_date.strftime("%b %-d")
        cards.append({
            "value": f"POLY_CARD::{key}",
            "label": f"{status}  •  {group['card_name']}  •  {date_label}  •  {len(group['bouts'])} fights",
            "status": status,
            "start_utc": group["start_utc"],
            "bouts": group["bouts"],
        })
    priority = {"LIVE": 0, "TODAY": 1, "UPCOMING": 2}
    return sorted(cards, key=lambda item: (priority[item["status"]], item["start_utc"]))


def discover_event_options():
    """Browsable choices for the app; safe fallback keeps the app usable offline."""
    try:
        cards = _active_polymarket_ufc_cards()
        if cards:
            return [{"label": card["label"], "value": card["value"]} for card in cards]
    except Exception:
        pass
    return [{"label": "UFC 330  •  Aug 15  •  12 fights", "value": "UFC 330"}]


def discover_card(event_search: str, refresh_results: bool = False):
    if str(event_search).startswith("POLY_CARD::"):
        try:
            selected = next(card for card in _active_polymarket_ufc_cards() if card["value"] == event_search)
            if selected["bouts"]:
                return selected["bouts"]
        except Exception:
            raise RuntimeError("That card is no longer active. Refresh the event list and choose another card.")
    single = re.match(r"^\s*(.+?)\s+vs\.?\s+(.+?)\s*$", event_search, re.I)
    if single:
        return [{
            "event": f"{single.group(1).strip()} vs {single.group(2).strip()}",
            "event_date": str(datetime.now().date()), "fighter_a": single.group(1).strip(),
            "fighter_b": single.group(2).strip(), "winner": "", "fight_url": "",
        }]
    query = canonical_name(re.sub(r"\blive\b", "", event_search, flags=re.I))
    if re.fullmatch(r"ufc 330", query) and not refresh_results:
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
    raise RuntimeError("Event not found. Choose a listed UFC card or type both names as Fighter A vs Fighter B.")


def parse_jsonish(value, default=None):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def card_match(card, text):
    target = canonical_name(text)
    scored = []
    for bout in card:
        direct = canonical_name(f"{bout['fighter_a']} vs {bout['fighter_b']}")
        reverse = canonical_name(f"{bout['fighter_b']} vs {bout['fighter_a']}")
        a_last = canonical_name(bout["fighter_a"]).split()[-1]
        b_last = canonical_name(bout["fighter_b"]).split()[-1]
        contains_pair = a_last in target.split() and b_last in target.split()
        score = max(fuzz.partial_ratio(direct, target), fuzz.partial_ratio(reverse, target))
        scored.append((score + (30 if contains_pair else 0), bout))
    if not scored:
        return None
    score, bout = max(scored, key=lambda item: item[0])
    return bout if score >= 92 else None


def polymarket_book(token_id):
    data = http_json("https://clob.polymarket.com/book", params={"token_id": str(token_id)})
    bids = [(safe_float(row.get("price")), safe_float(row.get("size"), 0)) for row in data.get("bids", [])]
    asks = [(safe_float(row.get("price")), safe_float(row.get("size"), 0)) for row in data.get("asks", [])]
    bids = [row for row in bids if 0 < row[0] < 1]
    asks = [row for row in asks if 0 < row[0] < 1]
    best_bid = max(bids, key=lambda row: row[0]) if bids else (np.nan, 0)
    best_ask = min(asks, key=lambda row: row[0]) if asks else (np.nan, 0)
    return {
        "bid": best_bid[0], "bid_size": best_bid[1],
        "ask": best_ask[0], "ask_size": best_ask[1],
        "timestamp": str(data.get("timestamp") or ""),
    }


def fighter_named_in_text(bout, text):
    target = canonical_name(text)
    choices = []
    for fighter in (bout["fighter_a"], bout["fighter_b"]):
        name = canonical_name(fighter)
        last = name.split()[-1]
        score = max(fuzz.partial_ratio(name, target), 100 if last in target.split() else 0)
        choices.append((score, fighter))
    choices.sort(reverse=True)
    return choices[0][1] if choices and choices[0][0] >= 80 and (len(choices) == 1 or choices[0][0] > choices[1][0]) else ""


def fetch_polymarket_prices(card):
    rows, specifications = [], []
    data = http_json("https://gamma-api.polymarket.com/events", params={
        "tag_slug": "ufc", "active": "true", "closed": "false",
        "order": "endDate", "ascending": "true", "limit": 200,
    })
    for event in data:
        event_title = str(event.get("title") or "")
        for market in event.get("markets", []):
            if str(market.get("sportsMarketType") or "").lower() != "moneyline":
                continue
            question = str(market.get("question") or "")
            bout = card_match(card, f"{event_title} {question}")
            if not bout:
                continue
            outcomes = [str(value) for value in (parse_jsonish(market.get("outcomes"), []) or [])]
            tokens = [str(value) for value in (parse_jsonish(market.get("clobTokenIds"), []) or [])]
            if len(outcomes) != 2 or len(tokens) != 2:
                continue
            normalized = [canonical_name(value) for value in outcomes]
            if set(normalized) == {"yes", "no"}:
                subject = fighter_named_in_text(bout, question)
                if not subject:
                    continue
                opponent = bout["fighter_b"] if canonical_name(subject) == canonical_name(bout["fighter_a"]) else bout["fighter_a"]
                yes_index = normalized.index("yes")
                no_index = normalized.index("no")
                fighter_tokens = {subject: tokens[yes_index], opponent: tokens[no_index]}
            else:
                fighter_tokens = {}
                for outcome, token in zip(outcomes, tokens):
                    fighter = fighter_named_in_text(bout, outcome)
                    if fighter:
                        fighter_tokens[fighter] = token
                if len(fighter_tokens) != 2:
                    continue
            specifications.append({
                "bout": bout, "tokens": fighter_tokens,
                "url": f"https://polymarket.com/event/{event.get('slug', '')}",
            })
    unique_tokens = sorted({token for item in specifications for token in item["tokens"].values()})
    books = {}
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(unique_tokens)))) as executor:
        futures = {executor.submit(polymarket_book, token): token for token in unique_tokens}
        for future in as_completed(futures):
            try:
                books[futures[future]] = future.result()
            except Exception:
                pass
    for item in specifications:
        bout = item["bout"]
        token_a = item["tokens"].get(bout["fighter_a"])
        token_b = item["tokens"].get(bout["fighter_b"])
        book_a, book_b = books.get(token_a, {}), books.get(token_b, {})
        ask_a, ask_b = safe_float(book_a.get("ask")), safe_float(book_b.get("ask"))
        if not (0 < ask_a < 1 and 0 < ask_b < 1):
            continue
        rows.append({
            "fighter_a": bout["fighter_a"], "fighter_b": bout["fighter_b"],
            "price_a": ask_a, "price_b": ask_b,
            "ask_a": ask_a, "ask_b": ask_b,
            "bid_a": safe_float(book_a.get("bid")), "bid_b": safe_float(book_b.get("bid")),
            "ask_size_a": safe_float(book_a.get("ask_size"), 0), "ask_size_b": safe_float(book_b.get("ask_size"), 0),
            "source": "Polymarket CLOB", "market_url": item["url"],
            "market_timestamp": book_a.get("timestamp") or book_b.get("timestamp") or "",
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
                    "price_a": values[0], "price_b": values[1],
                    "ask_a": values[0], "ask_b": values[1],
                    "bid_a": np.nan, "bid_b": np.nan, "source": "ESPN listed odds",
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
        futures = [
            executor.submit(fetch_polymarket_prices, card),
            executor.submit(fetch_kalshi_prices),
            executor.submit(fetch_espn_prices),
        ]
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
                "ask_a": american_to_price(manual[a]), "ask_b": american_to_price(manual[b]),
                "bid_a": np.nan, "bid_b": np.nan,
                "source": "Manual current odds",
            })
    return rows


def best_market_for_bout(bout, market_rows, polymarket_only=True):
    aligned = []
    for row in market_rows:
        if polymarket_only and row.get("source") != "Polymarket CLOB":
            continue
        if pair_key(row["fighter_a"], row["fighter_b"]) != pair_key(bout["fighter_a"], bout["fighter_b"]):
            continue
        same = canonical_name(row["fighter_a"]) == canonical_name(bout["fighter_a"])
        ask_a = safe_float(row.get("ask_a", row["price_a"]) if same else row.get("ask_b", row["price_b"]))
        ask_b = safe_float(row.get("ask_b", row["price_b"]) if same else row.get("ask_a", row["price_a"]))
        bid_a = safe_float(row.get("bid_a") if same else row.get("bid_b"))
        bid_b = safe_float(row.get("bid_b") if same else row.get("bid_a"))
        if 0 < ask_a < 1 and 0 < ask_b < 1:
            aligned.append({
                "price_a": ask_a, "price_b": ask_b, "bid_a": bid_a, "bid_b": bid_b,
                "source": row["source"], "market_url": row.get("market_url", ""),
                "market_timestamp": row.get("market_timestamp", ""),
            })
    if not aligned:
        return None
    best_a = min(aligned, key=lambda row: row["price_a"])
    best_b = min(aligned, key=lambda row: row["price_b"])
    return {
        "price_a": best_a["price_a"], "price_b": best_b["price_b"],
        "bid_a": best_a["bid_a"], "bid_b": best_b["bid_b"],
        "source_a": best_a["source"], "source_b": best_b["source"],
        "url_a": best_a["market_url"], "url_b": best_b["market_url"],
        "timestamp_a": best_a["market_timestamp"], "timestamp_b": best_b["market_timestamp"],
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
                 max_position_pct=0.02, min_prior_fights=3, convergence=0.50,
                 max_card_exposure=0.10, polymarket_only=True):
    research_texts = research_texts or []
    analyses = []
    for bout in card:
        vector, values, experience, raw_inputs = feature_vector(bout["fighter_a"], bout["fighter_b"], fighters, bundle["features"])
        core_a = float(calibrated_probability(bundle, vector)[0])
        shift = research_shift(research_texts, bout["fighter_a"], bout["fighter_b"])
        probability_a = float(sigmoid(math.log(core_a / (1 - core_a)) + shift))
        probability_b = 1 - probability_a
        likely_is_a = probability_a >= probability_b
        likely_winner = bout["fighter_a"] if likely_is_a else bout["fighter_b"]
        likely_probability = max(probability_a, probability_b)
        market = best_market_for_bout(bout, market_rows, polymarket_only=polymarket_only)
        if market:
            edge_a = probability_a - market["price_a"] - cost_buffer
            edge_b = probability_b - market["price_b"] - cost_buffer
            if edge_a >= edge_b:
                pick, probability, price, edge = bout["fighter_a"], probability_a, market["price_a"], edge_a
                source, market_url, market_timestamp, bid, pick_is_a = market["source_a"], market["url_a"], market["timestamp_a"], market["bid_a"], True
            else:
                pick, probability, price, edge = bout["fighter_b"], probability_b, market["price_b"], edge_b
                source, market_url, market_timestamp, bid, pick_is_a = market["source_b"], market["url_b"], market["timestamp_b"], market["bid_b"], False
            all_in = min(0.999, price + cost_buffer)
            full_kelly = max(0, (probability - all_in) / max(1e-9, 1 - all_in))
            action = "BET" if edge >= min_edge and experience >= min_prior_fights else "NO BET"
            position = round(bankroll * min(max_position_pct, kelly_fraction * full_kelly), 2) if action == "BET" else 0
            exit_target = min(probability, price + convergence * max(0, probability - price))
            scenario_move = exit_target - price
        else:
            pick_is_a = probability_a >= probability_b
            pick = bout["fighter_a"] if pick_is_a else bout["fighter_b"]
            probability = max(probability_a, probability_b)
            price, bid, edge, source, action, position = np.nan, np.nan, np.nan, "", "NO BET", 0
            market_url, market_timestamp, exit_target, scenario_move = "", "", np.nan, np.nan
        drivers = local_drivers(bundle, vector)
        for driver in drivers:
            driver["impact_pick"] = driver["impact_a"] if pick_is_a else -driver["impact_a"]
        top = sorted(drivers, key=lambda row: abs(row["impact_pick"]), reverse=True)[:3]
        if market is None:
            why = "No live Polymarket order book" if polymarket_only else "No current executable price"
        elif experience < min_prior_fights:
            why = f"Only {experience} prior UFC fights on the less-experienced side"
        else:
            prefix = "Underdog value trade. " if probability < 0.5 else ""
            why = prefix + "; ".join(f"{row['factor']} {row['impact_pick']:+.1%}" for row in top[:2])
        analyses.append({
            **bout, "pick": pick, "trade_side": pick,
            "likely_winner": likely_winner, "likely_probability": likely_probability,
            "model_probability": probability,
            "probability_a": probability_a, "probability_b": probability_b,
            "market_probability": price, "live_ask": price, "live_bid": bid,
            "exit_target": exit_target, "scenario_move": scenario_move,
            "scenario_return": scenario_move / price if 0 < price < 1 and np.isfinite(scenario_move) else np.nan,
            "american_odds": price_to_american(price),
            "net_edge": edge, "action": action, "position_dollars": position,
            "experience": experience, "why": why, "drivers": top,
            "raw_inputs": raw_inputs, "research_shift": shift, "market_source": source,
            "market_url": market_url, "market_timestamp": market_timestamp,
            "as_of_utc": datetime.now(timezone.utc).isoformat(), "model_version": bundle["version"],
        })
    total_position = sum(row["position_dollars"] for row in analyses if row["action"] == "BET")
    exposure_cap = bankroll * max_card_exposure
    if total_position > exposure_cap > 0:
        scale = exposure_cap / total_position
        for row in analyses:
            if row["action"] == "BET":
                row["position_dollars"] = round(row["position_dollars"] * scale, 2)
        rounded_total = sum(row["position_dollars"] for row in analyses if row["action"] == "BET")
        if rounded_total > exposure_cap:
            last_bet = next(row for row in reversed(analyses) if row["action"] == "BET")
            last_bet["position_dollars"] = round(last_bet["position_dollars"] - (rounded_total - exposure_cap), 2)
    return analyses


LOG_COLUMNS = [
    "prediction_id", "timestamp_utc", "event_date", "event", "fighter_a", "fighter_b",
    "pick", "model_probability", "market_probability", "net_edge", "action",
    "position_dollars", "entry_timestamp_utc", "entry_price", "entry_odds",
    "potential_profit", "status", "winner", "outcome", "settled_timestamp_utc",
    "exit_type", "exit_price", "return_pct", "pnl", "closing_reason", "model_version",
    "last_price_timestamp_utc", "live_bid", "live_ask", "target_price",
    "target_progress", "unrealized_return", "unrealized_pnl", "position_signal",
]

LOG_TEXT_COLUMNS = [
    "prediction_id", "timestamp_utc", "event_date", "event", "fighter_a", "fighter_b",
    "pick", "action", "entry_timestamp_utc", "status", "winner", "outcome",
    "settled_timestamp_utc", "exit_type", "closing_reason", "model_version",
    "last_price_timestamp_utc", "position_signal",
]

LOG_NUMERIC_COLUMNS = [
    "model_probability", "market_probability", "net_edge", "position_dollars",
    "entry_price", "entry_odds", "potential_profit", "exit_price", "return_pct", "pnl",
    "live_bid", "live_ask", "target_price", "target_progress",
    "unrealized_return", "unrealized_pnl",
]

MARKET_HISTORY_COLUMNS = [
    "timestamp_utc", "event", "fighter_a", "fighter_b", "trade_side",
    "model_probability", "live_bid", "live_ask", "exit_target", "market_source",
]


def normalize_prediction_log(log):
    """Upgrade older ledgers in place while preserving every recorded entry."""
    log = log.copy() if isinstance(log, pd.DataFrame) else pd.DataFrame()
    for column in LOG_COLUMNS:
        if column not in log.columns:
            log[column] = "" if column in LOG_TEXT_COLUMNS else np.nan
    for column in LOG_TEXT_COLUMNS:
        log[column] = log[column].fillna("").astype("object")
    for column in LOG_NUMERIC_COLUMNS:
        log[column] = pd.to_numeric(log[column], errors="coerce").astype(float)
    legacy_bets = (log["action"] == "BET") & ~np.isfinite(log["entry_price"])
    log.loc[legacy_bets, "entry_price"] = log.loc[legacy_bets, "market_probability"]
    missing_entry_time = legacy_bets & (log["entry_timestamp_utc"] == "")
    log.loc[missing_entry_time, "entry_timestamp_utc"] = log.loc[missing_entry_time, "timestamp_utc"]
    missing_profit = legacy_bets & ~np.isfinite(log["potential_profit"])
    if missing_profit.any():
        prices = log.loc[missing_profit, "entry_price"]
        stakes = log.loc[missing_profit, "position_dollars"].fillna(0)
        valid = prices.between(0.000001, 0.999999)
        calculated = pd.Series(np.nan, index=prices.index, dtype=float)
        calculated.loc[valid] = stakes.loc[valid] * (1 / prices.loc[valid] - 1)
        log.loc[missing_profit, "potential_profit"] = calculated
    return log[LOG_COLUMNS]


def load_prediction_log(state_dir: Path):
    path = state_dir / "prediction_log.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    return normalize_prediction_log(log)


def merge_prediction_log(state_dir: Path, incoming):
    """Restore a downloaded ledger without overwriting stronger completed/entered records."""
    state_dir.mkdir(parents=True, exist_ok=True)
    existing = load_prediction_log(state_dir)
    incoming = normalize_prediction_log(incoming)
    combined = pd.concat([existing, incoming], ignore_index=True) if len(existing) else incoming.copy()
    selected = []
    for _, group in combined.groupby("prediction_id", sort=False, dropna=False):
        completed = group[group["status"] == "COMPLETED"]
        if len(completed):
            ranked = completed.assign(_time=pd.to_datetime(completed["settled_timestamp_utc"], errors="coerce"))
            selected.append(ranked.sort_values("_time", na_position="first").iloc[-1].drop(labels=["_time"]))
            continue
        entered = group[group["action"] == "BET"]
        if len(entered):
            ranked = entered.assign(_time=pd.to_datetime(entered["entry_timestamp_utc"], errors="coerce"))
            selected.append(ranked.sort_values("_time", na_position="last").iloc[0].drop(labels=["_time"]))
            continue
        ranked = group.assign(_time=pd.to_datetime(group["timestamp_utc"], errors="coerce"))
        selected.append(ranked.sort_values("_time", na_position="first").iloc[-1].drop(labels=["_time"]))
    merged = normalize_prediction_log(pd.DataFrame(selected))
    merged.to_csv(state_dir / "prediction_log.csv", index=False)
    return merged


def _settle_prediction_row(log, index, winner, settled_at=None):
    settled_at = settled_at or datetime.now(timezone.utc).isoformat()
    row = log.loc[index]
    action = str(row.get("action") or "")
    if action == "BET":
        won = canonical_name(row.get("pick")) == canonical_name(winner)
        price = safe_float(row.get("entry_price"), safe_float(row.get("market_probability")))
        stake = safe_float(row.get("position_dollars"), 0)
        pnl = stake * (1 / price - 1) if won and 0 < price < 1 else -stake
        outcome = "WIN" if won else "LOSS"
        exit_price = 1.0 if won else 0.0
        return_pct = pnl / stake if stake else np.nan
        reason = f"{winner} won; binary contract settled at {exit_price:.0f}."
    else:
        pnl, outcome, exit_price, return_pct = 0.0, "NO BET", np.nan, np.nan
        reason = f"{winner} won; no paper position was opened."
    log.at[index, "status"] = "COMPLETED"
    log.at[index, "winner"] = winner
    log.at[index, "outcome"] = outcome
    log.at[index, "settled_timestamp_utc"] = settled_at
    log.at[index, "exit_type"] = "FIGHT RESULT"
    log.at[index, "exit_price"] = exit_price
    log.at[index, "return_pct"] = return_pct
    log.at[index, "pnl"] = round(pnl, 2)
    log.at[index, "closing_reason"] = reason


def grade_prediction_log(state_dir: Path, completed_bouts):
    """Settle previously recorded paper positions from official completed bouts."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "prediction_log.csv"
    log = load_prediction_log(state_dir)
    result_map = {
        pair_key(row.get("fighter_a", ""), row.get("fighter_b", "")): row.get("winner", "")
        for row in completed_bouts or [] if row.get("winner")
    }
    if result_map:
        for index, row in log.loc[log["status"] == "OPEN"].iterrows():
            winner = result_map.get(pair_key(row["fighter_a"], row["fighter_b"]))
            if winner:
                _settle_prediction_row(log, index, winner)
        log.to_csv(path, index=False)
    return log


def fetch_completed_results_for_log(log, max_events=30):
    """Find official UFCStats winners for due paper positions without hindsight entries."""
    log = normalize_prediction_log(log)
    open_bets = log[(log["status"] == "OPEN") & (log["action"] == "BET")].copy()
    if open_bets.empty:
        return []
    eastern_today = datetime.now(ZoneInfo("America/New_York")).date()
    event_dates = pd.to_datetime(open_bets["event_date"], errors="coerce")
    due_mask = event_dates.isna() | (event_dates.dt.date <= eastern_today)
    due = open_bets.loc[due_mask].copy()
    if due.empty:
        return []
    due_pairs = {pair_key(row["fighter_a"], row["fighter_b"]) for _, row in due.iterrows()}
    event_names = [canonical_name(value) for value in due["event"].dropna().astype(str)]
    valid_dates = pd.to_datetime(due["event_date"], errors="coerce").dropna()
    earliest = valid_dates.min().date() if len(valid_dates) else eastern_today
    completed = []
    events = list_ufcstats_events("completed")
    for event in events[:max_events]:
        listed_date = pd.to_datetime(event.get("date_text"), errors="coerce")
        if pd.notna(listed_date) and listed_date.date() < earliest - timedelta(days=3):
            continue
        event_name = canonical_name(event.get("event"))
        name_match = any(fuzz.WRatio(event_name, target) >= 55 for target in event_names)
        date_match = pd.notna(listed_date) and any(abs((listed_date.date() - value.date()).days) <= 1 for value in valid_dates)
        if not name_match and not date_match:
            continue
        try:
            card = parse_ufcstats_card(event)
        except Exception:
            continue
        for bout in card:
            if bout.get("winner") and pair_key(bout["fighter_a"], bout["fighter_b"]) in due_pairs:
                completed.append(bout)
                due_pairs.discard(pair_key(bout["fighter_a"], bout["fighter_b"]))
        if not due_pairs:
            break
    return completed


def update_market_history(state_dir: Path, analyses):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "market_history.csv"
    history = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=MARKET_HISTORY_COLUMNS)
    snapshots = []
    for row in analyses:
        if not np.isfinite(safe_float(row.get("live_ask"))):
            continue
        snapshots.append({
            "timestamp_utc": row["as_of_utc"], "event": row["event"],
            "fighter_a": row["fighter_a"], "fighter_b": row["fighter_b"],
            "trade_side": row["trade_side"], "model_probability": row["model_probability"],
            "live_bid": row["live_bid"], "live_ask": row["live_ask"],
            "exit_target": row["exit_target"], "market_source": row["market_source"],
        })
    if snapshots:
        incoming = pd.DataFrame(snapshots, columns=MARKET_HISTORY_COLUMNS)
        history = incoming if history.empty else pd.concat([history, incoming], ignore_index=True)
        history = history.drop_duplicates(
            subset=["timestamp_utc", "event", "fighter_a", "fighter_b", "trade_side"], keep="last"
        ).tail(5000)
        history.to_csv(path, index=False)
    return history


def update_prediction_log(state_dir: Path, analyses):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "prediction_log.csv"
    log = load_prediction_log(state_dir)
    result_map = {pair_key(row["fighter_a"], row["fighter_b"]): row["winner"] for row in analyses if row.get("winner")}
    for index, row in log.loc[log["status"] == "OPEN"].iterrows():
        winner = result_map.get(pair_key(row["fighter_a"], row["fighter_b"]))
        if winner:
            _settle_prediction_row(log, index, winner)
    for row in analyses:
        prediction_id = f"{canonical_name(row['event'])}|{'|'.join(pair_key(row['fighter_a'], row['fighter_b']))}"
        is_bet = row["action"] == "BET"
        entry_price = safe_float(row.get("market_probability")) if is_bet else np.nan
        stake = safe_float(row.get("position_dollars"), 0) if is_bet else 0
        potential_profit = stake * (1 / entry_price - 1) if is_bet and 0 < entry_price < 1 else np.nan
        record = {
            "prediction_id": prediction_id, "timestamp_utc": row["as_of_utc"],
            "event_date": row["event_date"], "event": row["event"],
            "fighter_a": row["fighter_a"], "fighter_b": row["fighter_b"], "pick": row["pick"],
            "model_probability": row["model_probability"], "market_probability": row["market_probability"],
            "net_edge": row["net_edge"], "action": row["action"], "position_dollars": row["position_dollars"],
            "entry_timestamp_utc": row["as_of_utc"] if is_bet else "",
            "entry_price": entry_price, "entry_odds": price_to_american(entry_price) if is_bet else np.nan,
            "potential_profit": potential_profit, "status": "OPEN", "winner": "",
            "outcome": "", "settled_timestamp_utc": "", "exit_type": "", "exit_price": np.nan,
            "return_pct": np.nan, "pnl": 0, "closing_reason": "", "model_version": row["model_version"],
            "last_price_timestamp_utc": "", "live_bid": np.nan, "live_ask": np.nan,
            "target_price": np.nan, "target_progress": np.nan,
            "unrealized_return": np.nan, "unrealized_pnl": np.nan, "position_signal": "",
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
            existing_index = existing[-1]
            existing_action = str(log.at[existing_index, "action"])
            # Preserve the original executable entry price once a BET is recorded.
            # A prior NO BET can become a new entry if the live price later creates an edge.
            if existing_action != "BET":
                for key, value in record.items():
                    log.loc[existing_index, key] = value
        else:
            incoming = pd.DataFrame([record], columns=LOG_COLUMNS)
            log = incoming if log.empty else pd.concat([log, incoming], ignore_index=True)
    log = normalize_prediction_log(log)
    log.to_csv(path, index=False)
    return log


def mark_prediction_log(state_dir: Path, position_rows):
    """Persist the latest executable mark for every open paper position on the loaded card."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "prediction_log.csv"
    log = load_prediction_log(state_dir)
    marked_at = datetime.now(timezone.utc).isoformat()
    for row in position_rows or []:
        prediction_id = str(row.get("prediction_id") or "")
        if not prediction_id:
            continue
        matches = log.index[
            (log["prediction_id"] == prediction_id)
            & (log["status"] == "OPEN")
            & (log["action"] == "BET")
        ].tolist()
        if not matches:
            continue
        index = matches[-1]
        values = {
            "last_price_timestamp_utc": marked_at,
            "live_bid": safe_float(row.get("current_bid")),
            "live_ask": safe_float(row.get("current_ask")),
            "target_price": safe_float(row.get("target_price")),
            "target_progress": safe_float(row.get("target_progress")),
            "unrealized_return": safe_float(row.get("unrealized_return")),
            "unrealized_pnl": safe_float(row.get("unrealized_pnl")),
            "position_signal": str(row.get("signal") or ""),
        }
        for column, value in values.items():
            log.at[index, column] = value
    log = normalize_prediction_log(log)
    log.to_csv(path, index=False)
    return log


def realized_metrics(log):
    log = normalize_prediction_log(log)
    completed = log[log["status"] == "COMPLETED"] if len(log) else pd.DataFrame()
    bets = completed[completed["action"] == "BET"] if len(completed) else pd.DataFrame()
    open_bets = log[(log["status"] == "OPEN") & (log["action"] == "BET")] if len(log) else pd.DataFrame()
    wins = int((bets["outcome"] == "WIN").sum()) if len(bets) else 0
    losses = int((bets["outcome"] == "LOSS").sum()) if len(bets) else 0
    staked = float(pd.to_numeric(bets.get("position_dollars", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(bets) else 0
    pnl = float(pd.to_numeric(bets.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(bets) else 0
    open_risk = float(pd.to_numeric(open_bets.get("position_dollars", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(open_bets) else 0
    open_pnl = float(pd.to_numeric(open_bets.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(open_bets) else 0
    marked_open = int(pd.to_numeric(open_bets.get("live_bid", pd.Series(dtype=float)), errors="coerce").notna().sum()) if len(open_bets) else 0
    total_staked = staked + open_risk
    total_pnl = pnl + open_pnl
    entry_times = pd.to_datetime(log.loc[log["action"] == "BET", "entry_timestamp_utc"], utc=True, errors="coerce") if len(log) else pd.Series(dtype="datetime64[ns, UTC]")
    return {
        "wins": wins, "losses": losses, "graded": len(bets), "open": len(open_bets),
        "pnl": pnl, "roi": pnl / staked if staked else np.nan,
        "win_rate": wins / len(bets) if len(bets) else np.nan,
        "staked": staked, "open_risk": open_risk,
        "open_pnl": open_pnl, "marked_open": marked_open,
        "total_pnl": total_pnl, "total_staked": total_staked,
        "net_return": total_pnl / total_staked if total_staked else np.nan,
        "inception": entry_times.min() if entry_times.notna().any() else pd.NaT,
    }
