from pathlib import Path

import numpy as np
import pandas as pd

from excel_report import build_excel
from model_engine import (
    binary_position_sizing,
    entry_timing,
    feature_vector,
    load_assets,
    local_drivers,
    normalize_prediction_log,
    price_to_american,
    safe_float,
)


ROOT = Path(__file__).resolve().parent


def release_analyses(log, bundle, fighters, bankroll=10_100, max_card_exposure=0.20):
    rows = []
    for record in log.to_dict("records"):
        vector, _, experience, raw_inputs = feature_vector(
            record["fighter_a"], record["fighter_b"], fighters, bundle["features"]
        )
        pick_is_a = record["pick"] == record["fighter_a"]
        drivers = local_drivers(bundle, vector)
        for driver in drivers:
            driver["impact_pick"] = driver["impact_a"] if pick_is_a else -driver["impact_a"]
        drivers = sorted(drivers, key=lambda item: abs(item["impact_pick"]), reverse=True)
        price = safe_float(record.get("market_probability"))
        model_p = safe_float(record.get("model_probability"))
        sizing = binary_position_sizing(model_p, price, bankroll, experience, cost_buffer=0.02)
        timing = entry_timing(
            model_p, price, cost_buffer=0.02, min_edge=0.03,
            experience=experience, min_prior_fights=3,
        )
        is_bet = record["action"] == "BET"
        rows.append({
            "event": record["event"], "event_date": record["event_date"],
            "fighter_a": record["fighter_a"], "fighter_b": record["fighter_b"],
            "winner": record["winner"], "fight_url": "", "pick": record["pick"],
            "trade_side": record["pick"], "likely_winner": record["pick"],
            "likely_probability": model_p, "model_probability": model_p,
            "probability_a": model_p if pick_is_a else 1 - model_p,
            "probability_b": 1 - model_p if pick_is_a else model_p,
            "market_probability": price, "live_ask": price,
            "live_bid": safe_float(record.get("exit_price")),
            "exit_target": safe_float(record.get("target_price")),
            "scenario_move": np.nan, "scenario_return": np.nan,
            "american_odds": price_to_american(price), "net_edge": sizing["net_edge"],
            "action": record["action"], "position_dollars": sizing["position_dollars"] if is_bet else 0,
            "model_odds": price_to_american(model_p), "entry_odds": price_to_american(price),
            "effective_entry": sizing["effective_entry"], "gross_edge": sizing["gross_edge"],
            "full_kelly": sizing["full_kelly"], "kelly_fraction": sizing["kelly_fraction"],
            "data_reliability": sizing["data_reliability"],
            "uncapped_position_fraction": sizing["uncapped_position_fraction"],
            "position_fraction": sizing["position_fraction"] if is_bet else 0,
            "portfolio_scale": 1.0 if is_bet else 0.0,
            "expected_profit": sizing["expected_profit"] if is_bet else 0.0,
            "max_entry_price": timing["max_entry_price"],
            "timing_signal": timing["timing_signal"], "timing_reason": timing["timing_reason"],
            "hours_to_event": timing["hours_to_event"], "bankroll_at_entry": bankroll,
            "experience": experience, "why": record.get("decision_reason", ""),
            "drivers": drivers[:3], "raw_inputs": raw_inputs, "research_shift": 0,
            "market_source": record.get("entry_source", ""),
            "market_url": record.get("entry_market_url", ""),
            "market_timestamp": record.get("entry_timestamp_utc", ""),
            "as_of_utc": record.get("timestamp_utc", ""),
            "model_version": record.get("model_version", bundle["version"]),
        })
    total_position = sum(row["position_dollars"] for row in rows if row["action"] == "BET")
    exposure_cap = bankroll * max_card_exposure
    if total_position > exposure_cap > 0:
        scale = exposure_cap / total_position
        for row in rows:
            if row["action"] == "BET":
                row["position_dollars"] = round(row["position_dollars"] * scale, 2)
                row["position_fraction"] = row["position_dollars"] / bankroll
                row["portfolio_scale"] = scale
                row["expected_profit"] *= scale
    return rows


def main():
    bundle, fighters = load_assets(ROOT)
    log = normalize_prediction_log(pd.read_csv(ROOT / "seed_prediction_log.csv"))
    analyses = release_analyses(log, bundle, fighters, bankroll=10_100, max_card_exposure=.20)
    output = build_excel(
        analyses, bundle, log, 10_100, "UFC 330: Makhachev vs. Machado Garry",
        cost_buffer=.02, min_edge=.03, min_prior_fights=3,
        max_card_exposure=.20, fighters=fighters,
    )
    destination = ROOT / "dist" / "UFC_Edge_Model_Audit.xlsx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output)
    print(destination)


if __name__ == "__main__":
    main()
