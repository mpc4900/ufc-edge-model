from pathlib import Path

import numpy as np
import pandas as pd

from excel_report import build_excel
from model_engine import (
    feature_vector,
    load_assets,
    local_drivers,
    normalize_prediction_log,
    price_to_american,
    safe_float,
)


ROOT = Path(__file__).resolve().parent


def release_analyses(log, bundle, fighters):
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
            "american_odds": price_to_american(price), "net_edge": safe_float(record.get("net_edge")),
            "action": record["action"], "position_dollars": safe_float(record.get("position_dollars"), 0),
            "experience": experience, "why": record.get("decision_reason", ""),
            "drivers": drivers[:3], "raw_inputs": raw_inputs, "research_shift": 0,
            "market_source": record.get("entry_source", ""),
            "market_url": record.get("entry_market_url", ""),
            "market_timestamp": record.get("entry_timestamp_utc", ""),
            "as_of_utc": record.get("timestamp_utc", ""),
            "model_version": record.get("model_version", bundle["version"]),
        })
    return rows


def main():
    bundle, fighters = load_assets(ROOT)
    log = normalize_prediction_log(pd.read_csv(ROOT / "seed_prediction_log.csv"))
    analyses = release_analyses(log, bundle, fighters)
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
