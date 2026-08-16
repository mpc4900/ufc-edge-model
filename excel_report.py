from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xlsxwriter

from model_engine import realized_metrics, safe_float


RED = "#D20A0A"
BLACK = "#090A0C"
DARK = "#151619"
MID = "#25272B"
WHITE = "#F4F4F1"
GRAY = "#8A8D93"
LIGHT = "#ECEDEF"
BORDER = "#C9CBD0"
GREEN = "#137A4A"
AMBER = "#C47A00"
INPUT_BLUE = "#0000FF"
FORMULA_GREEN = "#008000"


def formats(workbook):
    base = {"font_name": "Arial", "font_size": 10}
    return {
        "title": workbook.add_format({**base, "bold": True, "font_color": WHITE, "bg_color": BLACK, "font_size": 18, "align": "left", "valign": "vcenter"}),
        "subtitle": workbook.add_format({**base, "font_color": "#D7D8DB", "bg_color": BLACK, "font_size": 9, "align": "left", "valign": "vcenter"}),
        "section": workbook.add_format({**base, "bold": True, "font_color": WHITE, "bg_color": RED, "font_size": 11, "align": "left", "valign": "vcenter"}),
        "header": workbook.add_format({**base, "bold": True, "font_color": WHITE, "bg_color": DARK, "bottom": 2, "bottom_color": RED, "align": "center", "valign": "vcenter", "text_wrap": True}),
        "text": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "valign": "vcenter"}),
        "wrap": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "valign": "top", "text_wrap": True}),
        "small_wrap": workbook.add_format({**base, "font_size": 9, "font_color": "#45484E", "bottom": 1, "bottom_color": BORDER, "valign": "top", "text_wrap": True}),
        "pct": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "num_format": "0.0%;[Red](0.0%);-", "align": "right", "valign": "vcenter"}),
        "pct_input": workbook.add_format({**base, "font_color": INPUT_BLUE, "bottom": 1, "bottom_color": BORDER, "num_format": "0.0%;[Red](0.0%);-", "align": "right"}),
        "pct_formula": workbook.add_format({**base, "font_color": FORMULA_GREEN, "bottom": 1, "bottom_color": BORDER, "num_format": "0.0%;[Red](0.0%);-", "align": "right"}),
        "money": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "num_format": "$#,##0;[Red]($#,##0);-", "align": "right", "valign": "vcenter"}),
        "money_formula": workbook.add_format({**base, "font_color": FORMULA_GREEN, "bottom": 1, "bottom_color": BORDER, "num_format": "$#,##0;[Red]($#,##0);-", "align": "right"}),
        "num": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "num_format": "0.000", "align": "right", "valign": "vcenter"}),
        "num_input": workbook.add_format({**base, "font_color": INPUT_BLUE, "bottom": 1, "bottom_color": BORDER, "num_format": "0.000", "align": "right"}),
        "int": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "num_format": "#,##0", "align": "right", "valign": "vcenter"}),
        "int_input": workbook.add_format({**base, "font_color": INPUT_BLUE, "bottom": 1, "bottom_color": BORDER, "num_format": "#,##0", "align": "right"}),
        "date": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "num_format": "mmm d, yyyy", "align": "center", "valign": "vcenter"}),
        "datetime": workbook.add_format({**base, "bottom": 1, "bottom_color": BORDER, "num_format": "mmm d, yyyy h:mm AM/PM", "align": "center", "valign": "vcenter"}),
        "formula_text": workbook.add_format({**base, "font_color": FORMULA_GREEN, "bottom": 1, "bottom_color": BORDER, "valign": "vcenter"}),
        "bet": workbook.add_format({**base, "bold": True, "font_color": WHITE, "bg_color": RED, "align": "center", "valign": "vcenter"}),
        "no_bet": workbook.add_format({**base, "bold": True, "font_color": "#BFC1C5", "bg_color": MID, "align": "center", "valign": "vcenter"}),
        "pass": workbook.add_format({**base, "bold": True, "font_color": WHITE, "bg_color": GREEN, "align": "center", "valign": "vcenter"}),
        "review": workbook.add_format({**base, "bold": True, "font_color": BLACK, "bg_color": "#F6C453", "align": "center", "valign": "vcenter"}),
        "card_label": workbook.add_format({**base, "bold": True, "font_color": "#60636A", "bg_color": LIGHT, "top": 1, "bottom": 1, "top_color": BORDER, "bottom_color": BORDER, "align": "center", "valign": "vcenter"}),
        "card_value": workbook.add_format({**base, "bold": True, "font_size": 16, "font_color": BLACK, "bg_color": "#FFFFFF", "bottom": 3, "bottom_color": RED, "align": "center", "valign": "vcenter"}),
        "card_pct": workbook.add_format({**base, "bold": True, "font_size": 16, "font_color": BLACK, "bg_color": "#FFFFFF", "bottom": 3, "bottom_color": RED, "align": "center", "valign": "vcenter", "num_format": "0.0%"}),
        "card_money": workbook.add_format({**base, "bold": True, "font_size": 16, "font_color": BLACK, "bg_color": "#FFFFFF", "bottom": 3, "bottom_color": RED, "align": "center", "valign": "vcenter", "num_format": "$#,##0"}),
        "label": workbook.add_format({**base, "bold": True, "font_color": "#4B4E54", "bg_color": LIGHT, "bottom": 1, "bottom_color": BORDER, "valign": "top", "text_wrap": True}),
        "input_text": workbook.add_format({**base, "font_color": INPUT_BLUE, "bottom": 1, "bottom_color": BORDER, "valign": "top", "text_wrap": True}),
        "note": workbook.add_format({**base, "font_size": 9, "font_color": "#5E6168", "bg_color": "#F5F5F6", "text_wrap": True, "valign": "top"}),
        "formula_box": workbook.add_format({**base, "font_name": "Courier New", "font_color": BLACK, "bg_color": "#F5F5F6", "border": 1, "border_color": BORDER, "text_wrap": True, "valign": "vcenter"}),
    }


def _write_title(worksheet, title, subtitle, last_col, fmt):
    worksheet.hide_gridlines(2)
    worksheet.set_row(0, 32)
    worksheet.set_row(1, 20)
    worksheet.merge_range(0, 0, 0, last_col, title, fmt["title"])
    worksheet.merge_range(1, 0, 1, last_col, subtitle, fmt["subtitle"])


def _write_value(worksheet, row, column, value, kind, fmt):
    if kind in {"date", "datetime"}:
        date = pd.to_datetime(value, errors="coerce")
        if pd.notna(date):
            if getattr(date, "tzinfo", None) is not None:
                date = date.tz_convert(None)
            worksheet.write_datetime(row, column, date.to_pydatetime(), fmt[kind])
        else:
            worksheet.write_blank(row, column, None, fmt[kind])
    elif kind in {"pct", "money", "num", "int"}:
        number = safe_float(value)
        if np.isfinite(number):
            worksheet.write_number(row, column, number, fmt[kind])
        else:
            worksheet.write_blank(row, column, None, fmt[kind])
    else:
        worksheet.write(row, column, "" if pd.isna(value) else str(value), fmt.get(kind, fmt["text"]))


def _table_sheet(workbook, fmt, name, title, subtitle, columns, rows):
    worksheet = workbook.add_worksheet(name)
    _write_title(worksheet, title, subtitle, len(columns) - 1, fmt)
    worksheet.freeze_panes(4, 0)
    for column, (label, _, width, _) in enumerate(columns):
        worksheet.write(3, column, label, fmt["header"])
        worksheet.set_column(column, column, width)
    for row_index, record in enumerate(rows, start=4):
        worksheet.set_row(row_index, 24)
        for column, (_, key, _, kind) in enumerate(columns):
            _write_value(worksheet, row_index, column, record.get(key, ""), kind, fmt)
    if rows:
        worksheet.autofilter(3, 0, 3 + len(rows), len(columns) - 1)
    return worksheet


def _raw_feature_rows(analyses):
    rows = []
    for record in analyses:
        impact_map = {driver["factor"]: driver["impact_pick"] for driver in record.get("drivers", [])}
        for item in record.get("raw_inputs", []):
            rows.append({
                "event": record.get("event", ""),
                "fight": f"{record['fighter_a']} vs {record['fighter_b']}",
                "factor": item.get("Factor", ""),
                "fighter_a": record["fighter_a"],
                "a_value": item.get("Fighter A"),
                "fighter_b": record["fighter_b"],
                "b_value": item.get("Fighter B"),
                "difference": item.get("Model difference"),
                "trade_side": record.get("trade_side", ""),
                "driver_impact": impact_map.get(item.get("Factor", ""), np.nan),
            })
    return rows


def build_excel(
    analyses,
    bundle,
    prediction_log,
    bankroll,
    event_search,
    cost_buffer=0.02,
    min_edge=0.03,
    min_prior_fights=3,
    convergence=0.50,
    max_card_exposure=0.10,
    capture_target=0.65,
    minimum_exit_return=0.03,
    stop_loss=0.15,
    convergence_rows=None,
    market_history=None,
    fighters=None,
):
    """Create a formula-driven UFC pricing and audit workbook."""
    convergence_rows = convergence_rows or []
    market_history = market_history if isinstance(market_history, pd.DataFrame) else pd.DataFrame()
    prediction_log = prediction_log if isinstance(prediction_log, pd.DataFrame) else pd.DataFrame()
    fighters = fighters if isinstance(fighters, pd.DataFrame) else pd.DataFrame()
    now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True, "nan_inf_to_errors": True})
    workbook.set_properties({
        "title": "UFC Edge Model — Detailed Decision Report",
        "subject": "UFC fair value, executable market prices, edge, sizing and convergence",
        "company": "UFC Edge",
        "comments": "Hardcoded inputs are blue. Workbook formulas are green.",
    })
    fmt = formats(workbook)

    # Assumptions are intentionally visible and linked throughout the workbook.
    assumptions = workbook.add_worksheet("ASSUMPTIONS")
    _write_title(assumptions, "UFC EDGE — MODEL ASSUMPTIONS", f"Auditable inputs // generated {now_label}", 4, fmt)
    assumptions.set_column("A:A", 30)
    assumptions.set_column("B:B", 16)
    assumptions.set_column("C:C", 22)
    assumptions.set_column("D:E", 48)
    for col, label in enumerate(["Input", "Value", "Type", "How it is used", "Interpretation"]):
        assumptions.write(3, col, label, fmt["header"])
    assumption_rows = [
        ("Bankroll", bankroll, "money", "Position sizing denominator", "Capital base used only for sizing"),
        ("Cost buffer", cost_buffer, "pct", "Model P − live ask − cost buffer", "Allowance for spread, fees and slippage"),
        ("Minimum net edge", min_edge, "pct", "Minimum edge required for BET", "Below this level the decision is NO BET"),
        ("Minimum prior UFC fights", min_prior_fights, "int", "Minimum of both fighters' UFC experience", "Reduces action on thin UFC samples"),
        ("Convergence scenario", convergence, "pct", "Ask + convergence × (fair − ask)", "Illustrative exit target; not guaranteed"),
        ("Kelly fraction", 0.25, "pct", "Quarter Kelly", "Reduces full-Kelly volatility"),
        ("Per-fight position cap", 0.02, "pct", "Maximum share of bankroll in one fight", "Concentration control"),
        ("Card exposure cap", max_card_exposure, "pct", "Maximum total risk on one card", "Portfolio-level exposure control"),
        ("Gap capture target", capture_target, "pct", "Share of entry-to-fair gap required before SELL", "Measures observed convergence"),
        ("Minimum exit return", minimum_exit_return, "pct", "Required sellable return before SELL", "Prevents tiny, untradeable exit signals"),
        ("Loss review threshold", stop_loss, "pct", "Flags a position for REVIEW", "Not an automatic stop order"),
    ]
    for row, (label, value, kind, use, interpretation) in enumerate(assumption_rows, start=4):
        assumptions.write(row, 0, label, fmt["label"])
        value_fmt = {"pct": fmt["pct_input"], "money": fmt["money"], "int": fmt["int_input"]}[kind]
        assumptions.write_number(row, 1, value, value_fmt)
        assumptions.write(row, 2, {"pct": "Percentage", "money": "USD", "int": "Count"}[kind], fmt["input_text"])
        assumptions.write(row, 3, use, fmt["wrap"])
        assumptions.write(row, 4, interpretation, fmt["small_wrap"])
        assumptions.set_row(row, 34)
    assumptions.merge_range("A18:E19", "COLOR KEY: blue font = hardcoded input or live snapshot. Green font = Excel formula. Red blocks = BET/action emphasis. Market prices are executable order-book snapshots and can change after download.", fmt["note"])

    # Formula-driven pricing table.
    pricing = workbook.add_worksheet("FIGHT PRICING")
    _write_title(pricing, "UFC EDGE — FIGHT PRICING", f"{event_search} // live snapshot {now_label}", 17, fmt)
    pricing.freeze_panes(4, 3)
    pricing_headers = [
        "Fight", "Likely Winner", "Trade Side", "Model Fair P", "Live Bid", "Live Ask",
        "Cost Buffer", "Net Edge", "Prior UFC Fights", "Full Kelly", "Quarter Kelly",
        "Position $", "Exit Target", "Decision", "Decision Rationale", "Market Source", "Market URL", "As Of UTC",
    ]
    widths = [36, 23, 23, 13, 11, 11, 11, 12, 14, 12, 12, 13, 12, 12, 44, 19, 48, 22]
    for col, label in enumerate(pricing_headers):
        pricing.write(3, col, label, fmt["header"])
        pricing.set_column(col, col, widths[col])
    for row_index, record in enumerate(analyses, start=4):
        excel_row = row_index + 1
        pricing.set_row(row_index, 34)
        pricing.write(row_index, 0, f"{record['fighter_a']} vs {record['fighter_b']}", fmt["text"])
        pricing.write(row_index, 1, record["likely_winner"], fmt["text"])
        pricing.write(row_index, 2, record["trade_side"], fmt["text"])
        pricing.write_number(row_index, 3, safe_float(record["model_probability"]), fmt["pct_input"])
        _write_value(pricing, row_index, 4, record.get("live_bid"), "pct", fmt)
        _write_value(pricing, row_index, 5, record.get("live_ask"), "pct", fmt)
        pricing.write_formula(row_index, 6, "=ASSUMPTIONS!$B$6", fmt["pct_formula"], cost_buffer)
        edge = safe_float(record.get("net_edge"))
        pricing.write_formula(row_index, 7, f'=IF(ISNUMBER(F{excel_row}),D{excel_row}-F{excel_row}-G{excel_row},"")', fmt["pct_formula"], edge if np.isfinite(edge) else "")
        pricing.write_number(row_index, 8, int(record.get("experience", 0)), fmt["int_input"])
        ask = safe_float(record.get("live_ask"))
        full_kelly = max(0, edge / max(1e-9, 1 - ask - cost_buffer)) if np.isfinite(edge) and np.isfinite(ask) else np.nan
        pricing.write_formula(row_index, 9, f'=IFERROR(MAX(0,H{excel_row}/(1-F{excel_row}-G{excel_row})),0)', fmt["pct_formula"], full_kelly if np.isfinite(full_kelly) else 0)
        pricing.write_formula(row_index, 10, f"=J{excel_row}*ASSUMPTIONS!$B$10", fmt["pct_formula"], full_kelly * 0.25 if np.isfinite(full_kelly) else 0)
        pricing.write_number(row_index, 11, safe_float(record.get("position_dollars"), 0), fmt["money"])
        exit_target = safe_float(record.get("exit_target"))
        pricing.write_formula(row_index, 12, f'=IFERROR(MIN(D{excel_row},F{excel_row}+ASSUMPTIONS!$B$9*(D{excel_row}-F{excel_row})),"")', fmt["pct_formula"], exit_target if np.isfinite(exit_target) else "")
        decision_formula = f'=IF(AND(ISNUMBER(F{excel_row}),H{excel_row}>=ASSUMPTIONS!$B$7,I{excel_row}>=ASSUMPTIONS!$B$8),"BET","NO BET")'
        pricing.write_formula(row_index, 13, decision_formula, fmt["bet"] if record["action"] == "BET" else fmt["no_bet"], record["action"])
        pricing.write(row_index, 14, record.get("why", ""), fmt["wrap"])
        pricing.write(row_index, 15, record.get("market_source", ""), fmt["text"])
        url = str(record.get("market_url") or "")
        if url.startswith("http"):
            pricing.write_url(row_index, 16, url, string="Open market")
        else:
            pricing.write(row_index, 16, "", fmt["text"])
        pricing.write(row_index, 17, str(record.get("as_of_utc") or ""), fmt["small_wrap"])
    if analyses:
        pricing.autofilter(3, 0, 3 + len(analyses), 17)
        pricing.conditional_format(4, 7, 3 + len(analyses), 7, {"type": "3_color_scale", "min_color": "#F4CCCC", "mid_color": "#FFF2CC", "max_color": "#D9EAD3"})

    # Decision board links to the pricing sheet so the displayed math stays auditable.
    dashboard = workbook.add_worksheet("DECISION BOARD")
    _write_title(dashboard, "UFC EDGE — DECISION BOARD", f"{event_search} // generated {now_label}", 7, fmt)
    dashboard.freeze_panes(8, 0)
    bet_count = int((prediction_log.get("action", pd.Series(dtype=str)) == "BET").sum()) if len(prediction_log) else 0
    edges = [safe_float(row.get("net_edge")) for row in analyses if np.isfinite(safe_float(row.get("net_edge")))]
    top_edge = max(edges) if edges else np.nan
    card_risk = sum(safe_float(row.get("position_dollars"), 0) for row in analyses if row["action"] == "BET")
    dashboard_realized = realized_metrics(prediction_log)
    model_input_columns = [column for column in fighters.columns if column not in {"Fighter", "Canonical"}]
    data_points = int(len(fighters) * len(model_input_columns))
    cards = [
        ("A4:B4", "A5:B6", "NET MODEL P&L", dashboard_realized["total_pnl"], "card_money"),
        ("C4:D4", "C5:D6", "SETTLED WIN RATE", dashboard_realized["win_rate"], "card_pct"),
        ("E4:F4", "E5:F6", "RECORDED BETS", bet_count, "card_value"),
        ("G4:H4", "G5:H6", "FIGHTER STAT CELLS", data_points, "card_value"),
    ]
    for label_range, value_range, label, value, kind in cards:
        dashboard.merge_range(label_range, label, fmt["card_label"])
        dashboard.merge_range(value_range, value if np.isfinite(safe_float(value)) else "—", fmt[kind] if np.isfinite(safe_float(value)) else fmt["card_value"])
    headers = ["Fight", "Trade Side", "Model Fair", "Live Ask", "Net Edge", "BET / NO BET", "Position $", "Why"]
    dashboard_widths = [38, 24, 13, 12, 12, 15, 14, 46]
    for col, label in enumerate(headers):
        dashboard.write(7, col, label, fmt["header"])
        dashboard.set_column(col, col, dashboard_widths[col])
    for row_index, record in enumerate(analyses, start=8):
        source_row = row_index - 3
        dashboard.set_row(row_index, 34)
        links = [
            (0, f"='FIGHT PRICING'!A{source_row}", f"{record['fighter_a']} vs {record['fighter_b']}", fmt["formula_text"]),
            (1, f"='FIGHT PRICING'!C{source_row}", record["trade_side"], fmt["formula_text"]),
            (2, f"='FIGHT PRICING'!D{source_row}", record["model_probability"], fmt["pct_formula"]),
            (3, f"='FIGHT PRICING'!F{source_row}", record.get("live_ask", ""), fmt["pct_formula"]),
            (4, f"='FIGHT PRICING'!H{source_row}", record.get("net_edge", ""), fmt["pct_formula"]),
        ]
        for col, formula, value, cell_fmt in links:
            dashboard.write_formula(row_index, col, formula, cell_fmt, value)
        dashboard.write_formula(row_index, 5, f"='FIGHT PRICING'!N{source_row}", fmt["bet"] if record["action"] == "BET" else fmt["no_bet"], record["action"])
        dashboard.write_formula(row_index, 6, f"='FIGHT PRICING'!L{source_row}", fmt["money_formula"], record.get("position_dollars", 0))
        dashboard.write_formula(row_index, 7, f"='FIGHT PRICING'!O{source_row}", fmt["formula_text"], record.get("why", ""))
    if analyses:
        dashboard.autofilter(7, 0, 7 + len(analyses), 7)
    dashboard.merge_range(10 + len(analyses), 0, 11 + len(analyses), 7, "READ THE BOARD LEFT TO RIGHT: the model creates a statistical fair probability; the live ask is the executable entry price; net edge subtracts the cost buffer; BET requires sufficient edge and UFC experience; position size is quarter Kelly subject to fight and card caps.", fmt["note"])

    # Detailed model math and a worked example.
    guide = workbook.add_worksheet("MODEL GUIDE")
    _write_title(guide, "UFC EDGE — HOW THE MATH WORKS", "Plain-English model, price, decision and exit logic", 7, fmt)
    guide.set_column("A:A", 5)
    guide.set_column("B:B", 28)
    guide.set_column("C:F", 22)
    guide.set_column("G:H", 26)
    guide.merge_range("A4:H4", "1 // WHAT THE MODEL IS ANSWERING", fmt["section"])
    guide.merge_range("A5:H6", "The likely winner is the fighter with the higher calibrated win probability. The trade side is the outcome with the best price discount after costs. A fighter can be the underdog and still be the better trade if the market price is sufficiently below the model's probability.", fmt["note"])
    guide.merge_range("A8:H8", "2 // MODEL PIPELINE", fmt["section"])
    pipeline = [
        ("Pre-fight inputs", "Opponent-adjusted Elo, smoothed record, recent form, striking, takedowns, control time, reach and age."),
        ("Gradient boosting", "200 shallow trees learn nonlinear interactions from completed UFC fights."),
        ("Order symmetry", "P(A) = [GB(x) + 1 − GB(−x)] ÷ 2, preventing fighter ordering from changing the result."),
        ("Calibration", f"P(A) = logistic({bundle['calibration_slope']:.3f} × logit(raw P(A))). Calibration maps raw scores to observed frequencies."),
        ("Trade selection", "Compare both fighters' model probabilities with their executable asks and select the larger net edge."),
        ("Decision", "BET only if net edge and minimum-experience requirements both pass."),
        ("Sizing", "Full Kelly estimates the growth-optimal fraction; quarter Kelly and portfolio caps reduce volatility."),
    ]
    for row, (step, explanation) in enumerate(pipeline, start=8):
        guide.write(row, 0, row - 7, fmt["label"])
        guide.write(row, 1, step, fmt["label"])
        guide.merge_range(row, 2, row, 7, explanation, fmt["wrap"])
        guide.set_row(row, 34)
    guide.merge_range("A17:H17", "3 // ELO RATING — EXACT UPDATE MATH", fmt["section"])
    guide.merge_range("A18:H18", "Expected score = 1 ÷ [1 + 10^((Opponent Elo − Fighter Elo) ÷ 400)]", fmt["formula_box"])
    guide.merge_range("A19:H19", "New Elo = Old Elo + K × (Actual result − Expected score)", fmt["formula_box"])
    guide.merge_range("A20:H21", "Each completed UFC fight updates both fighters. Beating a stronger opponent earns more points than beating a weaker opponent; losing to a weaker opponent costs more. The model uses only the rating available before the predicted fight, which prevents future results from leaking into a historical prediction.", fmt["note"])
    guide.merge_range("A23:H23", "4 // WORKED PRICING EXAMPLE", fmt["section"])
    example = analyses[0] if analyses else {}
    model_p = safe_float(example.get("model_probability"))
    ask = safe_float(example.get("live_ask"))
    edge = safe_float(example.get("net_edge"))
    full_kelly = max(0, edge / max(1e-9, 1 - ask - cost_buffer)) if np.isfinite(edge) and np.isfinite(ask) else np.nan
    worked = [
        ("Model fair probability", model_p, "pct", "Calibrated probability for the selected trade side"),
        ("Executable live ask", ask, "pct", "Current price required to enter"),
        ("Cost buffer", cost_buffer, "pct", "Spread / fee / slippage allowance"),
        ("Net edge", edge, "pct", "Model fair − live ask − cost buffer"),
        ("Full Kelly", full_kelly, "pct", "Net edge ÷ (1 − live ask − cost buffer)"),
        ("Quarter Kelly", full_kelly * 0.25 if np.isfinite(full_kelly) else np.nan, "pct", "Full Kelly × 25%"),
        ("Actual position", safe_float(example.get("position_dollars")), "money", "Quarter Kelly after fight and card caps"),
        ("Exit target", safe_float(example.get("exit_target")), "pct", "Ask + convergence scenario × (fair − ask)"),
    ]
    for col, label in enumerate(["", "Metric", "Value", "Interpretation"]):
        guide.write(23, col, label, fmt["header"])
    for row, (label, value, kind, note) in enumerate(worked, start=24):
        guide.write(row, 0, row - 23, fmt["label"])
        guide.write(row, 1, label, fmt["label"])
        if np.isfinite(value):
            guide.write_number(row, 2, value, fmt[kind])
        else:
            guide.write_blank(row, 2, None, fmt[kind])
        guide.merge_range(row, 3, row, 7, note, fmt["wrap"])
        guide.set_row(row, 28)
    guide.merge_range("A34:H34", "5 // PRICE DISCOVERY AND PRE-FIGHT EXIT", fmt["section"])
    guide.merge_range("A35:H35", "Gap closed today = (Current midpoint − 8 AM price) ÷ (Model fair value − 8 AM price)", fmt["formula_box"])
    guide.merge_range("A36:H36", "Position capture = (Executable bid − Recorded entry) ÷ (Model fair value − Recorded entry)", fmt["formula_box"])
    guide.merge_range("A37:H39", "The convergence idea is tested, not assumed. The live executable bid determines what can actually be realized before the fight. SELL requires both a positive minimum return and sufficient observed gap capture; HOLD means statistical value remains; REVIEW flags a material drawdown. A prediction market can move away from model fair value, and convergence is never guaranteed.", fmt["note"])

    feature_rows = _raw_feature_rows(analyses)
    _table_sheet(workbook, fmt, "FEATURE INPUTS", "UFC EDGE — RAW FEATURE INPUTS", "All pre-fight values fed to the model; differences are Fighter A minus Fighter B except age advantage", [
        ("Event", "event", 20, "text"), ("Fight", "fight", 36, "text"), ("Factor", "factor", 27, "text"),
        ("Fighter A", "fighter_a", 22, "text"), ("A Value", "a_value", 13, "num"),
        ("Fighter B", "fighter_b", 22, "text"), ("B Value", "b_value", 13, "num"),
        ("Model Difference", "difference", 16, "num"), ("Trade Side", "trade_side", 22, "text"),
        ("Local Probability Impact", "driver_impact", 18, "pct"),
    ], feature_rows)

    # Price convergence and exit monitor.
    convergence_sheet = workbook.add_worksheet("CONVERGENCE")
    _write_title(convergence_sheet, "UFC EDGE — PRE-FIGHT CONVERGENCE", "Entry, morning price, executable exit and measured gap capture", 15, fmt)
    convergence_sheet.freeze_panes(4, 2)
    conv_headers = ["Fight", "Trade Side", "Entry", "8 AM Price", "Live Bid", "Live Mid", "Model Fair", "Morning Gap", "Gap Closed", "Sell Target", "Position $", "Return If Sold", "P&L If Sold", "Signal", "Reason", "Hours to Event"]
    conv_widths = [36, 23, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 12, 46, 14]
    for col, label in enumerate(conv_headers):
        convergence_sheet.write(3, col, label, fmt["header"])
        convergence_sheet.set_column(col, col, conv_widths[col])
    for row_index, record in enumerate(convergence_rows, start=4):
        excel_row = row_index + 1
        values = [record.get("fight"), record.get("trade_side"), record.get("entry_price"), record.get("morning_price"), record.get("current_bid"), record.get("current_mid"), record.get("fair_value")]
        convergence_sheet.write(row_index, 0, values[0], fmt["text"])
        convergence_sheet.write(row_index, 1, values[1], fmt["text"])
        for col, value in enumerate(values[2:], start=2):
            _write_value(convergence_sheet, row_index, col, value, "pct", fmt)
        morning_gap = safe_float(record.get("fair_value")) - safe_float(record.get("morning_price"))
        convergence_sheet.write_formula(row_index, 7, f'=IFERROR(G{excel_row}-D{excel_row},"")', fmt["pct_formula"], morning_gap if np.isfinite(morning_gap) else "")
        convergence_sheet.write_formula(row_index, 8, f'=IFERROR((F{excel_row}-D{excel_row})/(G{excel_row}-D{excel_row}),"")', fmt["pct_formula"], record.get("gap_closed", ""))
        _write_value(convergence_sheet, row_index, 9, record.get("target_price"), "pct", fmt)
        _write_value(convergence_sheet, row_index, 10, record.get("position_dollars"), "money", fmt)
        convergence_sheet.write_formula(row_index, 11, f'=IFERROR(E{excel_row}/C{excel_row}-1,"")', fmt["pct_formula"], record.get("unrealized_return", ""))
        convergence_sheet.write_formula(row_index, 12, f'=IFERROR(K{excel_row}*L{excel_row},"")', fmt["money_formula"], record.get("unrealized_pnl", ""))
        signal = str(record.get("signal") or "")
        signal_fmt = fmt["bet"] if signal == "SELL" else fmt["review"] if signal == "REVIEW" else fmt["no_bet"]
        convergence_sheet.write(row_index, 13, signal, signal_fmt)
        convergence_sheet.write(row_index, 14, record.get("reason", ""), fmt["wrap"])
        _write_value(convergence_sheet, row_index, 15, record.get("hours_to_event"), "num", fmt)
        convergence_sheet.set_row(row_index, 32)
    if not convergence_rows:
        convergence_sheet.merge_range("A5:P7", "No position records were available when this report was generated. Run the live card at least once to record entries and market history.", fmt["note"])
        convergence_sheet.hide()

    snapshot_rows = market_history.to_dict("records") if len(market_history) else []
    market_sheet = _table_sheet(workbook, fmt, "MARKET SNAPSHOTS", "UFC EDGE — MARKET SNAPSHOTS", "Each app refresh appends the observed executable bid and ask; these values do not auto-refresh inside a downloaded file", [
        ("Timestamp UTC", "timestamp_utc", 23, "text"), ("Event", "event", 20, "text"),
        ("Fighter A", "fighter_a", 22, "text"), ("Fighter B", "fighter_b", 22, "text"),
        ("Trade Side", "trade_side", 22, "text"), ("Model Fair", "model_probability", 12, "pct"),
        ("Live Bid", "live_bid", 11, "pct"), ("Live Ask", "live_ask", 11, "pct"),
        ("Exit Target", "exit_target", 12, "pct"), ("Market Source", "market_source", 20, "text"),
    ], snapshot_rows)
    if not snapshot_rows:
        market_sheet.hide()

    # Backtest and live record.
    realized = realized_metrics(prediction_log)
    backtest = workbook.add_worksheet("BACKTEST")
    _write_title(backtest, "UFC EDGE — PERFORMANCE", "Frozen unseen holdout plus recorded live decisions", 7, fmt)
    backtest.set_column("A:A", 30)
    backtest.set_column("B:B", 16)
    backtest.set_column("C:C", 34)
    backtest.merge_range("A4:C4", "OUT-OF-SAMPLE MODEL TEST", fmt["section"])
    performance = [
        ("Historical training fights", bundle["metrics"]["training_fights"], "int", "Used to fit the gradient-boosting model"),
        ("Unseen holdout fights", bundle["metrics"]["holdout_fights"], "int", "Not used for training or calibration"),
        ("Holdout accuracy", bundle["metrics"]["accuracy"], "pct", "Share of fights where the higher model probability won"),
        ("Holdout Brier score", bundle["metrics"]["brier"], "num", "Mean squared probability error; lower is better"),
        ("Holdout ROC AUC", bundle["metrics"]["auc"], "num", "Ranking quality; 0.5 is random and 1.0 is perfect"),
    ]
    for row, (metric, value, kind, meaning) in enumerate(performance, start=4):
        backtest.write(row, 0, metric, fmt["label"])
        backtest.write_number(row, 1, value, fmt[kind])
        backtest.write(row, 2, meaning, fmt["wrap"])
    backtest.merge_range("A11:C11", "RECORDED LIVE DECISIONS", fmt["section"])
    live_rows = [
        ("Open paper bets", realized["open"], "int"), ("Open paper risk", realized["open_risk"], "money"),
        ("Open mark-to-market P&L", realized["open_pnl"], "money"),
        ("Open positions with live marks", realized["marked_open"], "int"),
        ("Settled paper bets", realized["graded"], "int"), ("Wins", realized["wins"], "int"),
        ("Losses", realized["losses"], "int"), ("Win rate", realized["win_rate"], "pct"),
        ("Total settled stake", realized["staked"], "money"), ("Recorded ROI", realized["roi"], "pct"),
        ("Realized P&L", realized["pnl"], "money"),
        ("Total recorded stakes", realized["total_staked"], "money"),
        ("Net model P&L", realized["total_pnl"], "money"),
        ("Net model return", realized["net_return"], "pct"),
    ]
    for row, (metric, value, kind) in enumerate(live_rows, start=11):
        backtest.write(row, 0, metric, fmt["label"])
        if np.isfinite(safe_float(value)):
            backtest.write_number(row, 1, value, fmt[kind])
        else:
            backtest.write_blank(row, 1, None, fmt[kind])
    importance = sorted(bundle["importance"].items(), key=lambda item: item[1], reverse=True)
    backtest.merge_range("E4:F4", "GLOBAL FEATURE IMPORTANCE", fmt["section"])
    backtest.write("E5", "Factor", fmt["header"])
    backtest.write("F5", "Importance", fmt["header"])
    backtest.set_column("E:E", 28)
    backtest.set_column("F:F", 14)
    for row, (factor, value) in enumerate(importance, start=5):
        backtest.write(row, 4, factor.replace(" diff", ""), fmt["text"])
        backtest.write_number(row, 5, value, fmt["pct"])
    chart = workbook.add_chart({"type": "bar"})
    chart.add_series({"name": "Importance", "categories": ["BACKTEST", 5, 4, 4 + len(importance), 4], "values": ["BACKTEST", 5, 5, 4 + len(importance), 5], "fill": {"color": RED}, "border": {"none": True}})
    chart.set_title({"name": "What the model uses most"})
    chart.set_x_axis({"num_format": "0%", "major_gridlines": {"visible": False}})
    chart.set_y_axis({"major_gridlines": {"visible": False}})
    chart.set_legend({"none": True})
    chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    chart.set_plotarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    backtest.insert_chart("E15", chart, {"x_scale": 1.15, "y_scale": 1.05})

    log_rows = prediction_log.to_dict("records") if len(prediction_log) else []
    paper_rows = prediction_log[prediction_log["action"] == "BET"].to_dict("records") if len(prediction_log) and "action" in prediction_log.columns else []
    track_rows = []
    for record in paper_rows:
        completed = str(record.get("status")) == "COMPLETED"
        track_rows.append({
            **record,
            "fight": f"{record.get('fighter_a', '')} vs {record.get('fighter_b', '')}",
            "current_final_price": record.get("exit_price") if completed else record.get("live_bid"),
            "effective_return": record.get("return_pct") if completed else record.get("unrealized_return"),
            "effective_pnl": record.get("pnl") if completed else record.get("unrealized_pnl"),
            "track_status": record.get("outcome") if completed else (record.get("position_signal") or "OPEN"),
            "last_mark_utc": record.get("settled_timestamp_utc") if completed else record.get("last_price_timestamp_utc"),
        })
    _table_sheet(workbook, fmt, "TRACK RECORD", "UFC EDGE — MODEL TRACK RECORD", "Every recorded BET at its original entry; open positions use the executable bid and completed positions use the official result", [
        ("Entry UTC", "entry_timestamp_utc", 23, "text"), ("Event", "event", 17, "text"),
        ("Fight", "fight", 34, "text"), ("Bet Taken", "pick", 22, "text"),
        ("Stake", "position_dollars", 13, "money"), ("Entry", "entry_price", 11, "pct"),
        ("Current / Final", "current_final_price", 14, "pct"), ("Sell Target", "target_price", 13, "pct"),
        ("Target Progress", "target_progress", 15, "pct"), ("Net Return", "effective_return", 13, "pct"),
        ("Net P&L", "effective_pnl", 13, "money"), ("Status", "track_status", 12, "text"),
        ("Official Winner", "winner", 22, "text"), ("Last Mark UTC", "last_mark_utc", 23, "text"),
        ("Entry Source", "entry_source", 34, "text"), ("Why", "decision_reason", 46, "wrap"),
        ("Result Source", "result_source", 44, "wrap"),
    ], track_rows)
    _table_sheet(workbook, fmt, "PREDICTION LOG", "UFC EDGE — PREDICTION LOG", "Entries are recorded before results; completed rows are graded as wins or losses", [
        ("Timestamp UTC", "timestamp_utc", 23, "text"), ("Event Date", "event_date", 14, "date"),
        ("Event", "event", 19, "text"), ("Fighter A", "fighter_a", 22, "text"), ("Fighter B", "fighter_b", 22, "text"),
        ("Pick", "pick", 22, "text"), ("Model P", "model_probability", 12, "pct"),
        ("Market Price", "market_probability", 12, "pct"), ("Net Edge", "net_edge", 12, "pct"),
        ("Decision", "action", 12, "text"), ("Position $", "position_dollars", 13, "money"),
        ("Status", "status", 12, "text"), ("Winner", "winner", 22, "text"),
        ("Outcome", "outcome", 12, "text"), ("P&L", "pnl", 13, "money"), ("Model Version", "model_version", 18, "text"),
        ("Entry Source", "entry_source", 34, "text"), ("Decision Reason", "decision_reason", 46, "wrap"),
        ("Result Source", "result_source", 44, "wrap"),
    ], log_rows)

    # Event-level ledger: one compact line per card, tied to the recorded bets.
    event_rows = []
    if len(prediction_log):
        working = prediction_log.copy()
        for event_name, group in working.groupby("event", dropna=False):
            bets_group = group[group["action"] == "BET"]
            settled_group = bets_group[bets_group["status"] == "COMPLETED"]
            stake = pd.to_numeric(bets_group.get("position_dollars"), errors="coerce").fillna(0).sum()
            pnl_value = pd.to_numeric(bets_group.get("pnl"), errors="coerce").fillna(0).sum()
            wins_value = int((settled_group.get("outcome") == "WIN").sum()) if len(settled_group) else 0
            losses_value = int((settled_group.get("outcome") == "LOSS").sum()) if len(settled_group) else 0
            event_rows.append({
                "event_date": group.get("event_date", pd.Series(dtype=str)).iloc[0] if len(group) else "",
                "event": event_name, "status": "SETTLED" if len(settled_group) == len(bets_group) and len(bets_group) else "OPEN",
                "fights": len(group), "bets": len(bets_group), "wins": wins_value, "losses": losses_value,
                "win_rate": wins_value / len(settled_group) if len(settled_group) else np.nan,
                "stake": stake, "pnl": pnl_value, "roi": pnl_value / stake if stake else np.nan,
            })
    _table_sheet(workbook, fmt, "EVENT HISTORY", "UFC EDGE — EVENT HISTORY", "One line per event; P&L uses the original recorded Polymarket entry price", [
        ("Event Date", "event_date", 14, "date"), ("Event", "event", 38, "text"),
        ("Status", "status", 12, "text"), ("Fights Screened", "fights", 15, "int"),
        ("Bets", "bets", 10, "int"), ("Wins", "wins", 10, "int"), ("Losses", "losses", 10, "int"),
        ("Win Rate", "win_rate", 12, "pct"), ("Capital Used", "stake", 14, "money"),
        ("Net P&L", "pnl", 14, "money"), ("ROI", "roi", 12, "pct"),
    ], event_rows)

    # The workbook carries the complete fighter snapshot used for live feature
    # construction. This is intentionally a dense audit tab, not a dashboard.
    fighter_rows = fighters.drop(columns=["Canonical"], errors="ignore").to_dict("records") if len(fighters) else []
    _table_sheet(workbook, fmt, "FIGHTER DATABASE", "UFC EDGE — FIGHTER DATABASE", f"{len(fighter_rows):,} fighter snapshots // {data_points:,} populated stat fields carried into the model", [
        ("Fighter", "Fighter", 26, "text"), ("UFC Fights", "UFC fights", 12, "int"),
        ("Wins", "Wins", 9, "int"), ("Losses", "Losses", 9, "int"), ("Draws", "Draws", 9, "int"),
        ("Opponent-adjusted Elo", "Elo", 19, "num"), ("Smoothed Win %", "Smoothed win %", 16, "pct"),
        ("Recent 5", "Recent 5", 12, "pct"), ("Adj Strike Diff / Min", "Adj strike diff/min", 19, "num"),
        ("Adj TD Diff / 15", "Adj TD diff/15", 17, "num"), ("Adj Control Min / 15", "Adj control min/15", 20, "num"),
        ("Reach", "Reach", 10, "num"), ("Age", "Age", 10, "num"), ("Last Fight", "Last fight", 14, "date"),
    ], fighter_rows)

    dictionary_rows = [
        {"field":"UFC fights", "unit":"count", "meaning":"Completed UFC bouts included before the prediction date", "model_use":"Experience screen"},
        {"field":"Elo", "unit":"rating", "meaning":"Opponent-adjusted sequential strength rating", "model_use":"Elo difference"},
        {"field":"Smoothed win %", "unit":"probability", "meaning":"UFC record shrunk toward 50% for small samples", "model_use":"Record difference"},
        {"field":"Recent 5", "unit":"probability", "meaning":"Win rate across the fighter's five most recent eligible UFC bouts", "model_use":"Recent-form difference"},
        {"field":"Adj strike diff/min", "unit":"strikes/min", "meaning":"Opponent-adjusted significant-strike margin per minute", "model_use":"Striking difference"},
        {"field":"Adj TD diff/15", "unit":"takedowns/15 min", "meaning":"Opponent-adjusted takedown margin per 15 minutes", "model_use":"Takedown difference"},
        {"field":"Adj control min/15", "unit":"minutes/15 min", "meaning":"Opponent-adjusted control-time margin per 15 minutes", "model_use":"Control difference"},
        {"field":"Reach", "unit":"inches", "meaning":"Listed reach", "model_use":"Reach difference"},
        {"field":"Age", "unit":"years", "meaning":"Age at the model snapshot", "model_use":"Age advantage"},
        {"field":"Market price", "unit":"probability", "meaning":"Executable Polymarket ask at the recorded entry", "model_use":"Net edge and settlement P&L"},
    ]
    _table_sheet(workbook, fmt, "DATA DICTIONARY", "UFC EDGE — DATA DICTIONARY", "Plain-English definitions for the raw model fields", [
        ("Field", "field", 25, "text"), ("Unit", "unit", 20, "text"),
        ("What It Means", "meaning", 58, "wrap"), ("How the Model Uses It", "model_use", 32, "wrap"),
    ], dictionary_rows)

    # Automated workbook checks.
    checks = workbook.add_worksheet("CHECKS")
    _write_title(checks, "UFC EDGE — MODEL CHECKS", "Automated integrity tests for this report", 5, fmt)
    checks.set_column("A:A", 34)
    checks.set_column("B:D", 18)
    checks.set_column("E:E", 13)
    checks.set_column("F:F", 48)
    for col, label in enumerate(["Check", "Observed", "Requirement", "Difference", "Status", "Meaning"]):
        checks.write(3, col, label, fmt["header"])
    check_rows = [
        ("Total card exposure", card_risk / bankroll if bankroll else np.nan, max_card_exposure, "pct", "Must not exceed configured cap"),
        ("Probability range", min([min(r["probability_a"], r["probability_b"]) for r in analyses], default=np.nan), 0, "pct", "All probabilities must be at least zero"),
        ("Probability sums", max([abs(r["probability_a"] + r["probability_b"] - 1) for r in analyses], default=np.nan), 0.000001, "pct", "A and B probabilities must sum to 100%"),
        ("Fights with live asks", sum(np.isfinite(safe_float(r.get("live_ask"))) for r in analyses), len(analyses), "int", "Shows market coverage for the selected card"),
        ("Holdout sample size", bundle["metrics"]["holdout_fights"], 500, "int", "Evaluation should use a material unseen sample"),
    ]
    for row, (label, observed, requirement, kind, meaning) in enumerate(check_rows, start=4):
        excel_row = row + 1
        checks.write(row, 0, label, fmt["label"])
        if np.isfinite(safe_float(observed)):
            checks.write_number(row, 1, observed, fmt[kind])
        else:
            checks.write_blank(row, 1, None, fmt[kind])
        checks.write_number(row, 2, requirement, fmt[kind])
        if label in {"Total card exposure", "Probability range", "Probability sums"}:
            difference = safe_float(observed) - requirement
            checks.write_formula(row, 3, f"=B{excel_row}-C{excel_row}", fmt["pct_formula"] if kind == "pct" else fmt["num"], difference)
            condition = "<=" if label != "Probability range" else ">="
            status = "PASS" if ((observed <= requirement) if condition == "<=" else (observed >= requirement)) else "REVIEW"
            checks.write_formula(row, 4, f'=IF(B{excel_row}{condition}C{excel_row},"PASS","REVIEW")', fmt["pass"] if status == "PASS" else fmt["review"], status)
        else:
            difference = safe_float(observed) - requirement
            checks.write_formula(row, 3, f"=B{excel_row}-C{excel_row}", fmt["num"], difference)
            status = "PASS" if observed >= requirement else "REVIEW"
            checks.write_formula(row, 4, f'=IF(B{excel_row}>=C{excel_row},"PASS","REVIEW")', fmt["pass"] if status == "PASS" else fmt["review"], status)
        checks.write(row, 5, meaning, fmt["wrap"])
        checks.set_row(row, 30)

    sources = workbook.add_worksheet("SOURCES")
    _write_title(sources, "UFC EDGE — DATA SOURCES", "Public source map and timing controls", 4, fmt)
    sources.set_column("A:A", 25)
    sources.set_column("B:B", 54)
    sources.set_column("C:E", 38)
    for col, label in enumerate(["Component", "URL", "Used For", "Timing", "Important Limitation"]):
        sources.write(3, col, label, fmt["header"])
    source_rows = [
        ("UFCStats", "http://ufcstats.com/statistics/events/completed?page=all", "Completed fight results and bout statistics", "Only information available before each predicted bout enters its feature row", "Unofficial public statistics site; page structure can change"),
        ("Polymarket Gamma", "https://gamma-api.polymarket.com/events", "Active UFC event and outcome discovery", "Refreshed when the app loads the card", "A listed market is not necessarily liquid"),
        ("Polymarket CLOB", "https://clob.polymarket.com/book", "Executable best bid and ask", "Refreshed every 30 seconds while the card is active", "Prices can move; size and slippage still matter"),
        ("Polymarket history", "https://clob.polymarket.com/prices-history", "Observed price path and convergence measurement", "Fetched for the selected outcome token", "Historical midpoint is not guaranteed execution"),
        ("Scientific Reports", "https://www.nature.com/articles/s41598-020-79408-6", "Reference on combat-sport outcome modeling", "Research context only", "The production model is separately trained and calibrated"),
    ]
    for row, values in enumerate(source_rows, start=4):
        for col, value in enumerate(values):
            if col == 1:
                sources.write_url(row, col, value, string=value)
            else:
                sources.write(row, col, value, fmt["wrap"] if col >= 2 else fmt["text"])
        sources.set_row(row, 52)

    # Put the decision sheet first without hiding the audit tabs.
    dashboard.activate()
    dashboard.select()
    workbook.close()
    output.seek(0)
    return output.getvalue()
