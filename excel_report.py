from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xlsxwriter

from model_engine import realized_metrics, safe_float


def formats(workbook):
    navy, blue, green, red, border = "#08192E", "#173B5E", "#0B6B4F", "#B42318", "#D5DCE5"
    return {
        "title": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "font_size": 17, "align": "left", "valign": "vcenter"}),
        "subtitle": workbook.add_format({"font_color": "#DCE6F2", "bg_color": navy, "font_size": 10, "align": "left", "valign": "vcenter"}),
        "header": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": blue, "bottom": 1, "bottom_color": border, "align": "center", "valign": "vcenter", "text_wrap": True}),
        "text": workbook.add_format({"bottom": 1, "bottom_color": border, "valign": "vcenter"}),
        "wrap": workbook.add_format({"bottom": 1, "bottom_color": border, "valign": "vcenter", "text_wrap": True}),
        "pct": workbook.add_format({"bottom": 1, "bottom_color": border, "num_format": "0.0%;[Red](0.0%);-", "align": "right", "valign": "vcenter"}),
        "money": workbook.add_format({"bottom": 1, "bottom_color": border, "num_format": "$#,##0;[Red]($#,##0);-", "align": "right", "valign": "vcenter"}),
        "num": workbook.add_format({"bottom": 1, "bottom_color": border, "num_format": "0.000", "align": "right", "valign": "vcenter"}),
        "int": workbook.add_format({"bottom": 1, "bottom_color": border, "num_format": "#,##0", "align": "right", "valign": "vcenter"}),
        "date": workbook.add_format({"bottom": 1, "bottom_color": border, "num_format": "mmm d, yyyy", "align": "center", "valign": "vcenter"}),
        "bet": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": green, "align": "center", "valign": "vcenter"}),
        "no_bet": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": red, "align": "center", "valign": "vcenter"}),
        "card_label": workbook.add_format({"bold": True, "font_color": "#5B6573", "bg_color": "#E9EFF6", "top": 1, "bottom": 1, "top_color": border, "bottom_color": border, "align": "center", "valign": "vcenter"}),
        "card_value": workbook.add_format({"bold": True, "font_size": 15, "font_color": navy, "bg_color": "#FFFFFF", "bottom": 1, "bottom_color": border, "align": "center", "valign": "vcenter"}),
        "card_pct": workbook.add_format({"bold": True, "font_size": 15, "font_color": navy, "bg_color": "#FFFFFF", "bottom": 1, "bottom_color": border, "align": "center", "valign": "vcenter", "num_format": "0.0%"}),
        "card_money": workbook.add_format({"bold": True, "font_size": 15, "font_color": navy, "bg_color": "#FFFFFF", "bottom": 1, "bottom_color": border, "align": "center", "valign": "vcenter", "num_format": "$#,##0"}),
        "positive": workbook.add_format({"font_color": green, "bottom": 1, "bottom_color": border, "num_format": "+0.0%;[Red]-0.0%;-", "align": "right"}),
        "negative": workbook.add_format({"font_color": red, "bottom": 1, "bottom_color": border, "num_format": "+0.0%;[Red]-0.0%;-", "align": "right"}),
    }


def write_value(worksheet, row, column, value, kind, fmt):
    if kind == "date":
        date = pd.to_datetime(value, errors="coerce")
        if pd.notna(date):
            worksheet.write_datetime(row, column, date.to_pydatetime(), fmt["date"])
        else:
            worksheet.write_blank(row, column, None, fmt["date"])
    elif kind in {"pct", "money", "num", "int"}:
        number = safe_float(value)
        if np.isfinite(number):
            worksheet.write_number(row, column, number, fmt[kind])
        else:
            worksheet.write_blank(row, column, None, fmt[kind])
    else:
        worksheet.write(row, column, "" if pd.isna(value) else str(value), fmt.get(kind, fmt["text"]))


def table_sheet(workbook, fmt, name, title, columns, rows):
    worksheet = workbook.add_worksheet(name)
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(4, 0)
    worksheet.set_row(0, 30)
    worksheet.merge_range(0, 0, 0, len(columns) - 1, title, fmt["title"])
    worksheet.merge_range(1, 0, 1, len(columns) - 1, f"As of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", fmt["subtitle"])
    for column, (label, _, width, _) in enumerate(columns):
        worksheet.write(3, column, label, fmt["header"])
        worksheet.set_column(column, column, width)
    for row_index, record in enumerate(rows, start=4):
        worksheet.set_row(row_index, 24)
        for column, (_, key, _, kind) in enumerate(columns):
            write_value(worksheet, row_index, column, record.get(key, ""), kind, fmt)
    if rows:
        worksheet.autofilter(3, 0, 3 + len(rows), len(columns) - 1)


def build_excel(analyses, bundle, prediction_log, bankroll, event_search, cost_buffer=0.02, min_edge=0.03, min_prior_fights=3):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    workbook.set_properties({"title": "UFC Edge Model Report", "company": "UFC Edge"})
    fmt = formats(workbook)

    worksheet = workbook.add_worksheet("DASHBOARD")
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(8, 0)
    worksheet.set_row(0, 32)
    worksheet.merge_range("A1:H1", "UFC EDGE — DECISION DASHBOARD", fmt["title"])
    worksheet.merge_range("A2:H2", f"{event_search}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", fmt["subtitle"])
    bet_count = sum(row["action"] == "BET" for row in analyses)
    edges = [safe_float(row["net_edge"]) for row in analyses if np.isfinite(safe_float(row["net_edge"]))]
    top_edge = max(edges) if edges else np.nan
    cards = [
        ("A4:B4", "A5:B6", "BET SIGNALS", bet_count, "card_value"),
        ("C4:D4", "C5:D6", "TOP NET EDGE", top_edge, "card_pct"),
        ("E4:F4", "E5:F6", "BANKROLL", bankroll, "card_money"),
        ("G4:H4", "G5:H6", "HOLDOUT ACCURACY", bundle["metrics"]["accuracy"], "card_pct"),
    ]
    for label_range, value_range, label, value, kind in cards:
        worksheet.merge_range(label_range, label, fmt["card_label"])
        worksheet.merge_range(value_range, value if np.isfinite(safe_float(value)) else "—", fmt[kind] if np.isfinite(safe_float(value)) else fmt["card_value"])
    headers = ["Fight", "Pick", "Model P", "Market P", "Net Edge", "Decision", "Position $", "Why"]
    widths = [38, 24, 12, 12, 12, 12, 14, 40]
    for column, header in enumerate(headers):
        worksheet.write(7, column, header, fmt["header"])
        worksheet.set_column(column, column, widths[column])
    for row_index, record in enumerate(analyses, start=8):
        worksheet.set_row(row_index, 30)
        worksheet.write(row_index, 0, f"{record['fighter_a']} vs {record['fighter_b']}", fmt["text"])
        worksheet.write(row_index, 1, record["pick"], fmt["text"])
        worksheet.write_number(row_index, 2, record["model_probability"], fmt["pct"])
        if np.isfinite(safe_float(record["market_probability"])):
            worksheet.write_number(row_index, 3, record["market_probability"], fmt["pct"])
            excel_row = row_index + 1
            worksheet.write_formula(row_index, 4, f"=C{excel_row}-D{excel_row}-'MODEL'!$B$15", fmt["pct"], record["net_edge"])
        else:
            worksheet.write_blank(row_index, 3, None, fmt["pct"])
            worksheet.write_blank(row_index, 4, None, fmt["pct"])
        worksheet.write(row_index, 5, record["action"], fmt["bet"] if record["action"] == "BET" else fmt["no_bet"])
        worksheet.write_number(row_index, 6, record["position_dollars"], fmt["money"])
        worksheet.write(row_index, 7, record["why"], fmt["wrap"])
    if analyses:
        worksheet.autofilter(7, 0, 7 + len(analyses), 7)
        worksheet.conditional_format(8, 4, 7 + len(analyses), 4, {
            "type": "3_color_scale", "min_color": "#F4CCCC", "mid_color": "#FFF2CC", "max_color": "#D9EAD3",
        })

    why_rows = []
    for record in analyses:
        for rank, driver in enumerate(record["drivers"], start=1):
            why_rows.append({
                "fight": f"{record['fighter_a']} vs {record['fighter_b']}", "pick": record["pick"],
                "rank": rank, "factor": driver["factor"], "value": driver["value"],
                "impact": driver["impact_pick"], "experience": record["experience"],
            })
    table_sheet(workbook, fmt, "WHY", "WHY THE MODEL LEANS THIS WAY", [
        ("Fight", "fight", 38, "text"), ("Pick", "pick", 24, "text"), ("Rank", "rank", 9, "int"),
        ("Driver", "factor", 28, "text"), ("Feature Difference", "value", 18, "num"),
        ("Probability Sensitivity", "impact", 20, "pct"), ("Min. Prior UFC Fights", "experience", 20, "int"),
    ], why_rows)

    worksheet = workbook.add_worksheet("MODEL")
    worksheet.hide_gridlines(2)
    worksheet.set_column("A:A", 31)
    worksheet.set_column("B:B", 29)
    worksheet.set_column("D:E", 25)
    worksheet.merge_range("A1:E1", "MODEL MATH", fmt["title"])
    worksheet.merge_range("A2:E2", "Calibrated, order-symmetric gradient boosting trained on completed UFC fights.", fmt["subtitle"])
    worksheet.write("A4", "Step", fmt["header"])
    worksheet.write("B4", "Math", fmt["header"])
    rows = [
        ("Inputs", "8 pre-fight A-minus-B feature differences"),
        ("Gradient boosting", "score(x) = base + 0.04 × Σ 200 shallow-tree outputs"),
        ("Order symmetry", "raw P(A) = [GB(x) + 1 − GB(−x)] ÷ 2"),
        ("Calibration", f"P(A) = logistic({bundle['calibration_slope']:.3f} × logit(raw P(A)))"),
        ("Net edge", "Model P − market P − cost buffer"),
        ("Decision", "BET when net edge clears threshold and experience minimum"),
        ("Full Kelly", "Net edge ÷ (1 − market P − cost buffer)"),
        ("Position", "Quarter Kelly, capped at 2.0% of bankroll"),
    ]
    for row_index, (step, formula) in enumerate(rows, start=4):
        worksheet.write(row_index, 0, step, fmt["text"])
        worksheet.write(row_index, 1, formula, fmt["wrap"])
        worksheet.set_row(row_index, 31)
    worksheet.write("A14", "Assumption", fmt["header"])
    worksheet.write("B14", "Value", fmt["header"])
    assumptions = [("Cost buffer", cost_buffer, "pct"), ("Minimum net edge", min_edge, "pct"), ("Minimum prior fights", min_prior_fights, "int")]
    for row_index, (label, value, kind) in enumerate(assumptions, start=14):
        worksheet.write(row_index, 0, label, fmt["text"])
        worksheet.write_number(row_index, 1, value, fmt[kind])
    worksheet.write("D4", "Global Factor", fmt["header"])
    worksheet.write("E4", "Importance", fmt["header"])
    importance = sorted(bundle["importance"].items(), key=lambda item: item[1], reverse=True)
    for row_index, (factor, value) in enumerate(importance, start=4):
        worksheet.write(row_index, 3, factor.replace(" diff", ""), fmt["text"])
        worksheet.write_number(row_index, 4, value, fmt["pct"])
    chart = workbook.add_chart({"type": "bar"})
    chart.add_series({"name": "Feature importance", "categories": ["MODEL", 4, 3, 3 + len(importance), 3], "values": ["MODEL", 4, 4, 3 + len(importance), 4], "fill": {"color": "#173B5E"}})
    chart.set_title({"name": "Global feature importance"})
    chart.set_x_axis({"num_format": "0%"})
    chart.set_legend({"none": True})
    chart.set_style(10)
    worksheet.insert_chart("D14", chart, {"x_scale": 1.15, "y_scale": 1.10})

    realized = realized_metrics(prediction_log)
    worksheet = workbook.add_worksheet("PERFORMANCE")
    worksheet.hide_gridlines(2)
    worksheet.set_column("A:A", 28)
    worksheet.set_column("B:B", 16)
    worksheet.merge_range("A1:B1", "PERFORMANCE", fmt["title"])
    worksheet.merge_range("A2:B2", "Frozen holdout plus recorded live decisions.", fmt["subtitle"])
    worksheet.write("A4", "Metric", fmt["header"])
    worksheet.write("B4", "Value", fmt["header"])
    performance = [
        ("Historical training fights", bundle["metrics"]["training_fights"], "int"),
        ("Unseen holdout fights", bundle["metrics"]["holdout_fights"], "int"),
        ("Holdout accuracy", bundle["metrics"]["accuracy"], "pct"),
        ("Holdout Brier score", bundle["metrics"]["brier"], "num"),
        ("Holdout ROC AUC", bundle["metrics"]["auc"], "num"),
        ("Recorded bets", realized["graded"], "int"), ("Recorded wins", realized["wins"], "int"),
        ("Recorded losses", realized["losses"], "int"), ("Recorded ROI", realized["roi"], "pct"),
        ("Recorded P&L", realized["pnl"], "money"),
    ]
    for row_index, (label, value, kind) in enumerate(performance, start=4):
        worksheet.write(row_index, 0, label, fmt["text"])
        if np.isfinite(safe_float(value)):
            worksheet.write_number(row_index, 1, value, fmt[kind])
        else:
            worksheet.write_blank(row_index, 1, None, fmt[kind])

    open_bets = prediction_log[(prediction_log.get("status", "") == "OPEN") & (prediction_log.get("action", "") == "BET")].to_dict("records") if len(prediction_log) else []
    results = prediction_log[prediction_log.get("status", "") == "COMPLETED"].sort_values("event_date", ascending=False).to_dict("records") if len(prediction_log) else []
    for record in open_bets + results:
        record["fight"] = f"{record.get('fighter_a', '')} vs {record.get('fighter_b', '')}"
    columns = [
        ("Event Date", "event_date", 15, "date"), ("Fight", "fight", 38, "text"),
        ("Pick", "pick", 24, "text"), ("Model P", "model_probability", 12, "pct"),
        ("Market P", "market_probability", 12, "pct"), ("Net Edge", "net_edge", 12, "pct"),
        ("Position $", "position_dollars", 14, "money"), ("Outcome", "outcome", 12, "text"),
        ("P&L", "pnl", 14, "money"),
    ]
    table_sheet(workbook, fmt, "OPEN BETS", "OPEN BETS", columns[:7], open_bets)
    table_sheet(workbook, fmt, "RESULTS", "RECORDED RESULTS", columns, results)
    workbook.close()
    output.seek(0)
    return output.getvalue()
