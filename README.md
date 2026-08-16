# UFC Edge Ledger

A clean Streamlit ledger for a calibrated UFC gradient-boosting model. It keeps one canonical record per event, separates open positions from settled history, and compares the model's probability with the executable Polymarket ask.

Active cards use Polymarket's public Gamma and CLOB APIs. The order book refreshes automatically every 30 seconds and can also be refreshed manually. Historical entry prices remain locked so later market moves never rewrite the original paper trade.

## Deploy on Streamlit Community Cloud

1. Put this folder in a private GitHub repository.
2. Create a Streamlit Community Cloud app.
3. Set the entrypoint to `app.py`.

The app can also run locally with `streamlit run app.py` or through the included Dockerfile.

## Model

- 200 shallow gradient-boosting trees
- Chronological training through 2022
- Probability calibration on 2023–2024
- Frozen 2025–2026 holdout: 835 fights, 63.6% accuracy, 0.223 Brier score
- Order-symmetric predictions
- Quarter-Kelly sizing capped at 2% of bankroll

The app uses public UFCStats, Polymarket, Kalshi and ESPN endpoints. If no executable current price is available, the decision remains `NO BET`.
