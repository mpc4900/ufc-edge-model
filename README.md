# UFC Edge Model

A fast Streamlit version of the calibrated UFC gradient-boosting model. The model is pre-trained and cached, so each refresh only retrieves the fight card and current public prices.

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
