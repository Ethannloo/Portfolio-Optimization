"""
Train and predict next-period (daily) log returns with a multi-output
XGBRFRegressor.

Key defaults (tuned for daily data)
-----------------------------------
periods_per_year : 252          # trading days
lag_nums         : 5            # 5 daily lags
rolling_windows  : 5, 21, 63    # 1 w, 1 m, 3 m
"""

import argparse
import logging
import os
import joblib
from typing import List, Dict

import numpy as np
import pandas as pd
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def load_data(csv_file: str) -> pd.DataFrame:
    """Read a price CSV whose first column is the date index."""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Period-over-period (daily) continuously-compounded returns."""
    return np.log(prices / prices.shift(1)).dropna()


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def prepare_features(
    returns: pd.DataFrame,
    lag_nums: int,
    rolling_windows: List[int],
    window_label: str = "d"       # “d” = days, “m” = months, etc.
) -> pd.DataFrame:
    """
    Build a feature matrix consisting of:
    • lagged returns (t-1 … t-lag_nums)
    • rolling window mean / std / min / max statistics
    """
    lagged = [returns.shift(lag).add_prefix(f"lag{lag}_") for lag in range(1, lag_nums + 1)]

    rolling = []
    for window in rolling_windows:
        prefix = f"{window}{window_label}_"
        rolling.append(returns.rolling(window).mean().add_prefix(f"roll_mean_{prefix}"))
        rolling.append(returns.rolling(window).std().add_prefix(f"roll_std_{prefix}"))
        rolling.append(returns.rolling(window).min().add_prefix(f"roll_min_{prefix}"))
        rolling.append(returns.rolling(window).max().add_prefix(f"roll_max_{prefix}"))

    X = pd.concat(lagged + rolling, axis=1).dropna()
    return X


# ----------------------------------------------------------------------
# Model training / evaluation
# ----------------------------------------------------------------------
def train_evaluate_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    test_ratio: float,
    rf_params: Dict,
    model_path: str | None = None
) -> MultiOutputRegressor:
    """
    Split data chronologically into train/test, fit model, log metrics,
    and optionally persist the fitted estimator.
    """
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

    logging.info(f"Training on {len(X_train)} rows; testing on {len(X_test)} rows.")

    base = XGBRFRegressor(**rf_params)
    model = MultiOutputRegressor(base)
    model.fit(X_train.values, Y_train.values)

    # --- evaluation ---
    Y_pred = pd.DataFrame(model.predict(X_test.values),
                          index=Y_test.index, columns=Y_test.columns)

    mae = {c: mean_absolute_error(Y_test[c], Y_pred[c]) for c in Y.columns}
    r2  = {c: r2_score(Y_test[c], Y_pred[c]) for c in Y.columns}
    logging.info(
        "Avg MAE %.6f | Avg R² %.6f",
        np.mean(list(mae.values())), np.mean(list(r2.values()))
    )

    # --- optional persistence ---
    if model_path:
        try:
            joblib.dump(model, model_path)
            logging.info("Model saved ➜ %s", model_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not save model: %s", exc)

    return model


# ----------------------------------------------------------------------
# Main public function
# ----------------------------------------------------------------------
def predict_next_mu(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    lag_nums: int = 5,
    rolling_windows: List[int] = (5, 21, 63),
    test_ratio: float = 0.2,
    rf_params: Dict | None = None,
    model_path: str = "ml_model_daily.pkl"
) -> pd.Series:
    """
    Forecast the *next-period* (next day) log returns for every asset,
    then annualize by `periods_per_year`.

    Steps
    -----
    1. Build feature matrix X_t using info up to time *t*.
    2. Target Y_t is return_{t+1}.  (shift −1)
    3. Train/test split chronologically for evaluation.
    4. Retrain on full data.
    5. Predict on the most recent feature row.
    """
    if rf_params is None:
        rf_params = {"n_estimators": 100, "max_depth": 4, "random_state": 42}

    setup_logging()

    # ---------- features ----------
    X_all = prepare_features(returns, lag_nums, rolling_windows, window_label="d")

    # ---------- targets (next-period) ----------
    Y_all = returns.shift(-1).loc[X_all.index].dropna()
    X_all = X_all.loc[Y_all.index]        # align lengths exactly

    # ---------- fit / evaluate or load ----------
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logging.info("Loaded existing model ➜ %s", model_path)
        except Exception as exc:          # noqa: BLE001
            logging.warning("Load failed (%s); retraining.", exc)
            model = train_evaluate_model(X_all, Y_all, test_ratio, rf_params, model_path)
    else:
        model = train_evaluate_model(X_all, Y_all, test_ratio, rf_params, model_path)

    # ---------- retrain on full data ----------
    final_base   = XGBRFRegressor(**rf_params)
    final_model  = MultiOutputRegressor(final_base)
    final_model.fit(X_all.values, Y_all.values)

    # ---------- predict next period ----------
    latest_X = prepare_features(returns, lag_nums, rolling_windows, "d").tail(1).values
    next_r   = pd.Series(final_model.predict(latest_X)[0], index=Y_all.columns)

    return next_r * periods_per_year      # annualized μ


# ----------------------------------------------------------------------
# CLI runner (optional)
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBRF & forecast next-period returns (daily default)."
    )
    parser.add_argument("--data", default="data.csv",
                        help="CSV with price history (date index)")
    parser.add_argument("--lags", type=int, default=5,
                        help="Number of lagged return features")
    parser.add_argument("--roll", nargs="+", type=int, default=[5, 21, 63],
                        help="Rolling window sizes (periods)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Hold-out share for evaluation")
    parser.add_argument("--model_path", default="ml_model_daily.pkl",
                        help="File path to save/load model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--periods_per_year", type=int, default=252,
                        help="Periods per year for annualization")
    args = parser.parse_args()

    prices  = load_data(args.data)
    returns = compute_log_returns(prices)
    rf_par  = {"n_estimators": args.n_estimators,
               "max_depth":    args.max_depth,
               "random_state": args.random_state}

    mu = predict_next_mu(
        returns,
        periods_per_year=args.periods_per_year,
        lag_nums=args.lags,
        rolling_windows=args.roll,
        test_ratio=args.test_ratio,
        rf_params=rf_par,
        model_path=args.model_path
    )

    print("\nAnnualized expected returns (next period forecast):")
    print(mu.round(6))


if __name__ == "__main__":   # pragma: no cover
    main()
