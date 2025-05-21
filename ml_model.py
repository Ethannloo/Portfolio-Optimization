"""
Train and predict next-month log returns using a multi-output XGBRFRegressor.
"""
import argparse
import logging
import os
import pandas as pd
import numpy as np
from xgboost import XGBRFRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compounded log returns.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def prepare_features(
    returns: pd.DataFrame,
    lag_nums: int,
    rolling_windows: list[int]
) -> pd.DataFrame:
    """
    Generate lagged and rolling window features.
    """
    lagged = []
    for lag in range(1, lag_nums + 1):
        lagged.append(returns.shift(lag).add_prefix(f"lag{lag}_"))
    rolling = []
    for window in rolling_windows:
        rolling.append(returns.rolling(window=window).mean().add_prefix(f"roll_mean_{window}m_"))
        rolling.append(returns.rolling(window=window).std().add_prefix(f"roll_std_{window}m_"))
        rolling.append(returns.rolling(window=window).min().add_prefix(f"roll_min_{window}m_"))
        rolling.append(returns.rolling(window=window).max().add_prefix(f"roll_max_{window}m_"))
    X = pd.concat(lagged + rolling, axis=1).dropna()
    return X


def train_evaluate_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    test_ratio: float,
    rf_params: dict,
    model_path: str = None
) -> MultiOutputRegressor:
    """
    Split into train-test, fit model, evaluate performance, and persist if requested.
    """
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

    # Convert to numpy for XGBoost
    X_train_arr, X_test_arr = X_train.values, X_test.values
    Y_train_arr, Y_test_arr = Y_train.values, Y_test.values

    logging.info(f"Training on {len(X_train_arr)} samples, testing on {len(X_test_arr)} samples.")

    base_model = XGBRFRegressor(**rf_params)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_arr, Y_train_arr)

    # Predictions and evaluation
    Y_pred_arr = model.predict(X_test_arr)
    Y_pred = pd.DataFrame(Y_pred_arr, index=Y_test.index, columns=Y_test.columns)
    mae_scores = {col: mean_absolute_error(Y_test[col], Y_pred[col]) for col in Y.columns}
    r2_scores = {col: r2_score(Y_test[col], Y_pred[col]) for col in Y.columns}

    for col in Y.columns:
        logging.info(f"{col} - MAE: {mae_scores[col]:.6f}, R2: {r2_scores[col]:.6f}")
    logging.info(f"Average MAE: {np.mean(list(mae_scores.values())):.6f}, Average R2: {np.mean(list(r2_scores.values())):.6f}")

    # Persist model with error handling
    if model_path:
        try:
            joblib.dump(model, model_path)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            logging.warning(f"Could not save model to {model_path}: {e}")

    return model


def predict_next_mu(
    returns: pd.DataFrame,
    periods_per_year: int = 12,
    lag_nums: int = 3,
    rolling_windows: list[int] = [3, 6],
    test_ratio: float = 0.2,
    rf_params: dict = None,
    model_path: str = "ml_model.pkl"
) -> pd.Series:
    """
    Predict next-month returns (annualized) using a Random Forest model.

    - Evaluates on holdout
    - Retrains on full dataset for final prediction
    """
    if rf_params is None:
        rf_params = {"n_estimators": 100, "max_depth": 4, "random_state": 42}

    setup_logging()
    X = prepare_features(returns, lag_nums, rolling_windows)
    Y = returns.loc[X.index]

    # Load or train & evaluate
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logging.info(f"Loaded existing model from {model_path}")
        except Exception as e:
            logging.warning(f"Could not load model from {model_path}: {e}")
            model = train_evaluate_model(X, Y, test_ratio, rf_params, model_path)
    else:
        model = train_evaluate_model(X, Y, test_ratio, rf_params, model_path)

    # Retrain on full dataset
    X_arr, Y_arr = X.values, Y.values
    final_base = XGBRFRegressor(**rf_params)
    final_model = MultiOutputRegressor(final_base)
    final_model.fit(X_arr, Y_arr)

    # Predict next
    latest_X = X.tail(1).values
    pred = final_model.predict(latest_X)[0]
    mu_monthly = pd.Series(pred, index=Y.columns)
    return mu_monthly * periods_per_year


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Train RF and predict next-month returns.")
    parser.add_argument("--data", default="data.csv", help="CSV file with historical price data")
    parser.add_argument("--lags", type=int, default=3, help="Number of lagged return features")
    parser.add_argument("--roll", nargs='+', type=int, default=[3, 6], help="Rolling window sizes (months)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data to hold out for evaluation")
    parser.add_argument("--model_path", default="ml_model.pkl", help="File path to save/load the trained model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    prices = load_data(args.data)
    returns = compute_log_returns(prices)
    rf_params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "random_state": args.random_state}
    mu = predict_next_mu(
        returns,
        periods_per_year=12,
        lag_nums=args.lags,
        rolling_windows=args.roll,
        test_ratio=args.test_ratio,
        rf_params=rf_params,
        model_path=args.model_path
    )
    print("Predicted next-month expected returns (annualized):")
    print(mu)


if __name__ == "__main__":
    main()

