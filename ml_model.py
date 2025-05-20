import numpy as np
import pandas as pd
from xgboost import XGBRFRegressor  # Use XGBRFRegressor for random forests
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    CSV_FILE = 'data.csv'
    lag_nums = 3
    test_ratio = 0.2
    params = {
        'n_estimators': 100,  # Number of trees
        'max_depth': 4,       # Maximum depth of each tree
        'random_state': 42    # For reproducibility
    }  # Parameters for XGBRFRegressor

    # Read and sort data
    prices = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True).sort_index()
    returns = np.log(prices / prices.shift(1)).dropna()  # Calculate log returns

    # Calculate lagged log returns
    lagged_features = []
    for lag in range(1, lag_nums + 1):
        lag_returns = returns.shift(lag).add_prefix(f'lag{lag}_').dropna()
        lagged_features.append(lag_returns)

    # Calculate rolling statistics
    rolling_windows = [3, 6]
    rolling_features = []
    for window in rolling_windows:
        rolling_mean = returns.rolling(window=window).mean().add_prefix(f'roll_mean_{window}m_')
        rolling_features.append(rolling_mean)
        rolling_std = returns.rolling(window=window).std().add_prefix(f'roll_std_{window}m_')
        rolling_features.append(rolling_std)
        rolling_min = returns.rolling(window=window).min().add_prefix(f'roll_min_{window}m_')
        rolling_features.append(rolling_min)
        rolling_max = returns.rolling(window=window).max().add_prefix(f'roll_max_{window}m_')  # Fixed prefix
        rolling_features.append(rolling_max)

    # Concatenate lagged and rolling features
    all_features = lagged_features + rolling_features
    X = pd.concat(all_features, axis=1).dropna()
    Y = returns.loc[X.index]

    # Train-test split
    n_rows = X.shape[0]
    train_size = int(n_rows * (1 - test_ratio))  # Number of rows for training
    train_idx = X.index[:train_size]  # Indices for training (first 80%)
    test_idx = X.index[train_size:]   # Indices for testing (last 20%)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    Y_train = Y.loc[train_idx]
    Y_test = Y.loc[test_idx]

    # Verify shapes
    print(f"Total rows: {n_rows}")
    print(f"Training rows: {X_train.shape[0]}, Testing rows: {X_test.shape[0]}")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # Define and train model
    base_model = XGBRFRegressor(**params)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)

    # Make predictions for test set
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, index=Y_test.index, columns=Y_test.columns)  # Predicted mu for test set

    # Evaluate model
    mae_scores = {}
    r2_scores = {}
    for stock in Y_test.columns:
        mae = mean_absolute_error(Y_test[stock], Y_pred[stock])
        r2 = r2_score(Y_test[stock], Y_pred[stock])
        mae_scores[stock] = mae
        r2_scores[stock] = r2
        print(f"{stock} - MAE: {mae:.6f}, R²: {r2:.6f}")

    # Average metrics
    avg_mae = np.mean(list(mae_scores.values()))
    avg_r2 = np.mean(list(r2_scores.values()))
    print(f"\nAverage MAE: {avg_mae:.6f}")
    print(f"Average R²: {avg_r2:.6f}")

    # Display sample predicted mu for test set
    print("\nSample Predicted Mu (Test Set):")
    print(Y_pred.tail(5))

    # Save test set predictions
    Y_pred.to_csv('predicted_mu_test.csv')
    print("\nTest set predicted mu saved to 'predicted_mu_test.csv'")

    # Predict mu for the next month (beyond dataset)
    latest_features = X.tail(1)  # Most recent features
    next_mu = model.predict(latest_features)
    next_date = X.index[-1] + pd.offsets.MonthEnd(1)  # Next month's date
    next_mu_df = pd.DataFrame(next_mu, columns=Y.columns, index=[next_date])
    print("\nPredicted Mu for Next Month:")
    print(next_mu_df)

    # Save next month's prediction
    next_mu_df.to_csv('predicted_mu_next_month.csv')
    print("\nNext month's predicted mu saved to 'predicted_mu_next_month.csv'")

    # Feature importance (for first stock)
    feature_importance = model.estimators_[0].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (for first stock):")
    print(importance_df.head(10))

if __name__ == "__main__":
    main()
