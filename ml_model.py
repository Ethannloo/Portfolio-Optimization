import numpy as np
import pandas as pd
from xgboost import XGBRFRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    CSV_FILE = 'data.csv'
    lag_nums = 3
    test_ratio = .2
    params = {'n_estimators' : 100, 'max_depth' : 4, 'learning_rate' : .1,
               'objective' : 'reg:squarederror', 'random_state' : 42} # initialzing test parameters
    
    prices = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True).sort_index()
    returns  = np.log(prices / prices.shift(1)).dropna() # Calculating current log returns.
    
    lagged_features = []
    for lag in range(1, lag_nums + 1): # Calculating lagged log returns
        lag_returns = returns.shift(lag).add_prefix(f'lag{lag}').dropna()
        lagged_features.append(lag_returns)

    rolling_windows = [3, 6]
    rolling_features = []
    for window in rolling_windows:
        rolling_mean = returns.rolling(window=window).mean().add_prefix(f'roll_mean_{window}')
        rolling_features.append(rolling_mean) 

        rolling_std = returns.rolling(window=window).std().add_prefix(f'roll_std_{window}')
        rolling_features.append(rolling_std)

        rolling_min = returns.rolling(window=window).min().add_prefix(f'roll_min_{window}')
        rolling_features.append(rolling_min)

        rolling_max = returns.rolling(window=window).max().add_prefix(f'roll_min_{window}')
        rolling_features.append(rolling_max)
    
    #concatenate lagged and rollings features
    all_features = lagged_features + rolling_features
    X = pd.concat(all_features, axis=1).dropna()
    Y = returns.loc[X.index]

    # train test split
    n_rows = X.shape[0]
    train_size = int(n_rows * (1 - test_ratio)) # number of rows for training 
    train_idx = X.index[:train_size] # Indices for training (first 80%)
    test_idx = X.index[train_size:] # Indices for testing (last 20%)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    Y_train = Y.loc[train_idx]
    Y_test = Y.loc[test_idx]

    # defining and training model
    base_model = XGBRFRegressor(**params)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)

    # make predictions
    Y_pred = model.predict(X_test) 
    Y_pred = pd.DataFrame(Y_pred, index= Y_test.index, columns=Y_test.columns)

    mae_scores = {}
    r2_scores = {}
    for stock in Y_test.columns:
        mae = mean_absolute_error(Y_test[stock], Y_pred[stock])
        r2 = r2_score(Y_test[stock], Y_pred[stock])
        mae_scores[stock] = mae
        r2_scores[stock] = r2
        print(f"{stock} - MAE: {mae: .6f}, R^2: {r2: .6f}")

    avg_mae = np.mean(list(mae_scores.values()))
    avg_r2 = np.mean(list(r2_scores.values()))
    print(f"Average MAE: {avg_mae: .6f}")
    print(f"Average R^2: {avg_r2: .6f}")

    feature_importance = model.estimators_[0].feature_importances_  # First stock's model
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (for first stock):")
    print(importance_df.head(10))  # Top 10 features



if __name__ == "__main__":
    main()

