import numpy as np
import pandas as pd
from xgboost import XGBRFRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    CSV_FILE = 'data.csv'
    lag_nums = 3
    test_ratio = .2
    params = {'est_num' : 100, 'max_depth' : 4, 'learning_rate' : .1,
               'objective' : 'reg:squarederror', 'random_state' : 42}
    
    prices = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True).sort_index()
    returns  = np.log(prices / prices.shift(1)).dropna()

    feature_frames = []
    for lag in range(1, lag_nums + 1):
        lag_df = returns.shift(lag).add_prefix(f'lag{lag}')
        feature_frames.append(lag_df)
    X = pd.concat(feature_frames, axis=1).dropna()
    Y = returns.loc[X.index]

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

    base_model = XGBRFRegressor(**params)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse_vals = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
    r2_vals = r2_score(Y_test, Y_pred, multioutput='raw_values')

    last_feat = pd.concat([returns.shift(lag).iloc[-1].add_prefix(f'lag{lag}_')
                           for lag in range(1, lag_nums + 1)], axis=0)
    X_next = last_feat.values.reshape(1, -1)
    mu_next = model.predict(X_next)[0]
if __name__ == "__main__":
    main()
