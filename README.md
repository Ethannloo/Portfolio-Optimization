# Portfolio Optimization with Markowitz Model and Random Forest

## Project Overview

#### This project is designed to marry the classical Markowitz mean–variance framework with a modern machine‑learning approach for forecasting asset returns. The pipeline begins by ingesting historical monthly price data (from a CSV file), computing log returns, and then applying a multi‑output Random Forest model to predict next‑month returns. These machine‑learning forecasts are fed into a convex optimization routine that solves for the long‑only tangency portfolio, yielding asset weights that target a unit of excess return over the risk‑free rate while minimizing variance. A visual comparison of the efficient frontier under both ML‑predicted and historical‑mean expected returns provides insight into the benefits of adaptive, data‑driven allocation.

## Machine Learning Model

#### At the heart of the forecasting step is ml_model.py, which defines a streamlined workflow for feature engineering, model training, evaluation, and persistence. Feature engineering constructs two families of predictors: lagged returns (the previous one through n months) and rolling statistics (mean, standard deviation, minimum, and maximum over configurable windows). These features capture temporal dependencies and recent volatility trends. A MultiOutputRegressor wrapping an XGBRFRegressor (XGBoost’s Random Forest) is trained on a temporally ordered train/test split—by default, reserving the most recent 20% of observations for out‑of‑sample evaluation. Mean Absolute Error (MAE) and R² scores for each asset, along with aggregate averages, are logged to help assess model quality.

#### Model artifacts are saved and loaded via joblib, with robust error handling to prevent pipeline interruptions if file permissions change. After initial evaluation, the script retrains on the full dataset to produce final forecasts. These next‑month monthly returns are then annualized (multiplied by the number of periods per year) before being passed to the optimization routine.

## Portfolio Optimization

#### The optimizer, implemented in optimization.py, takes the ML‑predicted expected returns vector  alongside the historical covariance matrix , annualized to match the return forecasts. It formulates a quadratic program with decision variable , minimizing the portfolio variance  subject to two key constraints: unit excess return  and nonnegativity  to enforce long‑only positions. CVXPY handles the problem definition and delegates to a high‑performance solver (such as OSQP or ECOS), returning raw weights that are then scaled to sum to unity, producing a fully invested portfolio.

#### Once solved, the script reports each ticker’s weight, the expected annual return, annualized volatility, and resulting Sharpe ratio. To provide a benchmark, it also computes and plots the efficient frontier under the historical mean‑return assumption, enabling a side‑by‑side visual assessment of ML‑driven versus traditional allocations.

## Minimal Setup and Usage

#### After cloning the repository and installing the required Python packages (pandas, numpy, scikit‑learn, xgboost, cvxpy, joblib, and matplotlib), the user first runs ml_model.py to generate and evaluate model forecasts. This can be configured via command‑line arguments for data path, feature parameters, and model hyperparameters. Once the ML model is in place, executing optimization.py will automatically invoke the predictor, solve the tangency portfolio, and display results in the console and via an efficient‑frontier plot.

#### Together, these components deliver a flexible, end‑to‑end workflow that leverages predictive analytics to inform quantitative portfolio construction.


