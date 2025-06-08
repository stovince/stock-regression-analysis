import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

def train_and_tune_rf(X, y, cv_splits=3):
    "RandomizedSearchCV med Random Forest och tidsserie-CV."
    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(2, 10),
        "min_samples_leaf": randint(1, 10)
    }
    base_rf = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    rnd_search = RandomizedSearchCV(
        base_rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )
    rnd_search.fit(X, y)
    return rnd_search.best_estimator_, rnd_search.cv_results_

def train_multivariate(df, features, target="LogReturn", model_type="rf"):
    "Träning av Ridge eller optimerad Random Forest med tidsserie-CV."
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    split = int(len(X_s) * 0.8)
    X_train, X_test = X_s[:split], X_s[split:]
    y_train, y_test = y[:split], y[split:]
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
    else:
        model, _ = train_and_tune_rf(X_train, y_train, cv_splits=5)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
    return model, metrics, (X_test, y_test, y_pred)

def main():
    "Kommandoverktyg för att träna Ridge eller Random Forest."
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV-fil med features")
    parser.add_argument("--features", nargs="+", required=True, help="Lista över feature-kolumner")
    parser.add_argument("--target", default="LogReturn", help="Målkollumn")
    parser.add_argument("--model", choices=["ridge","rf"], default="rf", help="Modelltyp: ridge eller rf")
    args = parser.parse_args()
    df = pd.read_csv(args.input, parse_dates=["Date"]).dropna()
    model, metrics, _ = train_multivariate(df, args.features, args.target, args.model)
    print(f"R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.6f}")

if __name__ == "__main__":
    main()
