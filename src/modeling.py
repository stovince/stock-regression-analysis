import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def train_simple_lr(df, feature, target="LogReturn"):
    X = df[[feature]].values
    y = df[target].values
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "coef": model.coef_[0],
        "intercept": model.intercept_
    }

def train_multivariate(df, features, target="LogReturn"):
    X = df[features].values
    y = df[target].values
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Välj modell här, t.ex. Ridge eller RandomForest
    model = Ridge(alpha=1.0)
    # model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    return {
        "model": model,
        "scaler": scaler,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "coefs": model.coef_.tolist() if hasattr(model, "coef_") else None,
        "intercept": getattr(model, "intercept_", None)
    }

def main():
    parser = argparse.ArgumentParser("Träna regression")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", choices=["simple","multi"], default="simple")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", default="LogReturn")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Date"]).dropna()

    if args.mode == "simple":
        res = train_simple_lr(df, args.features[0], args.target)
    else:
        res = train_multivariate(df, args.features, args.target)

    print(f"R²: {res['r2']:.4f}, MSE: {res['mse']:.6f}")
    if args.mode == "simple":
        print(f"Coef: {res['coef']:.6f}, Intercept: {res['intercept']:.6f}")
    else:
        print(f"Coefs: {res['coefs']}, Intercept: {res['intercept']}")

if __name__ == "__main__":
    main()

