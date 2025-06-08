import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def train_simple_lr(df, feature, target="LogReturn"):
    "Tränar enkel linjär regression på en feature och returnerar model, metrics och (X_test, y_test, y_pred)."
    X = df[[feature]].values
    y = df[target].values
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "coef": model.coef_[0],
        "intercept": model.intercept_
    }
    # Returnera modell, metrics, och test‐tripel
    return model, metrics, (X_test, y_test, y_pred)

def train_multivariate(df, features, target="LogReturn"):
    "Tränar multivariat regression (Ridge som default) och returnerar model, metrics och (X_test, y_test, y_pred)."
    X = df[features].values
    y = df[target].values
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    # model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "coefs": model.coef_.tolist() if hasattr(model, "coef_") else None,
        "intercept": getattr(model, "intercept_", None)
    }
    return model, metrics, (X_test_s, y_test, y_pred)

def main():
    "CLI för att välja simple eller multi regression."
    parser = argparse.ArgumentParser("Träna regression")
    parser.add_argument("--input", required=True, help="CSV med feature‐engineerad data")
    parser.add_argument("--mode", choices=["simple","multi"], default="simple", help="simple eller multi")
    parser.add_argument("--features", nargs="+", required=True, help="Vilka kolumner att använda")
    parser.add_argument("--target", default="LogReturn", help="Target‐kolumn")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["Date"]).dropna()

    if args.mode == "simple":
        model, metrics, _ = train_simple_lr(df, args.features[0], args.target)
    else:
        model, metrics, _ = train_multivariate(df, args.features, args.target)

    print(f"R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.6f}")
    if args.mode == "simple":
        print(f"Coef: {metrics['coef']:.6f}, Intercept: {metrics['intercept']:.6f}")
    else:
        print(f"Coefs: {metrics['coefs']}, Intercept: {metrics['intercept']}")

if __name__ == "__main__":
    main()
