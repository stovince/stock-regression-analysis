import pandas as pd
import numpy as np
import ta
import argparse
import os

def add_returns(df: pd.DataFrame, price_col: str = "Adj_Close") -> pd.DataFrame:
    "Lägger till daglig logaritmisk avkastning (LogReturn)."
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))
    return df

def add_moving_averages(df: pd.DataFrame, price_col: str = "Adj_Close",
                        windows: list = [5, 20, 50]) -> pd.DataFrame:
    "Lägger till enkla glidande medelvärden (MA) för givna fönster."
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    for w in windows:
        df[f"MA_{w}"] = df[price_col].rolling(window=w).mean()
    return df

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    "Lägger till 20-dagars rullande volatilitet (årlig std av LogReturn)."
    df = df.copy()
    df["Volatility"] = df["LogReturn"].rolling(window=window).std() * np.sqrt(252)
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    "Lägger till RSI-indikator med givet fönster."
    df = df.copy()
    df[f"RSI_{window}"] = ta.momentum.rsi(df["Adj_Close"], window=window)
    return df

def add_ema(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    "Lägger till EMA-indikator med givet fönster."
    df = df.copy()
    df[f"EMA_{window}"] = ta.trend.ema_indicator(df["Adj_Close"], window=window)
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    "Lägger till MACD, signal- och diff-linjer."
    df = df.copy()
    macd = ta.trend.MACD(df["Adj_Close"], window_fast=fast, window_slow=slow, window_sign=signal)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    "Lägger till Bollinger Bands (mavg, övre och undre band)."
    df = df.copy()
    bb = ta.volatility.BollingerBands(df["Adj_Close"], window=window, window_dev=std)
    df["BB_mavg"] = bb.bollinger_mavg()
    df["BB_hband"] = bb.bollinger_hband()
    df["BB_lband"] = bb.bollinger_lband()
    return df

def add_lagged_returns(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    "Lägger till laggade LogReturn-värden för givna lags."
    df = df.copy()
    for lag in lags:
        df[f"Lag{lag}_LogReturn"] = df["LogReturn"].shift(lag)
    return df

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    "Lägger till interaktionsterm mellan MA_20 och Volatility."
    df = df.copy()
    df["MA20_x_Volatility"] = df["MA_20"] * df["Volatility"]
    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    "Kör alla feature-engineering-steg och tar bort NaN."
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_lagged_returns(df)
    df = add_interactions(df)
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    "Läser in rådata, förbereder features och sparar till CSV."
    parser = argparse.ArgumentParser(description="Förbereda data med utökad feature-engineering")
    parser.add_argument("--input", required=True, help="Sökväg till rå CSV")
    parser.add_argument("--output", required=True, help="Sökväg för sparad CSV")
    args = parser.parse_args()
    df = pd.read_csv(args.input, parse_dates=["Date"])
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    elif "Close" in df.columns:
        df.rename(columns={"Close": "Adj_Close"}, inplace=True)
    else:
        raise ValueError("Förväntar kolumn 'Adj Close' eller 'Close'.")
    df["Adj_Close"] = pd.to_numeric(df["Adj_Close"], errors="coerce")
    df_prepared = prepare_data(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_prepared.to_csv(args.output, index=False)
    print(f"Bearbetad data sparad som {args.output}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
