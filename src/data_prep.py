import pandas as pd
import numpy as np
import argparse
import os

def add_returns(df: pd.DataFrame, price_col: str = "Adj_Close") -> pd.DataFrame:
    "Lägger till daglig logaritmisk avkastning (LogReturn)."
    df = df.copy()
    # Skapa LogReturn; innan det, säkerställ att pris-kolumnen är numerisk
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))
    return df

def add_moving_averages(df: pd.DataFrame, price_col: str = "Adj_Close", windows: list = [5, 20, 50]) -> pd.DataFrame:
    "Lägger till enkla glidande medelvärden (SMA) för givna fönster (i dagar). Exempel: windows=[5, 20, 50] ger kolumnerna \"MA_5\", \"MA_20\", \"MA_50\"."
    df = df.copy()
    # Se till att pris-kolumnen är numerisk
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    for w in windows:
        df[f"MA_{w}"] = df[price_col].rolling(window=w).mean()
    return df

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    "Lägger till 20-dagars rullande volatilitet (std av LogReturn). Omvandlar till årlig volatilitet genom att multiplicera med sqrt(252)."
    df = df.copy()
    # Volatilitet baserad på redan beräknade LogReturn
    df["Volatility"] = df["LogReturn"].rolling(window=window).std() * np.sqrt(252)
    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    "Kör alla feature-engineering-steg (returns, MA, volatility) och droppar NaN."
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    # Ta bort rader som innehåller NaN (främst p.g.a. rolling() och eventuella invalid parsing)
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Förbereda data: log-returns, MA, volatility")
    parser.add_argument("--input", required=True, help="Sökväg till rå CSV (från data_fetch.py)")
    parser.add_argument("--output", required=True, help="Sökväg där den bearbetade CSV-filen ska sparas")
    args = parser.parse_args()

    # Läs in data
    df = pd.read_csv(args.input, parse_dates=["Date"])

    # Om kolumnen heter "Adj Close", byt namn till "Adj_Close" för att matcha price_col
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    # Om det istället bara finns kolumnen "Close", döp om den till "Adj_Close"
    elif "Close" in df.columns:
        df.rename(columns={"Close": "Adj_Close"}, inplace=True)
    else:
        raise ValueError(
            "Inget giltigt prisfält hittades. Förväntade 'Adj Close' eller 'Close'."
        )

    # Efter omdöpning: se till att pris-kolumnen finns och är numerisk
    df["Adj_Close"] = pd.to_numeric(df["Adj_Close"], errors="coerce")

    # Kör feature engineering
    df_prepared = prepare_data(df)

    # Skapa output-katalog om den inte finns
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_prepared.to_csv(args.output, index=False)
    print(f"Bearbetad data sparad som {args.output}")

if __name__ == "__main__":
    main()
