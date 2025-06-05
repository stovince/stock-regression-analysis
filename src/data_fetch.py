import yfinance as yf
import pandas as pd
import argparse
import os

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)
    # Byt namn på "Adj Close" till "Adj_Close" för enklare hantering
    df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Hämta historisk aktiedata med yfinance")
    parser.add_argument("--ticker", required=True, help="Aktiets ticker, t.ex. AAPL")
    parser.add_argument("--start", required=True, help="Startdatum, t.ex. 2020-01-01")
    parser.add_argument("--end", required=True, help="Slutdatum, t.ex. 2023-01-01")
    parser.add_argument(
        "--output",
        default="data/raw/{ticker}_{start}_{end}.csv",
        help="Sökväg där CSV-filen sparas (använd {ticker}, {start}, {end} i strängen)."
    )
    args = parser.parse_args()

    # Hämta DataFrame
    df = fetch_stock_data(args.ticker, args.start, args.end)

    # Skapa katalog om den inte finns
    output_path = args.output.format(ticker=args.ticker, start=args.start, end=args.end)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Spara till CSV
    df.to_csv(output_path, index=False)
    print(f"Hämtad data för {args.ticker} sparad som {output_path}")

if __name__ == "__main__":
    main()
