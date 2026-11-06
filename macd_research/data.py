# Imports Libraries

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from binance.spot import Spot as Client

# Directory Variables

ROOT_DIR    = Path(__file__).resolve().parent.parent
OHLC_DIR    = ROOT_DIR / "data" / "01_OHLC_data"
BACKTEST_DIR = ROOT_DIR / "data" / "02_Backtest_data"

# API Variables

API_URL     = 'https://api1.binance.com'
api         = Client(base_url = API_URL)

# Backtest Settings

cash        = 1_000_000
commission  = 0.001

# Sample Size Variables

data_start = datetime(2018, 1, 1, tzinfo=timezone.utc)    # first full year of BTCUSDT on Binance
end_date   = datetime(2026, 7, 1, tzinfo=timezone.utc)    # Last candle is 2026-06-30
warmup     = timedelta(weeks=52)                          # EMA burn-in, no signals or trades in period
eval_start = data_start + warmup
total_candles = (end_date - data_start).days    # 3103 daily candles


def data_retrieval(api, end_date, ticker, total_candles):
    
    ''' Retrieves and compiles data in batches of 1000 candles (API limit) '''

    print('\n Update: Downloading data from Binance API.')

    ohlc_data = []
    retrieved_data = []

    # Initiates a while loop to extract the amount of candles in total_candles.

    while total_candles > 0:
        if total_candles >= 1000:
            candles = 1000
        else:
            candles = total_candles

    # API requires UNIX data in milliseconds. Since .timestamp() returns seconds, we apply * 1000.

        from_date = int((end_date - timedelta(days=candles)).timestamp() * 1000)
        to_date = int(end_date.timestamp() * 1000) - 1

        api_output = api.klines(ticker,
                            '1d',
                            startTime  =  from_date,
                            endTime    =  to_date,
                            limit      =  candles)

        if not api_output:
            raise RuntimeError(f"No data returned before {end_date}. Check ticker or history length.")

        retrieved_data += api_output

    #After each batch is retrieved, parameters are updated to access the following 1000 candle batch.

        end_date -= timedelta(days=candles)
        total_candles -= len(api_output)

    # Compiling and reformatting our data to provide an OHLC dataframe.

    for candle in retrieved_data:
        ohlc_data.append(candle[:5])

    columns=['Timestamp', 'Open', 'High', 'Low', 'Close']
    ohlc_data = pd.DataFrame(ohlc_data, columns=columns)

    ohlc_data['Timestamp'] = pd.to_datetime(ohlc_data['Timestamp'], unit='ms', utc=True)
    ohlc_data = ohlc_data.set_index('Timestamp')

    ohlc_data = ohlc_data.sort_index()
    for i in ['Open', 'High', 'Low', 'Close']:
        ohlc_data[i] = ohlc_data[i].astype('float')

    # Saving our data

    OHLC_DIR.mkdir(parents=True, exist_ok=True)

    first_date = ohlc_data.index[0].strftime("%Y%m%d")
    last_date = ohlc_data.index[-1].strftime("%Y%m%d")
    filepath = OHLC_DIR / f"{ticker}_1d_{first_date}_{last_date}.csv"

    ohlc_data.to_csv(filepath)

    print(f"\n Update: Saved {ticker} OHLC data to {filepath}")

def find_ohlc(ticker):

    ''' Returns the saved CSV that ends on the configuration's last day and has exactly total_candles bars '''

    end = (end_date - timedelta(days=1)).strftime("%Y%m%d")

    for filepath in sorted(OHLC_DIR.glob(f"{ticker}_1d_*_{end}.csv")):
        with open(filepath) as file:
            bars = sum(1 for line in file) - 1
        if bars == total_candles:
            return filepath

    return None

def load_ohlc(ticker):

    ''' Loads the CSV that matches the current configuration, refusing anything else '''

    filepath = find_ohlc(ticker)
    assert filepath is not None, f"{ticker}: no saved CSV with {total_candles} bars. Run the download cell."

    data = pd.read_csv(filepath, index_col="Timestamp", parse_dates=True).sort_index()

    assert len(data) == total_candles, f"{ticker}: {len(data)} bars, configuration requires {total_candles}"
    assert data.index.is_unique, f"{ticker}: duplicate timestamps"
    assert data.index[0] == data_start, f"{ticker}: data starts {data.index[0]}, expected {data_start}"

    gaps = int((data.index.to_series().diff().dropna() != pd.Timedelta(days=1)).sum())

    print(f"\n Update: Loaded {filepath.name} ({len(data)} bars, {gaps} gaps)")
    print(f" Data starts          : {data.index[0]}")
    print(f" Evaluation from      : {eval_start}  to  {data.index[-1]}")

    return data
