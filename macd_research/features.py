import numpy as np
import pandas as pd

from macd_research.data import eval_start

def exponential_moving_average(data, span):
    ''' Returns an exponential moving average '''
    return pd.Series(data).ewm(span = span, adjust = False).mean()

def macd(close, fast, slow, signal_length):
    ''' Returns our MACD line and Signal line '''
    ema_fast = exponential_moving_average(close, fast)
    ema_slow = exponential_moving_average(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal_length)
    return macd_line, signal_line

def weekly_macd_features(close_daily, fast = 12, slow = 26, signal = 9):

    ''' Returns Weekly MACD state (+1/-1) and price-scaled histogram, shifted so each Monday holds the last completed week '''
  
    weekly = close_daily.resample('7D').last()
    macd_line, signal_line = macd(weekly, fast, slow, signal)
    hist = (macd_line - signal_line) / weekly                      # histogram scaled by price: comparable across different prices in 2018 and 2026

    return pd.DataFrame({'state': np.sign(macd_line - signal_line), 'hist': hist}).shift(1)

def daily_features(close_daily):

    ''' Returns a Pandas Dataframe where each past week's value is matched against the following week '''

    feats = weekly_macd_features(close_daily).reindex(close_daily.index, method = 'ffill')
    return pd.concat([close_daily.rename('close'), feats], axis=1).loc[eval_start:]

def run_lookahead_test(close, T = '2022-06-30', shock = 0.05, seed = 0): # To Read

    ''' Multiply every daily close AFTER day T by random noise; every feature labelled <= T must be
    unchanged, and something after T must change. '''

    T = pd.Timestamp(T, tz = 'UTC')
    rng = np.random.default_rng(seed)

    shocked = close.copy()
    after = shocked.index > T                                     # days after T
    shocked[after] = shocked[after].values * rng.lognormal(0, shock, after.sum())

    base, pert = daily_features(close), daily_features(shocked)
    rows = []
    for name in ['state', 'hist']:
        a, b = base[name], pert[name]
        before = np.allclose(a.loc[:T].values, b.loc[:T].values, rtol = 0, atol = 1e-12, equal_nan = True)
        changed = not np.allclose(a.loc[T + pd.Timedelta(days = 1):].values,
                                  b.loc[T + pd.Timedelta(days = 1):].values, rtol = 0, atol = 1e-12, equal_nan = True) #if anything changes, the first week after T is expected to be the same as macd for last week reprsents this week
        rows.append({'Series': name, 'Rows <= T': int((a.index <= T).sum()),
                     'Unchanged <= T': before, 'Changed > T': changed})

    res = pd.DataFrame(rows).set_index('Series')
    print(f'=== Lookahead test: closes after {T:%Y-%m-%d} shocked ===')
    print(res.to_string())
    ok = res['Unchanged <= T'].all() and res['Changed > T'].all()
    print('LOOKAHEAD TEST PASSED' if ok else 'LOOKAHEAD TEST FAILED')
    if not ok:
        raise AssertionError('lookahead test failed')
    return res