import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from macd_research.features import daily_features

def forward_returns(close, h):

    ''' % return from each day's close to the close h days later (NaN for the last h days). '''

    return (close.shift(-h) / close - 1) * 100 # Shifts the close prices h days up to show forward returns, returns in a pct form.


def state_regression(d, h):

    ''' Initiates a Regression a + b (bullish) and a - b (bearish) where a is the Crypto's general drift and b is the signal's effect.'''

    return sm.OLS(d['fwd'], sm.add_constant(d['state'])).fit(
        cov_type='HAC', cov_kwds={'maxlags': max(h - 1, 1)}) #maxlags refer to the Newey-West errors measuring the correlation up to h - 1 lags (days) and adds into standard error

def rank_ic(d, h):

    ''' Spearman rank IC of hist vs h-day forward return, with a Newey-West t-stat for overlapping returns. '''

    ic = spearmanr(d['hist'], d['fwd']).statistic

    rank = d['hist'].rank(pct=True) - 0.5 # Rank the histograms from  0 - 1 and then center at 0.
    fit = sm.OLS(d['fwd'], sm.add_constant(rank)).fit(
        cov_type='HAC', cov_kwds={'maxlags': max(h - 1, 1)}) # Ordinary Least Squares where y = d['fwd'] and constant of 1 and the rank.

    return {'Rank IC': ic,
            't IC (HAC)': fit.tvalues['hist'],
            'Top-bottom spread [%]': fit.params['hist']}

def quintile_table(d, n_bins=5):

    ''' Count, mean, and median h-day forward return per histogram quintile where 1 is the most bearish and 5 is the most bullish. '''

    bins = pd.qcut(d['hist'], n_bins, labels=False) + 1 # Based on d['hist'] - calculates which quintile each histogram value falls in. from  1 - 5. Note: Histogram value is the MACD- Signal line divided by price,

    table = d['fwd'].groupby(bins).agg(['count', 'mean', 'median']) # Groups the forward return based on the quintile, then taking account of mean, median per group.
    table.index.name = 'Quintile'
    return table

def spread_by_year(d):

    ''' Mean h-day forward return per calendar year in each state + the bull-minus-bear spread. '''

    by_year = d.groupby([d.index.year, 'state'])['fwd'].mean().unstack() # Unstack turns the inner index (+1, -1) into columns
    by_year = by_year.reindex(columns=[-1.0, 1.0]) 
    by_year['Spread'] = by_year[1.0] - by_year[-1.0] # Calculates the 2b spread, negative means bullish did worse after h days then bearish
    by_year.index.name = 'Year'
    return by_year

def placebo_bar(close, horizons=(5, 10, 20, 30, 40), n_paths=300, seed=0):

    ''' |t| that a useless signal reaches 5% of the time: reruns the state regression on price paths rebuilt from randomized daily returns (no trends) '''

    # Reminder that t = b/Standard Error. Measuring our effect B against its standard error

    rng = np.random.default_rng(seed) # Creates a random number generator object
    log_returns = np.log(close).diff().dropna().values # Log returns
    t_stats = {h: [] for h in horizons}

    for _ in range(n_paths):
        shuffled = rng.permutation(log_returns) # Shuffles the order of returns
        fake_close = pd.Series(close.iloc[0] * np.exp(np.r_[0, np.cumsum(shuffled)]), index=close.index) # Creates the culmulative sum of log returns, and multiplies them with starting price to create artificial price path
        features = daily_features(fake_close)

        for h in horizons:
            d = pd.concat([features['state'],
                           forward_returns(features['close'], h).rename('fwd')], axis=1).dropna() # Puts h day forward and state side by side.
            d = d[d['state'] != 0] # Removes days where the MACD line equals the signal line
            t_stats[h].append(abs(state_regression(d, h).tvalues['state'])) # appends |t| of b values

    table = pd.DataFrame({
        'P(|t| > 2) under null': {h: np.mean(np.array(t) > 2) for h, t in t_stats.items()}, # How many passed > 2
        '95th pct |t| (bar)':    {h: np.quantile(t, 0.95) for h, t in t_stats.items()}}) # 95th percentile
    table.index.name = 'Horizon [d]'
    return table

def performance(equity):
    ''' Total return, annualised return, volatility, Sharpe (rf = 0) and max drawdown of a daily equity series '''
    daily = equity.pct_change().dropna() #Drops the NaN on the first day that doesn't have pct_change()
    years = len(daily) / 365
    total = equity.iloc[-1] / equity.iloc[0] - 1
    return pd.Series({
        'Total return [%]':      total * 100,
        'Annual return [%]':     ((1 + total) ** (1 / years) - 1) * 100,
        'Annual volatility [%]': daily.std() * np.sqrt(365) * 100,
        'Sharpe':                daily.mean() / daily.std() * np.sqrt(365),
        'Max drawdown [%]':      (equity / equity.cummax() - 1).min() * 100})

