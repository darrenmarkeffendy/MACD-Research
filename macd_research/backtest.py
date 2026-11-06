# Imports Libraries

from backtesting import Strategy
from backtesting.lib import crossover, resample_apply

from macd_research.features import macd
from macd_research.data import BACKTEST_DIR, eval_start

def save_backtest(stats, ticker, name):

    ''' Saves the trade list, daily equity curve and summary stats of the backtest run '''

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    first_date = eval_start.strftime("%Y%m%d")
    last_date = stats['End'].strftime("%Y%m%d")
    stem = f"{ticker}_{name}_{first_date}_{last_date}"

    stats['_trades'].to_csv(BACKTEST_DIR / f"{stem}_trades.csv", index=False)
    stats['_equity_curve'].to_csv(BACKTEST_DIR / f"{stem}_equity.csv")
    stats.drop(['_strategy', '_equity_curve', '_trades']).to_csv(BACKTEST_DIR / f"{stem}_summary.csv")

class Naive_7DMACD_Strategy(Strategy):

    # MACD Variables
    fast_length = 12
    slow_length  = 26
    signal_length = 9

    def init(self):

        ''' Initiates MACD and Signal Line Variables for Naive Backtest '''

        self.macd, self.signal_line = resample_apply('7D',
                                    macd,
                                    self.data.Close,
                                    self.fast_length,
                                    self.slow_length,
                                    self.signal_length)
    
    def next(self):

        ''' Initiates the 7-Day MACD Trading Strategy based on bullish and bearish crossovers '''

        can_trade = self.data.index[-1] >= eval_start

        bullish_crossover = crossover(self.macd, self.signal_line)
        bearish_crossover = crossover(self.signal_line, self.macd)

        if can_trade:
            if bullish_crossover:
                if self.position:
                    self.position.close()
                self.buy()
            elif bearish_crossover:
                if self.position:
                    self.position.close()
                self.sell()
                