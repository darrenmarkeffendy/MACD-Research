import matplotlib.pyplot as plt

from macd_research.data import eval_start

def plot_equity_to_closing_price(data, equity_curve, TICKER):
    ''' Plots the strategy's normalized equity curve against a Buy and Hold strategy's normalized equity '''

    # Extracts and normalizes closing price data starting from eval_start

    closing_price_list = data['Close'].loc[eval_start:]
    normalized_closing_price = closing_price_list / closing_price_list.iloc[0]

    # Normalizes the strategy's equity curve over the same period

    equity_list = equity_curve['Equity'].loc[eval_start:]
    normalized_equity_list = equity_list / equity_list.iloc[0]

    # Creates a dates list for the X-axis

    dates = []
    for date in equity_list.index:
        dates.append(date.strftime('%Y-%m-%d'))

    # Plots the Equity curve to Closing Price Visualization

    fig, ax = plt.subplots()
    fig.set_figwidth(13)

    ax.plot(normalized_closing_price.values, label='Normalized Closing Price (Buy & Hold)')
    ax.plot(normalized_equity_list.values, label='Normalized Equity (Naive 7D MACD)')
    ax.set_yscale('log')                       # BTC grows ~40x: a log scale keeps early years visible

    fig.suptitle(f"{TICKER} Naive 7D MACD against Buy & Hold")
    plt.ylabel("Growth of 1 (log scale)")
    plt.xlabel("Timeline")

    # Limits the visible ticks on the X-axis to 7

    labels = len(dates) // 7
    labels = max(labels, 1)
    ax.set_xticks(range(len(dates))[::labels])
    ax.set_xticklabels(dates[::labels])

    plt.legend()
    plt.show()
