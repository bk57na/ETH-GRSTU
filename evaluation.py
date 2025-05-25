import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def evaluate_strategy(log_returns: np.ndarray, regimes: np.ndarray, hedged: float, confidence: float = 0.95,) -> None:
    """
    Evaluate a trading strategy based on log returns and regime labels,
    for a given hedged percentage. The evaluations are outputted to the console.
    Args:
        log_returns (np.ndarray): Daily log returns of the asset.
        regimes (np.ndarray): Regime labels (0 or 1) for each day.
        hedged (float): Percentage of the strategy that is hedged.
        confidence (float): Confidence level for CVaR calculation.
    """
    strategy_log_returns = log_returns.copy()
    strategy_log_returns[regimes == 1] *= (1 - hedged)

    # Conditional Value at Risk (CVaR)
    var_cutoff = np.percentile(strategy_log_returns, (1 - confidence) * 100)
    cvar = np.mean(strategy_log_returns[strategy_log_returns <= var_cutoff])

    # Maximum Drawdown
    cum_returns = np.exp(np.cumsum(strategy_log_returns)) - 1
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / (1 + running_max)
    max_drawdown = np.max(drawdowns)

    # Total log return (annualized)
    total_log_return = np.sum(strategy_log_returns)
    annual_log_return = total_log_return * (252 / len(log_returns))

    # Volatility (annualized)
    daily_volatility = np.std(strategy_log_returns)
    annual_volatility = daily_volatility * np.sqrt(252)

    # Sharpe ratio (using log returns)
    excess_return = annual_log_return
    sharpe_ratio = excess_return / annual_volatility

    print(f'Statistics for {hedged * 100}% hedged:')
    print(f' CVaR ({confidence * 100}%): {cvar:.2%}')
    print(f' MDD: -{max_drawdown:.1%}')
    print(f' Annual Log Return: {annual_log_return:.2%}')
    print(f' Annual Volatility: {annual_volatility:.1%}')
    print(f' Sharpe Ratio: {sharpe_ratio:.3f}')


def apply_smoothing(regimes: np.ndarray, min_consecutive: int = 3) -> np.ndarray:
    smoothed = np.copy(regimes)

    for i in range(min_consecutive, len(smoothed)):
        window = regimes[i - min_consecutive:i]

        # If the count is less than min_consecutive, apply a majority vote
        if np.sum(window == regimes[i]) < min_consecutive:
            smoothed[i] = np.bincount(window).argmax()

    return smoothed

def plot_regimes(dates: np.ndarray, prices: np.ndarray, regimes: np.ndarray, fig_size=(7, 6), save_path: str = None):
    """
    Plot the index level with regimes shaded. The plot is saved as a .pgf file if save_path is provided.
    Args:
        dates (np.ndarray): Dates of the index.
        prices (np.ndarray): Prices of the index.
        regimes (np.ndarray): Regime labels (0 or 1) for each day.
        fig_size (tuple): Size of the figure.
        save_path (str): Path to save the plot as a .pgf file. If None, the plot is not saved.
    """
    if save_path is not None:
        matplotlib.use('pgf')
        matplotlib.rcParams.update({
            'pgf.texsystem': 'pdflatex',
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    y_max = prices.max() * 1.1
    _, ax = plt.subplots(figsize=fig_size)
    ax.semilogy()
    ax.plot(dates, prices, label='Index Level')
    ax.fill_between(dates, 0, y_max, where=regimes == 0, color='g', alpha=0.5, label='Regime 0')
    ax.fill_between(dates, 0, y_max, where=regimes == 1, color='r', alpha=0.5, label='Regime 1')
    ax.set_xlabel('Year')
    ax.set_ylabel('Index Level')
    ax.legend(loc='upper left')
    ax.set_xlim((dates[0], dates[-1]))
    ax.set_ylim(top=y_max)

    if save_path is not None:
        plt.savefig(f'plots/{save_path}.pgf', bbox_inches='tight')


def apply_smoothing(regimes: np.ndarray, min_consecutive: int = 3) -> np.ndarray:
    smoothed = np.copy(regimes)

    for i in range(min_consecutive, len(smoothed)):
        window = regimes[i - min_consecutive:i]

        # If the count is less than min_consecutive, apply a majority vote
        if np.sum(window == regimes[i]) < min_consecutive:
            smoothed[i] = np.bincount(window).argmax()

    return smoothed
