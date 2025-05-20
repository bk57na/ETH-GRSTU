import numpy as np

def metrics(returns, regime, hedged=1):
    # hedge type
    regime = np.where(regime == 0, 1, 1 - hedged)

    # R-S strategy
    strategy1 = regime
    rets1 = strategy1 * returns
    prices1 = np.zeros(len(returns))
    prices1[0] = 1000
    for t in range(1, len(returns)):
        prices1[t] = prices1[t - 1] * (1 + rets1[t] / 100)

    # hold strategy
    rets2 = returns
    prices2 = np.zeros(len(returns))
    prices2[0] = 1000
    for t in range(1, len(returns)):
        prices2[t] = prices2[t - 1] * (1 + rets2[t] / 100)

    # var strategy 1 and 2
    confidence_level = 0.95
    var1 = np.percentile(rets1, (1 - confidence_level) * 100)
    print(
        f"Value at Risk (VaR) at {confidence_level * 100}% confidence level, for a strategy hedged at {hedged:.1%} is {var1:.2%}")

    # cvar strategy 1 and 2
    var_threshold1 = np.percentile(rets1, (1 - confidence_level) * 100)
    tail_losses1 = rets1[rets1 < var_threshold1]
    cvar1 = tail_losses1.mean()
    print(
        f"Conditional Value at Risk (CVaR) at {confidence_level * 100}% confidence level, for a strategy hedged at {hedged:.1%} is {cvar1:.2%}")

    #log returns
    days_per_year = 252
    years = 20

    annual_log_returns = []

    for year in range(years):
        # Extract returns for one year
        start = year * days_per_year
        end = start + days_per_year
        yearly_returns = rets1[start:end]

        # Calculate log returns for the year
        log_returns = np.log(1 + yearly_returns)

        # Sum log returns to get annual log return
        annual_log_return = np.sum(log_returns)

        annual_log_returns.append(annual_log_return)

    annual_log_returns = np.mean(annual_log_returns)
    print(f"Annual log returns for each year: {annual_log_returns:.2%}")

    #Volatility
    volatility1 = np.std(rets1, ddof=1)
    volatility1 = volatility1 * np.sqrt(252)
    print(f"Volatility for a strategy hedged at {hedged:.1%}: {volatility1:.2%}")

    # Risk-free rate (annualized, e.g., 2% = 0.02)
    risk_free_rate_annual = 0.05


    # Excess returns
    excess_returns = rets1 - risk_free_rate_annual

    # Sharpe ratio (daily)
    mean_excess_return = np.mean(excess_returns)
    sharpe1 = mean_excess_return / volatility1
    print(f"Sharpe Ratio for a strategy hedged at {hedged:.1%}: {-1 * sharpe1:.3f}")

    # Step 1: Convert returns to cumulative price/index (start at 1)
    cumulative = np.cumprod(1 + rets1)

    # Step 2: Calculate running maximum of cumulative
    running_max = np.maximum.accumulate(cumulative)

    # Step 3: Calculate drawdowns
    drawdowns = (running_max - cumulative) / running_max

    # Step 4: Get the maximum drawdown (worst case)
    max_drawdown = np.max(drawdowns)

    print(f"Maximum Worst-Case Drawdown being hedge at {hedged:.1%}: {max_drawdown:.2%}")

