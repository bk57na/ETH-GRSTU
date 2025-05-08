import numpy as np

def metrics(returns, regime, hedged=1):
    # hedge type
    regime = np.where(regime == 0, 1 - hedged, 1)

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

    var2 = np.percentile(rets2, (1 - confidence_level) * 100)
    print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level, for a holding strategy is {var2:.2%}")

    # cvar strategy 1 and 2
    var_threshold1 = np.percentile(rets1, (1 - confidence_level) * 100)
    tail_losses1 = rets1[rets1 < var_threshold1]
    cvar1 = tail_losses1.mean()
    print(
        f"Conditional Value at Risk (CVaR) at {confidence_level * 100}% confidence level, for a strategy hedged at {hedged:.1%} is {cvar1:.2%}")

    var_threshold2 = np.percentile(rets2, (1 - confidence_level) * 100)
    tail_losses2 = rets2[rets2 < var_threshold2]
    cvar2 = tail_losses2.mean()
    print(
        f"Conditional Value at Risk (CVaR) at {confidence_level * 100}% confidence level, for a holding strategy is {cvar2:.2%}")

    # cumulative return
    cumulative_return1 = np.prod([1 + r for r in rets1]) - 1
    print(f"Cumulative return for a strategy hedged at {hedged:.1%}: {cumulative_return1:.2%}")

    cumulative_return2 = np.prod([1 + r for r in rets2]) - 1
    print(f"Cumulative return for a holding strategy: {cumulative_return2:.2%}")