import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd


def add_features(raw_returns: pd.Series) -> pd.DataFrame:
    features = {}
    hls = [5, 20, 60]

    for hl in hls:
        # Feature 1: EWM-ret
        features[f'ret_{hl}'] = raw_returns.ewm(halflife=hl).mean()
        # Feature 2: log(EWM-DD)
        sq_mean = np.minimum(raw_returns, 0.).pow(2).ewm(halflife=hl).mean()
        dd = np.sqrt(sq_mean)
        features[f'dd-log_{hl}'] = np.log(dd)
        # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD
        features[f'sortino_{hl}'] = features[f'ret_{hl}'].div(dd)

    return pd.DataFrame(features)


df = pd.read_csv('sp500.csv', parse_dates=['date'], index_col='date')
returns = df['returns']
X = add_features(returns)
close = df['close']

X_train = X.iloc[:5809]
X_test = X.iloc[5809:11012]
test_start, test_end = X_test.index[[0, -1]]

clipper = DataClipperStd(mul=3.)
scalar = StandardScalerPD()

X_train_scaled = scalar.fit_transform(clipper.fit_transform(X_train))
X_test_scaled = scalar.transform(clipper.transform(X_test))

jm = JumpModel(n_components=2, jump_penalty=50, cont=False)
jm.fit(X_train_scaled, returns, sort_by='cumret')

test_regimes = jm.predict_online(X_test_scaled)

prices = df['close'][test_start:test_end]
returns = df['returns'][test_start:test_end]
