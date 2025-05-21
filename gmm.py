import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

predict_retrain_years = 3
days_per_year = 252

df = pd.read_csv('sp500.csv')
df['date'] = pd.to_datetime(df['date'])

X = df['returns'].values.reshape(-1, 1)

train_size = 5809
test_size = 11012 - 5809

X_train = X[:train_size]
X_test = X[train_size:train_size + test_size]
dates_train = df['date'].iloc[:train_size]
dates_test = df['date'].iloc[train_size:train_size + test_size]

# Standardize the data (fit on training, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X)

# Train the GMM
gmm_model = GaussianMixture(n_components=2, n_init=10, random_state=42)
gmm_model.fit(X_train_scaled)

# Predict regimes for training and test data
train_regimes = gmm_model.predict(X_train_scaled)
test_regimes = np.zeros(test_size, dtype=int)

for t in range(0, test_size):
    X_test_scaled = X_scaled[:train_size + t]  # Past observations at time t
    test_regimes[t] = gmm_model.predict(X_test_scaled.reshape(-1, 1))[-1]

    # Retrain for every three years
    if t % (predict_retrain_years * days_per_year) == 0:
        print(f'Retraining after {t} days')
        gmm_model.fit(X_scaled[:t + train_size])
