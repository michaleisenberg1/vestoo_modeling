import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


class Arima(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.order = (0, 1, 1)

    def forward(self, x, n_samples):
        model = SARIMAX(x, order=self.order, trend='n').fit()
        return model.get_forecast(n_samples)


def generate_time_series(initial_state=2, alpha=0.4, theta=1, nsamples=20):
    ts = []
    n = np.random.normal(size=nsamples)
    for t in range(len(n)):
        y_t = initial_state + alpha * t + theta * np.cumsum(n)[t]
        ts.append(y_t)
    return pd.Series(ts)


def plot_arima_prediction(train, test, pred):
    x_test = test.index
    x_train = train.index
    ci = pred.conf_int()
    plt.plot(x_train, train, '.', label='train')
    plt.plot(x_test, test, '.', label='test')
    plt.plot(x_test, pred.predicted_mean, '.', label='pred')
    plt.fill_between(x_test, ci['lower y'], ci['upper y'], color='k', alpha=0.15)
    plt.legend()
    plt.show()


def observed_probability(observed, forecast_result):
    z_scores = (observed - forecast_result.predicted_mean) / forecast_result.se_mean
    neg_z_scores = [z if z <= 0 else -z for z in z_scores]
    return np.prod(st.norm.cdf(neg_z_scores) * 2)


if __name__ == '__main__':
    ts = generate_time_series()
    train = ts.loc[:13]
    test = ts.loc[14:]

    arima_model = Arima()
    pred = arima_model.forward(train, 6)
    plot_arima_prediction(train, test, pred)

    p = observed_probability(test.values, pred)
    print(
        'Probability of observing the test set given fitted model:',
        round(p, 5)
    )
