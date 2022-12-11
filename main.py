import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class Arima(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_samples):
        model = SARIMAX(x, order=(0, 1, 1), trend='t').fit()
        return model.get_forecast(n_samples)


def generate_time_series(initial_state=2, alpha=0.4, theta=1, nsamples=20):
    ts = []
    n = np.random.normal(size=nsamples)
    for t in range(len(n)):
        y_t = initial_state + alpha * t + theta * np.cumsum(n)[t]
        ts.append(y_t)
    return np.array(ts)


def plot_arima_prediction(train, test, x_train, x_test, pred):
    plt.plot(x_train, train, '.', label='train')
    plt.plot(x_test, test, '.', label='test')
    plt.plot(x_test, pred.predicted_mean, '.', label='pred')
    plt.fill_between(x_test, pred.conf_int()[:, 0], pred.conf_int()[:, 1], color='k', alpha=0.15)
    plt.legend()
    plt.show()


def observed_probability(test, arima_forecast_result):
    z_scores = (test - arima_forecast_result.predicted_mean) / arima_forecast_result.se_mean
    return np.prod(st.norm.cdf(z_scores))


if __name__ == '__main__':
    ts = generate_time_series()
    train = ts[:14]
    test = ts[14:]
    x_train = np.arange(14)
    x_test = np.arange(14, 20)

    arima_model = Arima()
    pred = arima_model.forward(train, 6)
    plot_arima_prediction(train, test, x_train, x_test, pred)

    p = observed_probability(test, pred)
    print('Probability of observing the test set given fitted model:', round(p, 5))

    train = np.append(ts[:7], ts[13:])
    # train = ts[:7]
    test = ts[7:13]

    model = SARIMAX(train, order=(0, 1, 1), trend='t').fit()
    pred = model.predict(7, 12)

    x_train = np.append(np.arange(7), np.arange(13, 20))
    x_test = np.arange(7, 13)
    plt.plot(x_train, train, '.', label='train')
    # plt.plot(x_train, np.append(ts[:7], ts[13:]), '.', label='train')
    plt.plot(x_test, test, '.', label='test')
    plt.plot(x_test, pred, '.', label='pred')
    plt.legend()
    plt.show()

    # 1.2.2) I would fit the ARIMA model with a Kalman filter instead of maximal likelihood,
    # which would deal better with the missing values.

