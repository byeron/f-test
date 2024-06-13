import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def standardize(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std


def chi2(data):
    _sum = 0
    for d in data:
        _sum += (d ** 2)
    return _sum


def case1(base_data, size, n_samples, mu, sigma):
    chi2s = []
    for i in range(n_samples):
        data = np.random.choice(base_data, size=size)
        print(data)
        data = standardize(data, mu, sigma)  # 既知の母平均と母標準偏差を使って標準化
        print(data)
        chi2_value = chi2(data)
        chi2s.append(chi2_value)
        print(chi2_value)
    return chi2s


def case2(base_data, size, n_samples, sigma):
    chi2s = []
    for i in range(n_samples):
        data = np.random.choice(base_data, size=size)
        data = standardize(data, std=sigma)  # 母平均を使わずに標準化
        chi2_value = chi2(data)
        chi2s.append(chi2_value)
    return chi2s


def case3(base_data, size, n_samples, sigma):
    chi2s = []
    for i in range(n_samples):
        dof = size - 1

        data = np.random.choice(base_data, size=size)
        chi2_value = (dof * np.var(data, ddof=1)) / (sigma ** 2)
        chi2s.append(chi2_value)
    return chi2s


def case4(base_data, size, n_samples, sigma):
    chi2s = []
    for i in range(n_samples):
        data = np.random.choice(base_data, size=size)
        chi2_value = size * np.var(data) / (sigma ** 2)
        chi2s.append(chi2_value)
    return chi2s


def plot(histgram, ax):
    ax.hist(histgram, bins=50, color='blue', alpha=0.7)
    ax.set_ylim(0, 200)


if __name__ == '__main__':
    np.random.seed(0)

    mu = 0
    sigma = 10
    base_data = np.random.normal(mu, sigma, 90000)
    size = 100
    n_samples = 1000
    fig = plt.figure(figsize=(6, 9))

    # 母平均が既知の場合
    chi2s = case1(
        base_data,
        size=size,
        n_samples=n_samples,
        mu=mu,
        sigma=sigma,
    )

    # histgram
    ax = fig.add_subplot(411)
    plot(chi2s, ax)

    # 母平均が未知の場合
    # 自由度がn-1のカイ二乗分布に従う
    chi2s = case2(
        base_data,
        size=size,
        n_samples=n_samples,
        sigma=sigma,
    )

    # histgram
    ax = fig.add_subplot(412)
    plot(chi2s, ax)

    # 不偏分散を使った場合
    chi2s = case3(
        base_data,
        size=size,
        n_samples=n_samples,
        sigma=sigma,
    )
    # histgram
    ax = fig.add_subplot(413)
    plot(chi2s, ax)

    # 標本分散を使った場合
    chi2s = case4(
        base_data,
        size=size,
        n_samples=n_samples,
        sigma=sigma,
    )

    # histgram
    ax = fig.add_subplot(414)
    plot(chi2s, ax)
    plt.tight_layout()
    fig.savefig('hist.png')

    # 区間推定してみる
    alpha = 0.05
    dof = size - 1
    lower = stats.chi2.ppf(alpha / 2, dof)
    upper = stats.chi2.ppf(1 - alpha / 2, dof)
    lower = (dof * np.var(base_data, ddof=1)) / lower
    upper = (dof * np.var(base_data, ddof=1)) / upper

    # standard deviation
    lower = np.sqrt(lower)
    upper = np.sqrt(upper)
    print(upper, lower)
