import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# 不偏分散を求める関数
def unbiased_var(data):
    _data = data.copy()
    n = len(data)
    mean = sum(data) / n
    for i in range(n):
        _data[i] = (_data[i] - mean) ** 2
    return sum(_data) / (n - 1)


def culc_mad(data):
    median = np.median(data)

    diffs = []
    for d in data:
        d = abs(d - median)
        diffs.append(d)
    return np.median(diffs)


if __name__ == "__main__":
    np.random.seed(0)

    # 2つの乱数データ群を生成
    size = (300, 300)
    mu = (0, 0)
    sigma = (1.0, 1.1)
    A = np.random.normal(mu[0], sigma[0], size[0])
    B = np.random.normal(mu[1], sigma[1], size[1])

    # 不偏分散を求める
    var_A = unbiased_var(A)
    var_B = unbiased_var(B)

    # F値を求める
    F = var_B / var_A
    print("F value: ", F)

    # 自由度がsizeのF分布を生成
    alpha = 0.025
    dfn = size[0] - 1
    dfd = size[1] - 1
    rv = stats.f(dfn, dfd)
    lower_f = rv.ppf(1-alpha)
    upper_f = rv.ppf(alpha)

    # MADを求める
    mad_A = culc_mad(A)
    mad_corrected_A = 1.4826 * mad_A
    print("MAD: ", mad_A)
    print("MAD corrected: ", mad_corrected_A)
    print("Standard deviation: ", np.std(A))

    mad_B = culc_mad(B)
    mad_corrected_B = 1.4826 * mad_B
    print("MAD: ", mad_B)
    print("MAD corrected: ", mad_corrected_B)
    print("Standard deviation: ", np.std(B))

    # MADを用いたF値を求める
    # 外れ値に鈍感になることをF値の分布から確認
    F_robust = mad_corrected_B ** 2 / mad_corrected_A ** 2

    fig = plt.figure(figsize=(10, 6))
    x = np.linspace(rv.ppf(0.0001), rv.ppf(0.9999), 1000)
    y = rv.pdf(x)
    ax = fig.add_subplot(211)
    ax.plot(x, y, label="F dist(dfn: %d, dfd: %d)" % (dfn, dfd))
    ax.fill_between(x, y, where=(x > rv.ppf(1-alpha)), color="red", alpha=0.5)
    ax.fill_between(x, y, where=(x < rv.ppf(alpha)), color="red", alpha=0.5)
    ax.axvline(
        F,
        color="black",
        linestyle="--",
        label="F value",
        ymin=0,
        ymax=rv.pdf(F) + 0.5,
    )

    ax.axvline(
        F_robust,
        color="red",
        linestyle="--",
        label="F value (robust)",
        ymin=0,
        ymax=rv.pdf(F_robust) + 0.5,
    )
    ax.set_xlabel("F value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_ylim(0, max(rv.pdf(x)) + 0.1)

    # cdf をプロット
    ax = fig.add_subplot(212)
    y = rv.cdf(x)
    ax.plot(x, y, label="F dist(dfn: %d, dfd: %d)" % (dfn, dfd))
    ax.axvline(F, color="black", linestyle="--", label="F value")
    ax.set_xlabel("F value")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    fig.savefig("robust_f_test.png")
