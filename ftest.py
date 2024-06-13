import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# 不偏分散を求める関数
def unbiased_var(data):
    n = len(data)
    mean = sum(data) / n
    for i in range(n):
        data[i] = (data[i] - mean) ** 2
    return sum(data) / (n - 1)


if __name__ == "__main__":
    np.random.seed(0)

    # 2つの乱数データ群を生成
    size = (100, 300)
    mu = (0, 1)
    sigma = (1, 1.2)
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
    print(f"{upper_f} < F range(dfn: {dfn}, dfd: {dfd}) < {lower_f}")

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
        ymax=rv.pdf(F),
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

    # F値が棄却域に入るか判定
    if rv.ppf(1-alpha) < F or F < rv.ppf(alpha):
        print("Reject H0")
    else:
        print("Accept H0")

    fig.savefig("f_test.png")

    chi2_A = (size[0] - 1) * var_A / sigma[0] ** 2
    chi2_B = (size[1] - 1) * var_B / sigma[1] ** 2
    F = chi2_B / chi2_A
    print(F)
