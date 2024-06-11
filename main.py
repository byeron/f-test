import numpy as np


def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def chi2(data):
    _sum = 0
    for d in data:
        _sum += (d ** 2)
    return _sum


if __name__ == '__main__':
    np.random.seed(0)

    size = 3
    dof = size - 1

    # 母分散が等しい2群のデータを生成
    base_data = np.random.normal(0, 10, size)
    same_var = np.random.normal(5, 10, size)

    # 母分散が異なる2群のデータを生成
    diff_var = np.random.normal(0, 20, size)

    # chi^2 統計量を計算, 標準化なし
    # 標準化しないと、母分散が同じでも異なる値になる
    base_chi2 = chi2(base_data)
    same_var_chi2 = chi2(same_var)
    print(base_chi2)
    print(same_var_chi2)

    # chi^2 統計量を計算, 標準化あり
    # 標準化すると、母分散が同じなら近い値になる
    base_chi2 = chi2(standardize(base_data))
    same_var_chi2 = chi2(standardize(same_var))
    print(base_chi2)
    print(same_var_chi2)
