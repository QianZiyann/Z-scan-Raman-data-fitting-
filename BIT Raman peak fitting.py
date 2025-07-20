import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

# ------ Voigt 型峰函数和多峰+基线 ------
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def double_voigt_bg(x, *params):
    y = (
        voigt(x, params[0], params[1], params[2], params[3]) +
        voigt(x, params[4], params[5], params[6], params[7]) +
        params[8] + params[9] * x  # 线性基线
    )
    return y

# ------ 读取数据 ------
your_file_path = r"D:\OneDrive - HKUST Connect\HKUST\experiment\Ge-Si\BIT\2025.07.17\Ge\Ge_2_-2_1.txt"  # 修改为你的实际路径

# 1. 读原始txt为DataFrame
raw = pd.read_csv(your_file_path, sep=None, engine='python', header=None)
raman_shift = raw.iloc[0, 1:].astype(float).values  # 波数坐标，去除首列NaN
z_values = raw.iloc[1:, 0].astype(float).values     # z坐标
intensity = raw.iloc[1:, 1:].astype(float).values   # shape: (z数, 波数数)

# 2. 拟合并记录结果
results = []

for idx, (z, y) in enumerate(zip(z_values, intensity)):
    x = raman_shift
    # --- 初始参数猜测（你可根据实际调整） ---
    # 取y最大最小点附近为两个峰初值
    peak1_idx = np.argmax(y)
    peak2_idx = np.argmax(y * (abs(x - x[peak1_idx]) > 30))  # 距离主峰30以上的最大点
    # 保证peak1_idx < peak2_idx
    if x[peak1_idx] > x[peak2_idx]:
        peak1_idx, peak2_idx = peak2_idx, peak1_idx
    # 初值
    amp1 = y[peak1_idx]
    ctr1 = x[peak1_idx]
    amp2 = y[peak2_idx]
    ctr2 = x[peak2_idx]
    # 初始参数 [amp1, ctr1, sigma1, gamma1, amp2, ctr2, sigma2, gamma2, bg0, bg1]
    p0 = [amp1, ctr1, 2, 2, amp2, ctr2, 2, 2, np.min(y), 0]
    # 参数下界和上界
    lb = [0, ctr1-10, 0.2, 0.2, 0, ctr2-10, 0.2, 0.2, -np.inf, -np.inf]
    ub = [np.inf, ctr1+10, 20, 20, np.inf, ctr2+10, 20, 20, np.inf, np.inf]
    try:
        popt, _ = curve_fit(
            double_voigt_bg, x, y, p0=p0, bounds=(lb, ub), maxfev=20000
        )
        # 排序，保证peak1 < peak2
        centers = sorted([popt[1], popt[5]])
        results.append([z, f'{centers[0]:.3f}', f'{centers[1]:.3f}'])
    except Exception as e:
        results.append([z, 'fit_fail', 'fit_fail'])
        continue

# 3. 输出为csv
df_result = pd.DataFrame(results, columns=['z', 'peak1', 'peak2'])
df_result.to_csv('fitted_peaks.csv', index=False)
print('拟合已完成，结果已保存为 fitted_peaks.csv')