import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数设置 ----------
dipole_file = "siesta.TDDIPOL"
direction = 3  # 1: x, 2: y, 3: z
hbar_ev_fs = 0.6582119  # ℏ in eV·fs

# ---------- 读取偶极数据 ----------
data = np.loadtxt(dipole_file)
time_fs = data[:, 0]
mu_t_raw = data[:, direction]
dt = time_fs[1] - time_fs[0]
N = len(time_fs)
T = time_fs[-1] - time_fs[0]

# ---------- 预处理 dipole 数据 ----------
mu_t_raw -= np.mean(mu_t_raw)  # 去DC成分

# ---------- 设定多个 Voigt 参数组合 ----------
eta_list = [0.05, 0.1, 0.2]  # Lorentzian 宽度 (eV)
sigma_list = [T / 5, T / 8, T / 10]  # Gaussian 宽度 (fs)

# ---------- 画图准备 ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
})
plt.figure(figsize=(8, 5))

# ---------- 多组参数下绘制光谱 ----------
for eta in eta_list:
    for sigma in sigma_list:
        # 构造 Voigt 窗口
        gaussian_window = np.exp(-0.5 * ((time_fs - T/2) / sigma)**2)
        lorentzian_decay = np.exp(-eta * time_fs / hbar_ev_fs)
        voigt_window = gaussian_window * lorentzian_decay

        # 应用窗口并做 FFT
        mu_t = mu_t_raw * voigt_window
        mu_fft = np.fft.fft(mu_t)
        freq = np.fft.fftfreq(N, d=dt)
        pos_freq_idx = freq > 0
        freq_pos = freq[pos_freq_idx]
        spectrum = np.abs(mu_fft[pos_freq_idx])**2
        spectrum /= np.max(spectrum)  # 归一化

        # 转换为能量单位
        energy_ev = 2 * np.pi * freq_pos * hbar_ev_fs

        # 绘图
        label_str = f"η={eta}, σ={sigma:.1f} fs"
        plt.plot(energy_ev, spectrum, label=label_str)

# ---------- 最终绘图 ----------
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity (a.u.)")
plt.title("TDDFT Spectrum: Voigt Broadening Comparison")
plt.grid(True)
plt.xlim(0.0, 8.0)
#plt.ylim(0.0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum_voigt_comparison.png", dpi=300)
plt.show()

