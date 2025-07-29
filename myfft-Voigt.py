import numpy as np
import matplotlib.pyplot as plt

'''
  Total simulation time 20 fs will be great!!!
  Adjust to difference parameters: eta and sigma.
'''
# ---------- 参数设置 ----------
dipole_file = "siesta.TDDIPOL"
direction = 3  # 1: x, 2: y, 3: z
eta = 0.2  # Lorentzian 展宽参数，单位 eV,range:  0.05-0.3eV
hbar_ev_fs = 0.6582119  # ℏ in eV·fs

# ---------- 读取偶极数据 ----------
data = np.loadtxt(dipole_file)
time_fs = data[:, 0]
mu_t = data[:, direction]
dt = time_fs[1] - time_fs[0]
N = len(time_fs)
T = time_fs[-1] - time_fs[0]

# ---------- 去掉直流偏移 ----------
mu_t -= np.mean(mu_t)

# ---------- 构造 Voigt 时间窗（Gaussian × Lorentzian） ----------
sigma = T / 15.  # 可调节宽度（Gaussian 窗）,range T/5-T/10
gaussian_window = np.exp(-0.5 * ((time_fs - T/2) / sigma)**2)
lorentzian_decay = np.exp(-eta * time_fs / hbar_ev_fs)
voigt_window = gaussian_window * lorentzian_decay

# ---------- 应用 Voigt 窗并进行傅里叶变换 ----------
windowed_mu = mu_t * voigt_window
mu_fft = np.fft.fft(windowed_mu)
freq = np.fft.fftfreq(N, d=dt)
pos_freq_idx = freq > 0
freq = freq[pos_freq_idx]
spectrum = np.abs(mu_fft[pos_freq_idx])**2
spectrum /= np.max(spectrum)

# ---------- 转为能量单位 ----------
energy_ev = 2 * np.pi * freq * hbar_ev_fs

# ---------- 输出与绘图 ----------
np.savetxt("spectrum.dat", np.column_stack((energy_ev, spectrum)),
           header="Energy (eV)    Intensity (a.u.)")

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
})

plt.plot(energy_ev, spectrum, label=f'Z-dir, Voigt (η={eta} eV, σ={sigma:.1f} fs)')
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity (a.u.)")
plt.xlim(0.0, 8.0)
#plt.ylim(0.0, 0.03)
plt.title("TDDFT Absorption Spectrum (Voigt broadened)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum.png", dpi=300)
plt.show()

