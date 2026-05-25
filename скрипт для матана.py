import numpy as np
import matplotlib.pyplot as plt

# 1. Параметры
np.random.seed(42)   # чтобы каждый раз был один и тот же шум
N = 2048             # сколько точек
K0 = 31              # номер гармоники
A = 1.0
PHI = 0.3
sigma = 0.35         # шум

# время для графика
fs = 8000
t = np.arange(N) / fs

# 2. Сигнал s и зашумленный r
n = np.arange(N)
s = A * np.sin(2 * np.pi * K0 * n / N + PHI)
noise = np.random.normal(0, sigma, N)
r = s + noise

# 3. БПФ
r_fft = np.fft.fft(r)
s_fft = np.fft.fft(s)
s_k = s_fft / N   # коэффициенты ряда

# 4. Считаем MSE для разных K
max_K = 60
mse_exp = []      # эксперимент
mse_theo = []     # теория

for K in range(max_K + 1):
    # маска: оставляем гармоники с |k| <= K
    mask = np.abs(np.fft.fftfreq(N)) <= (K / N)

    # обнуляем лишние коэффициенты и делаем обратное БПФ
    r_new = r_fft.copy()
    r_new[~mask] = 0
    s_hat = np.fft.ifft(r_new).real

    # экспериментальный MSE
    mse = np.mean((s - s_hat) ** 2)
    mse_exp.append(mse)

    # теоретический MSE по формуле
    poterya = np.sum(np.abs(s_k[~mask]) ** 2)
    shum = (2 * K + 1) * (sigma ** 2) / N
    mse_theo.append(poterya + shum)

# где MSE минимальная
K_opt_exp = np.argmin(mse_exp)
K_opt_theo = np.argmin(mse_theo)
print("K_opt (эксперимент):", K_opt_exp)
print("K_opt (теория):", K_opt_theo)

# 5. Графики
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial"]
plt.figure(figsize=(12, 8))

# 5.1 сигнал во времени
plt.subplot(2, 2, 1)
plt.plot(t, s, label="полезный s(t)")
plt.plot(t, r, alpha=0.5, label="зашумленный r(t)")
plt.xlabel("время t, с")
plt.ylabel("амплитуда")
plt.title("Сигнал")
plt.legend()
plt.grid(True)

# 5.2 спектр
plt.subplot(2, 2, 2)
freq = np.fft.fftfreq(N, 1 / fs)
plt.stem(freq[freq >= 0], (np.abs(s_k[freq >= 0]) ** 2), basefmt=" ")
plt.xlim(0, 400)
plt.xlabel("частота f, Гц")
plt.ylabel("|c_k|^2")
plt.title("Спектр")
plt.grid(True)

# 5.3 восстановление при лучшем K
plt.subplot(2, 2, 3)
mask_best = np.abs(np.fft.fftfreq(N)) <= (K_opt_exp / N)
r_best = r_fft.copy()
r_best[~mask_best] = 0
s_hat_best = np.fft.ifft(r_best).real
plt.plot(t, s, label="s(t)")
plt.plot(t, s_hat_best, "--", label="восстановление, K=" + str(K_opt_exp))
plt.xlabel("время t, с")
plt.ylabel("амплитуда")
plt.title("Восстановление")
plt.legend()
plt.grid(True)

# 5.4 MSE от K
plt.subplot(2, 2, 4)
plt.plot(mse_exp, "o-", label="эксперимент")
plt.plot(mse_theo, "s--", label="теория")
plt.axvline(K_opt_exp, linestyle=":", label="K_opt = " + str(K_opt_exp))
plt.xlabel("число гармоник K")
plt.ylabel("MSE")
plt.title("MSE(K)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("grafik.png", dpi=150)
print("График сохранен: grafik.png")
plt.show()
