import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Frecuencia de muestreo
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Señal compuesta
signal_clean = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# Ruido blanco
noise = 0.5 * np.random.normal(size=len(t))

# Señal con ruido
signal_noisy = signal_clean + noise

# Filtro pasa bajas
b_lp, a_lp = signal.butter(4, 10, btype='low', fs=fs)

# Filtro pasa altas
b_hp, a_hp = signal.butter(4, 20, btype='high', fs=fs)

# Filtro pasa bandas
b_bp, a_bp = signal.butter(4, [10, 40], btype='bandpass', fs=fs)

# Aplicación de filtros
signal_lp = signal.filtfilt(b_lp, a_lp, signal_noisy)
signal_hp = signal.filtfilt(b_hp, a_hp, signal_noisy)
signal_bp = signal.filtfilt(b_bp, a_bp, signal_noisy)

# Gráficas
plt.figure()
plt.plot(t, signal_noisy)
plt.title("Señal original con ruido")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

plt.figure()
plt.plot(t, signal_lp)
plt.title("Señal filtrada - Pasa Bajas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

plt.figure()
plt.plot(t, signal_hp)
plt.title("Señal filtrada - Pasa Altas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

plt.figure()
plt.plot(t, signal_bp)
plt.title("Señal filtrada - Pasa Bandas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

# Respuesta en frecuencia
plt.figure()
w, h = signal.freqz(b_lp, a_lp, fs=fs)
plt.plot(w, abs(h))
plt.title("Respuesta en frecuencia - Pasa Bajas")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.show()

plt.figure()
w, h = signal.freqz(b_hp, a_hp, fs=fs)
plt.plot(w, abs(h))
plt.title("Respuesta en frecuencia - Pasa Altas")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.show()

plt.figure()
w, h = signal.freqz(b_bp, a_bp, fs=fs)
plt.plot(w, abs(h))
plt.title("Respuesta en frecuencia - Pasa Bandas")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.show()
