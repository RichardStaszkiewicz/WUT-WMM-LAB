import numpy as np
import matplotlib.pyplot as plt

N = 48
A = 2
n = [0, N//4, N//2, (3*N)//4]

def amplitude_spectrum(signal):
    return np.abs(np.fft.fft(signal)) #tutaj ew można podzielić przez N a potem konsekwentnie wszystko skalować

def phase_spectrum(signal):
    return np.angle(np.fft.fft(signal))


def plot_amplitude_spectrum(a_spec, x):
    plt.stem(a_spec)
    plt.suptitle("Amplitude spectrum")
    plt.title(f"n0 = {x/100}N")
    plt.xlabel("spectrum idx")
    plt.ylabel("spectrum amplitude")
    plt.gcf().savefig(f"lab1-signal/charts/zad2/n={x}-amplitude-spc.png", format="png")
    plt.clf()

def plot_phase_spectrum(p_spec, x):
    plt.stem(p_spec)
    plt.suptitle("Phase spectrum")
    plt.title(f"n0 = {x/100}N")
    plt.gcf().savefig(f"lab1-signal/charts/zad2/n={x}-phase-spc.png", format="png")
    plt.clf()

def make_signal(signal, delta):
    return signal[delta:] + signal[:delta]


if __name__ == "__main__":
    s0 = [A * np.cos(2*np.pi*x/N) for x in range(N)]
    for x in n:
        signal = make_signal(s0, x)
        conv = 1e-6

        amplitude_s = amplitude_spectrum(signal)
        amplitude_s = [0 if a < conv else a for a in amplitude_s]

        phase_s = phase_spectrum(signal)
        phase_s = [0 if (np.abs(p) < conv or a < conv) else p for a, p in zip(amplitude_s, phase_s)]

        plot_phase_spectrum(phase_s, round((x*100)/N))
        plot_amplitude_spectrum(amplitude_s, round((x*100)/N))


