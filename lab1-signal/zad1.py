import numpy as np
import matplotlib.pyplot as plt

SIGNAL1 = [3, 1, 0, 2]
SIGNAL2 = [0, 1, 0, 3]


def amplitude_spectrum(signal):
    return np.abs(np.fft.fft(signal)) #tutaj ew można podzielić przez N a potem konsekwentnie wszystko skalować

def phase_spectrum(signal):
    return np.angle(np.fft.fft(signal))

def power(signal):
    power_spectrum = amplitude_spectrum(signal)**2
    return sum(power_spectrum)

def Parseval(signal):
    return sum([i**2 for i in amplitude_spectrum(signal)])

def plot_amplitude_spectrum(signal):
    plt.stem(amplitude_spectrum(signal))
    plt.title("Amplitude spectrum")
    plt.xlabel("spectrum idx")
    plt.ylabel("spectrum amplitude")
    plt.gcf().savefig(f"lab1-signal/charts/zad1/{signal}-amplitude-spc.png", format="png")
    plt.clf()

def plot_phase_spectrum(signal):
    plt.title("Phase spectrum")
    plt.stem(phase_spectrum(signal))
    plt.gcf().savefig(f"lab1-signal/charts/zad1/{signal}-phase-spc.png", format="png")
    plt.clf()

def cyclic_convolution_def(signal1, signal2):
    if len(signal1) != len(signal2):
        return
    N = len(signal1)
    convolution = []
    for n in range(N):
        result = 0
        for m in range(N):
            if n - m < 0:
                x = N + (n - m)
            elif n - m == 0:
                x = 0
            else:
                x = n - m
            result += signal1[m] * signal2[x]
        convolution.append(result)
    return convolution

def cyclic_convolution_short(signal1, signal2):
    return np.real(np.fft.ifft(np.fft.fft(signal1) * np.fft.fft(signal2)))



if __name__ == "__main__":
    plot_amplitude_spectrum(SIGNAL1)
    plot_amplitude_spectrum(SIGNAL2)
    plot_phase_spectrum(SIGNAL1)
    plot_phase_spectrum(SIGNAL2)

    print("a.")

    for s in [SIGNAL1, SIGNAL2]:
        print(f"Moc sygnału wg. definicji: P{s} = {power(s)}")
        print(f"Moc sygnału wg. Tw. Parsevala: P{s} = {Parseval(s)}")

    print("\nb.")

    print(f"A**B wg. definicji: {cyclic_convolution_def(SIGNAL1, SIGNAL2)}")
    print(f"A**B wg. DTF: {cyclic_convolution_short(SIGNAL1, SIGNAL2)}")
