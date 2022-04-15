import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

A = [0.1, 0.4, 0.8]
f = [3000, 4000, 10000]

fs = 48000

N1 = 2048
N2 = 3 * N1 // 2

def sigval(time):
    return sum([_a * np.sin(2*np.pi*_f*time) for _a, _f in zip(A, f)])


if __name__ == "__main__":
    for n in (N1, N2):
        signal = sum([_a * np.sin(2*np.pi*_f*np.arange(n) / fs) for _a, _f in zip(A, f)])
        fper, Pxx = sig.periodogram(signal, fs, 'hamming', n, scaling="density")
        plt.semilogy(fper, Pxx)
        plt.xlim(0, 15000)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Spectrum power density")
        plt.suptitle("Signal spectrum power density")
        plt.title(f"N = {n}")
        plt.gcf().savefig(f"lab1-signal/charts/zad4/N={n}-Power-density-spectrum.png", format="png")
        plt.clf()

    window = fs / np.gcd.reduce(f)
    print(f"Window = {window}\nN1 % window = {N1%window}\nN2 % window = {N2%window}")
