import readline
import matplotlib.pyplot as plt
import numpy as np

def hist_tab(dir):
    x = []
    with open(dir, "r+") as handle:
        c = handle.readline()
        while c:
            c = float(c)
            x.append(c)
            c = handle.readline()
    x = np.array(x)
    print(max(x))
    plt.figure()
    plt.plot(x, color="blue")
    plt.xlim([0, 256])
    plt.ylim(top=20000)
    plt.title("Mono orginal histogram")
    plt.savefig("lab3-photo-static-features/charts/original/mono-orginal-hist.png")
    plt.show()

def diff_hist(dir):
    x = []
    with open(dir, "r+") as handle:
        c = handle.readline()
        while c:
            c = float(c)
            x.append(c)
            c = handle.readline()
    x = np.array(x)
    print(max(x))
    plt.figure()
    plt.plot(np.arange(-255, 256, 1), x, color="red") ### jawne podane wartości 'x' i 'y', żeby zmienić opisy na osi poziomej
    plt.title("Mono diff histogram")
    plt.xlim([-255, 255])
    plt.savefig("lab3-photo-static-features/charts/differential/mono-diff-hist.png")
    plt.show()

def velv_hist(dirs):
    fig = plt.figure()
    fig.set_figheight(fig.get_figheight()*2) ### zwiększenie rozmiarów okna
    fig.set_figwidth(fig.get_figwidth()*2)
    plt.subplot(2, 2, 1)
    t = []
    for dir in dirs:
        x = []
        with open(dir, "r+") as handle:
            c = handle.readline()
            while c:
                c = float(c)
                x.append(c)
                c = handle.readline()
        t.append(np.array(x))
    hist_ll, hist_lh, hist_hl, hist_hh = t
    plt.plot(hist_ll, color="blue")
    plt.title("hist_ll")
    plt.xlim([0, 255])
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(-255, 256, 1), hist_lh, color="red")
    plt.title("hist_lh")
    plt.xlim([-255, 255])
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(-255, 256, 1), hist_hl, color="red")
    plt.title("hist_hl")
    plt.xlim([-255, 255])
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(-255, 256, 1), hist_hh, color="red")
    plt.title("hist_hh")
    plt.xlim([-255, 255])
    plt.show()

# diff_hist('lab3-photo-static-features/mono_diff-hist.txt')
velv_hist(["lab3-photo-static-features/hist_ll.txt", "lab3-photo-static-features/hist_lh.txt", "lab3-photo-static-features/hist_hl.txt", "lab3-photo-static-features/hist_hh.txt"])