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
    plt.savefig("./charts/hist/orginal-hist.png")
    plt.show()

hist_tab('dump1.txt')