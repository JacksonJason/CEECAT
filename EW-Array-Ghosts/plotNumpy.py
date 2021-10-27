import numpy as np
import matplotlib.pyplot as plt


def plot_and_save(gauss, point, label):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(label)
    ax1.plot(gauss[1], gauss[0])
    ax1.set_title("Gaussian")
    ax2.plot(point[1], point[0])
    ax2.set_title("Point")

    plt.savefig("vs_graphs/" + label + ".png")

    plt.clf()
    plt.cla()


G_gauss = np.load("data\G_gauss.npy")
GT_gauss = np.load("data\GT_gauss.npy")
M_gauss = np.load("data\M_gauss.npy")
R_gauss = np.load("data\R_gauss.npy")

G_point = np.load("data\G_point.npy")
GT_point = np.load("data\GT_point.npy")
M_point = np.load("data\M_point.npy")
R_point = np.load("data\R_point.npy")

plot_and_save(G_gauss, G_point, "G")
plot_and_save(GT_gauss, GT_point, "GT")
plot_and_save(M_gauss, M_point, "M")
plot_and_save(R_gauss, R_point, "R")
