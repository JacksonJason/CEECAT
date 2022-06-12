import numpy as np
import matplotlib.pyplot as plt


def plot_and_save(gauss, point, label, label1="Gaussian", label2="Point"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(label)
    ax1.plot(gauss[1], gauss[0])
    ax1.set_title(label1)
    ax2.plot(point[1], point[0])
    ax2.set_title(label2)

    plt.savefig("vs_graphs/" + label + ".png")

    plt.clf()
    plt.cla()

def plot_and_save_same_graph(gauss, point, label):
    gauss = gauss[0]
    point = point[0]
    plt.plot(gauss, "b")
    plt.plot(point, "r")

    plt.savefig("vs_graphs/" + label + ".png")

    plt.clf()
    plt.cla()

def plot_and_save_triple(gauss, point, theory, label, label1="Gaussian", label2="Point", label3="Theory"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(label)
    ax1.plot(gauss[1], gauss[0])
    ax1.set_title(label1)
    ax2.plot(point[1], point[0])
    ax2.set_title(label2)
    ax3.plot(theory[1], theory[0])
    ax3.set_title(label3)
    
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


import os.path
if os.path.isfile('data\G_theory_gauss.npy') and os.path.isfile("data\GT_theory_gauss.npy"):
    G_theory = np.load("data\G_theory_gauss.npy")
    GT_theory = np.load("data\GT_theory_gauss.npy")
    plot_and_save_triple(G_gauss, G_point, G_theory, "G_theory")
    plot_and_save_triple(GT_gauss, GT_point, GT_theory, "GT_theory")

if os.path.isfile('data\G_theory_vis_gauss.npy') and os.path.isfile("data\GT_theory_vis_gauss.npy"):
    G_theory = np.load("data\G_theory_vis_gauss.npy")
    GT_theory = np.load("data\GT_theory_vis_gauss.npy")
    G_gauss = np.load("data\G_vis_gauss.npy")
    GT_gauss = np.load("data\GT_vis_gauss.npy")
    plot_and_save(G_gauss, G_theory, "G_vis_theory", "Gaussian", "Theory")
    plot_and_save(GT_gauss, GT_theory, "GT_vis_theory", "Gaussian", "Theory")

    plot_and_save_same_graph(G_gauss, G_theory, "G_vis_theory_same")
    plot_and_save_same_graph(GT_gauss, GT_theory, "GT_vis_theory_same")


