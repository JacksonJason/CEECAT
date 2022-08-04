from email.mime import base
import time

# from threading import active_count
import uuid
import numpy as np
import EW_theoretical_derivation
import sys

import signal
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp

# from multiprocessing.managers import SyncManager
from tqdm import tqdm
import random
import pickle

import warnings

warnings.filterwarnings("ignore")

"""
This class produces the theoretical ghost patterns of a simple two source case. It is based on a very simple EW array layout.
EW-layout: (0)---3---(1)---2---(2)

"""
print_lock = mp.Lock()


class T_ghost:
    """
    This function initializes the theoretical ghost object
    """

    def __init__(self):
        pass

    def create_G_stef(self, R, M, imax, tau, temp, no_auto):
        """This function finds argmin G ||R-GMG^H|| using StEFCal.
        R is your observed visibilities matrx.
        M is your predicted visibilities.
        imax maximum amount of iterations.
        tau stopping criteria.
        g the antenna gains.
        G = gg^H."""
        N = R.shape[0]
        g_temp = np.ones((N,), dtype=complex)
        if no_auto:
            R = R - R * np.eye(R.shape[0])
            M = M - M * np.eye(M.shape[0])
        for k in range(imax):
            g_old = np.copy(g_temp)
            for p in range(N):
                z = g_old * M[:, p]
                g_temp[p] = np.sum(np.conj(R[:, p]) * z) / (np.sum(np.conj(z) * z))

            if k % 2 == 0:
                if (
                    np.sqrt(np.sum(np.absolute(g_temp - g_old) ** 2))
                    / np.sqrt(np.sum(np.absolute(g_temp) ** 2))
                    <= tau
                ):
                    break
                else:
                    g_temp = (g_temp + g_old) / 2

        G_m = np.dot(np.diag(g_temp), temp)
        G_m = np.dot(G_m, np.diag(g_temp.conj()))

        g = g_temp
        G = G_m

        return g, G

    def plt_circle_grid(self, grid_m):
        rad = np.arange(1, 1 + grid_m, 1)
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 1, 500)

        x_c = np.cos(2 * np.pi * x)
        y_c = np.sin(2 * np.pi * y)
        for k in range(len(rad)):
            plt.plot(rad[k] * x_c, rad[k] * y_c, "k", ls=":", lw=0.5)

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on
    """

    def extrapolation_function(
        self,
        baseline,
        true_sky_model,
        cal_sky_model,
        Phi,
        image_s,
        s,
        resolution,
        pid,
        b0=36,
        f=1.45e9,
    ):
        temp = np.ones(Phi.shape, dtype=complex)

        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        delta_v = delta_u
        delta_l = resolution * (1.0 / 3600.0) * (np.pi / 180.0)
        N = int(np.ceil(1 / (delta_l * delta_u))) + 1

        if (N % 2) == 0:
            N = N + 1

        u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
        v = np.linspace(-(N - 1) / 2 * delta_v, (N - 1) / 2 * delta_v, N)
        uu, vv = np.meshgrid(u, v)
        u_dim = uu.shape[0]
        v_dim = uu.shape[1]
        #######################################################
        r_pq = np.zeros((u_dim, v_dim), dtype=complex)
        g_pq = np.zeros((u_dim, v_dim), dtype=complex)
        g_pq_t = np.zeros((u_dim, v_dim), dtype=complex)
        g_pq_t_inv = np.zeros((u_dim, v_dim), dtype=complex)
        m_pq = np.zeros((u_dim, v_dim), dtype=complex)

        R = np.zeros(Phi.shape, dtype=complex)
        M = np.zeros(Phi.shape, dtype=complex)

        str_baseline = str(baseline[0]) + " " + str(baseline[1])
        with tqdm(total=u_dim, desc=str_baseline, position=pid, leave=False) as pbar:
            for i in range(u_dim):
                pbar.update(1)
                for j in range(v_dim):
                    ut = u[i]
                    vt = v[j]
                    u_m = (Phi * ut) / (1.0 * Phi[baseline[0], baseline[1]])
                    v_m = (Phi * vt) / (1.0 * Phi[baseline[0], baseline[1]])
                    R = np.zeros(Phi.shape, dtype=complex)
                    M = np.zeros(Phi.shape, dtype=complex)
                    for k in range(len(true_sky_model)):
                        s = true_sky_model[k]
                        if len(s) <= 3:
                            R += s[0] * np.exp(
                                -2
                                * np.pi
                                * 1j
                                * (
                                    u_m * (s[1] * np.pi / 180.0)
                                    + v_m * (s[2] * np.pi / 180.0)
                                )
                            )
                        else:
                            sigma = s[3] * (np.pi / 180)
                            g_kernal = (
                                2
                                * np.pi
                                * sigma**2
                                * np.exp(
                                    -2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2)
                                )
                            )
                            R += (
                                s[0]
                                * np.exp(
                                    -2
                                    * np.pi
                                    * 1j
                                    * (
                                        u_m * (s[1] * np.pi / 180.0)
                                        + v_m * (s[2] * np.pi / 180.0)
                                    )
                                )
                                * g_kernal
                            )

                    for k in range(len(cal_sky_model)):
                        s = cal_sky_model[k]
                        if len(s) <= 3:
                            M += s[0] * np.exp(
                                -2
                                * np.pi
                                * 1j
                                * (
                                    u_m * (s[1] * np.pi / 180.0)
                                    + v_m * (s[2] * np.pi / 180.0)
                                )
                            )
                        else:
                            sigma = s[3] * (np.pi / 180)
                            g_kernal = (
                                2
                                * np.pi
                                * sigma**2
                                * np.exp(
                                    -2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2)
                                )
                            )
                            M += (
                                s[0]
                                * np.exp(
                                    -2
                                    * np.pi
                                    * 1j
                                    * (
                                        u_m * (s[1] * np.pi / 180.0)
                                        + v_m * (s[2] * np.pi / 180.0)
                                    )
                                )
                                * g_kernal
                            )
                    g_stef, G = self.create_G_stef(R, M, 200, 1e-8, temp, no_auto=False)

                    (
                        g_pq_t[i, j],
                        g_pq_t_inv[i, j],
                    ) = EW_theoretical_derivation.derive_from_theory(
                        true_sky_model[0][3],
                        N,
                        Phi,
                        baseline[0],
                        baseline[1],
                        true_sky_model[0][0],
                        ut,
                        vt,
                    )

                    r_pq[j, i] = R[baseline[0], baseline[1]]
                    m_pq[j, i] = M[baseline[0], baseline[1]]
                    g_pq[j, i] = G[baseline[0], baseline[1]]

        lam = (1.0 * 3 * 10**8) / f
        b_len = b0 * Phi[baseline[0], baseline[1]]
        fwhm = 1.02 * lam / (b_len)
        sigma_kernal = fwhm / (2 * np.sqrt(2 * np.log(2)))
        g_kernal = (
            2
            * np.pi
            * sigma_kernal**2
            * np.exp(-2 * np.pi**2 * sigma_kernal**2 * (uu**2 + vv**2))
        )

        return (
            r_pq,
            g_pq,
            g_pq_t,
            g_pq_t_inv,
            g_kernal,
            sigma_kernal,
            delta_u,
            delta_v,
            delta_l,
        )

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on
    """

    def extrapolation_function_linear(
        self,
        baseline,
        true_sky_model,
        cal_sky_model,
        Phi,
        vis_s,
        resolution,
        pid,
        b0=36,
        f=1.45e9,
    ):
        temp = np.ones(Phi.shape, dtype=complex)

        N = int(np.ceil(vis_s * 2 / resolution))

        if (N % 2) == 0:
            N = N + 1
        u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)
        r_pq = np.zeros(u.shape, dtype=complex)
        g_pq = np.zeros(u.shape, dtype=complex)
        g_pq_t = np.zeros(u.shape, dtype=complex)
        g_pq_t_inv = np.zeros(u.shape, dtype=complex)
        m_pq = np.zeros(u.shape, dtype=complex)

        R = np.zeros(Phi.shape, dtype=complex)
        M = np.zeros(Phi.shape, dtype=complex)

        str_baseline = str(baseline[0]) + " " + str(baseline[1])
        with tqdm(total=len(u), desc=str_baseline, position=pid, leave=False) as pbar:
            for i in range(len(u)):
                pbar.update(1)
                ut = u[i]
                vt = 0
                u_m = (Phi * ut) / (1.0 * Phi[baseline[0], baseline[1]])
                v_m = (Phi * vt) / (1.0 * Phi[baseline[0], baseline[1]])
                R = np.zeros(Phi.shape, dtype=complex)
                M = np.zeros(Phi.shape, dtype=complex)
                for k in range(len(true_sky_model)):
                    s = true_sky_model[k]
                    if len(s) <= 3:
                        R += s[0] * np.exp(
                            -2
                            * np.pi
                            * 1j
                            * (
                                u_m * (s[1] * np.pi / 180.0)
                                + v_m * (s[2] * np.pi / 180.0)
                            )
                        )
                    else:
                        sigma = s[3] * (np.pi / 180)
                        g_kernal = (
                            2
                            * np.pi
                            * sigma**2
                            * np.exp(
                                -2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2)
                            )
                        )
                        R += (
                            s[0]
                            * np.exp(
                                -2
                                * np.pi
                                * 1j
                                * (
                                    u_m * (s[1] * np.pi / 180.0)
                                    + v_m * (s[2] * np.pi / 180.0)
                                )
                            )
                            * g_kernal
                        )

                for k in range(len(cal_sky_model)):
                    s = cal_sky_model[k]
                    if len(s) <= 3:
                        M += s[0] * np.exp(
                            -2
                            * np.pi
                            * 1j
                            * (
                                u_m * (s[1] * np.pi / 180.0)
                                + v_m * (s[2] * np.pi / 180.0)
                            )
                        )
                    else:
                        sigma = s[3] * (np.pi / 180)
                        g_kernal = (
                            2
                            * np.pi
                            * sigma**2
                            * np.exp(
                                -2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2)
                            )
                        )
                        M += (
                            s[0]
                            * np.exp(
                                -2
                                * np.pi
                                * 1j
                                * (
                                    u_m * (s[1] * np.pi / 180.0)
                                    + v_m * (s[2] * np.pi / 180.0)
                                )
                            )
                            * g_kernal
                        )
                g_stef, G = self.create_G_stef(R, M, 200, 1e-8, temp, no_auto=False)

                (
                    g_pq_t[i],
                    g_pq_t_inv[i],
                    B,
                ) = EW_theoretical_derivation.derive_from_theory_linear(
                    true_sky_model[0][3],
                    N,
                    Phi,
                    baseline[0],
                    baseline[1],
                    true_sky_model[0][0],
                    ut,
                    vt,
                )

                r_pq[i] = R[baseline[0], baseline[1]]
                m_pq[i] = M[baseline[0], baseline[1]]
                g_pq[i] = G[baseline[0], baseline[1]]

        lam = (1.0 * 3 * 10**8) / f
        b_len = b0 * Phi[baseline[0], baseline[1]]
        fwhm = 1.02 * lam / (b_len)
        sigma_kernal = fwhm / (2 * np.sqrt(2 * np.log(2)))
        g_kernal = (
            2
            * np.pi
            * sigma_kernal**2
            * np.exp(-2 * np.pi**2 * sigma_kernal**2 * (u**2))
        )

        return r_pq, g_pq, g_pq_t, g_pq_t_inv, g_kernal, sigma_kernal, u, B


def another_exp(phi, size_gauss=0.02, K1=30.0, K2=3.0, N=4):
    s_size = size_gauss  # size of Gaussian in degrees
    r = (s_size * 3600) / (1.0 * K1)  # resolution
    siz = s_size * (K2 * 1.0)

    if N == 5:
        phi = np.delete(phi, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        phi = np.delete(phi, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 6:
        phi = np.delete(phi, [1, 3, 5, 6, 7, 8, 11, 12], axis=0)
        phi = np.delete(phi, [1, 3, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 7:
        phi = np.delete(phi, [1, 3, 5, 7, 8, 11, 12], axis=0)
        phi = np.delete(phi, [1, 3, 5, 7, 8, 11, 12], axis=1)
    if N == 8:
        phi = np.delete(phi, [1, 3, 5, 7, 11, 12], axis=0)
        phi = np.delete(phi, [1, 3, 5, 7, 11, 12], axis=1)
    if N == 9:
        phi = np.delete(phi, [1, 3, 5, 7, 12], axis=0)
        phi = np.delete(phi, [1, 3, 5, 7, 12], axis=1)
    if N == 10:
        phi = np.delete(phi, [1, 3, 5, 7], axis=0)
        phi = np.delete(phi, [1, 3, 5, 7], axis=1)
    if N == 11:
        phi = np.delete(phi, [3, 5, 7], axis=0)
        phi = np.delete(phi, [3, 5, 7], axis=1)
    if N == 12:
        phi = np.delete(phi, [5, 7], axis=0)
        phi = np.delete(phi, [5, 7], axis=1)
    if N == 13:
        phi = np.delete(phi, [7], axis=0)
        phi = np.delete(phi, [7], axis=1)
    if N == 4:
        phi = np.delete(phi, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        phi = np.delete(phi, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)
    every_baseline(phi, s_size, r, siz, K1, K2, N)


def every_baseline(phi, s_size=0.02, r=30.0, siz=3.0, K1=None, K2=None, N=None):

    mp.freeze_support()

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(mp.cpu_count(), initargs=(mp.RLock(),), initializer=tqdm.set_lock)
    signal.signal(signal.SIGINT, original_sigint_handler)
    m = mp.Manager()
    shared_array = m.dict()
    pid = 0
    try:
        for k in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                time.sleep(0.1)
                if len(shared_array) >= mp.cpu_count():
                    pids = [pid for pid, running in shared_array.items() if not running]
                    while len(pids) == 0:
                        time.sleep(0.1)
                        pids = [
                            pid for pid, running in shared_array.items() if not running
                        ]

                    pid = shared_array.keys().index(pids[0])
                res = pool.starmap_async(
                    process_baseline,
                    [
                        (
                            k,
                            j,
                            phi,
                            siz,
                            r,
                            s_size,
                            shared_array,
                            pid,
                            K1,
                            K2,
                            N,
                        )
                    ],
                )
                # process_baseline(k, j, phi, siz, r, s_size, shared_array, pid, counter, K1, K2, N)
                pid += 1

    except KeyboardInterrupt:
        print("CTRL+C")
    except Exception as e:
        print(e)
        f = open("crashReport.txt", "a")
        f.write("Crash report: " + e)
        f.close()
    finally:
        if not res.get()[0]:
            print(res.get())

    while True:
        time.sleep(5)
        pids = [pid for pid, running in shared_array.items() if running]

        if len(pids) == 0:
            print("Program finished")
            pool.terminate()
            pool.join()
            break


def process_baseline(
    k, j, phi, siz, r, s_size, shared_array, pid, K1=None, K2=None, N=None
):
    try:
        shared_array[pid] = True
        if j > k:
            baseline = [k, j]
            true_sky_model = np.array([[1.0, 0, 0, s_size]])
            cal_sky_model = np.array([[1, 0, 0]])

            t = T_ghost()

            file_name = ""
            if N is not None:
                file_name = (
                    "data/10_baseline/"
                    + str(N)
                    + "_10_baseline_"
                    + str(k)
                    + "_"
                    + str(j)
                    + "_"
                    + str(phi[k, j])
                    + ".p"
                )
            else:
                file_name = (
                    "data/G/g_"
                    + str(k)
                    + "_"
                    + str(j)
                    + "_"
                    + str(phi[k, j])
                    + ".png"
                )

            if N is not None:
                (
                    r_pq,
                    g_pq,
                    g_pq_t,
                    g_pq_t_inv,
                    g_kernal,
                    sigma_kernal,
                    delta_u,
                    delta_v,
                    delta_l,
                ) = t.extrapolation_function(
                    baseline=baseline,
                    true_sky_model=true_sky_model,
                    cal_sky_model=cal_sky_model,
                    Phi=phi,
                    image_s=siz,
                    s=1,
                    resolution=r,
                    pid=pid,
                )
                f = open(file_name, "wb")
                pickle.dump(g_pq, f)
                pickle.dump(r_pq, f)
                pickle.dump(g_kernal, f)
                pickle.dump(g_pq_t_inv, f)
                pickle.dump(g_pq_t, f)
                pickle.dump(sigma_kernal, f)
                pickle.dump(delta_u, f)
                pickle.dump(delta_l, f)
                pickle.dump(s_size, f)
                pickle.dump(siz, f)
                pickle.dump(r, f)
                pickle.dump(phi, f)
                if K1 is not None:
                    pickle.dump(K1, f)
                if K2 is not None:
                    pickle.dump(K2, f)
                f.close()
            else:
                (
                    r_pq,
                    g_pq,
                    g_pq_t,
                    g_pq_t_inv,
                    g_kernal,
                    sigma_kernal,
                    u,
                    B,
                ) = t.extrapolation_function_linear(
                    baseline=baseline,
                    true_sky_model=true_sky_model,
                    cal_sky_model=cal_sky_model,
                    Phi=phi,
                    vis_s=s_size,
                    resolution=r,
                    pid=pid,
                )
                f = open(file_name[:-2], "wb")
                pickle.dump(g_pq_t, f)
                pickle.dump(g_pq, f)
                pickle.dump(g_pq_t_inv, f)
                pickle.dump(r_pq, f)
                pickle.dump(g_kernal, f)
                pickle.dump(sigma_kernal, f)
                pickle.dump(B, f)
                if K1 is not None:
                    pickle.dump(K1, f)
                if K2 is not None:
                    pickle.dump(K2, f)
                f.close()
                plt.clf()
                plt.plot(u, np.absolute(g_pq_t ** (-1) * r_pq), "b")
                plt.plot(u, np.absolute(g_pq ** (-1) * r_pq), "r")
                plt.savefig(file_name)
    except Exception as e:
        print(e)
        f = open("crashReport.txt", "a")
        f.write("Crash report: " + e)
        f.close()
    finally:
        shared_array[pid] = False
        return not (j > k)


def process_pickle_files_g(phi=np.array([])):
    counter2 = 0

    vis_s = 5000
    resolution = 1

    N = int(np.ceil(vis_s * 2 / resolution))
    if (N % 2) == 0:
        N = N + 1
    u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)

    idx_M = np.zeros(phi.shape, dtype=int)
    c = 0
    for k in range(len(phi)):
        for j in range(len(phi)):
            c += 1
            idx_M[k, j] = c

    for k in range(len(phi)):
        for j in range(len(phi)):
            counter2 += 1
            if j != k:
                if j > k:
                    name = (
                        "data/G/g_"
                        + str(k)
                        + "_"
                        + str(j)
                        + "_"
                        + str(phi[k, j])
                        + ".p"
                    )

                    pkl_file = open(name, "rb")
                    g_pq_t = pickle.load(pkl_file)
                    g_pq = pickle.load(pkl_file)
                    g_pq_inv = pickle.load(pkl_file)
                    B = pickle.load(pkl_file)

                    ax = plt.subplot(14, 14, counter2)
                    if (k != 0) or (j != 1):
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax.set_xlabel(r"$u$" + " [rad" + r"$^{-1}$" + "]")

                    if k == 0:
                        ax.set_title(str(j))

                    ax.plot(u, np.absolute(g_pq) / B, "r")
                    ax.plot(u, np.absolute(g_pq_t) / B, "b")
                    ax = plt.subplot(14, 14, idx_M[j, k])
                    ax.set_ylim([0.9, 2.0])

                    if (k != 1) or (j != 2):
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax.set_xticks([])
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.tick_right()

                    if k == 0:
                        ax.annotate(
                            str(j),
                            xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad, 0),
                            xycoords=ax.yaxis.label,
                            textcoords="offset points",
                            size="large",
                            ha="right",
                            va="center",
                        )

                    ax.plot(u, np.absolute(g_pq ** (-1)) * B, "r")
                    ax.plot(u, np.absolute(g_pq_t ** (-1)) * B, "b")
                    ax.plot(u, np.absolute(g_pq_inv) * B, "g")

            else:
                if k == 0:
                    ax = plt.subplot(14, 14, counter2)
                    plt.setp(ax, "frame_on", False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(str(j))
                    ax.annotate(
                        str(0),
                        xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad, 0),
                        xycoords=ax.yaxis.label,
                        textcoords="offset points",
                        size="large",
                        ha="right",
                        va="center",
                    )
    plt.savefig(fname="plots/g_.png")


def process_pickle_files_g2(phi=np.array([])):
    counter2 = 0

    vis_s = 5000
    resolution = 1

    N = int(np.ceil(vis_s * 2 / resolution))
    if (N % 2) == 0:
        N = N + 1
    u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)

    idx_M = np.zeros(phi.shape, dtype=int)
    c = 0
    for k in range(len(phi)):
        for j in range(len(phi)):
            c += 1
            idx_M[k, j] = c

    for k in range(len(phi)):
        for j in range(len(phi)):
            counter2 += 1
            if j != k:
                if j > k:
                    # print(phi)
                    name = (
                        "data/G/g_"
                        + str(k)
                        + "_"
                        + str(j)
                        + "_"
                        + str(phi[k, j])
                        + ".p"
                    )

                    pkl_file = open(name, "rb")
                    g_pq_t = pickle.load(pkl_file)
                    g_pq = pickle.load(pkl_file)
                    g_pq_inv = pickle.load(pkl_file)
                    r_pq = pickle.load(pkl_file)
                    r_pq = pickle.load(pkl_file)
                    B = pickle.load(pkl_file)

                    ax = plt.subplot(14, 14, counter2)

                    if (k != 0) or (j != 1):
                        ax.set_xticks([])
                    else:
                        ax.set_xticks([-5000, 5000])
                        ax.set_xlabel(r"$u$" + " [rad" + r"$^{-1}$" + "]")

                    plt_i_domain = True

                    if (k == 0) and (j == 1):
                        ax.set_ylim(0, 7)
                        plt_i_domain = False

                    if (k == 0) and (j == 2):
                        ax.set_ylim(0, 3)
                        plt_i_domain = False

                    if (k == 0) and (j == 3):
                        ax.set_ylim(0, 3)
                        plt_i_domain = False
                    if (k == 1) and (j == 2):
                        ax.set_ylim(0, 4)
                        plt_i_domain = False
                    if (k == 1) and (j == 3):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 1) and (j == 4):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 2) and (j == 3):
                        ax.set_ylim(0, 4)
                        plt_i_domain = False
                    if (k == 2) and (j == 4):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 3) and (j == 4):
                        ax.set_ylim(0, 4)
                        plt_i_domain = False
                    if (k == 3) and (j == 5):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 4) and (j == 5):
                        ax.set_ylim(0, 4)
                        plt_i_domain = False
                    if (k == 4) and (j == 6):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 5) and (j == 6):
                        ax.set_ylim(0, 4)
                        plt_i_domain = False
                    if (k == 5) and (j == 7):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 6) and (j == 7):
                        ax.set_ylim(0, 4)
                        plt_i_domain = False
                    if (k == 6) and (j == 8):
                        ax.set_ylim(0, 2)
                        plt_i_domain = False
                    if (k == 7) and (j == 8):
                        ax.set_ylim(0, 3)
                        plt_i_domain = False

                    if (k == 12) and (j == 13):
                        ax.set_ylim(0, 20)
                        plt_i_domain = False

                    if k == 0:
                        ax.set_title(str(j))
                        plt_i_domain = False

                    ax.plot(u, np.absolute(g_pq ** (-1) * r_pq), "r")
                    ax.plot(u, np.absolute(g_pq_t ** (-1) * r_pq), "b")
                    ax.plot(u, np.absolute(g_pq_inv * r_pq), "g")

                    name = (
                        "data/10_baseline/14_10_baseline_"
                        + str(k)
                        + "_"
                        + str(j)
                        + "_"
                        + str(phi[k, j])
                        + ".p"
                    )
                    pkl_file = open(name, "rb")
                    g_pq12 = pickle.load(pkl_file)
                    r_pq12 = pickle.load(pkl_file)
                    g_pq_inv12 = pickle.load(pkl_file)
                    g_pq_t12 = pickle.load(pkl_file)
                    delta_u = pickle.load(pkl_file)
                    siz = pickle.load(pkl_file)
                    # phi = pickle.load(pkl_file)
                    
                    pkl_file.close()

                    ax = plt.subplot(14, 14, idx_M[j, k])

                    if (k != 2) or (j != 3):
                        ax.set_xticks([])
                        ax.yaxis.tick_right()

                    else:
                        ax.set_xticks([-siz, siz])
                        ax.set_xlabel(r"$l$" + " [deg]")
                        ax.xaxis.set_label_position("top")
                        ax.xaxis.tick_top()
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.tick_right()

                    if k == 0:
                        ax.annotate(
                            str(j),
                            xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad, 0),
                            xycoords=ax.yaxis.label,
                            textcoords="offset points",
                            size="large",
                            ha="right",
                            va="center",
                        )

                    x = np.linspace(-1 * siz, siz, len(g_pq12))
                    if plt_i_domain:
                        ax.plot(
                            x,
                            cut(
                                img(
                                    np.absolute(g_pq12 ** (-1) * r_pq12) * B,
                                    delta_u,
                                    delta_u,
                                )
                            ),
                            "r",
                        )
                        ax.plot(
                            x,
                            cut(
                                img(
                                    np.absolute(g_pq_t12 ** (-1) * r_pq12) * B,
                                    delta_u,
                                    delta_u,
                                )
                            ),
                            "b",
                        )
                        ax.plot(
                            x,
                            cut(
                                img(
                                    np.absolute(g_pq_inv12 * r_pq12) * B,
                                    delta_u,
                                    delta_u,
                                )
                            ),
                            "g",
                        )
                        ax.plot(
                            x,
                            cut(img(np.absolute(r_pq12), delta_u, delta_u)),
                            "y",
                        )

            else:
                if k == 0:
                    ax = plt.subplot(14, 14, counter2)
                    plt.setp(ax, "frame_on", False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(str(j))
                    ax.annotate(
                        str(0),
                        xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad, 0),
                        xycoords=ax.yaxis.label,
                        textcoords="offset points",
                        size="large",
                        ha="right",
                        va="center",
                    )
    plt.subplots_adjust(wspace=1, hspace=0.3)
    plt.savefig(fname="plots/g2_.png")


def compute_division_matrix(P=np.array([]), N=14, peak_flux=2, peak_flux2=100):
    if N == 5:
        P = np.delete(P, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 6:
        P = np.delete(P, [1, 3, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 7:
        P = np.delete(P, [1, 3, 5, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 8, 11, 12], axis=1)
    if N == 8:
        P = np.delete(P, [1, 3, 5, 7, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 11, 12], axis=1)
    if N == 9:
        P = np.delete(P, [1, 3, 5, 7, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 12], axis=1)
    if N == 10:
        P = np.delete(P, [1, 3, 5, 7], axis=0)
        P = np.delete(P, [1, 3, 5, 7], axis=1)
    if N == 11:
        P = np.delete(P, [3, 5, 7], axis=0)
        P = np.delete(P, [3, 5, 7], axis=1)
    if N == 12:
        P = np.delete(P, [5, 7], axis=0)
        P = np.delete(P, [5, 7], axis=1)
    if N == 13:
        P = np.delete(P, [7], axis=0)
        P = np.delete(P, [7], axis=1)
    if N == 4:
        P = np.delete(P, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)

    final_matrix = np.zeros((N, N), dtype=float)
    amp_matrix = np.zeros((len(P), len(P), 3), dtype=float)

    curves = [np.array([]), np.array([]), np.array([])]
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if j > i:
                name = (
                    "data/10_baseline/"
                    + str(N)
                    + "_10_baseline_"
                    + str(i)
                    + "_"
                    + str(j)
                    + "_"
                    + str(P[i, j])
                    + ".p"
                )


                pkl_file = open(name, "rb")
                g_pq12 = pickle.load(pkl_file)
                r_pq12 = pickle.load(pkl_file)
                g_pq_inv12 = pickle.load(pkl_file)
                g_pq_t12 = pickle.load(pkl_file)
                delta_u = pickle.load(pkl_file)
                s_size = pickle.load(pkl_file)
                
                pkl_file.close()

                B = 2 * np.pi * (s_size * (np.pi / 180)) ** 2

                c1 = cut(
                    img(np.absolute(g_pq12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                )  # red
                c2 = cut(
                    img(np.absolute(g_pq_t12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                )  # blue
                c3 = cut(
                    img(np.absolute(g_pq_inv12 * r_pq12) * B, delta_u, delta_u)
                )  # green

                curves[0] = c1
                curves[1] = c2
                curves[2] = c3

                value = -1

                if (
                    np.max(curves[0]) <= peak_flux
                    and np.max(curves[1]) <= peak_flux
                    and np.max(curves[2]) <= peak_flux
                ):
                    value = 0
                elif (
                    np.max(curves[0]) <= peak_flux
                    and np.max(curves[1]) <= peak_flux2
                    and np.max(curves[1]) > peak_flux
                    and np.max(curves[2]) <= peak_flux
                ):
                    value = 2
                elif (
                    np.max(curves[1]) <= peak_flux
                    and np.max(curves[0]) <= peak_flux2
                    and np.max(curves[0]) > peak_flux
                    and np.max(curves[2]) <= peak_flux
                ):
                    value = 1
                elif (
                    np.max(curves[1]) <= peak_flux
                    and np.max(curves[0]) > peak_flux2
                    and np.max(curves[2]) <= peak_flux
                ):
                    value = 5
                elif (
                    np.max(curves[1]) > peak_flux
                    and np.max(curves[1]) <= peak_flux2
                    and np.max(curves[0]) <= peak_flux2
                    and np.max(curves[0]) > peak_flux
                    and np.max(curves[2]) <= peak_flux
                ):
                    value = 3
                elif (
                    np.max(curves[1]) > peak_flux
                    and np.max(curves[1]) <= peak_flux2
                    and np.max(curves[0]) > peak_flux2
                    and np.max(curves[2]) <= peak_flux
                ):
                    value = 4

                final_matrix[i, j] = value
                final_matrix[j, i] = value

                amp_matrix[i, j, 0] = np.max(c1)
                amp_matrix[i, j, 1] = np.max(c2)
                amp_matrix[i, j, 2] = np.max(c3)
                amp_matrix[j, i, :] = amp_matrix[i, j, :]

    return final_matrix, amp_matrix


def main_phi_plot(P=np.array([]), m=np.array([])):
    a = np.unique(np.absolute(P)).astype(int)

    amp_matrix = np.zeros((len(P), len(P), 3), dtype=float)
    size_matrix = np.zeros((len(P), len(P), 3), dtype=float)
    flux_matrix = np.zeros((len(P), len(P), 3), dtype=float)

    for k in range(len(P)):
        for j in range(len(P)):
            if j == k:
                amp_matrix[k, j, :] = np.nan
                size_matrix[k, j, :] = np.nan
                flux_matrix[k, j, :] = np.nan
            if j != k:
                if j > k:
                    name = (
                        "data/10_baseline/"
                        + str(14)
                        + "_10_baseline_"
                        + str(k)
                        + "_"
                        + str(j)
                        + "_"
                        + str(P[k, j])
                        + ".p"
                    )
                    pkl_file = open(name, "rb")
                    g_pq12 = pickle.load(pkl_file)
                    r_pq12 = pickle.load(pkl_file)
                    g_pq_inv12 = pickle.load(pkl_file)
                    g_pq_t12 = pickle.load(pkl_file)
                    delta_u = pickle.load(pkl_file)
                    s_size = pickle.load(pkl_file)
                    r = pickle.load(pkl_file)
                    
                    pkl_file.close()
                    B = 2 * np.pi * (s_size * (np.pi / 180)) ** 2
                    # PROCESS THE INFORMATION AS A FUNCTION OF PHI
                    c1 = cut(
                        img(np.absolute(g_pq12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                    )
                    c2 = cut(
                        img(
                            np.absolute(g_pq_t12 ** (-1) * r_pq12) * B, delta_u, delta_u
                        )
                    )
                    c3 = cut(
                        img(np.absolute(g_pq_inv12 * r_pq12) * B, delta_u, delta_u)
                    )

                    if include_baseline(k, j):
                        amp_matrix[k, j, 0] = np.max(c1)
                    else:
                        amp_matrix[k, j, 0] = np.nan
                    amp_matrix[k, j, 1] = np.max(c2)
                    amp_matrix[k, j, 2] = np.max(c3)
                    amp_matrix[j, k, :] = amp_matrix[k, j, :]

                    if include_baseline(k, j):
                        idx = c1 >= (amp_matrix[k, j, 0] / 2)
                        integer_map = map(int, idx)
                        size_matrix[k, j, 0] = (
                            np.sum(list(integer_map)) * r
                        )  # in arcseconds
                    else:
                        size_matrix[k, j, 0] = np.nan

                    idx = c2 >= (amp_matrix[k, j, 1] / 2)
                    integer_map = map(int, idx)
                    size_matrix[k, j, 1] = (
                        np.sum(list(integer_map)) * r
                    )  # in arcseconds

                    idx = c3 >= ((amp_matrix[k, j, 2] - np.min(c3)) / 2)
                    integer_map = map(int, idx)
                    size_matrix[k, j, 2] = (
                        np.sum(list(integer_map)) * r
                    )  # in arcseconds

                    size_matrix[j, k, :] = size_matrix[k, j, :]

                    if include_baseline(k, j):
                        flux_matrix[k, j, 0] = (
                            amp_matrix[k, j, 0]
                            * (
                                ((size_matrix[k, j, 0] / 3600.0) * (np.pi / 180))
                                / (2 * np.sqrt(2 * np.log(2)))
                            )
                            ** 2
                            * 2
                            * np.pi
                        )
                    else:
                        flux_matrix[k, j, 0] = np.nan
                    flux_matrix[k, j, 1] = (
                        amp_matrix[k, j, 1]
                        * (
                            ((size_matrix[k, j, 1] / 3600.0) * (np.pi / 180))
                            / (2 * np.sqrt(2 * np.log(2)))
                        )
                        ** 2
                        * 2
                        * np.pi
                    )
                    flux_matrix[k, j, 2] = (
                        amp_matrix[k, j, 2]
                        * (
                            ((size_matrix[k, j, 2] / 3600.0) * (np.pi / 180))
                            / (2 * np.sqrt(2 * np.log(2)))
                        )
                        ** 2
                        * 2
                        * np.pi
                    )

                    flux_matrix[j, k, :] = flux_matrix[k, j, :]

    size_matrix[amp_matrix > 100] = np.nan
    flux_matrix[amp_matrix > 100] = np.nan
    amp_matrix[amp_matrix > 100] = np.nan

    fwhm = (s_size * (np.pi / 180) * 2 * np.sqrt(2 * np.log(2))) * (180 / np.pi) * 3600

    P_new = np.zeros(P.shape, dtype=float)
    P_s = np.zeros(P.shape, dtype=float)
    P_new2 = np.zeros(P.shape, dtype=float)

    for k in range(P.shape[0]):
        for j in range(P.shape[1]):
            if j == k:
                P_new2[j, k] = np.nan
            if j > k:
                P_new[k, j] = (
                    np.mean(((P[k, :] ** 2 + P[k, j] ** 2) / P[k, j] ** 2) ** (-1))
                    + np.mean(((P[:, j] ** 2 + P[k, j] ** 2) / P[k, j] ** 2) ** (-1))
                ) / 2
                P_s[k, j] = (
                    np.std(((P[k, :] ** 2 + P[k, j] ** 2) / P[k, j] ** 2) ** (-1))
                    + np.std(((P[:, j] ** 2 + P[k, j] ** 2) / P[k, j] ** 2) ** (-1))
                ) / 2
                P_s[j, k] = P_s[k, j]
                P_new[j, k] = P_new[k, j]
                P_new2[k, j] = (
                    np.mean(((P[k, :] ** 2 + P[k, j] ** 2) / P[k, j] ** 2) ** (1))
                    + np.mean(((P[:, j] ** 2 + P[k, j] ** 2) / P[k, j] ** 2) ** (1))
                ) / 2
                P_new2[j, k] = P_new2[k, j]

    plt_imshow(np.absolute(P_s), P, l=r"std$(\hat{K}_{pq})$", name_file="s")
    plt_imshow(np.absolute(P), P, l=r"$\phi_{pq}$", name_file="phi")
    plt_imshow(P_new, P, l=r"$\hat{K}_{pq}$", name_file="k")
    plt_imshow(m, P, l="Category", name_file="cat")

    for k in range(3):

        plt_imshow(
            amp_matrix[:, :, k], P, l=r"c", name_file="pk" + str(k), vmin=1, vmax=2
        )
        plt_imshow(
            size_matrix[:, :, k] / fwhm,
            P,
            l=r"FWHM $/$ FWHM$_B$",
            name_file="fw" + str(k),
            vmin=0,
            vmax=1,
        )
        plt_imshow(
            flux_matrix[:, :, k] / B,
            P,
            l=r"Flux [Jy]",
            name_file="f" + str(k),
            vmin=1,
            vmax=2,
        )


def include_baseline(k=0, j=1):
    if (k == 0) and (j == 1):
        return False
    if (k == 0) and (j == 2):
        return False
    if (k == 0) and (j == 3):
        return False
    if (k == 1) and (j == 2):
        return False
    if (k == 1) and (j == 3):
        return False
    if (k == 1) and (j == 4):
        return False
    if (k == 2) and (j == 3):
        return False
    if (k == 2) and (j == 4):
        return False
    if (k == 2) and (j == 5):
        return False
    if (k == 3) and (j == 4):
        return False
    if (k == 3) and (j == 5):
        return False
    if (k == 4) and (j == 5):
        return False
    if (k == 4) and (j == 6):
        return False
    if (k == 5) and (j == 6):
        return False
    if (k == 5) and (j == 7):
        return False
    if (k == 6) and (j == 7):
        return False
    if (k == 6) and (j == 8):
        return False
    if (k == 7) and (j == 8):
        return False
    if (k == 12) and (j == 13):
        return False
    return True


def plt_imshow(
    data,
    P,
    l=r"Log(Peak Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$])",
    name_file="f",
    vmax=-1,
    vmin=-1,
):
    N = 20  # number of colors to extract from each of the base_cmaps below
    base_cmaps = ["YlGnBu", "Greys", "Purples", "Reds", "Oranges"]
    # we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
    colors = np.concatenate(
        [plt.get_cmap(name)(np.linspace(0.2, 0.8, N)) for name in base_cmaps]
    )
    fig, ax = plt.subplots()
    if vmax == vmin:
        im = ax.imshow(data, cmap="jet")
    else:
        im = ax.imshow(data, cmap="jet", vmax=vmax, vmin=vmin)
    cbar = fig.colorbar(im)
    cbar.set_label(l, labelpad=10)

    min_val, max_val, diff = 0.0, 14.0, 1.0

    # text portion
    ind_array = np.arange(min_val, max_val, diff)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val, p, d in zip(
        x.flatten(), y.flatten(), P.flatten(), data.flatten()
    ):
        c = str(int(p))

        if np.isnan(d):
            ax.text(
                x_val,
                y_val,
                int(np.absolute(p)),
                va="center",
                ha="center",
                color="black",
            )
        else:
            ax.text(
                x_val,
                y_val,
                int(np.absolute(p)),
                va="center",
                ha="center",
                color="white",
            )

    ax = plt.gca()
    xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    yticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("Antenna $q$")
    ax.set_ylabel("Antenna $p$")
    plt.savefig("plots/" + name_file + ".png")
    plt.close()


def cut(inp):
    return inp.real[int(inp.shape[0] / 2), :]


def img(inp, delta_u, delta_v):
    zz = inp
    zz = np.roll(zz, -int(zz.shape[0] / 2), axis=0)
    zz = np.roll(zz, -int(zz.shape[0] / 2), axis=1)

    zz_f = np.fft.fft2(zz) * (delta_u * delta_v)
    zz_f = np.roll(zz_f, -int(zz.shape[0] / 2) - 1, axis=0)
    zz_f = np.roll(zz_f, -int(zz.shape[0] / 2) - 1, axis=1)
    return zz_f.real


def get_average_response(P=np.array([]), N=14, peak_flux=100):
    if N == 5:
        P = np.delete(P, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 6:
        P = np.delete(P, [1, 3, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 7:
        P = np.delete(P, [1, 3, 5, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 8, 11, 12], axis=1)
    if N == 8:
        P = np.delete(P, [1, 3, 5, 7, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 11, 12], axis=1)
    if N == 9:
        P = np.delete(P, [1, 3, 5, 7, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 12], axis=1)
    if N == 10:
        P = np.delete(P, [1, 3, 5, 7], axis=0)
        P = np.delete(P, [1, 3, 5, 7], axis=1)
    if N == 11:
        P = np.delete(P, [3, 5, 7], axis=0)
        P = np.delete(P, [3, 5, 7], axis=1)
    if N == 12:
        P = np.delete(P, [5, 7], axis=0)
        P = np.delete(P, [5, 7], axis=1)
    if N == 13:
        P = np.delete(P, [7], axis=0)
        P = np.delete(P, [7], axis=1)
    if N == 4:
        P = np.delete(P, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)

    counter1 = 0
    counter2 = 0
    counter3 = 0

    average_curve1 = np.array([])
    average_curve2 = np.array([])
    average_curve3 = np.array([])

    max_peak = np.zeros((3,), dtype=float)
    width = np.zeros((3,), dtype=float)
    flux = np.zeros((3,), dtype=float)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if j > i:
                name = (
                    "data/10_baseline/"
                    + str(N)
                    + "_10_baseline_"
                    + str(i)
                    + "_"
                    + str(j)
                    + "_"
                    + str(P[i, j])
                    + ".p"
                )

                pkl_file = open(name, "rb")
                g_pq12 = pickle.load(pkl_file)
                r_pq12 = pickle.load(pkl_file)
                g_pq_inv12 = pickle.load(pkl_file)
                g_pq_t12 = pickle.load(pkl_file)
                delta_u = pickle.load(pkl_file)
                s_size = pickle.load(pkl_file)
                siz = pickle.load(pkl_file)
                r = pickle.load(pkl_file)
                
                pkl_file.close()

                B = 2 * np.pi * (s_size * (np.pi / 180)) ** 2
                fwhm = (
                    (s_size * (np.pi / 180) * 2 * np.sqrt(2 * np.log(2)))
                    * (180 / np.pi)
                    * 3600
                )

                c1 = cut(
                    img(np.absolute(g_pq12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                )  # red
                c2 = cut(
                    img(np.absolute(g_pq_t12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                )  # blue
                c3 = cut(
                    img(np.absolute(g_pq_inv12 * r_pq12) * B, delta_u, delta_u)
                )  # green

                if np.max(c1) < peak_flux:
                    if len(average_curve1) == 0:
                        average_curve1 = c1
                        counter1 = 1
                    else:
                        average_curve1 += c1
                        counter1 += 1

                if len(average_curve2) == 0:
                    average_curve2 = c2
                    counter2 = 1
                else:
                    average_curve2 += c2
                    counter2 += 1

                if len(average_curve3) == 0:
                    average_curve3 = c3
                    counter3 = 1
                else:
                    average_curve3 += c3
                    counter3 += 1

    average_curve1 /= counter1 * 1.0
    average_curve2 /= counter2 * 1.0
    average_curve3 /= counter3 * 1.0

    max_peak[0] = np.max(average_curve1)
    max_peak[1] = np.max(average_curve2)
    max_peak[2] = np.max(average_curve3)

    idx = average_curve1 >= (max_peak[0] / 2)
    integer_map = map(int, idx)
    width[0] = np.sum(list(integer_map)) * r  # in arcseconds

    idx = average_curve2 >= (max_peak[1] / 2)
    integer_map = map(int, idx)
    width[1] = np.sum(list(integer_map)) * r  # in arcseconds

    idx = average_curve3 >= (max_peak[2] / 2)
    integer_map = map(int, idx)
    width[2] = np.sum(list(integer_map)) * r  # in arcseconds

    flux[0] = (
        (max_peak[0] / B)
        * (((width[0] / 3600.0) * (np.pi / 180)) / (2 * np.sqrt(2 * np.log(2)))) ** 2
        * 2
        * np.pi
    )
    flux[1] = (
        (max_peak[1] / B)
        * (((width[1] / 3600.0) * (np.pi / 180)) / (2 * np.sqrt(2 * np.log(2)))) ** 2
        * 2
        * np.pi
    )
    flux[2] = (
        (max_peak[2] / B)
        * (((width[2] / 3600.0) * (np.pi / 180)) / (2 * np.sqrt(2 * np.log(2)))) ** 2
        * 2
        * np.pi
    )

    width = width / fwhm

    return max_peak, width, flux


def create_violin_plot(
    data=[],
    fc="green",
    labels=["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"],
    yl=r"Amp. Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$]",
    t=False,
    label="",
):
    fig, ax = plt.subplots()
    # Create a plot
    parts = ax.violinplot(data, showmeans=True, showmedians=True)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("green")
    parts["cmaxes"].set_color("magenta")

    for pc in parts["bodies"]:
        pc.set_facecolor(fc)
        pc.set_edgecolor("black")
        pc.set_alpha(0.2)

    set_axis_style(ax, labels)
    ax.set_ylabel(yl)

    if t:
        N = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        plt.plot(x, (2.0 * N - 1) / N, "r")
    plt.savefig(fname="plots/violin_" + label)


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(r"$N$")


def plot_as_func_of_N(P=np.array([]), N=14, peak_flux=2, peak_flux2=100):
    if N == 5:
        P = np.delete(P, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 6:
        P = np.delete(P, [1, 3, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 6, 7, 8, 11, 12], axis=1)
    if N == 7:
        P = np.delete(P, [1, 3, 5, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 8, 11, 12], axis=1)
    if N == 8:
        P = np.delete(P, [1, 3, 5, 7, 11, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 11, 12], axis=1)
    if N == 9:
        P = np.delete(P, [1, 3, 5, 7, 12], axis=0)
        P = np.delete(P, [1, 3, 5, 7, 12], axis=1)
    if N == 10:
        P = np.delete(P, [1, 3, 5, 7], axis=0)
        P = np.delete(P, [1, 3, 5, 7], axis=1)
    if N == 11:
        P = np.delete(P, [3, 5, 7], axis=0)
        P = np.delete(P, [3, 5, 7], axis=1)
    if N == 12:
        P = np.delete(P, [5, 7], axis=0)
        P = np.delete(P, [5, 7], axis=1)
    if N == 13:
        P = np.delete(P, [7], axis=0)
        P = np.delete(P, [7], axis=1)
    if N == 4:
        P = np.delete(P, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=0)
        P = np.delete(P, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12], axis=1)


    r_ampl = [np.array([]), np.array([]), np.array([])]
    r_size = [np.array([]), np.array([]), np.array([])]
    r_flux = [np.array([]), np.array([]), np.array([])]
    r_counter = np.array([0, 0, 0])

    r_ampl2 = [np.array([]), np.array([]), np.array([])]
    r_size2 = [np.array([]), np.array([]), np.array([])]
    r_flux2 = [np.array([]), np.array([]), np.array([])]
    r_counter2 = [np.array([]), np.array([]), np.array([])]

    r_phi = [np.array([]), np.array([]), np.array([])]
    r_phi2 = [np.array([]), np.array([]), np.array([])]
    curves = [np.array([]), np.array([]), np.array([])]

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if j > i:
                name = (
                    "data/10_baseline/"
                    + str(N)
                    + "_10_baseline_"
                    + str(i)
                    + "_"
                    + str(j)
                    + "_"
                    + str(P[i, j])
                    + ".p"
                )

                pkl_file = open(name, "rb")
                g_pq12 = pickle.load(pkl_file)
                r_pq12 = pickle.load(pkl_file)
                g_pq_inv12 = pickle.load(pkl_file)
                g_pq_t12 = pickle.load(pkl_file)
                delta_u = pickle.load(pkl_file)
                s_size = pickle.load(pkl_file)
                siz = pickle.load(pkl_file)
                r = pickle.load(pkl_file)
                
                pkl_file.close()

                B = 2 * np.pi * (s_size * (np.pi / 180)) ** 2
                fwhm = (
                    (s_size * (np.pi / 180) * 2 * np.sqrt(2 * np.log(2)))
                    * (180 / np.pi)
                    * 3600
                )
                x = np.linspace(-1 * siz, siz, len(g_pq12))

                c1 = cut(
                    img(np.absolute(g_pq12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                )
                c2 = cut(
                    img(np.absolute(g_pq_t12 ** (-1) * r_pq12) * B, delta_u, delta_u)
                )
                c3 = cut(img(np.absolute(g_pq_inv12 * r_pq12) * B, delta_u, delta_u))

                curves[0] = c1
                curves[1] = c2
                curves[2] = c3

                store = False
                if (
                    np.max(curves[0]) < peak_flux
                    and np.max(curves[1]) < peak_flux
                    and np.max(curves[2]) < peak_flux
                ):
                    store = True

                for c in range(len(curves)):
                    if np.max(curves[c]) < peak_flux:
                        r_ampl[c] = np.append(r_ampl[c], [np.max(curves[c])])
                        idx = curves[c] >= (np.max(curves[c]) / 2)
                        integer_map = map(int, idx)
                        r_size[c] = np.append(
                            r_size[c], [np.sum(list(integer_map)) * r]
                        )  # in arcseconds
                        r_flux[c] = np.append(
                            r_flux[c],
                            np.max(curves[c])
                            * (
                                (
                                    ((np.sum(list(integer_map)) * r) / 3600.0)
                                    * (np.pi / 180)
                                )
                                / (2 * np.sqrt(2 * np.log(2)))
                            )
                            ** 2
                            * 2
                            * np.pi,
                        )
                        r_counter[c] += 1
                        r_phi[c] = np.append(r_phi[c], [P[i, j]])
                    else:
                        if np.max(curves[c]) < peak_flux2:
                            r_ampl2[c] = np.append(r_ampl2[c], [np.max(curves[c])])
                            idx = curves[c] >= (np.max(curves[c]) / 2)
                            integer_map = map(int, idx)
                            r_size2[c] = np.append(
                                r_size2[c], [np.sum(list(integer_map)) * r]
                            )  # in arcseconds
                            r_flux2[c] = np.append(
                                r_flux2[c],
                                np.max(curves[c])
                                * (
                                    (
                                        ((np.sum(list(integer_map)) * r) / 3600.0)
                                        * (np.pi / 180)
                                    )
                                    / (2 * np.sqrt(2 * np.log(2)))
                                )
                                ** 2
                                * 2
                                * np.pi,
                            )
                            r_counter2[c] += 1
                            r_phi2[c] = np.append(r_phi2[c], [P[i, j]])

    return (
        curves,
        x,
        cut(img(np.absolute(r_pq12), delta_u, delta_u)),
        r_ampl,
        r_size,
        r_flux,
        r_counter,
        r_ampl2,
        r_size2,
        r_flux2,
        r_counter2,
        B,
        fwhm,
        r_phi,
        r_phi2,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=int,
        nargs="+",
        default=[0, 1],
        help="The baseline to calculate on",
    )

    parser.add_argument(
        "--performExp",
        action=argparse.BooleanOptionalAction,
        help="Re-run all baselines if true, if false run from existing files",
    )

    # parser.add_argument(
    #     "--performExp",
    #     type=bool,
    #     default=True,
    #     help="Re-run all baselines if true, if false run from existing files",
    # )

    global args
    args = parser.parse_args()
    t = T_ghost()

    baseline = np.array(args.baseline)
    phi = 4 * np.array(
        [
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),
            (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75),
            (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75),
            (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75),
            (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),
            (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75),
            (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75),
            (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),
            (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),
            (-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),
            (
                -9.25,
                -8.25,
                -7.25,
                -6.25,
                -5.25,
                -4.25,
                -3.25,
                -2.25,
                -1.25,
                -0.25,
                0,
                0.5,
                9,
                9.5,
            ),
            (
                -9.75,
                -8.75,
                -7.75,
                -6.75,
                -5.75,
                -4.75,
                -3.75,
                -2.75,
                -1.75,
                -0.75,
                -0.5,
                0,
                8.5,
                9,
            ),
            (
                18.25,
                -17.25,
                -16.25,
                -15.25,
                -14.25,
                -13.25,
                -12.25,
                -11.25,
                -10.25,
                -9.25,
                -9,
                -8.5,
                0,
                0.5,
            ),
            (
                -18.75,
                -17.75,
                -16.75,
                -15.75,
                -14.75,
                -13.75,
                -12.75,
                -11.75,
                -10.75,
                -9.75,
                -9.5,
                -9,
                -0.5,
                0,
            ),
        ]
    )

    if args.performExp:
        print("Running Experiment")
        print()
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=14)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=13)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=12)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=11)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=10)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=9)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=8)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=7)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=6)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=5)
        another_exp(phi, size_gauss=0.02, K1=30.0, K2=4.0, N=4)

        every_baseline(phi, 5000, 1)
    else:
        print("Saving images")
        process_pickle_files_g(phi=phi)
        process_pickle_files_g2(phi=phi)
        m, amp_matrix = compute_division_matrix(
            P=phi, N=14, peak_flux=2, peak_flux2=100
        )
        main_phi_plot(P=phi, m=m)
        n = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        peak = np.zeros((len(n), 3), dtype=float)
        width = np.zeros((len(n), 3), dtype=float)
        flux = np.zeros((len(n), 3), dtype=float)

        for k in range(len(n)):
            peak[k, :], width[k, :], flux[k, :] = get_average_response(
                P=phi, N=n[k], peak_flux=100
            )

        c = ["r", "b", "g"]
        for i in range(3):

            plt.semilogy(n, peak[:, i], c[i])
            plt.semilogy(n, width[:, i], c[i] + "--")
            plt.semilogy(n, flux[:, i], c[i] + ":")

        plt.xlabel(r"$N$")
        plt.ylabel(r"$c$ or FWHM/FWHM$_B$ or Flux [Jy]")
        plt.legend()
        plt.savefig(fname="semilogy")

        n = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        A1 = []
        A2 = []
        A3 = []

        S1 = []
        S2 = []
        S3 = []

        F1 = []
        F2 = []
        F3 = []

        P1 = []
        P2 = []
        P3 = []

        for k in range(len(n)):
            (
                curves,
                x,
                r,
                A,
                S,
                F,
                C,
                A_2,
                S_2,
                F_2,
                C_2,
                B,
                FW,
                PM,
                PM_2,
            ) = plot_as_func_of_N(P=phi, N=n[k], peak_flux=100, peak_flux2=100)
            A1.append(A[0])
            A2.append(A[1])
            A3.append(A[2])
            S1.append(S[0] / FW)
            S2.append(S[1] / FW)
            S3.append(S[2] / FW)
            F1.append(F[0] / B)
            F2.append(F[1] / B)
            F3.append(F[2] / B)

        create_violin_plot(data=A1, fc="red", yl="c", t=False, label="A1")
        create_violin_plot(data=A2, fc="blue", yl="c", t=False, label="A2")
        create_violin_plot(data=A3, fc="green", yl="c", t=True, label="A3")

        create_violin_plot(data=S1, fc="red", yl=r"FWHM/FWHM$_B$", label="S1")
        create_violin_plot(data=S2, fc="blue", yl=r"FWHM/FWHM$_B$", label="S2")
        create_violin_plot(data=S3, fc="green", yl=r"FWHM/FWHM$_B$", label="S3")

        create_violin_plot(data=F1, fc="red", yl=r"Flux [Jy]", label="F1")
        create_violin_plot(data=F2, fc="blue", yl=r"Flux [Jy]", label="F2")
        create_violin_plot(data=F3, fc="green", yl=r"Flux [Jy]", label="F3")
