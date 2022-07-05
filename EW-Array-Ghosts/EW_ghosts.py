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
                g_temp[p] = np.sum(np.conj(R[:, p]) * z) / \
                    (np.sum(np.conj(z) * z))

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
                            R += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1] * np.pi /
                                            180.0) + v_m * (s[2] * np.pi / 180.0)))
                        else:
                            sigma = s[3] * (np.pi / 180)
                            g_kernal = 2 * np.pi * sigma ** 2 * \
                                np.exp(-2 * np.pi ** 2 * sigma ** 2 * (u_m ** 2 + v_m ** 2))
                            R += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1] * np.pi /
                                            180.0) + v_m * (s[2] * np.pi / 180.0))) * g_kernal

                    for k in range(len(cal_sky_model)):
                        s = cal_sky_model[k]
                        if len(s) <= 3:
                            M += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1]
                                            * np.pi/180.0) + v_m * (s[2] * np.pi / 180.0)))
                        else:
                            sigma = s[3] * (np.pi / 180)
                            g_kernal = 2 * np.pi * sigma ** 2 * \
                                np.exp(-2 * np.pi ** 2 * sigma ** 2 *(u_m ** 2 + v_m ** 2))
                            M += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1] * np.pi /
                                            180.0) + v_m * (s[2] * np.pi / 180.0))) * g_kernal
                    g_stef, G = self.create_G_stef(
                        R, M, 200, 1e-8, temp, no_auto=False)

                    g_pq_t[i, j], g_pq_t_inv[i, j] = EW_theoretical_derivation.derive_from_theory(
                        true_sky_model[0][3], N, Phi, baseline[0], baseline[1], true_sky_model[0][0], ut, vt)

                    r_pq[j, i] = R[baseline[0], baseline[1]]
                    m_pq[j, i] = M[baseline[0], baseline[1]]
                    g_pq[j, i] = G[baseline[0], baseline[1]]
        
        lam = (1.0 * 3 * 10 ** 8) / f
        b_len = b0 * Phi[baseline[0], baseline[1]]
        fwhm = 1.02 * lam / (b_len)
        sigma_kernal = fwhm / (2 * np.sqrt(2 * np.log(2)))
        g_kernal = 2 * np.pi * sigma_kernal ** 2 * np.exp(-2 * np.pi ** 2 * sigma_kernal ** 2 * (uu ** 2 + vv ** 2))

        return r_pq, g_pq, g_pq_t, g_pq_t_inv, g_kernal, sigma_kernal,delta_u, delta_v, delta_l

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

        N = int(np.ceil(vis_s*2/resolution))

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
                        R += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1] * np.pi /
                                        180.0) + v_m * (s[2] * np.pi / 180.0)))
                    else:
                        sigma = s[3] * (np.pi / 180)
                        g_kernal = 2 * np.pi * sigma ** 2 * \
                            np.exp(-2 * np.pi ** 2 * sigma ** 2 * (u_m ** 2 + v_m ** 2))
                        R += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1] * np.pi /
                                        180.0) + v_m * (s[2] * np.pi / 180.0))) * g_kernal

                for k in range(len(cal_sky_model)):
                    s = cal_sky_model[k]
                    if len(s) <= 3:
                        M += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1]
                                        * np.pi/180.0) + v_m * (s[2] * np.pi / 180.0)))
                    else:
                        sigma = s[3] * (np.pi / 180)
                        g_kernal = 2 * np.pi * sigma ** 2 * \
                            np.exp(-2 * np.pi ** 2 * sigma ** 2 *(u_m ** 2 + v_m ** 2))
                        M += s[0] * np.exp(-2 * np.pi * 1j * (u_m * (s[1] * np.pi /
                                        180.0) + v_m * (s[2] * np.pi / 180.0))) * g_kernal
                g_stef, G = self.create_G_stef(
                    R, M, 200, 1e-8, temp, no_auto=False)

                g_pq_t[i], g_pq_t_inv[i], B = EW_theoretical_derivation.derive_from_theory_linear(
                    true_sky_model[0][3], N, Phi, baseline[0], baseline[1], true_sky_model[0][0], ut, vt)

                r_pq[i] = R[baseline[0], baseline[1]]
                m_pq[i] = M[baseline[0], baseline[1]]
                g_pq[i] = G[baseline[0], baseline[1]]
        
        lam = (1.0*3*10**8) / f
        b_len = b0 * Phi[baseline[0], baseline[1]]
        fwhm = 1.02 * lam / (b_len)
        sigma_kernal = fwhm / (2 * np.sqrt(2 * np.log(2)))
        g_kernal = 2 * np.pi * sigma_kernal ** 2 * np.exp(-2 * np.pi ** 2 * sigma_kernal ** 2 * (u ** 2))

        return r_pq, g_pq, g_pq_t, g_pq_t_inv, g_kernal, sigma_kernal, u, B


def another_exp(phi, size_gauss = 0.02, K1 = 30.0, K2 = 3.0, N = 4):
    s_size = size_gauss #size of Gaussian in degrees
    r = (s_size * 3600) / (1.0 * K1) #resolution
    siz = s_size * (K2 * 1.0)

    if (N == 5):
        phi = np.delete(phi,[1,3,4,5,6,7,8,11,12],axis=0)
        phi = np.delete(phi,[1,3,4,5,6,7,8,11,12],axis=1)
    if (N == 6):
        phi = np.delete(phi,[1,3,5,6,7,8,11,12],axis=0)
        phi = np.delete(phi,[1,3,5,6,7,8,11,12],axis=1)
    if (N == 7):
        phi = np.delete(phi,[1,3,5,7,8,11,12],axis=0)
        phi = np.delete(phi,[1,3,5,7,8,11,12],axis=1)
    if (N == 8):
        phi = np.delete(phi,[1,3,5,7,11,12],axis=0)
        phi = np.delete(phi,[1,3,5,7,11,12],axis=1)
    if (N == 9):
        phi = np.delete(phi,[1,3,5,7,12],axis=0)
        phi = np.delete(phi,[1,3,5,7,12],axis=1)
    if (N == 10):
        phi = np.delete(phi,[1,3,5,7],axis=0)
        phi = np.delete(phi,[1,3,5,7],axis=1)
    if (N == 11):
        phi = np.delete(phi,[3,5,7],axis=0)
        phi = np.delete(phi,[3,5,7],axis=1)
    if (N == 12):
        phi = np.delete(phi,[5,7],axis=0)
        phi = np.delete(phi,[5,7],axis=1)
    if (N == 13):
        phi = np.delete(phi,[7],axis=0)
        phi = np.delete(phi,[7],axis=1)
    if (N == 4):
        phi = np.delete(phi,[1,2,3,4,5,6,7,8,11,12],axis=0)
        phi = np.delete(phi,[1,2,3,4,5,6,7,8,11,12],axis=1)
    every_baseline(phi, s_size, r, siz, K1, K2, N)

def every_baseline(phi, s_size = 0.02, r = 30.0, siz = 3.0, K1=None, K2=None, N=None):
    
    mp.freeze_support()

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(mp.cpu_count(), initargs=(mp.RLock(),), initializer=tqdm.set_lock)
    signal.signal(signal.SIGINT, original_sigint_handler)
    m = mp.Manager()
    shared_array = m.dict()
    pid = 0
    try:
        counter = 0
        for k in range(len(phi)):
            for j in range(len(phi)):
                time.sleep(.1)
                if (len(shared_array) >= mp.cpu_count()):
                    pids = [pid for pid, running in shared_array.items() if not running]
                    while (len(pids) == 0):
                        time.sleep(.1)
                        pids = [pid for pid, running in shared_array.items() if not running]

                    pid = shared_array.keys().index(pids[0])
                res = pool.starmap_async(process_baseline, [(k, j, phi, siz, r, s_size, shared_array, pid, counter, K1, K2, N)])
                # process_baseline(k, j, phi, siz, r, s_size, shared_array, pid, counter, K1, K2, N)
                pid += 1
                counter += 1
               
    except KeyboardInterrupt:
        print("CTRL+C")
    except Exception as e:
        print(e)
        f = open("crashReport.txt", "a")
        f.write("Crash report: " + e)
        f.close()  
    finally:
        if (not res.get()[0]):
            print(res.get())

    while True:
        time.sleep(5)
        pids = [pid for pid, running in shared_array.items() if running]
        
        if (len(pids) == 0):
            print("Program finished")
            pool.terminate()
            pool.join()
            break


def process_baseline(k, j, phi, siz, r, s_size, shared_array, pid, counter, K1=None, K2=None, N=None):
    try:
        shared_array[pid] = True
        if j > k:
            baseline = [k, j]
            true_sky_model=np.array([[1.0, 0, 0, s_size]])
            cal_sky_model=np.array([[1, 0, 0]])

            t = T_ghost()

            file_name = ""
            if (N is not None):
                file_name = "data/10_baseline/" + str(N) + "_10_baseline_" + str(counter) + "_" + str(k) + "_" + str(j) + "_" + str(phi[k, j]) + ".p"
            else:
                file_name = "data/G/g_" + str(k) + "_" + str(counter) + "_" + str(j) + "_" + str(phi[k,j]) + ".png"

            if (N is not None):
                r_pq, g_pq, g_pq_t, g_pq_t_inv, g_kernal, sigma_kernal, delta_u, delta_v, delta_l= t.extrapolation_function(
                    baseline=baseline,
                    true_sky_model=true_sky_model,
                    cal_sky_model=cal_sky_model,
                    Phi=phi,
                    image_s=siz,
                    s=1,
                    resolution=r,
                    pid=pid)
                f = open(file_name, 'wb')
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
                if (K1 is not None):
                    pickle.dump(K1, f)
                if (K2 is not None):
                    pickle.dump(K2, f)
                f.close()  
            else:
                r_pq, g_pq, g_pq_t, g_pq_t_inv, g_kernal, sigma_kernal, u, B= t.extrapolation_function_linear(
                    baseline=baseline,
                    true_sky_model=true_sky_model,
                    cal_sky_model=cal_sky_model,
                    Phi=phi,
                    vis_s=s_size,
                    resolution=r,
                    pid=pid)
                f = open(file_name[:-2], 'wb')
                pickle.dump(g_pq_t,f)
                pickle.dump(g_pq,f)
                pickle.dump(g_pq_t_inv,f)
                pickle.dump(r_pq,f)
                pickle.dump(g_kernal,f)
                pickle.dump(sigma_kernal,f)
                pickle.dump(B,f)
                if (K1 is not None):
                    pickle.dump(K1, f)
                if (K2 is not None):
                    pickle.dump(K2, f)
                f.close()
                plt.clf()
                plt.plot(u,np.absolute(g_pq_t**(-1)*r_pq),"b")
                plt.plot(u,np.absolute(g_pq**(-1)*r_pq),"r")
                plt.savefig(file_name)
    except Exception as e:
        print(e) 
        f = open("crashReport.txt", "a")
        f.write("Crash report: " + e)
        f.close()     
    finally:
        shared_array[pid] = False
        return not (j > k)

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
        "--performExp", type=bool, default=True, help="Re-run all baselines if true, if false run from existing files")


    global args
    args = parser.parse_args()
    t = T_ghost()

    baseline = np.array(args.baseline)
    phi = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),
                    (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), 
                    (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), 
                    (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), 
                    (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),
                    (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), 
                    (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), 
                    (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),
                    (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75), 
                    (-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75), 
                    (-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5), 
                    (-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9), 
                    (18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), 
                    (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75, -9.5, -9, -0.5, 0)])
     
    if args.performExp:
        print("Running Experiment")
        print()
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 14)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 13)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 12)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 11)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 10)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 9)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 8)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 7)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 6)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 5)
        another_exp(phi,size_gauss=0.02,K1=30.0,K2=4.0, N = 4)

        every_baseline(phi, 5000, 1)
    else:
        pass
