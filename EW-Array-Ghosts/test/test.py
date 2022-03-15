import numpy as np
import pylab as plt
import pickle
import sys

import scipy.special
from scipy import optimize
import matplotlib as mpl
import argparse

"""
This class produces the theoretical ghost patterns of a simple two source case. It is based on a very simple EW array layout.
EW-layout: (0)---3---(1)---2---(2)
 
"""


class T_ghost():
    """
    This function initializes the theoretical ghost object
    """

    def __init__(self):
        pass

    def create_G_stef(self, R, M, imax, tau, temp, no_auto):
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

            if (k % 2 == 0):
                if (np.sqrt(np.sum(np.absolute(g_temp - g_old) ** 2)) / np.sqrt(
                        np.sum(np.absolute(g_temp) ** 2)) <= tau):
                    break
                else:
                    g_temp = (g_temp + g_old) / 2

        G_m = np.dot(np.diag(g_temp), temp)
        G_m = np.dot(G_m, np.diag(g_temp.conj()))

        g = g_temp
        G = G_m

        return g, G

    def c_func(self, r, s, p, q, A, B, N):
        answer = 0
        if ((r == p) and (s == q) and (r != s)):
            answer = (2*A*B*1.0)/N - (A*B*1.0)/(N**2)
        elif ((r == p) and (s != q) and (r != s)):
            answer = (A*B*1.0)/N - (A*B*1.0)/(N**2)
        elif ((r != p) and (s == q) and (r != s)):
            answer = (A*B*1.0)/N - (A*B*1.0)/(N**2)
        elif ((r != p) and (s != q) and (r != s)):
            answer = - (A*B*1.0)/(N**2)
        else:
            answer = 0
        return answer

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    """

    def theory_g(self, baseline=np.array([0, 1]), true_sky_model=np.array([[1, 0, 0, 0.1]]), cal_sky_model=np.array([[1, 0, 0]]), Phi=np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]), image_s=3, s=1, resolution=100):
        # temporary adding some constant values for the sky
        # l0 = 1 * (np.pi / 180)#temporary
        #m0 = 0 * (np.pi / 180)
        # print(true_sky_model)
        temp = np.ones(Phi.shape, dtype=complex)
        s_old = s

        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        delta_v = delta_u
        delta_l = resolution * (1.0 / 3600.0) * (np.pi / 180.0)
        delta_m = delta_l
        N = int(np.ceil(1 / (delta_l * delta_u))) + 1

        if (N % 2) == 0:
            N = N + 1

        delta_l_new = 1 / ((N - 1) * delta_u)
        delta_m_new = delta_l_new
        u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
        v = np.linspace(-(N - 1) / 2 * delta_v, (N - 1) / 2 * delta_v, N)
        l_cor = np.linspace(-1 / (2 * delta_u), 1 / (2 * delta_u), N)
        m_cor = np.linspace(-1 / (2 * delta_v), 1 / (2 * delta_v), N)
        uu, vv = np.meshgrid(u, v)
        u_dim = uu.shape[0]
        v_dim = uu.shape[1]
        #######################################################
        # EXTRACTING THE IMPORTANT PARAMETERS
        g_pq = np.zeros((u_dim, v_dim), dtype=complex)
        A = true_sky_model[0, 0]
        sigma = true_sky_model[0, 3]*(np.pi/180)
        B = 2*sigma**2*np.pi
        p = baseline[0]
        q = baseline[1]
        len_N = Phi.shape[0]

        for i in range(u_dim):
            for j in range(v_dim):
                ut = u[i]
                vt = v[j]
                for r in range(len_N):
                    for s in range(len_N):
                        if r != s:
                            c_pq = self.c_func(r, s, p, q, A, B, len_N)
                            A_pq = A*((Phi[p, q]**2*1.0)/Phi[r, s]**2)*c_pq
                            s_pq2 = ((Phi[p, q]**2*1.0)/Phi[r, s]**2)*sigma**2
                            g_pq[i, j] += A_pq*(2*np.pi*s_pq2) * \
                                np.exp(-2*np.pi**2*s_pq2*(ut**2+vt**2))

                #g_pq[i,j] += (A*B*1.0)/len_N

        # plt.imshow((g_pq.real))
        # plt.show()
        fig, ax = plt.subplots()
        im = ax.imshow(g_pq.real)
        fig.colorbar(im, ax=ax)
        plt.show()
        print(np.max(g_pq.real))
        return(g_pq.real[int(np.round(g_pq.shape[0]/2)), :])

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    """

    def extrapolation_function(self, baseline=np.array([0, 1]), true_sky_model=np.array([[1, 0, 0], [0.2, 1, 0]]), cal_sky_model=np.array([[1, 0, 0]]), Phi=np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]), image_s=3, s=1, resolution=100, kernel=True, type_plot="GT-1"):
        # temporary adding some constant values for the sky
        # l0 = 1 * (np.pi / 180)#temporary
        #m0 = 0 * (np.pi / 180)
        # print(true_sky_model)
        temp = np.ones(Phi.shape, dtype=complex)
        s_old = s

        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        delta_v = delta_u
        delta_l = resolution * (1.0 / 3600.0) * (np.pi / 180.0)
        delta_m = delta_l
        N = int(np.ceil(1 / (delta_l * delta_u))) + 1

        if (N % 2) == 0:
            N = N + 1

        delta_l_new = 1 / ((N - 1) * delta_u)
        delta_m_new = delta_l_new
        u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
        v = np.linspace(-(N - 1) / 2 * delta_v, (N - 1) / 2 * delta_v, N)
        l_cor = np.linspace(-1 / (2 * delta_u), 1 / (2 * delta_u), N)
        m_cor = np.linspace(-1 / (2 * delta_v), 1 / (2 * delta_v), N)
        uu, vv = np.meshgrid(u, v)
        u_dim = uu.shape[0]
        v_dim = uu.shape[1]
        #######################################################
        r_pq = np.zeros((u_dim, v_dim), dtype=complex)
        g_pq = np.zeros((u_dim, v_dim), dtype=complex)
        m_pq = np.zeros((u_dim, v_dim), dtype=complex)

        R = np.zeros(Phi.shape, dtype=complex)
        M = np.zeros(Phi.shape, dtype=complex)

        for i in range(u_dim):
            for j in range(v_dim):
                ut = u[i]
                vt = v[j]
                u_m = (Phi*ut)/(1.0*Phi[baseline[0], baseline[1]])
                v_m = (Phi*vt)/(1.0*Phi[baseline[0], baseline[1]])
                R = np.zeros(Phi.shape, dtype=complex)
                M = np.zeros(Phi.shape, dtype=complex)
                for k in range(len(true_sky_model)):
                    s = true_sky_model[k]
                    # print(s)
                    if len(s) <= 3:
                        # pass
                        R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]
                                         * np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                        sigma = s[3]*(np.pi/180)
                        g_kernal = 2*np.pi*sigma**2 * \
                            np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                        R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi /
                                         180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
                        #(2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                        # print(s[3])
                for k in range(len(cal_sky_model)):
                    s = cal_sky_model[k]
                    # print(s)
                    if len(s) <= 3:
                        # pass
                        M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]
                                         * np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                        sigma = s[3]*(np.pi/180)
                        g_kernal = 2*np.pi*sigma**2 * \
                            np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                        M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi /
                                         180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
                        #(2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                        # print(s[3])
                g_stef, G = self.create_G_stef(
                    R, M, 200, 1e-9, temp, no_auto=False)

                #R = np.exp(-2*np.pi*1j*(u_m*l0+v_m*m0))
                r_pq[j, i] = R[baseline[0], baseline[1]]
                m_pq[j, i] = M[baseline[0], baseline[1]]
                g_pq[j, i] = G[baseline[0], baseline[1]]

        #sigma = 0.05*(np.pi/180)
        #g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
        # plt.imshow(g_pq.real)
        # plt.show()
        # g_pq = g_pq-1#distilation
        # if type_plot == "GT-1":
        #   g_pq = (g_pq)**(-1)-1
        # else:
        #   g_pq = (g_pq)**(-1)*r_pq - r_pq

        # if kernel:
        #   g_pq = g_pq*g_kernal
        #g_pq = g_pq[:,::-1]
        # g_pq = g_pq#-1

        fig, ax = plt.subplots()
        im = ax.imshow(g_pq.real)
        fig.colorbar(im, ax=ax)
        plt.show()
        print(np.max(g_pq.real))
        return(g_pq.real[int(np.round(g_pq.shape[0]/2)), :])

        '''
        #IMAGING QUICKLY
        zz = g_pq
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

        zz_f = np.fft.fft2(zz) * (delta_u*delta_v)
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2), axis=0)
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2), axis=1)

        fig, ax = plt.subplots()
        #print(image_s)
        #print(s)
        im = ax.imshow(zz_f.real/0.2,extent=[-s_old*image_s,s_old*image_s,-s_old*image_s,s_old*image_s],vmin=-0.7)
        fig.colorbar(im, ax=ax) 
        plt.show()
        #print(R)
        
        #plt.imshow(R[0,1])
        #plt.show()
        #plt.imsho
        #for k in range(len(self.true_point_sources)):
        #            R = R + self.true_point_sources[k, 0] * np.exp(-2 * 1j * np.pi * (
        #                    u_t_m * self.true_point_sources[k, 1] + v_t_m * self.true_point_sources[k, 2])) * Gauss(
        #                self.true_point_sources[k, 3], u_t_m, v_t_m)

        #for i in range(u_dim)):
        #    for j in range(v_dim)):
        #print("hallo")
        '''


if __name__ == "__main__":
    # print('hallo')
    t = T_ghost()

    #print(t.c_func(0, 1, 1, 3, 5, 7, 9))

    g1 = t.theory_g(baseline=np.array([0, 1]), true_sky_model=np.array([[1, 0, 0, 0.5]]), cal_sky_model=np.array(
        [[1, 0, 0]]), Phi=np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]), image_s=3, s=1, resolution=100)
    g2 = t.extrapolation_function(baseline=np.array([0, 1]), true_sky_model=np.array([[1, 0, 0, 0.5]]), cal_sky_model=np.array(
        [[1, 0, 0]]), Phi=np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]), image_s=3, s=1, resolution=100)
    plt.plot(g1/np.max(g1), "r")
    plt.plot(g2/np.max(g2), "b")
    plt.show()

    # point source case GT-1
    # t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,type_plot="GT-1")

    # point source case GTR-R
    # t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,type_plot="GTR-R")

    # extended source case GT-1
    # t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0,0.1],[0.2,1,0,0.2]]),cal_sky_model=np.array([[1,0,0,0.1]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,type_plot="GT-1")

    # extended source case GTR-R
    # t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0,0.1],[0.2,1,0,0.2]]),cal_sky_model=np.array([[1,0,0,0.1]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=False,type_plot="GTR-R")
