import numpy as np
import pylab as plt
import pickle
import sys

import scipy.special
from scipy import optimize
import matplotlib as mpl
import argparse

"""
This class produces the theoretical ghost patterns of a simple two source case. It can handle any array layout.
It requires that you set up the array geometry matrices. Two have been setup by default. You have a simple
3 element interferometer and WSRT to choose from (you can add any other array that you wish).
"""


class T_ghost():
    """This function initializes the theoretical ghost object
       point_sources --- contain the two source model (see main for an example)
       antenna --- you can select a subset of the antennas, ignore some if you wish, "all" for all antennas, [1,2,3], for subset
       MS --- selects your antenna layout, "EW_EXAMPLE" or "WSRT"
    """

    def __init__(self,
                 true_point_sources=np.array([]),
                 model_point_sources=np.array([]),
                 antenna="",
                 MS=""):
        self.antenna = antenna
        self.true_point_sources = true_point_sources
        self.model_point_sources = model_point_sources
        self.A_1 = true_point_sources[0, 0]
        if len(true_point_sources) > 1:
            self.A_2 = true_point_sources[1, 0]
            self.l_0 = true_point_sources[1, 1]
            self.m_0 = true_point_sources[1, 2]
        else:
            self.A_2 = 0
            self.l_0 = 0
            self.m_0 = 0

        # Here you can add your own antenna layout, default is WSRT and EW-EXAMPLE
        if MS == "EW_EXAMPLE":
            self.ant_names = [0, 1, 2]
            self.a_list = self.get_antenna(self.antenna, self.ant_names)
            # The 3 geometry matrices of the simple three element example
            self.b_m = np.zeros((3, 3))
            self.theta_m = np.zeros((3, 3))
            self.phi_m = np.array([(0, 3, 5), (-3, 0, 2), (-5, -2, 0)])
            self.sin_delta = None
            self.wave = 3e8 / 1.45e9
            self.dec = np.pi / 2.0
        elif MS == "WSRT":  # traditional (36,108,1332,1404) WSRT configuration
            self.ant_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            # The 3 geometry matrices of WSRT
            self.a_list = self.get_antenna(self.antenna, self.ant_names)
            self.b_m = np.zeros((14, 14))
            self.theta_m = np.zeros((14, 14))
            self.phi_m = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),
                                       (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75),
                                       (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75),
                                       (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75),
                                       (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),
                                       (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75),
                                       (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75),
                                       (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),
                                       (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),
                                       (-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),
                                       (-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9,
                                        9.5),
                                       (-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0,
                                        8.5, 9),
                                       (-18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25,
                                        -9, -8.5, 0, 0.5),
                                       (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,
                                        -9.5, -9, -0.5, 0)])
            self.sin_delta = None
            self.wave = 3e8 / 1.45e9  # observational wavelenght
            self.dec = np.pi / 2.0  # declination
        elif MS == "KAT7":
            file_name = "./Pickle/KAT7_1445_1x16_12h_antnames.p"
            self.ant_names = pickle.load(open(file_name, "rb"))

            # print "ant_names = ",self.ant_names
            self.a_list = self.get_antenna(self.antenna, self.ant_names)
            # print "a_list = ",self.a_list

            file_name = "./Pickle/KAT7_1445_1x16_12h_phi_m.p"
            self.phi_m = pickle.load(open(file_name, "rb"))
            # print self.phi_m
            # print "self.phi_m = ",self.phi_m
            # self.phi_m =  pickle.load(open(MS[2:-4]+"_phi_m.p","rb"))

            file_name = "./Pickle/KAT7_1445_1x16_12h_b_m.p"
            self.b_m = pickle.load(open(file_name, "rb"))
            # print "self.b_m = ",self.b_m
            # self.b_m = pickle.load(open(MS[2:-4]+"_b_m.p","rb"))

            file_name = "./Pickle/KAT7_1445_1x16_12h_theta_m.p"
            self.theta_m = pickle.load(open(file_name, "rb"))
            # print "self.theta_m = ",self.theta_m
            # self.theta_m = pickle.load(open(MS[2:-4]+"_theta_m.p","rb"))

            self.sin_delta = None

            file_name = "./Pickle/KAT7_1445_1x16_12h_wave.p"
            self.wave = pickle.load(open(file_name, "rb"))
            file_name = "./Pickle/KAT7_1445_1x16_12h_declination.p"
            self.dec = pickle.load(open(file_name, "rb"))
            # print "self.dec = ",self.dec
            # self.dec = np.pi/2.0 #declination
        elif MS == "LOFAR":
            file_name = "./Pickle/L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.calibrated_antnames.p"
            self.ant_names = pickle.load(open(file_name, "rb"))

            # print "ant_names = ",self.ant_names
            self.a_list = self.get_antenna(self.antenna, self.ant_names)
            # print "a_list = ",self.a_list

            file_name = "./Pickle/L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.calibrated_phi_m.p"
            self.phi_m = pickle.load(open(file_name, "rb"))
            # self.phi_m =  pickle.load(open(MS[2:-4]+"_phi_m.p","rb"))

            file_name = "./Pickle/L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.calibrated_b_m.p"
            self.b_m = pickle.load(open(file_name, "rb"))
            # self.b_m = pickle.load(open(MS[2:-4]+"_b_m.p","rb"))

            file_name = "./Pickle/L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.calibrated_theta_m.p"
            self.theta_m = pickle.load(open(file_name, "rb"))
            # self.theta_m = pickle.load(open(MS[2:-4]+"_theta_m.p","rb"))

            self.sin_delta = None

            file_name = "./Pickle/L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.calibrated_wave.p"
            self.wave = pickle.load(open(file_name, "rb"))
            self.dec = np.pi / 2.0  # declination

    """Function processes your antenna selection string
    """

    def get_antenna(self, ant, ant_names):
        if isinstance(ant[0], int):
            return np.array(ant)
        if ant == "all":
            return np.arange(len(ant_names))
        new_ant = np.zeros((len(ant),))
        for k in range(len(ant)):
            for j in range(len(ant_names)):
                if (ant_names[j] == ant[k]):
                    new_ant[k] = j
        return new_ant

    """Function calculates your delete list
    """

    def calculate_delete_list(self):
        if self.antenna == "all":
            return np.array([])
        d_list = list(range(self.phi_m.shape[0]))
        for k in range(len(self.a_list)):
            d_list.remove(self.a_list[k])
        return d_list

    '''Unpolarized direction independent phase-only calibration entails finding the G(theta) that minimizes ||R-GMG^H||. 
    This function evaluates R-G(theta)MG^H(theta).
    theta is a vactor containing the phase of the antenna gains.
    r is a vector containing a vecotrized R (observed visibilities), real and imaginary
    m is a vector containing a vecotrized M (predicted), real and imaginary   
    '''

    def err_func_theta(self, theta, r, m):
        Nm = len(r) / 2
        N = len(theta)
        G = np.diag(np.exp(-1j * theta))
        R = np.reshape(r[0:Nm], (N, N)) + np.reshape(r[Nm:], (N, N)) * 1j
        M = np.reshape(m[0:Nm], (N, N)) + np.reshape(m[Nm:], (N, N)) * 1j
        T = np.dot(G, M)
        T = np.dot(T, G.conj())
        Y = R - T
        y_r = np.ravel(Y.real)
        y_i = np.ravel(Y.imag)
        y = np.hstack([y_r, y_i])
        return y

    '''This function finds argmin phase ||R-G(phase)MG(phase)^H|| using Levenberg-Marquardt. It uses the optimize.
    leastsq scipy to perform
    the actual minimization
    R is your observed visibilities matrx
    M is your predicted visibilities
    g the antenna gains
    G = gg^H'''

    def create_G_LM_phase_only(self, R, M):
        N = R.shape[0]
        temp = np.ones((R.shape[0], R.shape[1]), dtype=complex)
        G = np.zeros(R.shape, dtype=complex)
        theta = np.zeros((R.shape[0],), dtype=float)

        theta_0 = np.zeros((N,))
        r_r = np.ravel(R.real)
        r_i = np.ravel(R.imag)
        r = np.hstack([r_r, r_i])
        m_r = np.ravel(M.real)
        m_i = np.ravel(M.imag)
        m = np.hstack([m_r, m_i])
        theta_lstsqr_temp = optimize.leastsq(self.err_func_theta, theta_0, args=(r, m))
        theta_lstsqr = theta_lstsqr_temp[0]

        G_m = np.dot(np.diag(np.exp(-1j * theta_lstsqr)), temp)
        G_m = np.dot(G_m, np.diag(np.exp(-1j * theta_lstsqr)).conj())

        theta = theta_lstsqr
        G = G_m

        return theta, G

    def create_G_GN1_phase_only(self, R, M):
        G = np.zeros(R.shape, dtype=complex)
        theta_v = np.zeros((R.shape[0],), dtype=float)

        Rt = R - R * np.eye(R.shape[0])

        for k in range(R.shape[0]):
            theta_v[k] = -1 * (np.sum(Rt[k, :])).imag / (R.shape[0] - 1)

        for k in range(R.shape[0]):
            for j in range(R.shape[1]):
                G[k, j] = 1 + (-1) * (1j) * (theta_v[k] - theta_v[j])

        return theta_v, G

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
                g_temp[p] = np.sum(np.conj(R[:, p]) * z) / (np.sum(np.conj(z) * z))

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

    def create_G_stef_norm1(self, R, M, imax, tau, no_auto):
        N = R.shape[0]
        if no_auto:
            R = R - R * np.eye(R.shape[0])
            M = M - M * np.eye(M.shape[0])
        temp = np.ones((R.shape[0], R.shape[1]), dtype=complex)
        G = np.zeros(R.shape, dtype=complex)
        g = np.zeros((R.shape[0],), dtype=complex)

        g_temp = np.ones((N,), dtype=complex)
        for i in range(imax):
            g_old = np.copy(g_temp)
            for p in range(N):
                z = g_old * M[:, p]
                g_temp[p] = np.sum(np.conj(R[:, p]) * z) / (np.sum(np.conj(z) * z))
            if (i % 2 == 0):
                g_temp = (g_temp + g_old) / 2
            g_temp = g_temp / np.absolute(g_temp)
            if (np.sqrt(np.sum(np.absolute(g_temp - g_old) ** 2)) / np.sqrt(np.sum(np.absolute(g_temp) ** 2)) <= tau):
                break
        print("Norm i = ", i)
        G_m = np.dot(np.diag(g_temp), temp)
        G_m = np.dot(G_m, np.diag(g_temp.conj()))

        g = g_temp
        G = G_m

        return g, G

    def create_G_stef_norm2(self, R, M, imax, tau, no_auto):
        N = R.shape[0]
        if no_auto:
            R = R - R * np.eye(R.shape[0])
            M = M - M * np.eye(M.shape[0])
        temp = np.ones((R.shape[0], R.shape[1]), dtype=complex)
        G = np.zeros(R.shape, dtype=complex)
        g = np.zeros((R.shape[0],), dtype=complex)

        g_temp = np.ones((N,), dtype=complex)
        for i in range(imax):
            g_old = np.copy(g_temp)
            for p in range(N):
                z = g_old * M[:, p]
                g_temp[p] = np.sum(np.conj(R[:, p]) * z) / (np.sum(np.conj(z) * z))
            if (i % 2 == 0):
                if (np.sqrt(np.sum(np.absolute(g_temp - g_old) ** 2)) / np.sqrt(
                        np.sum(np.absolute(g_temp) ** 2)) <= tau):
                    break
                else:
                    g_temp = (g_temp + g_old) / 2
        g_temp = g_temp / np.absolute(g_temp)
        G_m = np.dot(np.diag(g_temp), temp)
        G_m = np.dot(G_m, np.diag(g_temp.conj()))

        g = g_temp
        G = G_m

        return g, G

    def create_G_phase_stef(self, R, M, imax, tau, no_auto):
        N = R.shape[0]
        if no_auto:
            R = R - R * np.eye(R.shape[0])
            M = M - M * np.eye(M.shape[0])
        temp = np.ones((R.shape[0], R.shape[1]), dtype=complex)

        phase = np.zeros((R.shape[0],), dtype=float)
        phase_delta = np.zeros((R.shape[0],), dtype=float)
        m = np.zeros((R.shape[0],), dtype=float)
        # model_inverse = ((np.sum(np.conj(M[:,p])*M[:,p])).real)**(-1)
        g = np.exp(-1j * phase)
        for i in range(imax):
            # print "i_p = ",i
            for p in range(N):
                if i == 0:
                    m[p] = ((np.sum(np.conj(M[:, p]) * M[:, p])).real) ** (-1)
                z = g * M[:, p]
                phase_delta[p] = (-1 * np.conj(g[p]) * np.sum(np.conj(R[:, p]) * z)).imag * m[p]

            if (i % 2 == 0):
                phase_new = phase + phase_delta
            else:
                phase_new = phase + phase_delta / 2.

            if (np.sqrt(np.sum(np.absolute(phase_new - phase) ** 2)) / np.sqrt(
                    np.sum(np.absolute(phase_new) ** 2)) <= tau):
                break
            else:
                phase = phase_new
                g = np.exp(-1j * phase)
        # print "Phase i = ",i
        G = np.dot(np.diag(g), temp)
        G = np.dot(G, np.diag(g.conj()))
        return phase, G

    """Generates your extrapolated visibilities 2
    baseline --- which baseline do you whish to image [0,1], selects baseline 01
    u,v --- if you do not want to create extrapolated visibilities you can give an actual uv-track here
    resolution --- the resolution of the pixels in your final image in arcseconds
    image_s --- the final extend of your image in degrees
    wave --- observational wavelength in meters
    dec --- declination of your observation
    """

    # resolution --- arcsecond, image_s --- degrees
    def visibilities_pq_2D_LM(self, baseline, u=None, v=None, resolution=0, image_s=0, s=0, wave=None, dec=None,
                              algo="STEFCAL", no_auto=True, sigma=0):
        if wave == None:
            wave = self.wave
        if dec == None:
            dec = self.dec
        # SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline, self.ant_names)
        # print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        # print "d_list = ",d_list

        phi = self.phi_m[b_list[0], b_list[1]]
        delta_b = (self.b_m[b_list[0], b_list[1]] / wave) * np.cos(dec)
        theta = self.theta_m[b_list[0], b_list[1]]

        p = np.ones(self.phi_m.shape, dtype=int)
        p = np.cumsum(p, axis=0) - 1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p, d_list, axis=0)
            p_new = np.delete(p_new, d_list, axis=1)
            q_new = np.delete(q, d_list, axis=0)
            q_new = np.delete(q_new, d_list, axis=1)

            phi_new = np.delete(self.phi_m, d_list, axis=0)
            phi_new = np.delete(phi_new, d_list, axis=1)

            b_new = np.delete(self.b_m, d_list, axis=0)
            b_new = np.delete(b_new, d_list, axis=1)

            b_new = (b_new / wave) * np.cos(dec)

            theta_new = np.delete(self.theta_m, d_list, axis=0)
            theta_new = np.delete(theta_new, d_list, axis=1)
        #####################################################

        # print "theta_new = ",theta_new
        # print "b_new = ",b_new
        # print "phi_new = ",phi_new
        # print "delta_sin = ",self.sin_delta

        # print "phi = ",phi
        # print "delta_b = ",delta_b
        # print "theta = ",theta*(180/np.pi)

        if u != None:
            u_dim1 = len(u)
            u_dim2 = 1
            uu = u
            vv = v
            l_cor = None
            m_cor = None
        else:
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
            u_dim1 = uu.shape[0]
            u_dim2 = uu.shape[1]
            #######################################################

        # DO CALIBRATION
        #######################################################

        V_R_pq = np.zeros(uu.shape, dtype=complex)
        V_G_pq = np.zeros(uu.shape, dtype=complex)
        temp = np.ones(phi_new.shape, dtype=complex)

        for i in range(u_dim1):
            # print("u_dim1 = ", u_dim1)
            progress_bar(i, u_dim1)
            # print("i = ", i)
            for j in range(u_dim2):
                # print "u_dim1 = ",u_dim1
                # print "i = ",i
                # print("j = ",j)
                if u_dim2 != 1:
                    u_t = uu[i, j]
                    v_t = vv[i, j]
                else:
                    u_t = uu[i]
                    v_t = vv[i]
                # BASELINE CORRECTION (Single operation)
                #####################################################
                # ADDITION
                v_t = v_t - delta_b
                # SCALING
                u_t = u_t / phi
                v_t = v_t / (np.sin(dec) * phi)
                # ROTATION (Clockwise)
                u_t_r = u_t * np.cos(theta) + v_t * np.sin(theta)
                v_t_r = -1 * u_t * np.sin(theta) + v_t * np.cos(theta)
                # u_t_r = u_t
                # v_t_r = v_t
                # NON BASELINE TRANSFORMATION (NxN) operations
                #####################################################
                # ROTATION (Anti-clockwise)
                u_t_m = u_t_r * np.cos(theta_new) - v_t_r * np.sin(theta_new)
                v_t_m = u_t_r * np.sin(theta_new) + v_t_r * np.cos(theta_new)
                # u_t_m = u_t_r
                # v_t_m = v_t_r
                # SCALING
                u_t_m = phi_new * u_t_m
                v_t_m = phi_new * np.sin(dec) * v_t_m
                # ADDITION
                v_t_m = v_t_m + b_new

                # print "u_t_m = ",u_t_m
                # print "v_t_m = ",v_t_m

                # NB --- THIS IS WHERE YOU ADD THE NOISE
                # if scale == None:
                #   R = self.A_1 + self.A_2*np.exp(-2*1j*np.pi*(u_t_m*self.l_0+v_t_m*self.m_0))
                # else:
                #   R = self.A_1 + self.A_2*np.exp(-2*1j*np.pi*(u_t_m*self.l_0+v_t_m*self.m_0)) + np.random.normal(size=u_t_m.shape,scale=scale)
                # u_t_m[np.absolute(u_t_m)>4000] = 0
                # v_t_m[np.absolute(v_t_m)>4000] = 0
                R = np.zeros(u_t_m.shape)

                Gauss = lambda sigma, uu, vv: (2 * np.pi * sigma ** 2) * np.exp(
                    -2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                # print(Gauss(self.true_point_sources, 0, 0.05))

                for k in range(len(self.true_point_sources)):
                    R = R + self.true_point_sources[k, 0] * np.exp(-2 * 1j * np.pi * (
                            u_t_m * self.true_point_sources[k, 1] + v_t_m * self.true_point_sources[k, 2])) * Gauss(
                        sigma, u_t_m, v_t_m) #later sigma will be true_point_sources[k, 3]

                M = np.zeros(u_t_m.shape)
                for k in range(len(self.model_point_sources)):
                    M = M + self.model_point_sources[k, 0] * np.exp(-2 * 1j * np.pi * (
                            u_t_m * self.model_point_sources[k, 1] + v_t_m * self.model_point_sources[k, 2])) * Gauss(
                        sigma, u_t_m, v_t_m)

                if algo == "STEFCAL":
                    g_stef, G = self.create_G_stef(R, M, 200, 1e-9, temp, no_auto=no_auto)
                elif algo == "PHASE":
                    theta_calibration, G = self.create_G_LM_phase_only(np.copy(R), np.copy(M))
                elif algo == "PHASE_STEF":
                    g_stef, G = self.create_G_phase_stef(R, M, 200, 1e-9, no_auto=no_auto)
                elif algo == "PHASE_STEF_NORM1":
                    g_stef, G = self.create_G_stef_norm1(R, M, 200, 1e-9, no_auto=no_auto)
                elif algo == "PHASE_STEF_NORM2":
                    g_stef, G = self.create_G_stef_norm2(R, M, 200, 1e-9, no_auto=no_auto)
                else:
                    theta_calibration, G = self.create_G_GN1_phase_only(np.copy(R), np.copy(M))

                if self.antenna == "all":
                    if u_dim2 != 1:
                        V_R_pq[i, j] = R[b_list[0], b_list[1]]
                        V_G_pq[i, j] = G[b_list[0], b_list[1]]
                    else:
                        V_R_pq[i] = R[b_list[0], b_list[1]]
                        V_G_pq[i] = G[b_list[0], b_list[1]]
                else:
                    for k in range(p_new.shape[0]):
                        for l in range(p_new.shape[1]):
                            if (p_new[k, l] == b_list[0]) and (q_new[k, l] == b_list[1]):
                                if u_dim2 != 1:
                                    V_R_pq[i, j] = R[k, l]
                                    V_G_pq[i, j] = G[k, l]
                                else:
                                    V_R_pq[i] = R[k, l]
                                    V_G_pq[i] = G[k, l]

        return u, v, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor, m_cor

    """Generates your extrapolated visibilities
    baseline --- which baseline do you whish to image [0,1], selects baseline 01
    u,v --- if you do not want to create extrapolated visibilities you can give an actual uv-track here
    resolution --- the resolution of the pixels in your final image in arcseconds
    image_s --- the final extend of your image in degrees
    wave --- observational wavelength in meters
    dec --- declination of your observation
    approx --- if true we use Stefan Wijnholds approximation instead of ALS calibration
    scale --- standard deviation of your noise (very simplistic noise, please make rigorous by yourself) 
    """

    # resolution --- arcsecond, image_s --- degrees
    def visibilities_pq_2D(self, baseline, u=None, v=None, resolution=0, image_s=0, s=0, wave=None, dec=None,
                           approx=False, scale=None):
        if wave == None:
            wave = self.wave
        if dec == None:
            dec = self.dec
        # SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline, self.ant_names)
        # print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        # print "d_list = ",d_list

        phi = self.phi_m[b_list[0], b_list[1]]
        delta_b = (self.b_m[b_list[0], b_list[1]] / wave) * np.cos(dec)
        theta = self.theta_m[b_list[0], b_list[1]]

        p = np.ones(self.phi_m.shape, dtype=int)
        p = np.cumsum(p, axis=0) - 1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p, d_list, axis=0)
            p_new = np.delete(p_new, d_list, axis=1)
            q_new = np.delete(q, d_list, axis=0)
            q_new = np.delete(q_new, d_list, axis=1)

            phi_new = np.delete(self.phi_m, d_list, axis=0)
            phi_new = np.delete(phi_new, d_list, axis=1)

            b_new = np.delete(self.b_m, d_list, axis=0)
            b_new = np.delete(b_new, d_list, axis=1)

            b_new = (b_new / wave) * np.cos(dec)

            theta_new = np.delete(self.theta_m, d_list, axis=0)
            theta_new = np.delete(theta_new, d_list, axis=1)
        #####################################################

        # print "theta_new = ",theta_new
        # print "b_new = ",b_new
        # print "phi_new = ",phi_new
        # print "delta_sin = ",self.sin_delta

        # print "phi = ",phi
        # print "delta_b = ",delta_b
        # print "theta = ",theta*(180/np.pi)

        if u != None:
            u_dim1 = len(u)
            u_dim2 = 1
            uu = u
            vv = v
            l_cor = None
            m_cor = None
        else:
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
            u_dim1 = uu.shape[0]
            u_dim2 = uu.shape[1]
            #######################################################

        # DO CALIBRATION
        #######################################################

        V_R_pq = np.zeros(uu.shape, dtype=complex)
        V_G_pq = np.zeros(uu.shape, dtype=complex)
        temp = np.ones(phi_new.shape, dtype=complex)

        for i in range(u_dim1):
            for j in range(u_dim2):
                if u_dim2 != 1:
                    u_t = uu[i, j]
                    v_t = vv[i, j]
                else:
                    u_t = uu[i]
                    v_t = vv[i]
                # BASELINE CORRECTION (Single operation)
                #####################################################
                # ADDITION
                v_t = v_t - delta_b
                # SCALING
                u_t = u_t / phi
                v_t = v_t / (np.absolute(np.sin(dec)) * phi)
                # ROTATION (Clockwise)
                u_t_r = u_t * np.cos(theta) + v_t * np.sin(theta)
                v_t_r = -1 * u_t * np.sin(theta) + v_t * np.cos(theta)
                # u_t_r = u_t
                # v_t_r = v_t
                # NON BASELINE TRANSFORMATION (NxN) operations
                #####################################################
                # ROTATION (Anti-clockwise)
                u_t_m = u_t_r * np.cos(theta_new) - v_t_r * np.sin(theta_new)
                v_t_m = u_t_r * np.sin(theta_new) + v_t_r * np.cos(theta_new)
                # u_t_m = u_t_r
                # v_t_m = v_t_r
                # SCALING
                u_t_m = phi_new * u_t_m
                v_t_m = phi_new * np.absolute(np.sin(dec)) * v_t_m
                # ADDITION
                v_t_m = v_t_m + b_new

                # print "u_t_m = ",u_t_m
                # print "v_t_m = ",v_t_m

                # NB --- THIS IS WHERE YOU ADD THE NOISE
                if scale == None:
                    R = self.A_1 + self.A_2 * np.exp(-2 * 1j * np.pi * (u_t_m * self.l_0 + v_t_m * self.m_0))
                else:
                    R = self.A_1 + self.A_2 * np.exp(
                        -2 * 1j * np.pi * (u_t_m * self.l_0 + v_t_m * self.m_0)) + np.random.normal(size=u_t_m.shape,
                                                                                                    scale=scale)

                if not approx:
                    d, Q = np.linalg.eigh(R)
                    D = np.diag(d)
                    Q_H = Q.conj().transpose()
                    abs_d = np.absolute(d)
                    index = abs_d.argmax()
                    if (d[index] >= 0):
                        g = Q[:, index] * np.sqrt(d[index])
                    else:
                        g = Q[:, index] * np.sqrt(np.absolute(d[index])) * 1j
                    G = np.dot(np.diag(g), temp)
                    G = np.dot(G, np.diag(g.conj()))
                    if self.antenna == "all":
                        if u_dim2 != 1:
                            V_R_pq[i, j] = R[b_list[0], b_list[1]]
                            V_G_pq[i, j] = G[b_list[0], b_list[1]]
                        else:
                            V_R_pq[i] = R[b_list[0], b_list[1]]
                            V_G_pq[i] = G[b_list[0], b_list[1]]
                    else:
                        for k in range(p_new.shape[0]):
                            for l in range(p_new.shape[1]):
                                if (p_new[k, l] == b_list[0]) and (q_new[k, l] == b_list[1]):
                                    if u_dim2 != 1:
                                        V_R_pq[i, j] = R[k, l]
                                        V_G_pq[i, j] = G[k, l]
                                    else:
                                        V_R_pq[i] = R[k, l]
                                        V_G_pq[i] = G[k, l]
                else:
                    R1 = (R - self.A_1) / self.A_2
                    P = R1.shape[0]
                    if self.antenna == "all":
                        G = self.A_1 + ((0.5 * self.A_2) / P) * (np.sum(R1[b_list[0], :]) + np.sum(R1[:, b_list[1]]))
                        G = (G + ((0.5 * self.A_2) / P) ** 2 * R1[b_list[0], b_list[1]] * np.sum(R1))
                        if u_dim2 != 1:
                            V_R_pq[i, j] = R[b_list[0], b_list[1]]
                            V_G_pq[i, j] = G
                        else:
                            V_R_pq[i] = R[b_list[0], b_list[1]]
                            V_G_pq[i] = G
                    else:
                        for k in range(p_new.shape[0]):
                            for l in range(p_new.shape[1]):
                                if (p_new[k, l] == b_list[0]) and (q_new[k, l] == b_list[1]):
                                    G = self.A1 + ((0.5 * self.A2) / P) * (np.sum(R1[k, :]) + np.sum(R1[:, l]))
                                    G = (G + ((0.5 * self.A2) / P) ** 2 * R1[k, l] * np.sum(R1))
                                    if u_dim2 != 1:
                                        V_R_pq[i, j] = R[k, l]
                                        V_G_pq[i, j] = G
                                    else:
                                        V_R_pq[i] = R[k, l]
                                        V_G_pq[i] = G
        return u, v, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor, m_cor

    def vis_function(self, type_w, avg_v, V_G_pq, V_G_qp, V_R_pq, take_conj=False):
        if type_w == "R":
            print("Hallo R")
            vis = V_R_pq
        elif type_w == "RT":
            if take_conj:
                vis = np.conj(V_R_pq)
            else:
                vis = V_R_pq ** (-1)
        elif type_w == "R-1":
            vis = V_R_pq - 1
        elif type_w == "RT-1":
            if take_conj:
                vis = np.conj(V_R_pq)
            else:
                vis = V_R_pq ** (-1) - 1
        elif type_w == "G":
            if avg_v:
                vis = (V_G_pq + V_G_qp) / 2
            else:
                vis = V_G_pq
        elif type_w == "G-1":
            if avg_v:
                vis = (V_G_pq + V_G_qp) / 2 - 1
            else:
                vis = V_G_pq - 1
        elif type_w == "GT":
            if avg_v:
                if take_conj:
                    vis = (np.conj(V_G_pq) + np.conj(V_G_qp)) / 2
                else:
                    vis = (V_G_pq ** (-1) + V_G_qp ** (-1)) / 2
            else:
                if take_conj:
                    vis = np.conj(V_G_pq)
                else:
                    vis = V_G_pq ** (-1)
        elif type_w == "GT-1":
            if avg_v:
                if take_conj:
                    vis = (np.conj(V_G_pq) + np.conj(V_G_qp)) / 2 - 1
                else:
                    vis = (V_G_pq ** (-1) + V_G_qp ** (-1)) / 2 - 1
            else:
                if take_conj:
                    vis = np.conj(V_G_pq) - 1
                else:
                    vis = V_G_pq ** (-1) - 1
        elif type_w == "GTR-R":
            if avg_v:
                if take_conj:
                    vis = ((np.conj(V_G_pq) + np.conj(V_G_qp)) / 2) * V_R_pq - V_R_pq
                else:
                    vis = ((V_G_pq ** (-1) + V_G_qp ** (-1)) / 2) * V_R_pq - V_R_pq
            else:
                if take_conj:
                    vis = np.conj(V_G_pq) * V_R_pq - V_R_pq
                else:
                    vis = V_G_pq ** (-1) * V_R_pq - V_R_pq
        elif type_w == "GTR":
            if avg_v:
                if take_conj:
                    vis = ((np.conj(V_G_pq) + np.conj(V_G_qp)) / 2) * V_R_pq
                else:
                    vis = ((V_G_pq ** (-1) + V_G_qp ** (-1)) / 2) * V_R_pq
            else:
                if take_conj:
                    vis = np.conj(V_G_pq) * V_R_pq
                else:
                    vis = V_G_pq ** (-1) * V_R_pq
        elif type_w == "GTR-1":
            if avg_v:
                if take_conj:
                    vis = ((np.conj(V_G_pq) + np.conj(V_G_qp)) / 2) * V_R_pq - 1
                else:
                    vis = ((V_G_pq ** (-1) + V_G_qp ** (-1)) / 2) * V_R_pq - 1
            else:
                if take_conj:
                    vis = np.conj(V_G_pq) * V_R_pq - 1
                else:
                    vis = V_G_pq ** (-1) * V_R_pq - 1
        return vis

    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_2D(self, resolution, image_s, s, sigma=None, type_w="G-1", avg_v=False, plot=False, mask=False, wave=None,
               dec=None, approx=False, algo="STEFCAL", pickle_file=None, no_auto=True):
        if wave == None:
            wave = self.wave
        if dec == None:
            dec = self.dec
        ant_len = len(self.a_list)
        counter = 0
        baseline = [0, 0]

        for k in range(ant_len):
            for j in range(k + 1, ant_len):
                baseline[0] = self.a_list[k]
                baseline[1] = self.a_list[j]
                counter = counter + 1
                print("counter = ", counter)
                print("baseline = ", baseline)
                print("pickle_file = ", pickle_file)
                if avg_v:
                    baseline_new = [0, 0]
                    baseline_new[0] = baseline[1]
                    baseline_new[1] = baseline[0]
                    u, v, V_G_qp, V_R_qp, phi, delta_b, theta, l_cor, m_cor = self.visibilities_pq_2D_LM(baseline_new,
                                                                                                         resolution=resolution,
                                                                                                         image_s=image_s,
                                                                                                         s=s, wave=wave,
                                                                                                         dec=dec,
                                                                                                         algo=algo,
                                                                                                         no_auto=no_auto,
                                                                                                         sigma=sigma)
                else:
                    V_G_qp = 0

                u, v, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor, m_cor = self.visibilities_pq_2D_LM(baseline,
                                                                                                     resolution=resolution,
                                                                                                     image_s=image_s,
                                                                                                     s=s, wave=wave,
                                                                                                     dec=dec, algo=algo,
                                                                                                     no_auto=no_auto,
                                                                                                     sigma=sigma)

                if (k == 0) and (j == 1):
                    vis = self.vis_function(type_w, avg_v, V_G_pq, V_G_qp, V_R_pq)
                else:
                    vis = vis + self.vis_function(type_w, avg_v, V_G_pq, V_G_qp, V_R_pq)

        vis = vis / counter

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)

        N = l_cor.shape[0]

        delta_u = u[1] - u[0]
        delta_v = v[1] - v[0]

        if sigma != None:

            uu, vv = np.meshgrid(u, v)

            sigma = (np.pi / 180) * sigma

            g_kernal = (2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))

            vis = vis * g_kernal

            vis = np.roll(vis, -1 * (N - 1) / 2, axis=0)
            vis = np.roll(vis, -1 * (N - 1) / 2, axis=1)

            image = np.fft.fft2(vis) * (delta_u * delta_v)
        else:

            image = np.fft.fft2(vis) / N ** 2

        # ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image, 1 * (N - 1) / 2, axis=0)
        image = np.roll(image, 1 * (N - 1) / 2, axis=1)

        image = image[:, ::-1]
        # image = image[::-1,:]

        # image = (image/1)*100

        if plot:

            l_cor = l_cor * (180 / np.pi)
            m_cor = m_cor * (180 / np.pi)

            fig = plt.figure()
            cs = plt.imshow((image.real / self.A_2) * 100, interpolation="bicubic", cmap="jet",
                            extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
            cb = fig.colorbar(cs)
            cb.set_label(r"Flux [% of $A_2$]")
            self.plt_circle_grid(image_s)

            # print "amax = ",np.amax(image.real)
            # print "amax = ",np.amax(np.absolute(image))

            plt.xlim([-image_s, image_s])
            plt.ylim([-image_s, image_s])

            if mask:
                self.create_mask_all(plot_v=True, dec=dec)

            # self.create_mask(baseline,plot_v = True)

            plt.xlabel("$l$ [degrees]")
            plt.ylabel("$m$ [degrees]")
            plt.show()

            fig = plt.figure()
            cs = plt.imshow((image.imag / self.A_2) * 100, interpolation="bicubic", cmap="jet",
                            extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
            cb = fig.colorbar(cs)
            cb.set_label(r"Flux [% of $A_2$]")
            self.plt_circle_grid(image_s)

            plt.xlim([-image_s, image_s])
            plt.ylim([-image_s, image_s])

            if mask:
                self.create_mask_all(plot_v=True)
            # self.create_mask(baseline,plot_v = True)

            plt.xlabel("$l$ [degrees]")
            plt.ylabel("$m$ [degrees]")
            plt.show()
        if pickle_file != None:
            f = open(pickle_file, 'wb')
            pickle.dump(self.A_2, f)
            pickle.dump(l_cor, f)
            pickle.dump(m_cor, f)
            pickle.dump(image, f)
            pickle.dump(image_s, f)
            f.close()

        return image, l_old, m_old

    """Generates your ghost map
    baseline --- which baseline do you whish to image [0,1], selects baseline 01
    resolution --- the resolution of the pixels in your final image in arcseconds
    image_s --- the final extend of your image in degrees
    s --- oversampling rate
    sigma --- size of kernal used to fatten the ghosts otherwise they would be delta functions
    type --- which matrix do you wish to image
    avg --- average between baseline pq and qp
    plot --- plot the image
    mask --- plot the theoretical derived positions with crosses
    label_v --- labels them --- NOT SUPPROTED
    wave --- observational wavelength in meters
    dec --- declination of your observation
    save_fig --- saves figure
    approx --- if true we use Stefan Wijnholds approximation instead of ALS calibration
    difference --- between stefans approach and standard
    scale --- standard deviation of your noise (very simplistic noise, please make rigorous by yourself) 
    """

    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_pq_2D_LM(self, baseline, resolution, image_s, s, sigma=None, type_w="G-1", avg_v=False, plot=False,
                     mask=False, wave=None, dec=None, label_v=False, save_fig=False, algo="PHASE", algo2=None,
                     no_auto=True, take_conj1=False, take_conj2=False, pickle_file=None):
        if wave == None:
            wave = self.wave
        if dec == None:
            dec = self.dec

        if avg_v:
            baseline_new = [0, 0]
            baseline_new[0] = baseline[1]
            baseline_new[1] = baseline[0]
            u, v, V_G_qp, V_R_qp, phi, delta_b, theta, l_cor, m_cor = self.visibilities_pq_2D_LM(baseline_new,
                                                                                                 resolution=resolution,
                                                                                                 image_s=image_s, s=s,
                                                                                                 wave=wave, dec=dec,
                                                                                                 algo=algo,
                                                                                                 no_auto=no_auto,
                                                                                                 sigma=sigma)
        else:
            V_G_qp = 0

        u, v, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor, m_cor = self.visibilities_pq_2D_LM(baseline,
                                                                                             resolution=resolution,
                                                                                             image_s=image_s, s=s,
                                                                                             wave=wave, dec=dec,
                                                                                             algo=algo, no_auto=no_auto,
                                                                                             sigma=sigma)

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)
        print("pickle_file = ", pickle_file)
        N = l_cor.shape[0]

        vis = self.vis_function(type_w, avg_v, V_G_pq, V_G_qp, V_R_pq, take_conj1)
        delta_u = u[1] - u[0]
        delta_v = v[1] - v[0]

        if algo2 != None:

            if avg_v:
                baseline_new = [0, 0]
                baseline_new[0] = baseline[1]
                baseline_new[1] = baseline[0]
                u, v, V_G_qp, V_R_qp, phi, delta_b, theta, l_cor2, m_cor2 = self.visibilities_pq_2D_LM(baseline_new,
                                                                                                       resolution=resolution,
                                                                                                       image_s=image_s,
                                                                                                       s=s, wave=wave,
                                                                                                       dec=dec,
                                                                                                       algo=algo2,
                                                                                                       no_auto=no_auto,
                                                                                                       sigma=sigma)
            else:
                V_G_qp = 0

            u2, v2, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor2, m_cor2 = self.visibilities_pq_2D_LM(baseline,
                                                                                                     resolution=resolution,
                                                                                                     image_s=image_s,
                                                                                                     s=s, wave=wave,
                                                                                                     dec=dec,
                                                                                                     algo=algo2,
                                                                                                     no_auto=no_auto,
                                                                                                     sigma=sigma)

            vis2 = self.vis_function(type_w, avg_v, V_G_pq, V_G_qp, V_R_pq, take_conj2)
            vis = vis - vis2

        # vis = V_G_pq-1

        if sigma != None:

            uu, vv = np.meshgrid(u, v)

            sigma = (np.pi / 180) * sigma

            g_kernal = (2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))

            vis = vis * g_kernal

            vis = np.roll(vis, -1 * (N - 1) / 2, axis=0)
            vis = np.roll(vis, -1 * (N - 1) / 2, axis=1)

            image = np.fft.fft2(vis) * (delta_u * delta_v)
        else:

            image = np.fft.fft2(vis) / N ** 2

        # ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image, 1 * (N - 1) / 2, axis=0)
        image = np.roll(image, 1 * (N - 1) / 2, axis=1)

        # JASON I HAD TO ADD THIS LINE FLIPPED IN X AND Y
        image = image[::-1, ::-1]

        # image = image[::-1,:]

        # image = (image/(50-7.5))*100

        if plot:

            l_cor = l_cor * (180 / np.pi)
            m_cor = m_cor * (180 / np.pi)

            fig = plt.figure()
            # cs = plt.imshow((image.real/self.A_2)*100,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]],vmax=8.1,vmin=-27.1)
            cs = plt.imshow((image.real / self.A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                            extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
            # cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
            cb = fig.colorbar(cs)
            cb.set_label(r"Flux [% of $A_2$]")
            # cb.set_label("Jy")
            self.plt_circle_grid(image_s)
            if label_v:
                self.plot_source_labels_pq(baseline, im=image_s, plot_x=False)

            print("amax_real = ", np.amax((image.real / self.A_2) * 100))
            print("amin_real = ", np.amin((image.real / self.A_2) * 100))
            # print "amax = ",np.amax(np.absolute(image))

            plt.xlim([-image_s, image_s])
            plt.ylim([-image_s, image_s])

            if mask:
                p = self.create_mask(baseline, plot_v=True, dec=dec)

            # for k in xrange(len(p)):
            #    plt.plot(p[k,1]*(180/np.pi),p[k,2]*(180/np.pi),"kv")

            plt.xlabel("$l$ [degrees]")
            plt.ylabel("$m$ [degrees]")
            plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Real")

            if save_fig:
                plt.savefig("Figure_R_pq" + str(baseline[0]) + str(baseline[1]) + ".png", format="png",
                            bbox_inches="tight")
                plt.clf()
            else:
                plt.show()

            fig = plt.figure()
            # cs = plt.imshow((image.imag/self.A_2)*100,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]],vmax=12.3,vmin=-12.3)
            # JASON I HAD TO ADD THE -1 HERE (FOR SOME REASON IT WAS PLOTTING THE CONJUGATE)
            cs = plt.imshow(-1 * (image.imag / self.A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                            extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
            # cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
            cb = fig.colorbar(cs)
            cb.set_label(r"Flux [% of $A_2$]")
            # cb.set_label("Flux [\% of A_2]")

            print("amax_imag = ", np.amax((image.imag / self.A_2) * 100))
            print("amin_imag = ", np.amin((image.imag / self.A_2) * 100))

            self.plt_circle_grid(image_s)
            if label_v:
                self.plot_source_labels_pq(baseline, im=image_s, plot_x=False)

            plt.xlim([-image_s, image_s])
            plt.ylim([-image_s, image_s])

            if mask:
                self.create_mask(baseline, plot_v=True, dec=dec)

            plt.xlabel("$l$ [degrees]")
            plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Imag")
            plt.ylabel("$m$ [degrees]")
            if save_fig:
                plt.savefig("Figure_I_pq" + str(baseline[0]) + str(baseline[1]) + ".png", format="png",
                            bbox_inches="tight")
                plt.clf()
            else:
                plt.show()
        if pickle_file != None:
            f = open(pickle_file, 'wb')
            pickle.dump(self.A_2, f)
            pickle.dump(l_cor, f)
            pickle.dump(m_cor, f)
            pickle.dump(image, f)
            pickle.dump(image_s, f)
            pickle.dump(baseline, f)
            f.close()

        return image, l_old, m_old

    def plt_circle_grid(self, grid_m):
        rad = np.arange(1, 1 + grid_m, 1)
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 1, 500)

        x_c = np.cos(2 * np.pi * x)
        y_c = np.sin(2 * np.pi * y)
        for k in range(len(rad)):
            plt.plot(rad[k] * x_c, rad[k] * y_c, "k", ls=":", lw=0.5)

    def create_mask_all(self, plot_v=False, dec=None):
        if dec == None:
            dec = self.dec
        sin_delta = np.sin(dec)

        point_sources = np.array([(1, 0, 0)])
        point_sources = np.append(point_sources, [(1, self.l_0, -1 * self.m_0)], axis=0)
        point_sources = np.append(point_sources, [(1, -1 * self.l_0, 1 * self.m_0)], axis=0)

        # SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        d_list = self.calculate_delete_list()

        p = np.ones(self.phi_m.shape, dtype=int)
        p = np.cumsum(p, axis=0) - 1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p, d_list, axis=0)
            p_new = np.delete(p_new, d_list, axis=1)
            q_new = np.delete(q, d_list, axis=0)
            q_new = np.delete(q_new, d_list, axis=1)

            phi_new = np.delete(self.phi_m, d_list, axis=0)
            phi_new = np.delete(phi_new, d_list, axis=1)

            b_new = np.delete(self.b_m, d_list, axis=0)
            b_new = np.delete(b_new, d_list, axis=1)

            theta_new = np.delete(self.theta_m, d_list, axis=0)
            theta_new = np.delete(theta_new, d_list, axis=1)
        #####################################################
        if plot_v == True:
            plt.plot(0, 0, "rx")
            plt.plot(self.l_0 * (180 / np.pi), self.m_0 * (180 / np.pi), "rx")
            plt.plot(-1 * self.l_0 * (180 / np.pi), -1 * self.m_0 * (180 / np.pi), "rx")

        len_a = len(self.a_list)
        b_list = [0, 0]

        first = True

        for h in range(len_a):
            for i in range(h + 1, len_a):
                b_list[0] = self.a_list[h]
                b_list[1] = self.a_list[i]
                phi = self.phi_m[b_list[0], b_list[1]]
                delta_b = self.b_m[b_list[0], b_list[1]]
                theta = self.theta_m[b_list[0], b_list[1]]
                for j in range(theta_new.shape[0]):
                    for k in range(j + 1, theta_new.shape[0]):
                        if not np.allclose(phi_new[j, k], phi):
                            l_cordinate = phi_new[j, k] / phi * (
                                    np.cos(theta_new[j, k] - theta) * self.l_0 + sin_delta * np.sin(
                                theta_new[j, k] - theta) * self.m_0)
                            m_cordinate = phi_new[j, k] / phi * (
                                    np.cos(theta_new[j, k] - theta) * self.m_0 - sin_delta ** (-1) * np.sin(
                                theta_new[j, k] - theta) * self.l_0)
                            if plot_v == True:
                                plt.plot(l_cordinate * (180 / np.pi), m_cordinate * (180 / np.pi), "rx")
                                plt.plot(-1 * l_cordinate * (180 / np.pi), -1 * m_cordinate * (180 / np.pi), "rx")
                            point_sources = np.append(point_sources, [(1, l_cordinate, -1 * m_cordinate)], axis=0)
                            point_sources = np.append(point_sources, [(1, -1 * l_cordinate, 1 * m_cordinate)], axis=0)

        return point_sources

    def create_mask(self, baseline, plot_v=False, dec=None, plot_markers=False):
        if dec == None:
            dec = self.dec
        sin_delta = np.sin(dec)
        point_sources = np.array([(1, 0, 0)])
        point_sources_labels = np.array([(0, 0, 0, 0)])
        point_sources = np.append(point_sources, [(1, self.l_0, -1 * self.m_0)], axis=0)
        point_sources_labels = np.append(point_sources_labels, [(baseline[0], baseline[1], baseline[0], baseline[1])],
                                         axis=0)
        point_sources = np.append(point_sources, [(1, -1 * self.l_0, 1 * self.m_0)], axis=0)
        point_sources_labels = np.append(point_sources_labels, [(baseline[1], baseline[0], baseline[0], baseline[1])],
                                         axis=0)

        # SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline, self.ant_names)
        # print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        # print "d_list = ",d_list

        phi = self.phi_m[b_list[0], b_list[1]]
        delta_b = self.b_m[b_list[0], b_list[1]]
        theta = self.theta_m[b_list[0], b_list[1]]

        p = np.ones(self.phi_m.shape, dtype=int)
        p = np.cumsum(p, axis=0) - 1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p, d_list, axis=0)
            p_new = np.delete(p_new, d_list, axis=1)
            q_new = np.delete(q, d_list, axis=0)
            q_new = np.delete(q_new, d_list, axis=1)

            phi_new = np.delete(self.phi_m, d_list, axis=0)
            phi_new = np.delete(phi_new, d_list, axis=1)

            b_new = np.delete(self.b_m, d_list, axis=0)
            b_new = np.delete(b_new, d_list, axis=1)

            theta_new = np.delete(self.theta_m, d_list, axis=0)
            theta_new = np.delete(theta_new, d_list, axis=1)
        #####################################################
        if plot_v == True:
            if plot_markers:
                mk_string = self.return_color_marker([0, 0])
                plt.plot(0, 0, self.return_color_marker([0, 0]), label="(0,0)", mfc='none', ms=5)
                mk_string = self.return_color_marker(baseline)
                plt.plot(self.l_0 * (180 / np.pi), self.m_0 * (180 / np.pi), self.return_color_marker(baseline),
                         label="(" + str(baseline[0]) + "," + str(baseline[1]) + ")", mfc='none', mec=mk_string[0],
                         ms=5)
                mk_string = self.return_color_marker([baseline[1], baseline[0]])
                plt.plot(-1 * self.l_0 * (180 / np.pi), -1 * self.m_0 * (180 / np.pi),
                         self.return_color_marker([baseline[1], baseline[0]]),
                         label="(" + str(baseline[1]) + "," + str(baseline[0]) + ")", mfc='none', mec=mk_string[0],
                         ms=5)
            else:
                plt.plot(0, 0, "rx")
                plt.plot(self.l_0 * (180 / np.pi), self.m_0 * (180 / np.pi), "rx")
                plt.plot(-1 * self.l_0 * (180 / np.pi), -1 * self.m_0 * (180 / np.pi), "gx")
        for j in range(theta_new.shape[0]):
            for k in range(j + 1, theta_new.shape[0]):
                # print "Hallo:",j," ",k
                if not np.allclose(phi_new[j, k], phi):
                    # print "phi = ",phi_new[j,k]/phi
                    l_cordinate = (phi_new[j, k] * 1.0) / (1.0 * phi) * (
                            np.cos(theta_new[j, k] - theta) * self.l_0 + sin_delta * np.sin(
                        theta_new[j, k] - theta) * self.m_0)
                    # print "l_cordinate = ",l_cordinate*(180/np.pi)
                    m_cordinate = (phi_new[j, k] * 1.0) / (phi * 1.0) * (
                            np.cos(theta_new[j, k] - theta) * self.m_0 - sin_delta ** (-1) * np.sin(
                        theta_new[j, k] - theta) * self.l_0)
                    # print "m_cordinate = ",m_cordinate*(180/np.pi)
                    if plot_v == True:
                        if plot_markers:
                            mk_string = self.return_color_marker([j, k])
                            plt.plot(l_cordinate * (180 / np.pi), m_cordinate * (180 / np.pi),
                                     self.return_color_marker([j, k]), label="(" + str(j) + "," + str(k) + ")",
                                     mfc='none', mec=mk_string[0], ms=5)
                            mk_string = self.return_color_marker([k, j])
                            plt.plot(-1 * l_cordinate * (180 / np.pi), -1 * m_cordinate * (180 / np.pi),
                                     self.return_color_marker([k, j]), label="(" + str(k) + "," + str(j) + ")",
                                     mfc='none', mec=mk_string[0], ms=5)
                            plt.legend(loc=8, ncol=9, numpoints=1, prop={"size": 7}, columnspacing=0.1)
                        else:
                            plt.plot(l_cordinate * (180 / np.pi), m_cordinate * (180 / np.pi), "rx")
                            plt.plot(-1 * l_cordinate * (180 / np.pi), -1 * m_cordinate * (180 / np.pi), "gx")
                    point_sources = np.append(point_sources, [(1, l_cordinate, -1 * m_cordinate)], axis=0)
                    point_sources_labels = np.append(point_sources_labels, [(j, k, baseline[0], baseline[1])], axis=0)
                    point_sources = np.append(point_sources, [(1, -1 * l_cordinate, 1 * m_cordinate)], axis=0)
                    point_sources_labels = np.append(point_sources_labels, [(k, j, baseline[0], baseline[1])], axis=0)

        return point_sources, point_sources_labels

    # window is in degrees, l,m,point_sources in radians, point_sources[k,:] ---> kth point source
    def extract_flux(self, image, l, m, window, point_sources, plot):
        window = window * (np.pi / 180)
        point_sources_real = np.copy(point_sources)
        point_sources_imag = np.copy(point_sources)
        for k in range(len(point_sources)):
            l_0 = point_sources[k, 1]
            m_0 = point_sources[k, 2] * (-1)

            l_max = l_0 + window / 2.0
            l_min = l_0 - window / 2.0
            m_max = m_0 + window / 2.0
            m_min = m_0 - window / 2.0

            m_rev = m[::-1]

            # ll,mm = np.meshgrid(l,m)

            image_sub = image[:, (l < l_max) & (l > l_min)]
            # ll_sub = ll[:,(l<l_max)&(l>l_min)]
            # mm_sub = mm[:,(l<l_max)&(l>l_min)]

            if image_sub.size != 0:
                # print "Problem..."
                image_sub = image_sub[(m_rev < m_max) & (m_rev > m_min), :]
                # ll_sub = ll_sub[(m_rev<m_max)&(m_rev>m_min),:]
                # mm_sub = mm_sub[(m_rev<m_max)&(m_rev>m_min),:]

            # PLOTTING SUBSET IMAGE
            if plot:
                l_new = l[(l < l_max) & (l > l_min)]
                if l_new.size != 0:
                    m_new = m[(m < m_max) & (m > m_min)]
                    if m_new.size != 0:
                        l_cor = l_new * (180 / np.pi)
                        m_cor = m_new * (180 / np.pi)

                        # plt.contourf(ll_sub*(180/np.pi),mm_sub*(180/np.pi),image_sub.real)
                        # plt.show()

                        # fig = plt.figure()
                        # cs = plt.imshow(mm*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                        # fig.colorbar(cs)
                        # plt.show()
                        # fig = plt.figure()
                        # cs = plt.imshow(ll*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                        # fig.colorbar(cs)
                        # plt.show()

                        fig = plt.figure()
                        cs = plt.imshow(image_sub.real, interpolation="bicubic", cmap="jet",
                                        extent=[l_cor[0], l_cor[-1], m_cor[0], m_cor[-1]])
                        # plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")
                        fig.colorbar(cs)
                        plt.title("REAL")
                        plt.show()
                        fig = plt.figure()
                        cs = plt.imshow(image_sub.imag, interpolation="bicubic", cmap="jet",
                                        extent=[l_cor[0], l_cor[-1], m_cor[0], m_cor[-1]])
                        fig.colorbar(cs)
                        plt.title("IMAG")
                        plt.show()
            # print "image_sub = ",image_sub
            if image_sub.size != 0:
                max_v_r = np.amax(image_sub.real)
                # print "max_v_r = ",max_v_r
                max_v_i = np.amax(image_sub.imag)
                min_v_r = np.amin(image_sub.real)
                min_v_i = np.amin(image_sub.imag)
                if np.absolute(max_v_r) > np.absolute(min_v_r):
                    point_sources_real[k, 0] = max_v_r
                else:
                    point_sources_real[k, 0] = min_v_r
                if np.absolute(max_v_i) > np.absolute(min_v_i):
                    point_sources_imag[k, 0] = max_v_i
                else:
                    point_sources_imag[k, 0] = min_v_i

            else:
                # print "PROBLEM 2"
                point_sources_real[k, 0] = 0
                point_sources_imag[k, 0] = 0

        return point_sources_real, point_sources_imag

    def extract_deteuro_mask(self, baseline, mask, p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_not(np.logical_or(p_labels[:, 0] == p, p_labels[:, 1] == q))
        mask2 = np.logical_not(np.logical_and(p_labels[:, 0] == 0, p_labels[:, 1] == 0))
        mask3 = np.logical_not(np.logical_and(p_labels[:, 0] == q, p_labels[:, 1] == p))
        deteuro_mask = np.logical_and(mask1, mask2)
        deteuro_mask = np.logical_and(deteuro_mask, mask3)
        return mask[deteuro_mask], p_labels[deteuro_mask]

    def extract_deteuro_mask_all(self, baseline, mask, p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_not(np.logical_or(p_labels[:, 0] == p, p_labels[:, 1] == q))
        mask2 = np.logical_not(np.logical_and(p_labels[:, 0] == 0, p_labels[:, 1] == 0))
        deteuro_mask = np.logical_and(mask1, mask2)
        return mask[deteuro_mask], p_labels[deteuro_mask]

    def extract_proto_mask_all(self, baseline, mask, p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_or(p_labels[:, 0] == p, p_labels[:, 1] == q)
        mask2 = np.logical_and(p_labels[:, 0] == 0, p_labels[:, 1] == 0)
        proto_mask = np.logical_or(mask1, mask2)
        return mask[proto_mask], p_labels[proto_mask]

    def plot_ghost_mask_paper(self, baseline, dec=-74.66 * (np.pi / 180), im_s=4, pickle_file=None):
        mask, p_labels = self.create_mask(baseline, plot_v=False, dec=dec, plot_markers=False)
        # self.plt_circle_grid(abs(im_s))
        # plt.title("Baseline: "+str(baseline[0])+str(baseline[1]))
        # plt.xlabel("$l$ [degrees]")
        # plt.ylabel("$m$ [degrees]")
        # plt.axis("image")
        # plt.xlim([-1*abs(im_s),abs(im_s)])
        # plt.ylim([-1*abs(im_s),abs(im_s)])
        # plt.show()

        proto_mask, proto_labels = self.extract_proto_mask_all(baseline, mask, p_labels)
        deteuro_mask, deteuro_labels = self.extract_deteuro_mask_all(baseline, mask, p_labels)
        plt.plot(proto_mask[:, 1] * (180 / np.pi), -1 * proto_mask[:, 2] * (180 / np.pi), "o",
                 label="Blue Proto-ghosts", mfc='none', mec="b", ms=7)
        plt.plot(deteuro_mask[:, 1] * (180 / np.pi), -1 * deteuro_mask[:, 2] * (180 / np.pi), "s",
                 label="Red Deteuro-ghosts", mfc='none', mec="r", ms=7)
        self.plt_circle_grid(abs(im_s))
        plt.legend(loc=8, ncol=2, numpoints=1, prop={"size": 12})

        plt.title("Full-Complex, Baseline: " + str(baseline[0]) + str(baseline[1]))
        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.axis("image")
        plt.xlim([-1 * abs(im_s), abs(im_s)])
        plt.ylim([-1 * abs(im_s), abs(im_s)])
        plt.show()

        proto_mask, proto_labels = self.extract_proto_mask_phase(baseline, mask, p_labels)
        # deteuro_mask, deteuro_labels = self.extract_deteuro_mask_all(baseline,mask,p_labels)
        plt.plot(proto_mask[:, 1] * (180 / np.pi), -1 * proto_mask[:, 2] * (180 / np.pi), "o",
                 label="Blue Proto-ghosts", mfc='none', mec="b", ms=7)
        plt.plot(-1 * proto_mask[:, 1] * (180 / np.pi), proto_mask[:, 2] * (180 / np.pi), "o", label="Red Proto-ghosts",
                 mfc='none', mec="r", ms=7)
        # plt.plot(deteuro_mask[:,1]*(180/np.pi),-1*deteuro_mask[:,2]*(180/np.pi),"s",label="Deteuro-ghosts",mfc='none',mec="g",ms=7)
        self.plt_circle_grid(abs(im_s))
        plt.legend(loc=8, ncol=2, numpoints=1, prop={"size": 12})

        plt.title("Phase-Only, Baseline: " + str(baseline[0]) + str(baseline[1]))
        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.axis("image")
        plt.xlim([-1 * abs(im_s), abs(im_s)])
        plt.ylim([-1 * abs(im_s), abs(im_s)])
        plt.show()
        if pickle_file != None:
            f = open(pickle_file, 'wb')
            pickle.dump(proto_mask, f)
            pickle.dump(im_s, f)
            pickle.dump(baseline, f)
            f.close()

    def extract_proto_mask_phase(self, baseline, mask, p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_or(p_labels[:, 0] == p, p_labels[:, 1] == q)
        # mask2 = np.logical_not(np.logical_and(p_labels[:,0]==p,p_labels[:,1]==q))
        # proto_mask = np.logical_and(mask1,mask2)
        proto_mask = mask1
        return mask[proto_mask], p_labels[proto_mask]

    def extract_proto_mask(self, baseline, mask, p_labels):
        p = baseline[0]
        q = baseline[1]
        mask1 = np.logical_or(p_labels[:, 0] == p, p_labels[:, 1] == q)
        mask2 = np.logical_not(np.logical_and(p_labels[:, 0] == p, p_labels[:, 1] == q))
        proto_mask = np.logical_and(mask1, mask2)
        # proto_mask = mask1
        return mask[proto_mask], p_labels[proto_mask]


def experiment(A_min, A_max, num, file_name, algo="PHASE", type_w="GT-1"):
    A_v = np.linspace(A_min, A_max, num)
    result = np.zeros((6, len(A_v)))
    point_sources = np.zeros((6, 3))
    point_sources[0, :] = (1.0, 0, 0)
    point_sources[1, :] = (1.0, (1 * np.pi) / 180, (0 * np.pi) / 180)
    point_sources[2, :] = (1.0, (-1 * np.pi) / 180, (0 * np.pi) / 180)
    for k in range(len(A_v)):
        print("************")
        print("k = ", k)
        print("algo = ", algo)
        print("type_w =", type_w)
        print("experiment1")
        print("************")
        # Left out
        point_sources_true = np.array([(1.0, 0, 0), (A_v[k], (1 * np.pi) / 180, (0 * np.pi) / 180)])
        # point_sources_extract = np.array([(88.7,0,0),(30,(1*np.pi)/180,(0*np.pi)/180),(30,(-1*np.pi)/180,(0*np.pi)/180)])
        point_sources_model = np.array([(1.0, 0, 0)])
        ant_list = "all"
        # ant_list = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
        t = T_ghost(point_sources_true, point_sources_model, ant_list, "KAT7")
        image, l_v, m_v = t.sky_pq_2D_LM([4, 5], 150, 3, 2, sigma=0.05, type_w=type_w, plot=False, mask=True, algo=algo)
        mask, p_labels = t.create_mask([4, 5], plot_v=False, dec=None, plot_markers=False)
        point_sources_temp, point_labels = t.extract_proto_mask([4, 5], mask, p_labels)
        point_sources[3, :] = point_sources_temp[0, :]
        point_sources[4, :] = point_sources_temp[0, :]
        point_sources[4, 1:] = point_sources[4, 1:] * (-1)
        point_sources[5, :] = (1.0, (2 * np.pi) / 180, (0 * np.pi) / 180)
        point_real, point_imag = t.extract_flux(image, l_v, m_v, 0.5, point_sources, False)
        result[:, k] = np.sqrt(point_real[:, 0] ** 2 + point_imag[:, 0] ** 2)
    # print "result = ",(result/0.5)*100
    # for k in xrange(3):
    #    plt.plot(A_v,result[k,:])
    #    plt.hold('on')
    # plt.show()
    f = open(file_name, 'wb')
    pickle.dump(A_v, f)
    pickle.dump(result, f)
    pickle.dump(point_sources_true, f)
    pickle.dump(point_sources_model, f)
    f.close()
    return result


def experiment2(A_min, A_max, num, file_name, algo="PHASE", type_w="GT-1"):
    A_v = np.linspace(A_min, A_max, num)
    result = np.zeros((6, len(A_v)))
    point_sources = np.zeros((6, 3))
    point_sources[0, :] = (1.0, 0, 0)
    point_sources[1, :] = (1.0, (1 * np.pi) / 180, (0 * np.pi) / 180)
    point_sources[2, :] = (1.0, (-1 * np.pi) / 180, (0 * np.pi) / 180)
    for k in range(len(A_v)):
        print("************")
        print("k = ", k)
        print("algo = ", algo)
        print("type_w =", type_w)
        print("experiment 2")
        print("************")
        # Left out
        point_sources_true = np.array([(1.0, 0, 0), (A_v[k], (1 * np.pi) / 180, (0 * np.pi) / 180)])
        # point_sources_extract = np.array([(88.7,0,0),(30,(1*np.pi)/180,(0*np.pi)/180),(30,(-1*np.pi)/180,(0*np.pi)/180)])
        point_sources_model = np.array([(1.0, 0, 0)])
        ant_list = "all"
        # ant_list = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
        t = T_ghost(point_sources_true, point_sources_model, ant_list, "KAT7")
        image, l_v, m_v = t.sky_2D([4, 5], 150, 3, 2, sigma=0.05, type_w=type_w, plot=False, mask=True, algo=algo,
                                   avg_v=True)
        mask, p_labels = t.create_mask([4, 5], plot_v=False, dec=None, plot_markers=False)
        point_sources_temp, point_labels = t.extract_proto_mask([4, 5], mask, p_labels)
        point_sources[3, :] = point_sources_temp[0, :]
        point_sources[4, :] = point_sources_temp[0, :]
        point_sources[4, 1:] = point_sources[4, 1:] * (-1)
        point_real, point_imag = t.extract_flux(image, l_v, m_v, 0.5, point_sources, False)
        point_sources[5, :] = (1.0, (2 * np.pi) / 180, (0 * np.pi) / 180)
        result[:, k] = np.sqrt(point_real[:, 0] ** 2 + point_imag[:, 0] ** 2)
    # print "result = ",(result/0.5)*100
    # for k in xrange(3):
    #    plt.plot(A_v,result[k,:])
    #    plt.hold('on')
    # plt.show()
    f = open(file_name, 'wb')
    pickle.dump(A_v, f)
    pickle.dump(result, f)
    pickle.dump(point_sources_true, f)
    pickle.dump(point_sources_model, f)
    f.close()
    return result


def plot_results(file_name, algo="STEF", type_plot="G", avg_baseline=21, N=7.0, ylim1=[0, 35], ylim2=[0, 0.8]):
    # self.phi_m = pickle.load(open(file_name,"rb"))
    f = open(file_name, 'rb')
    A_v = pickle.load(f)
    result = pickle.load(f)
    c = ["b", "r", "g"]
    one = np.ones(A_v.size)
    # print "result = ",result

    if (algo == "STEF"):
        l = [r"$\mathbf{0}$", r"$\mathbf{s}_0$", r"-$\mathbf{s}_0$", "Blue Proto-ghost", "Red Deteuro-ghost"]
        y_primary = A_v / N
        y_secondary = 2 * A_v / N - A_v / N ** 2
        y_anti = A_v / N ** 2
        y_proto = A_v / N - A_v / N ** 2
        y_deteuro = A_v / N ** 2
        if (type_plot == "R"):
            y_primary = y_primary + A_v * y_anti
            y_secondary = y_secondary + A_v * y_primary
    else:
        y_secondary = A_v / (N)
        y_proto = A_v / (2 * N)
        l = [r"$\mathbf{0}$", r"$\mathbf{s}_0$", r"-$\mathbf{s}_0$", "Blue Proto-ghost", "Red Proto-ghost"]
        # y_anti =

    label_size = 15
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    # T_bright = one*(1./6)*100
    # T_dim = one*(1./(12*21))*100
    # plt.plot(A_v,T_bright,"b--",lw=1.5,label="T")
    # plt.hold('on')
    for k in range(3):
        # if k < 3:
        plt.plot(A_v, (result[k, :] / A_v) * 100, c[k], lw=1.5, label=l[k])
        # else:
        #  plt.plot(A_v,(result[k,:]/(21*A_v))*100)
        plt.hold('on')

    plt.legend(loc=1, prop={'size': 15})
    if algo == "STEF":
        plt.plot(A_v, (y_primary / A_v) * 100, c[0] + "--", lw=1.5)
        plt.plot(A_v, (y_secondary / A_v) * 100, c[1] + "--", lw=1.5)
        plt.plot(A_v, (y_anti / A_v) * 100, c[2] + "--", lw=1.5)
    else:
        plt.plot(A_v, (y_secondary / A_v) * 100, "k" + "--", lw=1.5)

    plt.grid('on')
    plt.ylim(ylim1)
    plt.xlabel("$A_2$", fontsize=label_size)
    plt.ylabel("% of $A_2$", fontsize=label_size)
    # plt.ylim([-5,35])
    plt.show()

    # plt.plot(A_v,T_dim,"b--",lw=1.5,label="T")
    plt.hold('on')
    for k in range(3, 5):
        # if k < 3:
        plt.plot(A_v, (result[k, :] / (avg_baseline * A_v)) * 100, c[k - 3], lw=1.5, label=l[k])
        # else:
        #  plt.plot(A_v,(result[k,:]/(21*A_v))*100)
        plt.hold('on')
    # plt.plot(A_v,((result[3,:]+result[4,:])/(21*A_v))*100,lw=1.5,label="TEST")
    plt.legend(loc=1, prop={'size': 15})
    if algo == "STEF":
        avg_bas_theory = (N ** 2 - N) / 2.
        plt.plot(A_v, (y_proto / (avg_bas_theory * A_v)) * 100, c[0] + "--", lw=1.5)
        plt.plot(A_v, (y_deteuro / (avg_bas_theory * A_v)) * 100, c[1] + "--", lw=1.5)
    else:
        avg_bas_theory = (N ** 2 - N) / 2.
        plt.plot(A_v, (y_proto / (avg_bas_theory * A_v)) * 100, "k--", lw=1.5)
    plt.grid('on')
    plt.ylim(ylim2)
    plt.xlabel("$A_2$", fontsize=label_size)
    plt.ylabel("% of $A_2$", fontsize=label_size)
    plt.show()


def plot_image(pickle_file, precentage=False, convert_to_degrees=False):
    t = T_ghost(np.array([(1, 0, 0)]), np.array([(1, 0, 0)]), "all", "KAT7")
    f = open(pickle_file, 'rb')
    A_2 = pickle.load(f)
    if convert_to_degrees:
        l_cor = pickle.load(f) * (180 / np.pi)
        m_cor = pickle.load(f) * (180 / np.pi)
    else:
        l_cor = pickle.load(f)
        m_cor = pickle.load(f)
    image = pickle.load(f)
    # print "Image = ",image
    print("Image.max = ", np.amax(image.real) / A_2 * 100)
    print("image.min = ", np.amin(image.real) / A_2 * 100)
    # print "l_cor = ",l_cor
    # print "m_cor = ",m_cor
    # plt.imshow(image.real,extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
    # plt.show()
    image_s = pickle.load(f)
    baseline = pickle.load(f)
    fig = plt.figure()
    # cs = plt.imshow((image.real/self.A_2)*100,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]],vmax=8.1,vmin=-27.1)
    if precentage:
        cs = plt.imshow((image.real / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                        extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
        cb = fig.colorbar(cs)
        cb.set_label(r"Flux [% of $A_2$]")
    else:
        cs = plt.imshow(image.real, interpolation="bicubic", cmap="cubehelix",
                        extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
        cb = fig.colorbar(cs)
        cb.set_label(r"Flux [% of $A_2$]")

    t.plt_circle_grid(image_s)
    plt.xlabel("$l$ [degrees]")
    plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Real")
    plt.ylabel("$m$ [degrees]")
    plt.xlim([-image_s, image_s])
    plt.ylim([-image_s, image_s])
    plt.show()

    fig = plt.figure()
    # cs = plt.imshow((image.real/self.A_2)*100,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]],vmax=8.1,vmin=-27.1)
    if precentage:
        cs = plt.imshow((image.imag / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                        extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
        cb = fig.colorbar(cs)
        cb.set_label(r"Flux [% of $A_2$]")
    else:
        cs = plt.imshow(image.imag, interpolation="bicubic", cmap="cubehelix",
                        extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
        cb = fig.colorbar(cs)
        cb.set_label(r"Flux [Jy]")

    t.plt_circle_grid(image_s)
    plt.xlabel("$l$ [degrees]")
    plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Imag")
    plt.ylabel("$m$ [degrees]")
    plt.xlim([-image_s, image_s])
    plt.ylim([-image_s, image_s])
    plt.show()


def plot_image_full(pickle_file):
    t = T_ghost(np.array([(1, 0, 0)]), np.array([(1, 0, 0)]), "all", "KAT7")
    f = open(pickle_file, 'rb')
    A_2 = pickle.load(f)
    l_cor = pickle.load(f) * (180 / np.pi)
    m_cor = pickle.load(f) * (180 / np.pi)
    image = pickle.load(f)
    image_s = pickle.load(f)
    f.close()
    # baseline = pickle.load(f)
    fig = plt.figure()
    # cs = plt.imshow((image.real/self.A_2)*100,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]],vmax=8.1,vmin=-27.1)
    cs = plt.imshow((image.real / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                    extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
    # cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
    cb = fig.colorbar(cs)
    cb.set_label(r"Flux [% of $A_2$]")
    t.plt_circle_grid(image_s)
    plt.xlabel("$l$ [degrees]")
    # plt.title("Baseline "+str(baseline[0])+str(baseline[1])+" --- Real")
    plt.title("Real")
    plt.ylabel("$m$ [degrees]")
    plt.xlim([-image_s, image_s])
    plt.ylim([-image_s, image_s])
    plt.show()

    fig = plt.figure()
    # cs = plt.imshow((image.real/self.A_2)*100,interpolation = "bicubic", cmap = "cubehelix", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]],vmax=8.1,vmin=-27.1)
    cs = plt.imshow((image.imag / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                    extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
    # cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
    cb = fig.colorbar(cs)
    cb.set_label(r"Flux [% of $A_2$]")
    t.plt_circle_grid(image_s)
    plt.xlabel("$l$ [degrees]")
    # plt.title("Baseline "+str(baseline[0])+str(baseline[1])+" --- Imag")
    plt.ylabel("$m$ [degrees]")
    plt.xlim([-image_s, image_s])
    plt.ylim([-image_s, image_s])
    plt.show()


def plot_ghost_pat_p(pat1, pat2, pat3, t):
    f = open(pat1, 'rb')
    proto_mask = pickle.load(f)
    im_s = pickle.load(f)
    baseline = pickle.load(f)
    f.close()
    plt.plot(proto_mask[:, 1] * (180 / np.pi), -1 * proto_mask[:, 2] * (180 / np.pi), "o", mfc='none', mec="b", ms=7)
    plt.plot(-1 * proto_mask[:, 1] * (180 / np.pi), proto_mask[:, 2] * (180 / np.pi), "o", mfc='none', mec="r", ms=7)
    f = open(pat2, 'rb')
    proto_mask = pickle.load(f)
    im_s = pickle.load(f)
    f.close()
    plt.plot(proto_mask[:, 1] * (180 / np.pi), -1 * proto_mask[:, 2] * (180 / np.pi), "^", mfc='none', mec="b", ms=7)
    plt.plot(-1 * proto_mask[:, 1] * (180 / np.pi), proto_mask[:, 2] * (180 / np.pi), "^", mfc='none', mec="r", ms=7)
    f = open(pat3, 'rb')
    proto_mask = pickle.load(f)
    im_s = pickle.load(f)
    f.close()
    plt.plot(proto_mask[:, 1] * (180 / np.pi), -1 * proto_mask[:, 2] * (180 / np.pi), "<", mfc='none', mec="b", ms=7)
    plt.plot(-1 * proto_mask[:, 1] * (180 / np.pi), proto_mask[:, 2] * (180 / np.pi), "<", mfc='none', mec="r", ms=7)
    t.plt_circle_grid(im_s)
    plt.axis("image")
    plt.xlim([-im_s, im_s])
    plt.ylim([-im_s, im_s])
    plt.title("Phase-Only, Baseline: " + str(baseline[0]) + str(baseline[1]))
    plt.xlabel("$l$ [degrees]")
    plt.ylabel("$m$ [degrees]")
    plt.show()

def progress_bar(count, total):
    """
    Taken from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    # print('[%s] %s%s iteration %s\r' % (bar, percents, '%', count))

    sys.stdout.write('[%s] %s%s iteration %s\r' % (bar, percents, '%', count))
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",
                        type=int,
                        nargs='+',
                        default=[2, 3],
                        help="The baseline to calculate on")
    parser.add_argument("--algo",
                        type=str,
                        default="STEFCAL",
                        help="The algorithm to use, in all CAPS")
    parser.add_argument("--sigma",
                        type=float,
                        default=0.05,
                        help="Larger values increases the size of the point sources, a smaller value decreases the size")
    parser.add_argument("--pickle_file",
                        type=str,
                        default="PHASE_STEF_45_image.p",
                        help="The pickle file to use")
    parser.add_argument("--mask",
                        action="store_true",
                        help="Set to apply the mask to the image")
    parser.add_argument("--dont_save",
                        action="store_false",
                        help="Set to apply the mask to the image")
    parser.add_argument("--type",
                        type=str,
                        default="GT-1",
                        help="The type of experiment to run")
    parser.add_argument("--radius",
                        type=int,
                        default=3,
                        help="The radius of the circle on the image")

    global args
    args = parser.parse_args()

    # experiment(0.001,0.5,20,"KAT7_STEFGT1.p",algo="STEFCAL",type_w="GT-1")
    # experiment2(0.001,0.5,20,"KAT7_STEFGT2.p",algo="STEFCAL",type_w="GT-1")
    # experiment(0.001,0.5,20,"KAT7_PHASEGT1.p",algo="PHASE_STEF",type_w="GT-1")
    # experiment2(0.001,0.5,20,"KAT7_PHASEGT2.p",algo="PHASE_STEF",type_w="GT-1")

    # experiment(0.001,0.5,20,"KAT7_STEFR1.p",algo="STEFCAL",type_w="GTR-R")
    # experiment2(0.001,0.5,20,"KAT7_STEFR2.p",algo="STEFCAL",type_w="GTR-R")
    # experiment(0.001,0.5,20,"KAT7_PHASER1.p",algo="PHASE_STEF",type_w="GTR-R")
    # experiment2(0.001,0.5,20,"KAT7_PHASER2.p",algo="PHASE_STEF",type_w="GTR-R")

    # experiment(0.001,0.5,20,"KAT7_PHASE_STEF.p",algo="PHASE_STEF")
    # experiment(0.001,0.5,20,"KAT7_PHASE_STEF2.p",algo="PHASE_STEF")
    # plot_results("KAT7_STEFGT1.p",type_plot="G")
    # plot_results("KAT7_STEFR1.p",type_plot="R")
    # plot_results("KAT7_PHASEGT1.p",algo="PHASE",type_plot="G")
    # plot_results("KAT7_PHASER1.p",algo="PHASE",type_plot="R")
    # plot_results("KAT7_STEFR1.p",type_plot="R")
    # plot_results("KAT7_PHASER1.p",type_plot="R",algo="PHASE")
    # plot_results("EW_EXAMPLE.p")
    # plt.show()
    # two source model, can only support two point sources (at center and 1 degree right of origin) [flux,l coordinate,m coordinate]
    # plot_image("PHASE_STEF_45_image.p")
    # plot_image("STEF_45_image.p",convert_to_degrees = False)
    # plot_image("PHASE_STEF_3S_45_image.p")
    # plot_image_full("PHASE_STEF_image.p")
    # plot_image_full("STEF_image.p")
    # plot_image_full("PHASE_STEF_3S_image.p")
    # plot_image("PHASE_STEF_IN_45_image.p",precentage=False)
    # plot_image("PHASE_IN_45_image.p",precentage=False)

    # Left out
    # point_sources_true = np.array([(1,0,0),(0.5,(1*np.pi)/180,(0*np.pi)/180),(1,(0.5*np.pi)/180,(0.5*np.pi)/180),(0.5,(-0.5*np.pi)/180,(0*np.pi)/180),(0.5,(-np.sqrt(2)*np.pi)/180,(1.3*np.pi)/180)])
    # point_sources_model = np.array([(1,0,0),(1,(0.5*np.pi)/180,(0.5*np.pi)/180)])

    # SYMMETRIC SOURCE GONE?
    # point_sources_true = np.array([(1,0,0),(0.8,(np.sqrt(2)*np.pi)/180,(0*np.pi)/180),(0.2,(-1*np.pi)/180,(0.3*np.pi)/180),(0.2,(1*np.pi)/180,(1.3*np.pi)/180)])
    # point_sources_model = np.array([(1,0,0),(0.8,(np.sqrt(2)*np.pi)/180,(0*np.pi)/180)])

    # point_sources_true = np.array([(1,0,0),(0.1,(1*np.pi)/180,(0*np.pi)/180),(0.1,(np.sqrt(2)*np.pi)/180,(np.sqrt(2)*np.pi)/180),(0.1,(-2*np.pi)/180,(1*np.pi)/180)])
    # point_sources_true = np.array([(1,0,0),(0.2,(1*np.pi)/180,(0*np.pi)/180)])
    # point_sources_model = np.array([(1,0,0)])

    # LOFAR POSITIONS
    # point_sources_true = np.array([(60,0,0),(10,(-2.87454196*np.pi)/180,(1.45538186*np.pi)/180),(10,(2.7595439*np.pi)/180,(-1.56301629*np.pi)/180)])
    # point_sources_model = np.array([(60,0,0),(50,(-2.87454196*np.pi)/180,(1.45538186*np.pi)/180)])

    # SYMMETRIC SOURCE GONE?
    # point_sources_true = np.array([(60,0,0),(10,(1*np.pi)/180,(0*np.pi)/180),(10,(-1*np.pi)/180,(0*np.pi)/180)])
    # point_sources_model = np.array([(60,0,0),(30,(1*np.pi)/180,(0*np.pi)/180)])
    print("EXECUTING....")
    # initializes ghost object
    # ant_list = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
    ant_list = "all"
    point_sources_true = np.array([(1, 0, 0), (0.2, (1 * np.pi) / 180, (0 * np.pi) / 180)])
    point_sources_model = np.array([(1, 0, 0)])
    t = T_ghost(point_sources_true, point_sources_model, ant_list, "KAT7")
    # t.plot_ghost_mask_paper([4,5],pickle_file="pat1.p")

    # point_sources_true = np.array([(1,0,0),(0.2,(-2*np.pi)/180,(1*np.pi)/180)])
    # point_sources_model = np.array([(1,0,0)])
    # t = T_ghost(point_sources_true,point_sources_model,ant_list,"KAT7")
    # 3t.plot_ghost_mask_paper([4,5],pickle_file="pat2.p")

    # point_sources_true = np.array([(1,0,0),(0.2,(np.sqrt(2)*np.pi)/180,(np.sqrt(2)*np.pi)/180)])
    # point_sources_model = np.array([(1,0,0)])
    # t = T_ghost(point_sources_true,point_sources_model,ant_list,"KAT7")
    # t.plot_ghost_mask_paper([4,5],pickle_file="pat3.p")

    # plot_ghost_pat_p("pat1.p","pat2.p","pat3.p",t)
    # plots ghost map for baseline 01 (resolution 150 arcseconds, extend 3 degrees)
    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GTR-R",plot=True,mask=False,algo="STEFCAL")
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma=0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE",algo2="PHASE_STEF",no_auto=True,take_conj1=False,take_conj2=False)
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GTR",plot=True,mask=False,algo="PHASE",no_auto=False)
    # image,l_v,m_v = t.sky_pq_2D_LM([0,1],150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="STEFCAL",no_auto=True,pickle_file="EW_01.p")
    # plot_image("EW_01.p")
    image, l_v, m_v = t.sky_pq_2D_LM(args.baseline, 150, args.radius, 2, sigma=args.sigma, type_w=args.type, plot=True,
                                     mask=args.mask,
                                     algo=args.algo, no_auto=True, pickle_file=args.pickle_file,
                                     save_fig=args.dont_save)
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=False,mask=False,algo="PHASE_STEF",no_auto=True,pickle_file="PHASE_STEF_45_image.p")
    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GT-1",plot=False,mask=False,algo="PHASE_STEF",pickle_file="PHASE_STEF_image.p",no_auto=True)
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=False,mask=False,algo="STEFCAL",no_auto=False,pickle_file="STEF_45_image.p")
    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GT-1",plot=False,mask=False,algo="STEFCAL",pickle_file="STEF_image.p",no_auto=False)
    # plot_image("PHASE_STEF_45_image.p")

    # point_sources_true = np.array([(1,0,0),(0.1,(1*np.pi)/180,(0*np.pi)/180),(0.1,(np.sqrt(2)*np.pi)/180,(np.sqrt(2)*np.pi)/180),(0.1,(-2*np.pi)/180,(1*np.pi)/180)])
    # t = T_ghost(point_sources_true,point_sources_model,ant_list,"KAT7")
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE_STEF",no_auto=True,pickle_file="PHASE_STEF_3S_45_image.p")
    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE_STEF",pickle_file="PHASE_STEF_3S_image.p")

    # point_sources_true = np.array([(1,0,0),(1.5,(1*np.pi)/180,(0*np.pi)/180)])
    # t = T_ghost(point_sources_true,point_sources_model,ant_list,"KAT7")
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE_STEF",no_auto=True,pickle_file="PHASE_STEF_IN_45_image.p")
    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE_STEF",pickle_file="PHASE_STEF_IN_image.p")

    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="STEFCAL")
    # image,l_v,m_v = t.sky_2D(150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE_STEF")
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE",algo2="PHASE_STEF",no_auto=True)
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="PHASE",algo2="PHASE_STEF_NORM1",no_auto=True)
    # t.plot_ghost_mask_paper([4,5])
    # image,l_v,m_v = t.sky_pq_2D_LM([2,3],150,8,2,sigma = 0.1,type_w="GT-1",plot=True,mask=False,algo="STEFCAL",no_auto=True)
    # image,l_v,m_v = t.sky_pq_2D_LM([2,3],150,8,2,sigma = 0.1,type_w="GT-1",plot=True,mask=False,algo="STEFCAL",no_auto=False)

    # image,l_v,m_v = t.sky_pq_2D_LM([1,5],90,2,2,sigma = 0.021,type_w="GT-1",plot=True,mask=False,algo="STEFCAL",no_auto=True)
    # image,l_v,m_v = t.sky_pq_2D_LM([1,5],90,2,2,sigma = 0.021,type_w="GT-1",plot=True,mask=False,algo="STEFCAL",no_auto=False)
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GT-1",plot=True,mask=False,algo="STEFCAL",no_auto=False)
    # image,l_v,m_v = t.sky_pq_2D_LM([4,5],150,4,2,sigma = 0.05,type_w="GTR-R",plot=True,mask=False,algo="PHASE")
    # image,l_v,m_v = t.sky_pq_2D_LM([0,1],250,3,2,sigma = 0.05,type_w="GTR-R",plot=True,mask=True)
    # plots ghost map for baseline 12 (resolution 150 arcseconds, extend 3 degrees)
    # image,l_v,m_v = t.sky_pq_2D_LM([1,2],150,3,2,sigma = 0.05,type_w="GTR",plot=True,mask=True)

    # NB naive noise implementation verify if correct yourself....
