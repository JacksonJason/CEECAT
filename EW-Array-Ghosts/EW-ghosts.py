import numpy as np
import pylab as plt
import sys

import matplotlib.pyplot as plt
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

    def plt_circle_grid(self, grid_m):
        rad = np.arange(1, 1 + grid_m, 1)
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 1, 500)

        x_c = np.cos(2 * np.pi * x)
        y_c = np.sin(2 * np.pi * y)
        for k in range(len(rad)):
            plt.plot(rad[k] * x_c, rad[k] * y_c, "k", ls=":", lw=0.5)

    def plot_image(self, type_plot, kernel, g_pq, r_pq, delta_u, delta_v, s_old, image_s, uu, vv, baseline):

        sigma = 0.05*(np.pi/180)
        g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
        if type_plot == "GT-1":
            vis = (g_pq)**(-1)-1
        elif type_plot == "R":
            vis = r_pq
        else:
            vis = (g_pq)**(-1)*r_pq - r_pq

        if kernel:
            vis = vis*g_kernal
        vis = vis[:, ::-1]

        # IMAGING QUICKLY
        zz = vis
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

        zz_f = np.fft.fft2(zz) * (delta_u*delta_v)
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2), axis=0)
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2), axis=1)

        fig, ax = plt.subplots()
        im = ax.imshow(
            zz_f.real, cmap="cubehelix", extent=[-s_old*image_s, s_old*image_s, -s_old*image_s, s_old*image_s])
        fig.colorbar(im, ax=ax)
        self.plt_circle_grid(image_s)

        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        plt.title("Baseline " +
                  str(baseline[0]) + str(baseline[1]) + " --- Real")

        plt.savefig("images/Figure_Real_pq" + str(baseline[0]) + str(
            baseline[1]) + " " + type_plot + ".png", format="png", bbox_inches="tight")

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on
    """

    def extrapolation_function(self, baseline, true_sky_model, cal_sky_model, Phi, image_s, s, resolution, kernel):
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
            progress_bar(i, u_dim)
            for j in range(v_dim):
                ut = u[i]
                vt = v[j]
                u_m = (Phi*ut)/(1.0*Phi[baseline[0], baseline[1]])
                v_m = (Phi*vt)/(1.0*Phi[baseline[0], baseline[1]])
                R = np.zeros(Phi.shape, dtype=complex)
                M = np.zeros(Phi.shape, dtype=complex)
                for k in range(len(true_sky_model)):
                    s = true_sky_model[k]
                    if len(s) <= 3:
                        R += s[0]*np.exp(-2*np.pi*1j*(u_m*s[1])+v_m*s[2])
                    else:
                        sigma = s[3]
                        g_kernal = 2*np.pi*sigma**2 * \
                            np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                        R += s[0]*np.exp(-2*np.pi*1j *
                                         (u_m*s[1]+v_m*s[2]))*g_kernal
                for k in range(len(cal_sky_model)):
                    s = cal_sky_model[k]
                    if len(s) <= 3:
                        M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]
                                         * np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                        sigma = s[3]
                        g_kernal = 2*np.pi*sigma**2 * \
                            np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                        M += s[0]*np.exp(-2*np.pi*1j *
                                         (u_m*s[1]+v_m*s[2]))*g_kernal
                g_stef, G = self.create_G_stef(
                    R, M, 200, 1e-9, temp, no_auto=False)

                r_pq[j, i] = R[baseline[0], baseline[1]]
                m_pq[j, i] = M[baseline[0], baseline[1]]
                g_pq[j, i] = G[baseline[0], baseline[1]]
        self.plot_image("GT-1", kernel, g_pq, r_pq, delta_u,
                        delta_v, s_old, image_s, uu, vv, baseline)
        self.plot_image("GTR-R", kernel, g_pq, r_pq, delta_u,
                        delta_v, s_old, image_s, uu, vv, baseline)
        self.plot_image("R", kernel, g_pq, r_pq, delta_u,
                        delta_v, s_old, image_s, uu, vv, baseline)


def progress_bar(count, total):
    """
    Taken from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s iteration %s\r' % (bar, percents, '%', count))
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",
                        type=int,
                        nargs='+',
                        default=[0, 1],
                        help="The baseline to calculate on")
    parser.add_argument("--experiment",
                        type=str,
                        default="1.1",
                        help="Run a pre-defined experiment")

    global args
    args = parser.parse_args()
    t = T_ghost()

    baseline = np.array(args.baseline)

    # point source case GT-1
    # t.extrapolation_function(baseline=baseline, true_sky_model=np.array([[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]), cal_sky_model=np.array(
    #     [[1*np.pi/180.0, 0, 0]]), Phi=np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]), image_s=3, s=1, resolution=100, kernel=True)

    if (args.experiment == "1.1"):
        # Experiment 1.1
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "1.2"):
        # Experiment 1.2
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "1.5"):
        # Experiment 1.5
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "1.6"):
        # Experiment 1.6
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "1.7"):
        # Experiment 1.7
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "2.1"):
        # Experiment 2.1
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "2.2"):
        # Experiment 2.2
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "2.4"):
        # Experiment 2.4
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.2*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "2.5"):
        # Experiment 2.5
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "2.7"):
        # Experiment 2.7
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "3.1"):
        # Experiment 3.1
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "3.2"):
        # Experiment 3.2
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "3.4"):
        # Experiment 3.4
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.2*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "3.6"):
        # Experiment 3.6
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "3.7"):
        # Experiment 3.7
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0], [0.2*np.pi/180.0, 1*np.pi/180.0, 0, 0.2*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)
    elif (args.experiment == "4.2"):
        # Experiment 4.2
        t.extrapolation_function(baseline=baseline,
                                 true_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0, 0.1*np.pi/180.0]]),
                                 cal_sky_model=np.array(
                                     [[1*np.pi/180.0, 0, 0]]),
                                 Phi=np.array(
                                     [[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
                                 image_s=3,
                                 s=1,
                                 resolution=100,
                                 kernel=True)

    print()
