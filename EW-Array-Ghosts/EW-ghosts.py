import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import math
import scipy


def antenna_layout():
    layout = np.array([[0, 0, 0],
                       [50, 0, 0],
                       [-50, 0, 0]])
    plt.scatter(layout[:, 0], layout[:, 1])
    plt.xlabel('E-W [m]')
    plt.ylabel('N-S [m]')
    plt.title('Array Layout')
    plt.show()
    return layout


# def create_G_stef(N, R, M, temp, imax, tau):
#     '''This function finds argmin G ||R-GMG^H|| using StEFCal.
#      R is your observed visibilities matrx.
#      M is your predicted visibilities.
#      imax maximum amount of iterations.
#      tau stopping criteria.
#      g the antenna gains.
#      G = gg^H.'''
#     g_temp = np.ones((N,), dtype=complex)
#     for k in range(imax):
#         g_old = np.copy(g_temp)
#         for p in range(N):
#             z = g_old * M[:][p]
#             g_temp[p] = np.sum(np.conj(R[:, p]) * z) / (np.sum(np.conj(z) * z))

#         if (k % 2 == 0):
#             if (np.sqrt(np.sum(np.absolute(g_temp - g_old) ** 2)) / np.sqrt(
#                     np.sum(np.absolute(g_temp) ** 2)) <= tau):
#                 break
#             else:
#                 g_temp = (g_temp + g_old) / 2

#     G_m = np.dot(np.diag(g_temp), temp)
#     G_m = np.dot(G_m, np.diag(g_temp.conj()))

#     g = g_temp
#     G = G_m

#     return g, G

def create_G_stef(R, M, temp, imax, tau):
    N = R.shape[0]
    g_temp = np.ones((N,), dtype=complex)
    for k in range(imax):
        g_old = np.copy(g_temp)
        for p in range(N):
            z = g_old * M[:][p]
            g_temp[p] = np.sum(np.conj(R[:][p]) * z) / (np.sum(np.conj(z) * z))

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


def get_B(b_ENU, L):
    """
    Converts the xyz form of the baseline into the coordinate system XYZ

    :param b_ENU: The baseline to convert
    :type b_ENU: float array

    :param L: The latitude of the interferometer
    :type L: float

    :returns: The baseline in XYZ
    """
    D = math.sqrt(np.sum((b_ENU)**2))
    A = np.arctan2(b_ENU[0], b_ENU[1])
    E = np.arcsin(b_ENU[2]/D)
    B = np.array([D * (math.cos(L)*math.sin(E) - math.sin(L) * math.cos(E)*math.cos(A)),
                  D * (math.cos(E)*math.sin(A)),
                  D * (math.sin(L)*math.sin(E) + math.cos(L) * math.cos(E)*math.cos(A))])
    return B


def get_lambda(f):
    """
    Gets the wavelength for calculations

    :param f: The frequency of the interferometer
    :type f: float

    :returns: Lambda, wavelength
    """
    c = scipy.constants.c
    lam = c/f
    return lam


def visibilities(true_sources, model_sources, radius, baseline, resolution, s):
    delta_u = 1 / (2 * s * radius * (np.pi / 180))
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

    phi_m = np.array([(0, 3, 5), (-3, 0, 2), (-5, -2, 0)])
    phi = phi_m[baseline[0], baseline[1]]
    dec = np.pi / 2.0

    V_R_pq = np.zeros(uu.shape, dtype=complex)
    V_G_pq = np.zeros(uu.shape, dtype=complex)
    temp = np.ones(phi_m.shape, dtype=complex)

    for i in range(u_dim1):
        progress_bar(i, u_dim1)
        for j in range(u_dim2):
            if u_dim2 != 1:
                u_t = uu[i][j]
                v_t = vv[i][j]
            else:
                u_t = uu[i]
                v_t = vv[i]

            # SCALING
            u_t = u_t / phi
            v_t = v_t / (np.sin(dec) * phi)

            u_t_m = phi_m * u_t
            v_t_m = phi_m * np.sin(dec) * v_t

            R = np.zeros(u_t_m.shape)

            def Gauss(sigma, uu, vv): return (2 * np.pi * sigma ** 2) * np.exp(
                -2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))

            for k in range(len(true_sources)):
                R = R + true_sources[k][0] * np.exp(-2 * 1j * np.pi * (
                    u_t_m * true_sources[k][1] + v_t_m * true_sources[k][2]))
                R *= Gauss(true_sources[k][3], u_t_m, v_t_m) if len(
                    true_sources[k]) > 3 else 1

            M = np.zeros(u_t_m.shape)
            for k in range(len(model_sources)):
                M = M + model_sources[k][0] * np.exp(-2 * 1j * np.pi * (
                    u_t_m * model_sources[k][1] + v_t_m * model_sources[k][2]))
                M *= Gauss(model_sources[k][3], u_t_m, v_t_m) if len(
                    model_sources[k]) > 3 else 1

            g_stef, G = create_G_stef(R, M, temp, 200, 1e-9)

            if u_dim2 != 1:
                V_R_pq[i][j] = R[baseline[0], baseline[1]]
                V_G_pq[i][j] = G[baseline[0], baseline[1]]
            else:
                V_R_pq[i] = R[baseline[0], baseline[1]]
                V_G_pq[i] = G[baseline[0], baseline[1]]

        return u, v, V_G_pq, V_R_pq, phi, l_cor, m_cor


def plt_circle_grid(grid_m):
    rad = np.arange(1, 1 + grid_m, 1)
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)

    x_c = np.cos(2 * np.pi * x)
    y_c = np.sin(2 * np.pi * y)
    for k in range(len(rad)):
        plt.plot(rad[k] * x_c, rad[k] * y_c, "k", ls=":", lw=0.5)


def plot_image(image, l_cor, m_cor, radius, baseline, A_2):
    l_cor = l_cor * (180 / np.pi)
    m_cor = m_cor * (180 / np.pi)

    fig = plt.figure()
    cs = plt.imshow((image.real / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                    extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
    cb = fig.colorbar(cs)
    cb.set_label(r"Flux [% of $A_2$]")
    plt_circle_grid(radius)
    # if label_v:
    #     self.plot_source_labels_pq(baseline, im=image_s, plot_x=False)

    print("amax_real = ", np.amax((image.real / A_2) * 100))
    print("amin_real = ", np.amin((image.real / A_2) * 100))
    # print "amax = ",np.amax(np.absolute(image))

    plt.xlim([-radius, radius])
    plt.ylim([-radius, radius])

    # if mask:
    #     p = self.create_mask(baseline, plot_v=True, dec=dec)

    plt.xlabel("$l$ [degrees]")
    plt.ylabel("$m$ [degrees]")
    plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Real")

    plt.savefig("images\Figure_R_pq" + str(baseline[0]) + str(baseline[1]) + ".png", format="png",
                bbox_inches="tight")
    plt.clf()

    fig = plt.figure()
    cs = plt.imshow(-1 * (image.imag / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
                    extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
    cb = fig.colorbar(cs)
    cb.set_label(r"Flux [% of $A_2$]")

    print("amax_imag = ", np.amax((image.imag / A_2) * 100))
    print("amin_imag = ", np.amin((image.imag / A_2) * 100))

    plt_circle_grid(radius)
    # if label_v:
    #     self.plot_source_labels_pq(baseline, im=radius, plot_x=False)

    plt.xlim([-radius, radius])
    plt.ylim([-radius, radius])

    # if mask:
    #     self.create_mask(baseline, plot_v=True, dec=dec)

    plt.xlabel("$l$ [degrees]")
    plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Imag")
    plt.ylabel("$m$ [degrees]")
    plt.savefig("images\Figure_I_pq" + str(baseline[0]) + str(baseline[1]) + ".png", format="png",
                bbox_inches="tight")
    plt.clf()


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
    radius = 3
    baseline = [1, 2]
    resolution = 150
    sigma = 0.05
    s = 2

    true_sources = np.array([[0, 0, 0], [0, 1, 2]])
    model_sources = np.array([[0, 0, 0]])
    layout = antenna_layout()
    u, v, V_G_pq, V_R_pq, phi, l_cor, m_cor = visibilities(
        model_sources, true_sources, radius, baseline, resolution, s)
    # vis = self.vis_function(type_w, avg_v, V_G_pq, V_G_qp, V_R_pq, take_conj1)
    vis = V_R_pq
    # V_G_qp = 0
    # vis = (V_G_pq ** (-1) + V_G_qp ** (-1)) / 2
    N = l_cor.shape[0]
    image = np.fft.fft2(vis) / N ** 2

    image = np.roll(image, int(1 * (N - 1) / 2), axis=0)
    image = np.roll(image, int(1 * (N - 1) / 2), axis=1)

    A_2 = true_sources[1][0]
    plot_image(image, l_cor, m_cor, radius, baseline, A_2)
