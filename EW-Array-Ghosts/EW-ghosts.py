import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import math
import scipy


def antenna_layout():
    layout = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [-1, 0, 0]])
    plt.scatter(layout[:, 0], layout[:, 1])
    plt.xlabel('E-W [m]')
    plt.ylabel('N-S [m]')
    plt.title('Array Layout')
    plt.savefig("images/Antenna_Layout")
    return layout


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


def visibilities(true_sources, model_sources, radius, baseline, resolution, s):
    Phi = np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]])

    temp = np.ones(Phi.shape, dtype=complex)
    s_old = s

    # FFT SCALING
    ######################################################
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
    u_dim = uu.shape[0]
    v_dim = uu.shape[1]
    # delta_u = 1 / (2 * s * radius * (np.pi / 180))
    # delta_v = delta_u

    # delta_l = resolution * (1.0 / 3600.0) * (np.pi / 180.0)
    # delta_m = delta_l

    # N = int(np.ceil(1 / (delta_l * delta_u))) + 1

    # if (N % 2) == 0:
    #     N = N + 1

    # delta_l_new = 1 / ((N - 1) * delta_u)
    # delta_m_new = delta_l_new

    # u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
    # v = np.linspace(-(N - 1) / 2 * delta_v, (N - 1) / 2 * delta_v, N)
    # l_cor = np.linspace(-1 / (2 * delta_u), 1 / (2 * delta_u), N)
    # m_cor = np.linspace(-1 / (2 * delta_v), 1 / (2 * delta_v), N)
    # uu, vv = np.meshgrid(u, v)
    # u_dim1 = uu.shape[0]
    # u_dim2 = uu.shape[1]

    # phi = Phi[baseline[0], baseline[1]]
    # dec = np.pi / 2.0

    r_pq = np.zeros(uu.shape, dtype=complex)
    g_pq = np.zeros(uu.shape, dtype=complex)
    # r_pq = np.zeros((u_dim, v_dim), dtype=complex)
    # g_pq = np.zeros((u_dim, v_dim), dtype=complex)
    m_pq = np.zeros((u_dim, v_dim), dtype=complex)
    temp = np.ones(Phi.shape, dtype=complex)

    for i in range(u_dim):
        progress_bar(i, u_dim)
        for j in range(v_dim):
            u_t = u[i]
            v_t = v[i]

            # SCALING
            # u_t = u_t / phi
            # v_t = v_t / (np.sin(dec) * phi)

            # u_t_m = Phi * u_t
            # v_t_m = Phi * np.sin(dec) * v_t

            u_m = (Phi*u_t)/(1.0*Phi[baseline[0], baseline[1]])
            v_m = (Phi*v_t)/(1.0*Phi[baseline[0], baseline[1]])

            R = np.zeros(u_m.shape, dtype=complex)

            for k in range(len(true_sources)):
                source = true_sources[k]
                R = R + source[0] * np.exp(-2 * 1j * np.pi * (u_m * source[1] + v_m * source[2]))
                R *= Gauss(source[3], u_m, v_m) if len(source) > 3 else 1
            # print(R)

            M = np.zeros(u_m.shape, dtype=complex)
            for k in range(len(model_sources)):
                source = model_sources[k]
                M = M + source[0] * np.exp(-2 * 1j * np.pi * (u_m * source[1] + v_m * source[2]))
                M *= Gauss(source[3], u_m, v_m) if len(source) > 3 else 1

            g_stef, G = create_G_stef(R, M, temp, 200, 1e-9)

            r_pq[j, i] = R[baseline[0], baseline[1]]
            m_pq[j, i] = M[baseline[0], baseline[1]]
            g_pq[j, i] = G[baseline[0], baseline[1]]

    return u, v, g_pq, r_pq, delta_u, delta_v


def Gauss(sigma, uu, vv):
    # print(np.exp(
    #     -2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2)))
    return (2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))


def plt_circle_grid(grid_m):
    rad = np.arange(1, 1 + grid_m, 1)
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)

    x_c = np.cos(2 * np.pi * x)
    y_c = np.sin(2 * np.pi * y)
    for k in range(len(rad)):
        plt.plot(rad[k] * x_c, rad[k] * y_c, "k", ls=":", lw=0.5)


def plot_image(image, radius, baseline, A_2, vis_type, s):
    fig = plt.figure()
    cs = plt.imshow(image.real, interpolation="bicubic", cmap="cubehelix", extent=[-s * radius, s * radius, -s * radius, s * radius])

    cb = fig.colorbar(cs)
    cb.set_label(r"Flux [% of $A_2$]")
    plt_circle_grid(radius)

    # plt.xlim([-radius, radius])
    # plt.ylim([-radius, radius])

    plt.xlabel("$l$ [degrees]")
    plt.ylabel("$m$ [degrees]")
    plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Real")

    plt.savefig("images/Figure_R_pq" + str(baseline[0]) + str(baseline[1]) + " " + vis_type + ".png", format="png",
                bbox_inches="tight")
    plt.clf()

    # fig = plt.figure()
    # cs = plt.imshow(-1 * (image.imag / A_2) * 100, interpolation="bicubic", cmap="cubehelix",
    #                 extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
    # cb = fig.colorbar(cs)
    # cb.set_label(r"Flux [% of $A_2$]")

    # plt_circle_grid(radius)

    # # plt.xlim([-radius, radius])
    # # plt.ylim([-radius, radius])

    # plt.xlabel("$l$ [degrees]")
    # plt.title("Baseline " + str(baseline[0]) + str(baseline[1]) + " --- Imag")
    # plt.ylabel("$m$ [degrees]")
    # plt.savefig("images/Figure_I_pq" + str(baseline[0]) + str(baseline[1]) + " " + vis_type + ".png", format="png",
    #             bbox_inches="tight")
    # plt.clf()


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


def vis_function(vis_function, g_pq, g_qp, r_pq):
    if vis_function == "R":
        vis = r_pq
    elif vis_function == "GT":
        vis = g_pq ** (-1)
    elif vis_function == "GT-1":
        vis = g_pq ** (-1) - 1
    return vis


if __name__ == "__main__":
    radius = 3
    baseline = [1, 2]
    resolution = 100
    sigma = 0.5
    s = 1
    vis_type = "GT-1"  # R or GT
    add_gaussian = True
    add_model_gaussian = False

    true_sources = np.array([])
    if add_gaussian:
        true_sources = np.array(
            [[1, 0, 0, (0.1 * np.pi) / 180], [0.2, (1 * np.pi) / 180, (0 * np.pi) / 180, (0.2 * np.pi) / 180]])
    else:
        true_sources = np.array(
            [[1, 0, 0], [0.2, (1 * np.pi) / 180, (0 * np.pi) / 180]])

    model_sources = np.array([])
    if add_model_gaussian:
        model_sources = np.array([[1, 0, 0, (sigma * np.pi) / 180]])
    else:
        model_sources = np.array([[1, 0, 0]])
    antenna_layout()
    u, v, g_pq, r_pq, delta_u, delta_v = visibilities(
        true_sources, model_sources, radius, baseline, resolution, s)
    g_qp = 0

    # R Image
    vis_type = "R"
    vis = vis_function(vis_type, g_pq, g_qp, r_pq)
    # N = l_cor.shape[0]


    # zz = g_pq
    # zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
    # zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

    # zz_f = np.fft.fft2(zz) * (delta_u*delta_v)
    # zz_f = np.roll(zz_f, -int(zz.shape[0]/2), axis=0)
    # zz_f = np.roll(zz_f, -int(zz.shape[0]/2), axis=1)

    image = vis[:, ::-1]
    image = np.roll(image, -int(image.shape[0] / 2), axis=0)
    image = np.roll(image, -int(image.shape[0] / 2), axis=1)

    image_f = np.fft.fft2(vis) * (delta_u*delta_v)
    image_f = np.roll(image, -int(image.shape[0] / 2), axis=0)
    image_f = np.roll(image, -int(image.shape[0] / 2), axis=1)

    A_2 = true_sources[1][0]
    plot_image(image_f, radius, baseline, A_2, vis_type, s)

    # GT-1 Image
    vis_type = "GT-1"
    vis = vis_function(vis_type, g_pq, g_qp, r_pq)

    image = vis[:, ::-1]
    image = np.roll(image, -int(image.shape[0] / 2), axis=0)
    image = np.roll(image, -int(image.shape[0] / 2), axis=1)

    image_f = np.fft.fft2(vis) * (delta_u*delta_v)
    image_f = np.roll(image, -int(image.shape[0] / 2), axis=0)
    image_f = np.roll(image, -int(image.shape[0] / 2), axis=1)

    A_2 = true_sources[1][0]
    plot_image(image, radius, baseline, A_2, vis_type, s)

    # GT Image
    vis_type = "GT"
    vis = vis_function(vis_type, g_pq, g_qp, r_pq)
    
    image = vis[:, ::-1]
    image = np.roll(image, -int(image.shape[0] / 2), axis=0)
    image = np.roll(image, -int(image.shape[0] / 2), axis=1)

    image_f = np.fft.fft2(vis) * (delta_u*delta_v)
    image_f = np.roll(image, -int(image.shape[0] / 2), axis=0)
    image_f = np.roll(image, -int(image.shape[0] / 2), axis=1)

    A_2 = true_sources[1][0]
    plot_image(image, radius, baseline, A_2, vis_type, s)
