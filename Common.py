import pickle
import numpy as np
import pylab as plt
import matplotlib.colors as colors


def magic_baseline(p=0, q=1):
    if (p == 0) and (q == 13):
        return True
    if (p == 7) and (q == 13):
        return True
    if (p == 9) and (q == 10):
        return True
    if (p == 11) and (q == 13):
        return True
    if (p == 2) and (q == 3):
        return True

    return False


def load_g_pickle(phi, k, j):
    name = "data/G/g_" + str(k) + "_" + str(j) + "_" + str(phi[k, j]) + ".p"

    pkl_file = open(name, "rb")
    g_pq_t = pickle.load(pkl_file)
    g_pq = pickle.load(pkl_file)
    g_pq_inv = pickle.load(pkl_file)
    r_pq = pickle.load(pkl_file)
    g_kernal = pickle.load(pkl_file)
    sigma_kernal = pickle.load(pkl_file)
    B = pickle.load(pkl_file)
    pkl_file.close()

    return g_pq_t, g_pq, g_pq_inv, r_pq, g_kernal, sigma_kernal, B


def load_14_10_pickle(phi, k, j):
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
    g_kernal = pickle.load(pkl_file)
    g_pq_inv12 = pickle.load(pkl_file)
    g_pq_t12 = pickle.load(pkl_file)
    sigma_b = pickle.load(pkl_file)
    delta_u = pickle.load(pkl_file)
    delta_l = pickle.load(pkl_file)
    s_size = pickle.load(pkl_file)
    siz = pickle.load(pkl_file)
    r = pickle.load(pkl_file)
    phi = pickle.load(pkl_file)
    K1 = pickle.load(pkl_file)
    K2 = pickle.load(pkl_file)

    pkl_file.close()
    return (
        g_pq12,
        r_pq12,
        g_kernal,
        g_pq_inv12,
        g_pq_t12,
        sigma_b,
        delta_u,
        delta_l,
        s_size,
        siz,
        r,
        phi,
        K1,
        K2,
    )


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


def image(image, psf, l_cor, delta_u, sigma, name, add_circle=False):
    sigma = sigma * (np.pi / 180)
    zz_psf = psf
    zz_psf = np.roll(zz_psf, -int(zz_psf.shape[0] / 2), axis=0)
    zz_psf = np.roll(zz_psf, -int(zz_psf.shape[0] / 2), axis=1)

    zz_f_psf = np.fft.fft2(zz_psf)
    mx = np.max(np.absolute(zz_f_psf))
    zz_f_psf = zz_f_psf / mx
    zz_f_psf = np.roll(zz_f_psf, -int(zz_psf.shape[0] / 2) - 1, axis=0)
    zz_f_psf = np.roll(zz_f_psf, -int(zz_psf.shape[0] / 2) - 1, axis=1)
    zz = image
    zz = np.roll(zz, -int(zz.shape[0] / 2), axis=0)
    zz = np.roll(zz, -int(zz.shape[0] / 2), axis=1)

    zz_f = np.fft.fft2(zz)
    zz_f = zz_f / mx
    zz_f = np.roll(zz_f, -int(zz.shape[0] / 2) - 1, axis=0)
    zz_f = np.roll(zz_f, -int(zz.shape[0] / 2) - 1, axis=1)
    cmap = colors.LinearSegmentedColormap.from_list(
        "nameofcolormap", ["b", "y", "r"], gamma=0.35
    )

    plt.rcParams["font.size"] = "16"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        np.absolute(zz_f_psf),
        extent=[
            l_cor[0] * (180.0 / np.pi),
            l_cor[-1] * (180.0 / np.pi),
            l_cor[0] * (180.0 / np.pi),
            l_cor[-1] * (180.0 / np.pi),
        ],
        cmap=cmap,
        interpolation="bicubic",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jy/beam", labelpad=10)
    ax.set_xlabel(r"$l$ [degrees]")
    ax.set_ylabel(r"$m$ [degrees]")
    plt.savefig(fname="plots/imaging_results/" + name + "_psf.pdf")
    plt.savefig(fname="plots/imaging_results/" + name + "_psf.png", dpi=200)
    plt.cla()
    plt.close()

    phi = np.linspace(0, 2 * np.pi, 100)
    x = 0.04 * np.cos(phi)
    y = 0.04 * np.sin(phi)

    x2 = 2 * 0.0019017550075500233 * np.cos(phi)
    y2 = 2 * 0.0019017550075500233 * np.sin(phi)
    x3 = 0.04*0.38219481012960854*np.cos(phi)
    y3 = 0.04*0.38219481012960854*np.sin(phi)  

    plt.rcParams["font.size"] = "16"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        np.absolute(zz_f),
        extent=[
            l_cor[0] * (180.0 / np.pi),
            l_cor[-1] * (180.0 / np.pi),
            l_cor[0] * (180.0 / np.pi),
            l_cor[-1] * (180.0 / np.pi),
        ],
        cmap="jet",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jy/beam", labelpad=10)
    ax.set_xlabel(r"$l$ [degrees]")
    ax.set_ylabel(r"$m$ [degrees]")

    if add_circle:
        ax.plot(x, y, "r", lw=2.0)
        ax.plot(x2, y2, "k", lw=2.0)
        ax.plot(x3,y3,"y",lw=2.0)

    plt.savefig(fname="plots/imaging_results/" + name + ".pdf")
    plt.savefig(fname="plots/imaging_results/" + name + ".png", dpi=200)
    plt.cla()
    plt.close()
    return zz_f_psf, zz_f


def create_G_stef(R, M, imax, tau, temp, no_auto):
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


def extrapolation_calculation(s, u_m, v_m):
    return s[0] * np.exp(
        -2 * np.pi * 1j * (u_m * (s[1] * np.pi / 180.0) + v_m * (s[2] * np.pi / 180.0))
    )

def extrapolation_loop(true_sky_model, cal_sky_model, u_m, v_m, R, M):
    for k in range(len(true_sky_model)):
        s = true_sky_model[k]
        if len(s) <= 3:
            R += extrapolation_calculation(s, u_m, v_m)
        else:
            sigma = s[3] * (np.pi / 180)
            g_kernal = g_kernel_calculation(sigma, u_m, v_m)
            R += extrapolation_calculation(s, u_m, v_m) * g_kernal
    
    for k in range(len(cal_sky_model)):
        s = cal_sky_model[k]
        if len(s) <= 3:
            M += extrapolation_calculation(s, u_m, v_m)
        else:
            sigma = s[3] * (np.pi / 180)
            g_kernal = g_kernel_calculation(sigma, u_m, v_m)
            M += extrapolation_calculation(s, u_m, v_m) * g_kernal
    return R, M

def g_kernel_calculation(sigma, u_m, v_m):
    return (
        2
        * np.pi
        * sigma**2
        * np.exp(-2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2))
    )
