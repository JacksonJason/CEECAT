import numpy as np
import pylab as plt
import matplotlib.colors as colors
import Common


def include_100_baseline(p=0, q=0):
    if (p == 0) and (q == 4):
        return True
    if (p == 0) and (q == 5):
        return True
    if (p == 0) and (q == 6):
        return True
    if (p == 0) and (q == 7):
        return True
    if (p == 0) and (q == 8):
        return True
    if (p == 0) and (q == 9):
        return True
    if (p == 0) and (q == 10):
        return True
    if (p == 0) and (q == 11):
        return True
    if (p == 0) and (q == 12):
        return True
    if (p == 0) and (q == 13):
        return True
    if (p == 1) and (q == 5):
        return True
    if (p == 1) and (q == 6):
        return True
    if (p == 1) and (q == 7):
        return True
    if (p == 1) and (q == 8):
        return True
    if (p == 1) and (q == 9):
        return True
    if (p == 1) and (q == 10):
        return True
    if (p == 1) and (q == 11):
        return True
    if (p == 1) and (q == 12):
        return True
    if (p == 1) and (q == 13):
        return True
    if (p == 2) and (q == 6):
        return True
    if (p == 2) and (q == 7):
        return True
    if (p == 2) and (q == 8):
        return True
    if (p == 2) and (q == 9):
        return True
    if (p == 2) and (q == 10):
        return True
    if (p == 2) and (q == 11):
        return True
    if (p == 2) and (q == 12):
        return True
    if (p == 2) and (q == 13):
        return True
    if (p == 3) and (q == 7):
        return True
    if (p == 3) and (q == 8):
        return True
    if (p == 3) and (q == 9):
        return True
    if (p == 3) and (q == 10):
        return True
    if (p == 3) and (q == 11):
        return True
    if (p == 3) and (q == 12):
        return True
    if (p == 3) and (q == 13):
        return True
    if (p == 4) and (q == 7):
        return True
    if (p == 4) and (q == 8):
        return True
    if (p == 4) and (q == 9):
        return True
    if (p == 4) and (q == 10):
        return True
    if (p == 4) and (q == 11):
        return True
    if (p == 4) and (q == 12):
        return True
    if (p == 4) and (q == 13):
        return True
    if (p == 5) and (q == 8):
        return True
    if (p == 5) and (q == 9):
        return True
    if (p == 5) and (q == 10):
        return True
    if (p == 5) and (q == 11):
        return True
    if (p == 5) and (q == 12):
        return True
    if (p == 5) and (q == 13):
        return True
    if (p == 6) and (q == 8):
        return True
    if (p == 6) and (q == 9):
        return True
    if (p == 6) and (q == 10):
        return True
    if (p == 6) and (q == 11):
        return True
    if (p == 6) and (q == 12):
        return True
    if (p == 6) and (q == 13):
        return True
    if (p == 7) and (q == 9):
        return True
    if (p == 7) and (q == 10):
        return True
    if (p == 7) and (q == 11):
        return True
    if (p == 7) and (q == 12):
        return True
    if (p == 7) and (q == 13):
        return True
    if (p == 8) and (q == 9):
        return True
    if (p == 8) and (q == 10):
        return True
    if (p == 8) and (q == 11):
        return True
    if (p == 8) and (q == 12):
        return True
    if (p == 8) and (q == 13):
        return True
    if (p == 9) and (q == 10):
        return True
    if (p == 9) and (q == 11):
        return True
    if (p == 9) and (q == 12):
        return True
    if (p == 9) and (q == 13):
        return True
    if (p == 10) and (q == 11):
        return True
    if (p == 10) and (q == 12):
        return True
    if (p == 10) and (q == 13):
        return True

    if (p == 11) and (q == 13):
        return True

    return False


def get_main_graphs(phi=np.array([])):
    counter = 0
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

    new_x = []
    new_r = []

    B_old = 0

    for k in range(len(phi)):
        for j in range(len(phi)):
            if j != k:
                if j > k:
                    if include_100_baseline(k, j):
                        counter += 1
                        (
                            g_pq_t,
                            g_pq,
                            g_pq_inv,
                            r_pq,
                            g_kernal,
                            sigma_kernal,
                            B,
                        ) = Common.load_g_pickle(phi, k, j)
                        B_old = B

                        (
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
                        ) = Common.load_14_10_pickle(phi, k, j)

                        if len(new_x) == 0:
                            new_x = np.absolute(g_pq12 ** (-1) * r_pq12)  # *B
                        else:
                            new_x += np.absolute(g_pq12 ** (-1) * r_pq12)  # *B
                        if len(new_r) == 0:
                            new_r = r_pq12
    N = g_pq12.shape[0]
    u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
    v = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
    uu, vv = np.meshgrid(u, v)
    u_dim = uu.shape[0]
    v_dim = uu.shape[1]

    f = np.zeros((u_dim, v_dim), dtype=complex)
    sigma = 0.0019017550075500233 * (np.pi / 180)

    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            f[i, j] = 1 * np.exp(
                -2 * np.pi**2 * sigma**2 * (uu[i, j] ** 2 + vv[i, j] ** 2)
            )

    fig, ax = plt.subplots()
    psf = Common.img(f, 1.0, 1.0)
    ma = np.max(psf)
    ma = 1 / ma

    phi = np.linspace(0, 2 * np.pi, 100)
    x = 0.04 * np.cos(phi)
    y = 0.04 * np.sin(phi)

    x2 = 2 * 0.0019017550075500233 * np.cos(phi)
    y2 = 2 * 0.0019017550075500233 * np.sin(phi)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Common.img(f, ma, 1.0), extent=[-siz, siz, -siz, siz], cmap="jet")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jy/beam", labelpad=10)
    ax.set_xlabel(r"$l$ [degrees]")
    ax.set_ylabel(r"$m$ [degrees]")
    
    plt.savefig(fname="plots/imaging_results/CleanBeam.pdf")
    plt.savefig(fname="plots/imaging_results/CleanBeam.png", dpi=200)
    plt.cla()
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        Common.img(f * (new_x / counter), ma, 1),
        extent=[-siz, siz, -siz, siz],
        cmap="jet",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jy/beam", labelpad=10)
    ax.set_xlabel(r"$l$ [degrees]")
    ax.set_ylabel(r"$m$ [degrees]")
    ax.plot(x, y, "r", lw=2.0)
    ax.plot(x2, y2, "k", lw=2.0)
    plt.savefig(fname="plots/imaging_results/CorrectedVis.pdf")
    plt.savefig(fname="plots/imaging_results/CorrectedVis.png", dpi=200)
    plt.cla()
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        Common.img(f * (new_r / B_old), ma, 1),
        extent=[-siz, siz, -siz, siz],
        cmap="jet",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jy/beam", labelpad=10)
    ax.set_xlabel(r"$l$ [degrees]")
    ax.set_ylabel(r"$m$ [degrees]")
    plt.savefig(fname="plots/imaging_results/Gaussian.pdf")
    plt.savefig(fname="plots/imaging_results/Gaussian.png", dpi=200)
    plt.cla()
    plt.close()

def generate_uv_tracks(P=np.array([]), freq=1.45e9, b0=36, time_slots=500, d=90):
    lam = 3e8 / freq  # observational wavelenth
    H = np.linspace(-6, 6, time_slots) * (np.pi / 12)  # Hour angle in radians
    delta = d * (np.pi / 180)  # Declination in radians

    u_m = np.zeros((len(P), len(P), len(H)), dtype=float)
    v_m = np.zeros((len(P), len(P), len(H)), dtype=float)

    for i in range(len(P)):
        for j in range(len(P)):
            if j > i:
                u_m[i, j] = lam ** (-1) * b0 * P[i, j] * np.cos(H)
                u_m[j, i] = -1 * u_m[i, j]

                v_m[i, j] = lam ** (-1) * b0 * P[i, j] * np.sin(H) * np.sin(delta)
                v_m[j, i] = -1 * v_m[i, j]

    plt.figure(figsize=(8, 6))
    for i in range(len(P)):
        for j in range(len(P)):
            if j > i:
                plt.plot(u_m[i, j, :], v_m[i, j, :], "r")
                plt.plot(-u_m[i, j, :], -v_m[i, j, :], "b")
    plt.xlabel("$u$ [rad$^{-1}$]", fontsize=18)
    plt.ylabel("$v$ [rad$^{-1}$]", fontsize=18)
    plt.title("$uv$-Coverage of WSRT", fontsize=18)
    plt.savefig(fname="plots/imaging_results/UV-Tracks.pdf")
    plt.savefig(fname="plots/imaging_results/UV-Tracks.png", dpi=200)
    return u_m, v_m


def create_vis_matrix(
    u_m=np.array([]),
    v_m=np.array([]),
    true_sky_model=np.array([[1, 0, 0], [0.2, 1, 0]]),
    cal_sky_model=np.array([[1, 0, 0]]),
):
    R = np.zeros(u_m.shape, dtype=complex)
    M = np.zeros(u_m.shape, dtype=complex)
    for k in range(len(true_sky_model)):
        s = true_sky_model[k]
        if len(s) <= 3:
            R += Common.extrapolation_calculation(s, u_m, v_m)
        else:
            sigma = s[3] * (np.pi / 180)
            g_kernal = (
                2
                * np.pi
                * sigma**2
                * np.exp(-2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2))
            )
            R += (
                s[0]
                * np.exp(
                    -2
                    * np.pi
                    * 1j
                    * (u_m * (s[1] * np.pi / 180.0) + v_m * (s[2] * np.pi / 180.0))
                )
                * g_kernal
            )
    for k in range(len(cal_sky_model)):
        s = cal_sky_model[k]
        if len(s) <= 3:
            M += Common.extrapolation_calculation(s, u_m, v_m)
        else:
            sigma = s[3] * (np.pi / 180)
            g_kernal = (
                2
                * np.pi
                * sigma**2
                * np.exp(-2 * np.pi**2 * sigma**2 * (u_m**2 + v_m**2))
            )
            M += (
                s[0]
                * np.exp(
                    -2
                    * np.pi
                    * 1j
                    * (u_m * (s[1] * np.pi / 180.0) + v_m * (s[2] * np.pi / 180.0))
                )
                * g_kernal
            )

    return R, M


def gridding(
    u_m=np.array([]),
    v_m=np.array([]),
    D=np.array([]),
    image_s=3,
    s=1,
    resolution=8,
    w=1,
    grid_all=True,
):
    delta_u = 1 / (2 * s * image_s * (np.pi / 180))
    delta_v = delta_u

    delta_l = resolution * (1.0 / 3600.0) * (np.pi / 180.0)
    N = int(np.ceil(1 / (delta_l * delta_u))) + 1

    if (N % 2) == 0:
        N = N + 1

    u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
    v = np.linspace(-(N - 1) / 2 * delta_v, (N - 1) / 2 * delta_v, N)

    l_cor = np.linspace(-1 / (2 * delta_u), 1 / (2 * delta_u), N)

    uu, vv = np.meshgrid(u, v)

    counter = np.zeros(uu.shape, dtype=int)

    grid_points = np.zeros(uu.shape, dtype=complex)

    for p in range(D.shape[0]):
        for q in range(D.shape[1]):
            if p != q:
                if grid_all:
                    for t in range(D.shape[2]):
                        idx_u = np.searchsorted(u, u_m[p, q, t])
                        idx_v = np.searchsorted(v, v_m[p, q, t])
                        if (idx_u != 0) and (idx_u != len(u)):
                            if (idx_v != 0) and (idx_v != len(v)):
                                grid_points[
                                    idx_u - w : idx_u + w, idx_v - w : idx_v + w
                                ] += D[p, q, t]
                                counter[
                                    idx_u - w : idx_u + w :, idx_v - w : idx_v + w
                                ] += 1

                else:
                    if include_100_baseline(p, q) or include_100_baseline(q, p):
                        for t in range(D.shape[2]):
                            idx_u = np.searchsorted(u, u_m[p, q, t])
                            idx_v = np.searchsorted(v, v_m[p, q, t])
                            if (idx_u != 0) and (idx_u != len(u)):
                                if (idx_v != 0) and (idx_v != len(v)):
                                    grid_points[
                                        idx_u - w : idx_u + w, idx_v - w : idx_v + w
                                    ] += D[p, q, t]
                                    counter[
                                        idx_u - w : idx_u + w :, idx_v - w : idx_v + w
                                    ] += 1

    grid_points[counter > 0] = grid_points[counter > 0] / counter[counter > 0]

    return grid_points, l_cor, u, delta_u


def calibrate(R=np.array([]), M=np.array([])):
    G = np.zeros(R.shape, dtype=complex)
    g = np.zeros((R.shape[0], R.shape[2]), dtype=complex)
    temp = np.ones((R.shape[0], R.shape[1]), dtype=complex)

    for t in range(R.shape[2]):
        g[:, t], G[:, :, t] = Common.create_G_stef(
            R[:, :, t], M[:, :, t], 200, 1e-20, temp, False
        )

    return g, G


def extrapolation_function(
    baseline=np.array([0, 1]),
    true_sky_model=np.array([[1, 0, 0], [0.2, 1, 0]]),
    cal_sky_model=np.array([[1, 0, 0]]),
    Phi=np.array([[0, 3, 5], [-3, 0, 2], [-5, -2, 0]]),
    u=0,
    v=0,
):
    temp = np.ones(Phi.shape, dtype=complex)
    ut = u
    vt = v
    u_m = (Phi * ut) / (1.0 * Phi[baseline[0], baseline[1]])
    v_m = (Phi * vt) / (1.0 * Phi[baseline[0], baseline[1]])
    R = np.zeros(Phi.shape, dtype=complex)
    M = np.zeros(Phi.shape, dtype=complex)
    R, M = Common.extrapolation_loop(true_sky_model, cal_sky_model, u_m, v_m, R, M)
    g_stef, G = Common.create_G_stef(R, M, 200, 1e-20, temp, no_auto=False)
    r_pq = R[baseline[0], baseline[1]]
    m_pq = M[baseline[0], baseline[1]]
    g_pq = G[baseline[0], baseline[1]]
    return g_pq
