import numpy as np
import pylab as plt
import pickle
import argparse


class T_ghost():
    def __init__(self,
                 point_sources=np.array([]),
                 antenna=""):
        self.antenna = antenna
        self.A_1 = point_sources[0, 0]
        self.A_2 = point_sources[1, 0]
        self.l_0 = point_sources[1, 1]
        self.m_0 = point_sources[1, 2]
        # v.PICKLENAME = "antnames"
        self.ant_names = pickle.load(open("KAT7_1445_1x16_12h_antnames.p", "rb"))

        # v.PICKLENAME = "phi_m"
        self.phi_m = pickle.load(open("KAT7_1445_1x16_12h_phi_m.p", "rb"))

        # v.PICKLENAME = "b_m"
        self.b_m = pickle.load(open("KAT7_1445_1x16_12h_b_m.p", "rb"))

        # v.PICKLENAME = "theta_m"
        self.theta_m = pickle.load(open("KAT7_1445_1x16_12h_theta_m.p", "rb"))

        # v.PICKLENAME = "sin_delta"
        self.sin_delta = pickle.load(open("KAT7_1445_1x16_12h_sin_delta.p", "rb"))

    def get_antenna(self, ant, ant_names):
        if isinstance(ant[0], int):
            return np.array(ant)
        if ant == "all":
            return np.arange(len(ant_names))
        new_ant = np.zeros((len(ant),))
        for k in xrange(len(ant)):
            for j in xrange(len(ant_names)):
                if (ant_names[j] == ant[k]):
                    new_ant[k] = j
        return new_ant

    def calculate_delete_list(self):
        if self.antenna == "all":
            return np.array([])
        d_list = list(xrange(len(self.ant_names)))
        for k in range(len(self.a_list)):
            d_list.remove(self.a_list[k])
        return d_list

    def create_mask(self, baseline, plot_v=False):
        point_sources = np.array([(1, 0, 0)])
        point_sources = np.append(point_sources, [(1, self.l_0, -1 * self.m_0)], axis=0)
        point_sources = np.append(point_sources, [(1, -1 * self.l_0, 1 * self.m_0)], axis=0)

        # SELECTING ONLY SPECIFIC INTERFEROMETERS
        b_list = self.get_antenna(baseline, self.ant_names)
        d_list = self.calculate_delete_list()

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

        if plot_v == True:
            plt.plot(0, 0, "rx")
            plt.plot(self.l_0 * (180 / np.pi), self.m_0 * (180 / np.pi), "rx")
            plt.plot(-1 * self.l_0 * (180 / np.pi), -1 * self.m_0 * (180 / np.pi), "rx")
        for j in xrange(theta_new.shape[0]):
            for k in xrange(j + 1, theta_new.shape[0]):
                if not np.allclose(phi_new[j, k], phi):
                    l_cordinate = phi_new[j, k] / phi * (
                            np.cos(theta_new[j, k] - theta) * self.l_0 + self.sin_delta * np.sin(
                        theta_new[j, k] - theta) * self.m_0)
                    m_cordinate = phi_new[j, k] / phi * (
                            np.cos(theta_new[j, k] - theta) * self.m_0 - self.sin_delta ** (-1) * np.sin(
                        theta_new[j, k] - theta) * self.l_0)
                    if plot_v == True:
                        plt.plot(l_cordinate * (180 / np.pi), m_cordinate * (180 / np.pi), "rx")
                        plt.plot(-1 * l_cordinate * (180 / np.pi), -1 * m_cordinate * (180 / np.pi), "gx")
                    point_sources = np.append(point_sources, [(1, l_cordinate, -1 * m_cordinate)], axis=0)
                    point_sources = np.append(point_sources, [(1, -1 * l_cordinate, 1 * m_cordinate)], axis=0)

        return point_sources

    def create_G_stef(self, N, R, M, temp, imax, tau):
        '''This function finds argmin G ||R-GMG^H|| using StEFCal.
         R is your observed visibilities matrx.
         M is your predicted visibilities.
         imax maximum amount of iterations.
         tau stopping criteria.
         g the antenna gains.
         G = gg^H.'''
        g_temp = np.ones((N,), dtype=complex)
        for k in xrange(imax):
            g_old = np.copy(g_temp)
            for p in xrange(N):
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

    # resolution --- arcsecond, image_s --- degrees
    def visibilities_pq_2D(self, baseline, u=None, v=None, resolution=0, image_s=0, s=0):
        # SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline, self.ant_names)
        d_list = self.calculate_delete_list()

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

        if u <> None:
            u_dim1 = len(u)
            u_dim2 = 1
            uu = u
            vv = v
            l_cor = None
            m_cor = None
        else:
            # FFT SCALING
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

        # DO CALIBRATION
        V_R_pq = np.zeros(uu.shape, dtype=complex)
        V_G_pq = np.zeros(uu.shape, dtype=complex)
        temp = np.ones(phi_new.shape, dtype=complex)

        for i in xrange(u_dim1):
            for j in xrange(u_dim2):
                if u_dim2 <> 1:
                    u_t = uu[i, j]
                    v_t = vv[i, j]
                else:
                    u_t = uu[i]
                    v_t = vv[i]

                # BASELINE CORRECTION (Single operation)
                # ADDITION
                v_t = v_t - delta_b
                # SCALING
                u_t = u_t / phi
                v_t = v_t / (self.sin_delta * phi)

                # ROTATION (Clockwise)
                u_t_r = u_t * np.cos(theta) + v_t * np.sin(theta)
                v_t_r = -1 * u_t * np.sin(theta) + v_t * np.cos(theta)
                # u_t_r = u_t
                # v_t_r = v_t
                # NON BASELINE TRANSFORMATION (NxN) operations
                # ROTATION (Anti-clockwise)
                u_t_m = u_t_r * np.cos(theta_new) - v_t_r * np.sin(theta_new)
                v_t_m = u_t_r * np.sin(theta_new) + v_t_r * np.cos(theta_new)
                # SCALING
                u_t_m = phi_new * u_t_m
                v_t_m = phi_new * self.sin_delta * v_t_m
                # ADDITION
                v_t_m = v_t_m + b_new
                R = self.A_1 + self.A_2 * np.exp(-2 * 1j * np.pi * (u_t_m * self.l_0 + v_t_m * self.m_0))
                M = self.A_1 * np.ones(R.shape, dtype=complex)

                d, Q = np.linalg.eigh(R)
                D = np.diag(d)
                Q_H = Q.conj().transpose()
                abs_d = np.absolute(d)
                index = abs_d.argmax()

                if args.stefcal:
                    N = R.shape[0]
                    g, G = self.create_G_stef(N, R, M, temp, 20, 1e-6)
                else:
                    if (d[index] >= 0):
                        g = Q[:, index] * np.sqrt(d[index])
                    else:
                        g = Q[:, index] * np.sqrt(np.absolute(d[index])) * 1j
                    G = np.dot(np.diag(g), temp)
                    G = np.dot(G, np.diag(g.conj()))
                if self.antenna == "all":
                    if u_dim2 <> 1:
                        V_R_pq[i, j] = R[b_list[0], b_list[1]]
                        V_G_pq[i, j] = G[b_list[0], b_list[1]]
                    else:
                        V_R_pq[i] = R[b_list[0], b_list[1]]
                        V_G_pq[i] = G[b_list[0], b_list[1]]
                else:
                    for k in xrange(p_new.shape[0]):
                        for l in xrange(p_new.shape[1]):
                            if (p_new[k, l] == b_list[0]) and (q_new[k, l] == b_list[1]):
                                if u_dim2 <> 1:
                                    V_R_pq[i, j] = R[k, l]
                                    V_G_pq[i, j] = G[k, l]
                                else:
                                    V_R_pq[i] = R[k, l] + R[l, k]
                                    V_G_pq[i] = G[k, l] + G[l, k]

        return u, v, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor, m_cor

    def vis_function(self, type_w, avg_v, V_G_pq, V_G_qp, V_R_pq):
        if type_w == "R":
            vis = V_R_pq
        elif type_w == "RT":
            vis = V_R_pq ** (-1)
        elif type_w == "R-1":
            vis = V_R_pq - 1
        elif type_w == "RT-1":
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
                vis = (V_G_pq ** (-1) + V_G_qp ** (-1)) / 2
            else:
                vis = V_G_pq ** (-1)
        elif type_w == "GT-1":
            if avg_v:
                vis = (V_G_pq ** (-1) + V_G_qp ** (-1)) / 2 - 1
            else:
                vis = V_G_pq ** (-1) - 1
        elif type_w == "GTR-R":
            if avg_v:
                vis = ((V_G_pq ** (-1) + V_G_qp ** (-1)) / 2) * V_R_pq - V_R_pq
            else:
                vis = V_G_pq ** (-1) * V_R_pq - V_R_pq
        elif type_w == "GTR":
            if avg_v:
                vis = ((V_G_pq ** (-1) + V_G_qp ** (-1)) / 2) * V_R_pq
            else:
                vis = V_G_pq ** (-1) * V_R_pq
        elif type_w == "GTR-1":
            if avg_v:
                vis = ((V_G_pq ** (-1) + V_G_qp ** (-1)) / 2) * V_R_pq - 1
            else:
                vis = V_G_pq ** (-1) * V_R_pq - 1
        return vis

    def plt_circle_grid(self, grid_m):
        rad = np.arange(1, 1 + grid_m, 1)
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 1, 500)

        x_c = np.cos(2 * np.pi * x)
        y_c = np.sin(2 * np.pi * y)
        for k in range(len(rad)):
            plt.plot(rad[k] * x_c, rad[k] * y_c, "k")

    def plotImage(self, image, l_cor, m_cor, radius, mask, baseline, fname):
        l_cor = l_cor * (180 / np.pi)
        m_cor = m_cor * (180 / np.pi)

        fig = plt.figure()
        cs = plt.imshow(image, interpolation="bicubic", cmap="cubehelix",
                        extent=[l_cor[0], -1 * l_cor[0], m_cor[0], -1 * m_cor[0]])
        fig.colorbar(cs)
        self.plt_circle_grid(radius)

        plt.xlim([-radius, radius])
        plt.ylim([-radius, radius])

        if mask:
            p = self.create_mask(baseline, plot_v=True)
            for k in xrange(len(p)):
                plt.plot(p[k, 1] * (180 / np.pi), p[k, 2] * (180 / np.pi), "kv")

        plt.xlabel("$l$ [degrees]")
        plt.ylabel("$m$ [degrees]")
        baseline = ''.join(map(str, args.baseline))
        plt.title("Baseline " + baseline + " --Real")
        plt.savefig("Baseline " + baseline + " --" + fname)
        plt.show()

    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_pq_2D(self, baseline, resolution, image_s, s, sigma=None, type_w="G-1", avg_v=False, plot=False,
                  mask=False):
        if avg_v:
            baseline_new = [0, 0]
            baseline_new[0] = baseline[1]
            baseline_new[1] = baseline[0]
            u, v, V_G_qp, V_R_qp, phi, delta_b, theta, l_cor, m_cor = self.visibilities_pq_2D(baseline_new,
                                                                                              resolution=resolution,
                                                                                              image_s=image_s, s=s)
        else:
            V_G_qp = 0

        u, v, V_G_pq, V_R_pq, phi, delta_b, theta, l_cor, m_cor = self.visibilities_pq_2D(baseline,
                                                                                          resolution=resolution,
                                                                                          image_s=image_s, s=s)

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)

        N = l_cor.shape[0]

        vis = self.vis_function(type_w, avg_v, V_G_pq, V_G_qp, V_R_pq)

        delta_u = u[1] - u[0]
        delta_v = v[1] - v[0]
        if sigma <> None:
            uu, vv = np.meshgrid(u, v)
            sigma = (np.pi / 180) * sigma
            g_kernal = (2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
            vis = vis * g_kernal
            vis = np.roll(vis, -1 * (N - 1) / 2, axis=0)
            vis = np.roll(vis, -1 * (N - 1) / 2, axis=1)
            # Changes will be needed here, implement gaussian here somehow
            image = np.fft.fft2(vis) * (delta_u * delta_v)
        else:
            image = np.fft.fft2(vis) / N ** 2

        image = image / self.A_2 * 100

        image = np.roll(image, 1 * (N - 1) / 2, axis=0)
        image = np.roll(image, 1 * (N - 1) / 2, axis=1)

        image = image[:, ::-1]

        if plot:
            self.plotImage(image.real, l_cor, m_cor, image_s, mask, baseline, "Real")
            self.plotImage(image.imag, l_cor, m_cor, image_s, mask, baseline, "Imaginary")

        return image, l_old, m_old


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stefcal",
                        action="store_true",
                        help="Choose to use StEFCal or not")
    parser.add_argument("--baseline",
                        type=int,
                        nargs='+',
                        default=[3, 5],
                        help="The baseline to calculate on")
    parser.add_argument("--resolution",
                        type=int,
                        default=250,
                        help="The resolution of the image")
    parser.add_argument("--radius",
                        type=int,
                        default=3,
                        help="The radius of the circle on the image")
    parser.add_argument("--mask",
                        action="store_true",
                        help="Set to apply the mask to the image")
    parser.add_argument("--sigma",
                        type=float,
                        default=0.05,
                        help="Larger values increases the size of the point sources, a smaller value decreases the size")

    global args
    args = parser.parse_args()

    point_sources = np.array([(1, 0, 0), (0.2, (1 * np.pi) / 180, (0 * np.pi) / 180)])  # creates your two point sources
    t = T_ghost(point_sources, "all")  # creates a T_ghost object instance
    image, l_v, m_v = t.sky_pq_2D(args.baseline, args.resolution, args.radius, 1, sigma=args.sigma, type_w="G-1", avg_v=False,
                                  plot=True, mask=args.mask)
    # ? Fourier Transform of a Gaussian:
    # if gaussian is F_x(x) = e**(-a*x**2)
    # Then transform is F_x(k) = np.sqrt(np.pi/a) * np.exp(-np.pi**2 * k**2 / a)
    # https: // mathworld.wolfram.com / FourierTransformGaussian.html

