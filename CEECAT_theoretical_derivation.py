import numpy as np

def get_c_pr_rs(r, s, p, q, A, B, N):
	"""
   	Get the pq'th entry of the gain matrix

    :param r: R value of the phi matrix
    :type r: Integer

	:param s: S value of the phi matrix
    :type s: Integer

	:param q: Q value of the visiblity matrix
    :type q: Integer

	:param p: P value of the visiblity matrix
    :type p: Integer

	:param A: The amplitude
    :type A: Float

	:param B: The kernel for the image
    :type B: Float

	:param N: The length of the PHI array
    :type N: Integer

    :returns: The pq'th entry of the gain matrix.
    """

	i1 = r == p and s == q and r != s
	i2 = r == p and s != q and r != s
	i3 = r != p and s == q and r != s
	i4 = r != p and s != q and r != s
	i5 = r == s

	if i1: 
		return (2 * A * B) / N - (A * B) / (N ** 2)
	elif i2:
		return (A * B) / N - (A * B) / (N ** 2)
	elif i3:
		return (A * B) / N - (A * B) / (N ** 2)
	elif i4:
		return -1 * (A * B) / (N ** 2)
	elif i5:
		return 0

def derive_from_theory(sigma, N, phi, p, q, amplitude, u, v):
	"""
   	Get the derivation visibilities

	:param sigma: The sigma value for the sky model
    :type sigma: Float

    :param N: The length of the phi matrix
    :type N: Integer

	:param phi: S value of the phi matrix
    :type phi: Integer

	:param q: Q value of the visiblity matrix
    :type q: Integer

	:param p: P value of the visiblity matrix
    :type p: Integer

	:param amplitude: The amplitude
    :type amplitude: Float

    :param u: The U value of the UV tracks
    :type u: Integer

    :param v: The V value of the UV tracks
    :type v: Integer

    :returns: The derived visibilties
    """

	sigma = sigma * (np.pi / 180)
	N = phi.shape[0]

	A = amplitude
	B = 2 * (sigma ** 2) * np.pi

	g_pq = 0 
	g_pq_inv = 0

	for r in range(N):
		for s in range(N):
			if (r != s):
				c_pq_rs = get_c_pr_rs(r, s, p, q, A, B, N)
				d_pq_rs = c_pq_rs * (A * B) ** (-1)
				a_pq_rs = B **(-1) * ((phi[p, q] ** 2 * 1.0) / phi[r, s] ** 2) * c_pq_rs
				sigma_2_pq_rs = ((phi[r, s] ** 2 * 1.0) / phi[p, q] ** 2) * sigma ** 2
				g_pq += a_pq_rs * (2 * np.pi * sigma_2_pq_rs) * np.exp(-2 * np.pi ** 2 * sigma_2_pq_rs * (u ** 2 + v ** 2))
				g_pq_inv -= (A * B) ** (-1) * d_pq_rs * np.exp(-2 * np.pi ** 2 * sigma_2_pq_rs * (u ** 2 + v ** 2))
	g_pq += (A * B * 1.0) / N
	g_pq_inv += (A  * B) ** (-1) * ((2.0 * N - 1) / (N))
	return g_pq, g_pq_inv

def derive_from_theory_linear(sigma, N, phi, p, q, amplitude, u, v):
	"""
   	Get the theory derivation visibilities

	:param sigma: The sigma value for the sky model
    :type sigma: Float

    :param N: The length of the phi matrix
    :type N: Integer

	:param phi: S value of the phi matrix
    :type phi: Integer

	:param q: Q value of the visiblity matrix
    :type q: Integer

	:param p: P value of the visiblity matrix
    :type p: Integer

	:param amplitude: The amplitude
    :type amplitude: Float

    :param u: The U value of the UV tracks
    :type u: Integer

    :param v: The V value of the UV tracks
    :type v: Integer

    :returns: The theory derived visibilties
    """

	sigma = sigma * (np.pi / 180)
	N = phi.shape[0]

	A = amplitude
	B = 2 * (sigma ** 2) * np.pi

	g_pq = 0 
	g_pq_inv = 0

	for r in range(N):
		for s in range(N):
			if (r != s):
				c_pq_rs = get_c_pr_rs(r, s, p, q, A, B, N)
				d_pq_rs = c_pq_rs * (A * B) ** (-1)
				a_pq_rs = B **(-1) * ((phi[p, q] ** 2 * 1.0) / phi[r, s] ** 2) * c_pq_rs
				sigma_2_pq_rs = ((phi[r, s] ** 2 * 1.0) / phi[p, q] ** 2) * sigma ** 2
				g_pq += a_pq_rs * (2 * np.pi * sigma_2_pq_rs) * np.exp(-2 * np.pi ** 2 * sigma_2_pq_rs * (u ** 2 + v ** 2))
				g_pq_inv -= (A * B) ** (-1) * d_pq_rs * np.exp(-2 * np.pi ** 2 * sigma_2_pq_rs * (u ** 2 + v ** 2))
	g_pq += (A * B * 1.0) / N
	g_pq_inv += (A  * B) ** (-1) * ((2.0 * N - 1) / (N))
	return g_pq, g_pq_inv, B, A


	
