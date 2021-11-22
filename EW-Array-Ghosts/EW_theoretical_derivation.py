import numpy as np

def get_c_pr_rs(r, s, p, q, A, B, N):
	i1 = r == p and s == q and r != s
	i2 = r == p and s != q and r != s
	i3 = r != p and s == q and r != s
	i4 = r != p and s != q and r != s
	i5 = r == s

	if i1: 
		return ((2 * A * B) / N) - ((A * B) / (N ** 2))
	elif i2:
		return ((A * B) / N) - ((A * B) / (N ** 2))
	elif i3:
		return ((A * B) / N) - ((A * B) / (N ** 2))
	elif i4:
		return -1 * (A * B) / (N ** 2)
	elif i5:
		return 0

def derive_from_theory(sigma, N, phi, p, q, amplitude, u, v):
	antennas = [0, 1, 2]

	A = amplitude
	B = 2 * (sigma ** 2) * np.pi

	c_pq_0 = 1 / 2 + (A * B) / N
	a_pq_0 = c_pq_0

	g_pq = a_pq_0

	for r in antennas:
		for s in antennas:
			if (r != s):
				c_pq_rs = get_c_pr_rs(r, s, p, q, A, B, N)
				d_pq_rs = c_pq_rs / (A * B)
				a_pq_rs = A * ((phi[p][q] ** 2) / (phi[r][s] ** 2)) * d_pq_rs
				sigma_2_pq_rs = (phi[p][q] ** 2) / (phi[r][s] ** 2) * sigma ** 2

				g_pq += a_pq_rs * 2 * np.pi * sigma_2_pq_rs * np.exp(-2 * np.pi ** 2 * sigma_2_pq_rs) * (u ** 2 + v ** 2)

	return g_pq


	
