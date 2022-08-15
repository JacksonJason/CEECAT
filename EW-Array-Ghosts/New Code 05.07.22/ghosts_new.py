import numpy as np
import pylab as plt
import pickle
import sys
from scipy.interpolate import UnivariateSpline

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
        #print(R)
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
                    #pass
                    g_temp = (g_temp + g_old) / 2

        G_m = np.dot(np.diag(g_temp), temp)
        G_m = np.dot(G_m, np.diag(g_temp.conj()))

        g = g_temp
        G = G_m

        return g, G

    #def create_G_ALS(self,R,M,temp):

    #    w,v = np.linalg.eig(R)
    #    print(w)
    #    print(v)
    #    sys.exit()

    def cal_G_eig(self,R):
        #D =np.zeros(R.shape, dtype=complex)
        Q=np.zeros(R.shape, dtype=complex)
        g=np.zeros((R.shape[0],) , dtype=complex)
        G=np.zeros(R.shape ,dtype=complex)
        temp =np.ones((R.shape[0],R.shape[0]) ,dtype=complex)
        d,Q = np.linalg.eigh(R)
        print(np.linalg.matrix_rank(R))
        Q_H = Q.conj().transpose()
        abs_d=np.absolute(d)
        index=abs_d.argmax()
        if (d[index] > 0):
           g=Q[:,index]*np.sqrt(d[index])
        else:
           g=Q[:,index]*np.sqrt(np.absolute(d[index]))*1j

        G = np.dot(np.diag(g),temp)
        G = np.dot(G,np.diag(g.conj()))

        return g,G


    def c_func(self, r, s, p, q, A, B, N):
        answer = 0
        if ((r == p) and (s==q) and (r != s)):
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
    resolution --- rad^{-1}
    vis_s --- overall extend of image in rad^{-1}

    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    
    baseline --- baseline to focus on  
    """
    def theory_g_linear(self,baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0,0.02]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),vis_s=5000,resolution=10):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
        temp = np.ones(Phi.shape, dtype=complex) 
        N = int(np.ceil(vis_s*2/resolution))
        if (N % 2) == 0:
           N = N + 1
        u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)

        g_pq = np.zeros(u.shape,dtype=complex)
        g_pq_inv = np.zeros(u.shape,dtype=complex)
        A = true_sky_model[0,0]
        sigma = true_sky_model[0,3]*(np.pi/180)
        B = 2*sigma**2*np.pi
        p = baseline[0]
        q = baseline[1]
        len_N = Phi.shape[0]

        for i in range(len(u)):
            ut = u[i]
            vt = 0
            for r in range(len_N):
                for s in range(len_N):
                    if r != s:
                       c_pq = self.c_func(r, s, p, q, A, B, len_N)
                       d_pq = c_pq*(A*B)**(-1)
                       A_pq = B**(-1)*((Phi[p,q]**2*1.0)/Phi[r,s]**2)*c_pq
                       s_pq2 = ((Phi[r,s]**2*1.0)/Phi[p,q]**2)*sigma**2
                       g_pq[i]+=A_pq*(2*np.pi*s_pq2)*np.exp(-2*np.pi**2*s_pq2*(ut**2+vt**2))
                       g_pq_inv[i] -= (A*B)**(-1)*d_pq*np.exp(-2*np.pi**2*s_pq2*(ut**2+vt**2))

        g_pq[i] += (A*B*1.0)/len_N
        g_pq_inv[i] += (A*B)**(-1)*((2.0*len_N-1)/(len_N))

        return g_pq,g_pq_inv,u,B





    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    """
    def theory_g(self,baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0,0.1]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
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
        #EXTRACTING THE IMPORTANT PARAMETERS
        g_pq = np.zeros((u_dim,v_dim),dtype=complex)
        g_pq_inv = np.zeros((u_dim,v_dim),dtype=complex)
        A = true_sky_model[0,0]
        sigma = true_sky_model[0,3]*(np.pi/180)
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
                           d_pq = c_pq*(A*B)**(-1)
                           A_pq = B**(-1)*((Phi[p,q]**2*1.0)/Phi[r,s]**2)*c_pq
                           s_pq2 = ((Phi[r,s]**2*1.0)/Phi[p,q]**2)*sigma**2
                           g_pq[i,j]+=A_pq*(2*np.pi*s_pq2)*np.exp(-2*np.pi**2*s_pq2*(ut**2+vt**2))
                           g_pq_inv[i,j] -= (A*B)**(-1)*d_pq*np.exp(-2*np.pi**2*s_pq2*(ut**2+vt**2))

        g_pq[i,j] += (A*B*1.0)/len_N
        g_pq_inv[i,j] += (A*B)**(-1)*((2.0*len_N-1)/(len_N))

        #plt.imshow((g_pq.real)) 
        #plt.show()
        #fig, ax = plt.subplots()
        #im = ax.imshow(g_pq.real)
        #fig.colorbar(im, ax=ax) 
        #plt.show()
        #print(np.max(g_pq.real))

        return g_pq,g_pq_inv,delta_u,delta_v
        
    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    """
    def gaussian_conv_clean_beam(self,A=1,sigma=1,sigma_b=0.5,image_s=3,s=1,resolution=100):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
        s_old = s      
        #print(baseline[0])
        #print(baseline[1])

        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        delta_v = delta_u
        print("gaussian")
        print(delta_u)
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
        f = np.zeros((u_dim,v_dim),dtype=complex)
        sigma = sigma*(np.pi/180)
        sigma_b = sigma_b*(np.pi/180)
        print("gaussian")
        print(sigma)

        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                f[i,j] = A*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))*(2*np.pi*sigma_b**2*np.exp(-2*np.pi**2*sigma_b**2*(uu[i,j]**2+vv[i,j]**2)))

        x = np.linspace(-1*delta_u*len(self.cut(f))/2,1*delta_u*len(self.cut(f))/2,len(self.cut(f)))
        plt.plot(x,self.cut(f))
        plt.show()

        zz = f
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

        zz_f2 = np.fft.fft2(zz) * (delta_u*delta_v)
        zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2)-1, axis=0)
        zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2)-1, axis=1)

        zz_f2 = zz_f2#*((sigma**2+sigma_b**2)/sigma_b**2)   

        fig, ax = plt.subplots()
        #print(image_s)
        #print(s)
        im = ax.imshow(zz_f2.real,extent=[-(N - 1) / 2 * delta_l * (180.0/np.pi), (N - 1) / 2 * delta_l * (180.0/np.pi), -(N - 1) / 2 * delta_l * (180.0/np.pi), (N - 1) / 2 * delta_l * (180.0/np.pi)])
        fig.colorbar(im, ax=ax) 
        plt.show()

        zz_f2 = zz_f2.real
        cut = zz_f2[zz.shape[0]/2,:]
        l = np.linspace(-(N - 1) / 2 * delta_l*(180/np.pi), (N - 1) / 2 * delta_l*(180/np.pi), N)
        plt.plot(l,cut)
        plt.show()
        return cut,l,delta_l,zz

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    """
    def gaussian(self,A=1,sigma=1,image_s=3,s=1,resolution=100):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
        s_old = s      
        #print(baseline[0])
        #print(baseline[1])

        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        delta_v = delta_u
        print("gaussian")
        print(delta_u)
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
        f = np.zeros((u_dim,v_dim),dtype=complex)
        sigma = sigma*(np.pi/180)
        print("gaussian")
        print(sigma)

        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                #f[i,j] = A*2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))
                f[i,j] = A*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))

        x = np.linspace(-1*delta_u*len(self.cut(f))/2,1*delta_u*len(self.cut(f))/2,len(self.cut(f)))
        plt.plot(x,self.cut(f))
        plt.show()

        zz = f
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

        zz_f2 = np.fft.fft2(zz) * (delta_u*delta_v)
        zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2)-1, axis=0)
        zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2)-1, axis=1)   

        fig, ax = plt.subplots()
        #print(image_s)
        #print(s)
        im = ax.imshow(zz_f2.real,extent=[-(N - 1) / 2 * delta_l * (180.0/np.pi), (N - 1) / 2 * delta_l * (180.0/np.pi), -(N - 1) / 2 * delta_l * (180.0/np.pi), (N - 1) / 2 * delta_l * (180.0/np.pi)])
        fig.colorbar(im, ax=ax) 
        plt.show()

        zz_f2 = zz_f2.real
        cut = zz_f2[zz.shape[0]/2,:]
        l = np.linspace(-(N - 1) / 2 * delta_l*(180/np.pi), (N - 1) / 2 * delta_l*(180/np.pi), N)
        plt.plot(l,cut)
        plt.show()  
        return cut,l,delta_l,zz
        
    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    """
    def rectangle(self,A=1,sigma=1,image_s=3,s=1,resolution=100):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
        s_old = s      
        #print(baseline[0])
        #print(baseline[1])

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
        f = np.zeros((u_dim,v_dim),dtype=complex)
        sigma = sigma*(np.pi/180)
        T = (sigma)**(-1)

        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                if uu[i,j] > -1*T:
                   if uu[i,j] < 1*T:
                      if vv[i,j] > -1*T:
                         if vv[i,j] < 1*T:
                            f[i,j] = A*(sigma**2/4)#(sigma**2/(4*np.pi**2))

        zz = f
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

        zz_f2 = np.fft.fft2(zz)* (delta_u*delta_v)
        zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2)-1, axis=0)
        zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2)-1, axis=1)   

        fig, ax = plt.subplots()
        #print(image_s)
        #print(s)
        im = ax.imshow(zz_f2.real,extent=[-(N - 1) / 2 * delta_l * (180.0/np.pi), (N - 1) / 2 * delta_l * (180.0/np.pi), -(N - 1) / 2 * delta_l * (180.0/np.pi), (N - 1) / 2 * delta_l * (180.0/np.pi)])
        fig.colorbar(im, ax=ax) 
        plt.show()

        zz_f2 = zz_f2.real

        cut = zz_f2[zz.shape[0]/2,:]
        l = np.linspace(-(N - 1) / 2 * delta_l*(180/np.pi), (N - 1) / 2 * delta_l*(180/np.pi), N)
        plt.plot(delta_l,l,cut)
        plt.show()  
    


    """
    resolution --- rad^{-1}
    vis_s --- 5000 rad^{-1}
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    b0 in m
    f in HZ
    """
    def extrapolation_function_linear(self,baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),vis_s=5000,resolution=10,kernel=True,b0=36,f=1.45e9,type_plot="GT-1"):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
        temp = np.ones(Phi.shape, dtype=complex) 
        
        N = int(np.ceil(vis_s*2/resolution))
        if (N % 2) == 0:
           N = N + 1
        u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)
        


        r_pq = np.zeros(u.shape,dtype=complex)
        g_pq = np.zeros(u.shape,dtype=complex)
        m_pq = np.zeros(u.shape,dtype=complex)
        
        R = np.zeros(Phi.shape,dtype=complex)
        M = np.zeros(Phi.shape,dtype=complex)

        for i in range(len(u)):
                ut = u[i]
                vt = 0
                u_m = (Phi*ut)/(1.0*Phi[baseline[0],baseline[1]])
                v_m = (Phi*vt)/(1.0*Phi[baseline[0],baseline[1]])
                R = np.zeros(Phi.shape,dtype=complex)
                M = np.zeros(Phi.shape,dtype=complex) 
                for k in range(len(true_sky_model)):
                    s = true_sky_model[k]
                    #print(s)
                    if len(s) <= 3:
                       #pass
                       R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                       sigma = s[3]*(np.pi/180)
                       g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                       R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
                       #(2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                       #print(s[3])
                for k in range(len(cal_sky_model)):
                    s = cal_sky_model[k]
                    #print(s)
                    if len(s) <= 3:
                       #pass
                       M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                       sigma = s[3]*(np.pi/180)
                       g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                       M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
                       #(2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                       #print(s[3])
                #if (i == j):
                #   print(R)
                g_stef, G = self.create_G_stef(R, M, 100, 1e-8, temp, no_auto=False) 
                #g,G = self.cal_G_eig(R)  

                #R = np.exp(-2*np.pi*1j*(u_m*l0+v_m*m0))
                r_pq[i] = R[baseline[0],baseline[1]]
                m_pq[i] = M[baseline[0],baseline[1]]
                g_pq[i] = G[baseline[0],baseline[1]]
 
                
        #print(f) 
        lam = (1.0*3*10**8)/f
        #print(lam)
        b_len = b0*Phi[baseline[0],baseline[1]]
        #print(b_len)
        fwhm = 1.02*lam/(b_len)
        #print(fwhm*(180/np.pi))
        sigma_kernal = fwhm/(2*np.sqrt(2*np.log(2)))
        #print(sigma_kernal*(180/np.pi))#printing degrees
        #print("extrapolation kernel")
        #print(sigma_kernal)
        
        #g_kernal = np.exp(-2*np.pi**2*sigma_kernal**2*(uu**2+vv**2))
        g_kernal = 2*np.pi*sigma_kernal**2*np.exp(-2*np.pi**2*sigma_kernal**2*(u**2))
       
        return r_pq,g_pq,u,g_kernal,sigma_kernal

    """
    resolution --- resolution in image domain in arcseconds
    images_s --- overall extend of image in degrees
    Phi --- geometry matrix
    true_skymodel --- true skymodel
    cal_skymodel --- model skymodel
    baseline --- baseline to focus on  
    b0 in m
    f in HZ
    """
    def extrapolation_function(self,baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,b0=36,f=1.45e9,type_plot="GT-1"):
        #temporary adding some constant values for the sky
        #l0 = 1 * (np.pi / 180)#temporary 
        #m0 = 0 * (np.pi / 180)
        #print(true_sky_model)
        temp = np.ones(Phi.shape, dtype=complex) 
        s_old = s      
        print(baseline[0])
        print(baseline[1])

        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        delta_v = delta_u
        #print("extrapolation")
        #print(delta_u)
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
        r_pq = np.zeros((u_dim,v_dim),dtype=complex)
        g_pq = np.zeros((u_dim,v_dim),dtype=complex)
        m_pq = np.zeros((u_dim,v_dim),dtype=complex)
        
        R = np.zeros(Phi.shape,dtype=complex)
        M = np.zeros(Phi.shape,dtype=complex)

        for i in range(u_dim):
            for j in range(v_dim):
                #print("*********")
                #print(u_dim)
                #print(v_dim)
                #print(i)
                #print(j)
                #print("*********") 
                ut = u[i]
                vt = v[j]
                u_m = (Phi*ut)/(1.0*Phi[baseline[0],baseline[1]])
                v_m = (Phi*vt)/(1.0*Phi[baseline[0],baseline[1]])
                R = np.zeros(Phi.shape,dtype=complex)
                M = np.zeros(Phi.shape,dtype=complex) 
                for k in range(len(true_sky_model)):
                    s = true_sky_model[k]
                    #print(s)
                    if len(s) <= 3:
                       #pass
                       R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                       sigma = s[3]*(np.pi/180)
                       g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                       R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
                       #(2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                       #print(s[3])
                for k in range(len(cal_sky_model)):
                    s = cal_sky_model[k]
                    #print(s)
                    if len(s) <= 3:
                       #pass
                       M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
                    else:
                       sigma = s[3]*(np.pi/180)
                       g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
                       M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
                       #(2 * np.pi * sigma ** 2) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (uu ** 2 + vv ** 2))
                       #print(s[3])
                #if (i == j):
                #   print(R)
                g_stef, G = self.create_G_stef(R, M, 100, 1e-8, temp, no_auto=False) 
                #g,G = self.cal_G_eig(R)  

                #R = np.exp(-2*np.pi*1j*(u_m*l0+v_m*m0))
                r_pq[j,i] = R[baseline[0],baseline[1]]
                m_pq[j,i] = M[baseline[0],baseline[1]]
                g_pq[j,i] = G[baseline[0],baseline[1]]
 
                
        #print(f) 
        lam = (1.0*3*10**8)/f
        #print(lam)
        b_len = b0*Phi[baseline[0],baseline[1]]
        #print(b_len)
        fwhm = 1.02*lam/(b_len)
        #print(fwhm*(180/np.pi))
        sigma_kernal = fwhm/(2*np.sqrt(2*np.log(2)))
        #print(sigma_kernal*(180/np.pi))#printing degrees
        #print("extrapolation kernel")
        #print(sigma_kernal)
        
        #g_kernal = np.exp(-2*np.pi**2*sigma_kernal**2*(uu**2+vv**2))
        g_kernal = 2*np.pi*sigma_kernal**2*np.exp(-2*np.pi**2*sigma_kernal**2*(uu**2+vv**2))
            
        #sigma = 0.05*(np.pi/180)
        #g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))
        #plt.imshow(g_pq.real)
        #plt.show()
        #g_pq = g_pq-1#distilation
        #if type_plot == "GT-1":
        #   g_pq = (g_pq)**(-1)-1
        #else:
        #   g_pq = (g_pq)**(-1)*r_pq - r_pq

        #r_pq_new = (g_pq.real)**(-1)*(r_pq.real)

        #if kernel:
        #   g_pq = g_pq*g_kernal
        #g_pq = g_pq[:,::-1]
        #g_pq = g_pq#-1
        
        #fig, ax = plt.subplots()
        #im = ax.imshow(g_pq.real)
        #fig.colorbar(im, ax=ax) 
        #plt.show()
        #print(np.max(g_pq.real))
        
        
        #zz2 = r_pq
        #zz2 = np.roll(zz2, -int(zz.shape[0]/2), axis=0)
        #zz2 = np.roll(zz2, -int(zz.shape[0]/2), axis=1)

        #zz_f2 = np.fft.fft2(zz2) * (delta_u*delta_v)
        #zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2), axis=0)
        #zz_f2 = np.roll(zz_f2, -int(zz.shape[0]/2), axis=1)

        #fig, ax = plt.subplots()
        #print(image_s)
        #print(s)
        #im = ax.imshow(zz_f2.real)
        #fig.colorbar(im, ax=ax) 
        #plt.show()

        
        return r_pq,g_pq,delta_u,delta_v,g_kernal,sigma_kernal,delta_l#radians

    #def sinc2D(A,sigma,l,m):
    #    return A*(np.sin(np.pi*sigma*l)/(np.pi*sigma*l))*(np.sin(np.pi*sigma*m)/(np.pi*sigma*m))

    def anotherExp(self,P,size_gauss=0.02,K1=30.0,K2=3.0, N = 4):
        #setting up the s_size, resoltuon and extent of the image
        #0.02 degrees is 10 times the resolution of WSRT
        s_size = size_gauss #size of Gaussian in degrees
        r = (s_size*3600)/(1.0*K1) #resolution
        siz = s_size * (K2*1.0) #total size of image in degrees
        #print(P)

        if (N == 5):
           P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=0)
           P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=1)
        if (N == 6):
           P = np.delete(P,[1,3,5,6,7,8,11,12],axis=0)
           P = np.delete(P,[1,3,5,6,7,8,11,12],axis=1)
        if (N == 7):
           P = np.delete(P,[1,3,5,7,8,11,12],axis=0)
           P = np.delete(P,[1,3,5,7,8,11,12],axis=1)
        if (N == 8):
           P = np.delete(P,[1,3,5,7,11,12],axis=0)
           P = np.delete(P,[1,3,5,7,11,12],axis=1)
        if (N == 9):
           P = np.delete(P,[1,3,5,7,12],axis=0)
           P = np.delete(P,[1,3,5,7,12],axis=1)
        if (N == 10):
           P = np.delete(P,[1,3,5,7],axis=0)
           P = np.delete(P,[1,3,5,7],axis=1)
        if (N == 11):
           P = np.delete(P,[3,5,7],axis=0)
           P = np.delete(P,[3,5,7],axis=1)
        if (N == 12):
           P = np.delete(P,[5,7],axis=0)
           P = np.delete(P,[5,7],axis=1)
        if (N == 13):
           P = np.delete(P,[7],axis=0)
           P = np.delete(P,[7],axis=1)
        if (N == 4):
           P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=0)
           P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=1)
        

        #print(P)
        counter = 0
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > i:
                   print("*******************")
                   print(counter)
                   print("Theory")
                   g_pq_temp,g_pq_inv_temp,delta_u,delta_v = t.theory_g(baseline=np.array([i,j]),true_sky_model=np.array([[1.0,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
                   print("Simulation")
                   r_pq,g_pq,delta_u,delta_v,g_kernal,sigma_b,delta_l = self.extrapolation_function(baseline=np.array([i,j]),true_sky_model=np.array([[1.0,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
                   print("*******************")
                   file_name = str(N)+"_10_baseline_"+str(counter)+"_"+str(i)+"_"+str(j)+"_"+str(P[i,j])+".p"
                   f = open(file_name, 'wb')
                   counter+=1
                   pickle.dump(g_pq,f)
                   pickle.dump(r_pq,f)
                   pickle.dump(g_kernal,f)
                   pickle.dump(g_pq_inv_temp,f)
                   pickle.dump(g_pq_temp,f)
                   pickle.dump(sigma_b,f)
                   pickle.dump(delta_u,f)
                   pickle.dump(delta_l,f)
                   pickle.dump(s_size,f)
                   pickle.dump(siz,f)
                   pickle.dump(r,f)
                   #pickle.dump(b0,f)
                   pickle.dump(K1,f)
                   pickle.dump(K2,f)
                   #pickle.dump(fr,f)
                   pickle.dump(P,f) 
                   f.close()  

    def plt_imshow_new(self,data,a,baselines,N_values,cr=np.array([]),l=r"Log(Peak Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$])",vmax=-1,vmin=-1):
        
        N = 20 # number of colors to extract from each of the base_cmaps below
        base_cmaps = ['YlGnBu','Greys','Purples','Reds','Oranges']

        n_base = len(base_cmaps)
        #we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
        colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2,0.8,N)) for name in base_cmaps])    
        cmap = mpl.colors.ListedColormap(colors)

        print("hallo")
        fig, ax = plt.subplots()
        if vmax==vmin:
           im = ax.imshow(data, cmap="jet")
        else:
           im = ax.imshow(data, cmap="jet",vmax=vmax,vmin=vmin)
        cbar = fig.colorbar(im)
        cbar.set_label(l,labelpad=10)

        

        #text portion 
        min_val, max_val, diff = 0., 11., 1.
        ind_array_y = np.arange(min_val, max_val, diff)
        min_val, max_val, diff = 0., len(a[1:]), 1.
        ind_array_x = np.arange(min_val, max_val, diff)
        x, y = np.meshgrid(ind_array_x, ind_array_y)

        for x_val, y_val,p,d in zip(x.flatten(), y.flatten(), cr.flatten(),data.flatten()):
            c = str(int(p))
            if int(p) > 0:
               if np.isnan(d): 
                  ax.text(x_val, y_val, 'x', va='center', ha='center',color="black")
               else:
                 ax.text(x_val, y_val, 'x', va='center', ha='center',color="white")
        
        ax = plt.gca()
        xticks = np.arange(0,len(a[1:]),1)
        yticks = np.arange(0,len(N_values),1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(a[1:].tolist())
        ax.set_yticklabels(N_values.tolist()) 
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$N$")
        plt.show()

    def anotherExpPicProcess(self,P,N_values = np.array([4,5,6,7,8,9,10,11,12,13,14]),max_peak = 100):

        a = np.unique(np.absolute(P)).astype(int)
        P_old = np.copy(P)
        results_ampl = np.zeros((len(N_values),len(a),3,5),dtype=float) #max,min,average,counter
         
        baselines = np.zeros(N_values.shape,dtype=int)

        for N in N_values:
            print(N)

            P = np.copy(P_old) 

            if (N == 5):
               P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=0)
               P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=1)
            if (N == 6):
               P = np.delete(P,[1,3,5,6,7,8,11,12],axis=0)
               P = np.delete(P,[1,3,5,6,7,8,11,12],axis=1)
            if (N == 7):
               P = np.delete(P,[1,3,5,7,8,11,12],axis=0)
               P = np.delete(P,[1,3,5,7,8,11,12],axis=1)
            if (N == 8):
               P = np.delete(P,[1,3,5,7,11,12],axis=0)
               P = np.delete(P,[1,3,5,7,11,12],axis=1)
            if (N == 9):
               P = np.delete(P,[1,3,5,7,12],axis=0)
               P = np.delete(P,[1,3,5,7,12],axis=1)
            if (N == 10):
               P = np.delete(P,[1,3,5,7],axis=0)
               P = np.delete(P,[1,3,5,7],axis=1)
            if (N == 11):
               P = np.delete(P,[3,5,7],axis=0)
               P = np.delete(P,[3,5,7],axis=1)
            if (N == 12):
               P = np.delete(P,[5,7],axis=0)
               P = np.delete(P,[5,7],axis=1)
            if (N == 13):
               P = np.delete(P,[7],axis=0)
               P = np.delete(P,[7],axis=1)
            if (N == 4):
               P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=0)
               P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=1)
                   
            #results_ampl = np.zeros((len(N_values),len(a),3,4),dtype=float) #max,min,average,counter
             
            counter = 0
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    if j > i:
                       file_name = str(N)+"_10_baseline_"+str(counter)+"_"+str(i)+"_"+str(j)+"_"+str(P[i,j])+".p"
                       f = open(file_name, 'rb')
                       counter+=1
                       g_pq = pickle.load(f)
                       r_pq = pickle.load(f)
                       g_kernal = pickle.load(f)
                       g_pq_inv = pickle.load(f)
                       g_pq_temp = pickle.load(f)
                       sigma_b = pickle.load(f)
                       delta_u = pickle.load(f)
                       delta_l = pickle.load(f)
                       s_size = pickle.load(f)
                       siz = pickle.load(f)
                       r = pickle.load(f)
                       K1 = pickle.load(f)
                       K2 = pickle.load(f)
                       P = pickle.load(f)
                       f.close()  

                       B = 2*np.pi*(s_size*(np.pi/180))**2
                       c1 = self.cut(self.img(np.absolute(g_pq**(-1)*r_pq)*B,delta_u,delta_u))
                       c2 = self.cut(self.img(np.absolute(g_pq_temp**(-1)*r_pq)*B,delta_u,delta_u))
                       c3 = self.cut(self.img(np.absolute(g_pq_inv*r_pq)*B,delta_u,delta_u))

                       #print(np.max(c3))

                       l = [c1,c2,c3] 

                       v = P[i,j]
                       result = np.where(a == v)[0][0]
                       #print(result)

                       #result_amp[:,:,:,1] = -1

                       for k in range(len(l)):
                            
                           if (np.max(l[k]) < max_peak):
                              if (np.max(l[k]) > results_ampl[N-4,result,k,0]): 
                                 #print(np.max(l[k]))
                                 results_ampl[N-4,result,k,0] = np.max(l[k])
                           else:
                              results_ampl[N-4,result,k,4] += 1
                           if (np.max(l[k]) < max_peak):
                              if ((int(results_ampl[N-4,result,k,1]) == 0) or (np.max(l[k]) < results_ampl[N-4,result,k,1])): 
                                 results_ampl[N-4,result,k,1] = np.max(l[k])
                           if (np.max(l[k]) < max_peak): 
                              results_ampl[N-4,result,k,2] += np.max(l[k])
                              results_ampl[N-4,result,k,3] += 1

            baselines[N-4] = counter
        #plt.imshow(results_ampl[:,:,0,4])

        #print(results_ampl[:,:,0,4])
        #print(results_ampl[:,:,0,2]/results_ampl[:,:,0,3])
        #plt.show()

        #plt.imshow(results_ampl[:,:,0,0])
        #plt.show()
        
        #for k in range(results_ampl.shape[1]):
        #    plt.plot(results_ampl[:,k,0,1])
            #plt.plot(results_ampl[:,,2,2]/results_ampl[:,-1,2,3],"b")

        #plt.show()

        self.plt_imshow_new(results_ampl[:,1:,0,2]/results_ampl[:,1:,0,3],a,baselines,N_values,results_ampl[:,1:,0,4],l=r"Log(Peak Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$])",vmax=1,vmin=2)
        self.plt_imshow_new(results_ampl[:,1:,1,2]/results_ampl[:,1:,1,3],a,baselines,N_values,results_ampl[:,1:,0,4],l=r"Log(Peak Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$])",vmax=1,vmin=2)
        self.plt_imshow_new(results_ampl[:,1:,2,2]/results_ampl[:,1:,2,3],a,baselines,N_values,results_ampl[:,1:,0,4],l=r"Log(Peak Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$])",vmax=1,vmin=2)
                     
        return results_ampl,a,counter               




    def conductExp(self,Phi,N,size_gauss,c):
        s_size = size_gauss #size of Gaussian in degrees
        r = (s_size*3600)/20.0 #resolution
        siz = s_size * 3 #total size of image in degrees

        sigma = s_size*(np.pi/180) #size of Gaussian in radians
        B1 = 2*sigma**2*np.pi #area below Gaussian curve with amp 1

        P = Phi[:N,:N]
        
        g_pq_temp,g_pq_inv_temp,delta_u,delta_v = t.theory_g(baseline=np.array([0,1]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
        
        Curves_theory = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)
        Curves_theory2 = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)
        Curves_real = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)
        r_real = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)
        g_theory = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)
        g_inv_theory = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)
        g_real = np.zeros((P.shape[0],P.shape[0],g_pq_temp.shape[0]),dtype=float)


        Amp_theory = np.zeros((P.shape[0],P.shape[0]),dtype=float)
        Amp_theory2 = np.zeros((P.shape[0],P.shape[0]),dtype=float)
        Width_theory = np.zeros((P.shape[0],P.shape[0]),dtype=float)
        Width_theory2 = np.zeros((P.shape[0],P.shape[0]),dtype=float) 
        Amp_real = np.zeros((P.shape[0],P.shape[0]),dtype=float)
        Width_real = np.zeros((P.shape[0],P.shape[0]),dtype=float)
        Flux_theory = np.zeros((P.shape[0],P.shape[0]),dtype=float)
        Flux_theory2 = np.zeros((P.shape[0],P.shape[0]),dtype=float)   
        Flux_real = np.zeros((P.shape[0],P.shape[0]),dtype=float)

        for i in range(P.shape[0]):
            for j in range(P.shape[0]):
                if j > i:
                   g_pq_t,g_pq_inv,delta_u,delta_v = self.theory_g(baseline=np.array([i,j]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
                   r_pq,g_pq,delta_u,delta_v =self.extrapolation_function(baseline=np.array([i,j]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
                   r_real[i,j,:] = t.cut(t.img(r_pq,delta_u,delta_v))
                   g_theory[i,j,:] = t.cut(t.img(g_pq_t,delta_u,delta_v))
                   g_inv_theory[i,j,:] = t.cut(t.img(g_pq_inv,delta_u,delta_v))                 
                   g_real[i,j,:] = t.cut(t.img(g_pq,delta_u,delta_v))
                   Curves_theory[i,j,:] = t.cut(t.img(g_pq_inv*r_pq,delta_u,delta_v))
                   Curves_theory2[i,j,:] = t.cut(t.img(g_pq_t**(-1)*r_pq,delta_u,delta_v))
                   Curves_real[i,j,:] = t.cut(t.img(g_pq**(-1)*r_pq,delta_u,delta_v))
                   Amp_theory[i,j] = np.max(Curves_theory[i,j,:])
                   Amp_theory2[i,j] = np.max(Curves_theory2[i,j,:])
                   Amp_real[i,j] = np.max(Curves_real[i,j,:])
                   half_theory = Amp_theory[i,j]/2.0
                   idx = (Curves_theory[i,j,:] >= half_theory)
                   integer_map = map(int, idx)
                   Width_theory[i,j] = np.sum(list(integer_map))*r #in arcseconds
                   half_theory2 = Amp_theory2[i,j]/2.0
                   idx = (Curves_theory2[i,j,:] >= half_theory2)
                   integer_map = map(int, idx)
                   Width_theory2[i,j] = np.sum(list(integer_map))*r #in arcseconds
                   half_real = Amp_real[i,j]/2.0
                   idx = (Curves_real[i,j,:] >= half_real)
                   integer_map = map(int, idx)
                   Width_real[i,j] = np.sum(list(integer_map))*r #in arcseconds
                   Flux_real[i,j] = Amp_real[i,j]*( ( (Width_real[i,j]/3600.0)*(np.pi/180) )/( 2*np.sqrt(2*np.log(2)) ) )**2*2*np.pi       
                   Flux_theory[i,j] =  Amp_theory[i,j]*(((Width_theory[i,j]/3600.0)*(np.pi/180))/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
                   Flux_theory2[i,j] =  Amp_theory2[i,j]*(((Width_theory2[i,j]/3600.0)*(np.pi/180))/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
                   
                    
        file_name = str(N)+str("_")+str(c)+".p"
        f = open(file_name, 'wb')
        pickle.dump(s_size,f)
        pickle.dump(r,f)
        pickle.dump(siz,f)
        pickle.dump(sigma,f)
        pickle.dump(B1,f)
        pickle.dump(P,f)
        #pickle.dump()

        pickle.dump(Curves_theory,f)
        pickle.dump(Curves_real,f) 
        pickle.dump(r_real,f)
        pickle.dump(g_theory,f)
        pickle.dump(g_inv_theory,f) 
        pickle.dump(g_real,f)


        pickle.dump(Amp_theory,f)
        pickle.dump(Width_theory,f)
        pickle.dump(Amp_real,f)
        pickle.dump(Width_real,f)
        pickle.dump(Flux_theory,f)
        pickle.dump(Flux_real,f)

        pickle.dump(Curves_theory2,f)
        pickle.dump(Amp_theory2,f)
        pickle.dump(Width_theory2,f)
        pickle.dump(Flux_theory2,f)
        

        f.close()

    def process_pickle_file(self,name="14_2.p"):
        pkl_file = open(name, 'rb')
        
        s_size=pickle.load(pkl_file)
        r=pickle.load(pkl_file)
        siz=pickle.load(pkl_file)
        sigma=pickle.load(pkl_file)
        B1=pickle.load(pkl_file)
        P=pickle.load(pkl_file)
        #print(P)

        a = np.unique(np.absolute(P)).astype(int)
        #print(a)
        

        values_real = np.zeros((len(a),),dtype=float)
        values_t1 = np.zeros((len(a),),dtype=float)
        values_t2 = np.zeros((len(a),),dtype=float)
        
        counter = np.zeros((len(a),),dtype=int) 
        
        Curves_theory=pickle.load(pkl_file)
        Curves_real=pickle.load(pkl_file) 
        r_real=pickle.load(pkl_file)
        g_theory=pickle.load(pkl_file)
        g_inv_theory=pickle.load(pkl_file)
        g_real=pickle.load(pkl_file)

        #print(g_real.shape) 
        #plt.plot(g_real)
        #plt.show()
        

        #print(Curves_theory)


        Amp_theory=pickle.load(pkl_file)
        Width_theory=pickle.load(pkl_file)
        Amp_real=pickle.load(pkl_file)
        Width_real=pickle.load(pkl_file)
        Flux_theory=pickle.load(pkl_file)
        Flux_real=pickle.load(pkl_file)

        Curves_theory2=pickle.load(pkl_file)
        Amp_theory2=pickle.load(pkl_file)
        Width_theory2=pickle.load(pkl_file)
        Flux_theory2=pickle.load(pkl_file)

        amp_matrix_real = np.zeros(P.shape,dtype=float)
        amp_matrix_t1 = np.zeros(P.shape,dtype=float)
        amp_matrix_t2 = np.zeros(P.shape,dtype=float)
        P_new = np.zeros(P.shape,dtype=float)

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   P_new[k,j] = np.sum(((P[k,:]**2+P[k,j]**2)/P[k,j]**2)**(-1)) + np.sum(((P[:,j]**2+P[k,j]**2)/P[k,j]**2)**(-1)) - 1

        for k in range(P.shape[0]):
            strv = ""
            for j in range(P.shape[1]):
                   strv += str(P_new[k,j])+","
            #print(strv[:-1])


        plt.imshow(np.log(P_new))
        plt.show()
        
        #print("Hallo P_new")
        #print(P_new)
        
        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   v = P[k,j]
                   result = np.where(a == v)[0][0]
                   amp_matrix_real[k,j] += Amp_real[k,j]
                   amp_matrix_t1[k,j] += Amp_theory[k,j]
                   amp_matrix_t2[k,j] += Amp_theory2[k,j]
                   
                   values_real[result] += Amp_real[k,j]
                   values_t1[result] += Amp_theory[k,j]
                   values_t2[result] += Amp_theory2[k,j]
                   counter[result] += 1
        values_real = values_real/counter
        values_t1 = values_t1/counter
        values_t2 = values_t2/counter
        values_real[values_real>20000] = np.nan
        amp_matrix_real[amp_matrix_real>20000] = np.nan
        plt.imshow(np.log(P))
        plt.show()
        plt.imshow(np.log(amp_matrix_real))
        plt.show()
        plt.imshow(np.log(amp_matrix_t1))
        plt.show()
        plt.imshow(np.log(amp_matrix_t2))
        plt.show()

        for k in range(P.shape[0]):
            strv = ""
            for j in range(P.shape[1]):
                   strv += str(amp_matrix_t2[k,j])+","
            #print(strv[:-1])

        for k in range(P.shape[0]):
            strv = ""
            for j in range(P.shape[1]):
                   strv += str(amp_matrix_t1[k,j])+","
            #print(strv[:-1])
  
        for k in range(P.shape[0]):
            strv = ""
            for j in range(P.shape[1]):
                   strv += str(amp_matrix_real[k,j])+","
            #print(strv[:-1])


        plt.semilogy(a,values_real,"ro")
        plt.semilogy(a,values_t1,"bs")
        plt.semilogy(a,values_t2,"gx")         
        plt.show()

        #print("HALLO")
        plt.plot(Curves_real[9,10,:],"b")
        plt.show()

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   #print(k)
                   #print(j)
                   #print(np.max(Curves_real[k,j,:]))
                   #print(P[k,j])
                   plt.clf()
                   plt.plot(Curves_theory[k,j,:],"--")
                   plt.plot(Curves_real[k,j,:])
                   plt.plot(Curves_theory2[k,j,:],".")
                   name = "baseline_"+str(k)+"_"+str(j)+".png"
                   plt.savefig(name)
        #plt.show()

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   plt.plot(P[k,j],Amp_theory2[k,j],"x")
                   plt.plot(P[k,j],Amp_theory[k,j],"o")
                   plt.plot(P[k,j],Amp_real[k,j],"s")
                   
        #plt.ylim(0,1e10)
        plt.show()

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   plt.plot(P[k,j],Width_theory2[k,j],"x")
                   plt.plot(P[k,j],Width_theory[k,j],"o")
                   plt.plot(P[k,j],Width_real[k,j],"s")
                   
        plt.ylim(0,1e9)
        
        plt.show()

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   plt.plot(P[k,j],Flux_theory2[k,j],"x")
                   plt.plot(P[k,j],Flux_theory[k,j],"o")
                   plt.plot(P[k,j],Flux_real[k,j],"s")
                   
        
        plt.show()


        #plt.show()      
               

    
    def cut(self,inp):
            return inp.real[inp.shape[0]/2,:]  

    def img(self,inp,delta_u,delta_v):
            zz = inp
            zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
            zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

            zz_f = np.fft.fft2(zz) * (delta_u*delta_v)
            zz_f = np.roll(zz_f, -int(zz.shape[0]/2)-1, axis=0)
            zz_f = np.roll(zz_f, -int(zz.shape[0]/2)-1, axis=1)

            #fig, ax = plt.subplots()
            #print(image_s)
            #print(s)
            #im = ax.imshow(zz_f.real)
            #fig.colorbar(im, ax=ax) 
            #plt.show()
            return zz_f.real


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
    def calc_information(self,source_conv, delta_l,sigma_b):
        max_value = np.max(source_conv)
        half_theory = np.max(max_value)/2.0
        idx = (source_conv >= half_theory)
        integer_map = map(int, idx)
        fwhm = np.sum(list(integer_map))*(delta_l * (180.0/np.pi)) #in degrees
        C1 = 2*np.sqrt(2*np.log(2))
        sigma_n = (fwhm/C1) * (np.pi/180.0)#in radians
        #print(sigma_n**2-sigma_b**2)
        sigma_s = np.sqrt(sigma_n**2-sigma_b**2)# in radians
        #print("answers")
        #print(sigma_n*(180.0/np.pi))
        #print(sigma_s*(180.0/np.pi))
        C = (sigma_n**2)/(sigma_b**2)
        B = (2*np.pi*sigma_s**2)
        flux = max_value*C
        #print(flux)
        peak_brightness = flux/(2*np.pi*sigma_s**2)
        return sigma_s,C,B,flux,peak_brightness

    def process_pickle_files_g(self,P=np.array([])):
        counter = 0
        counter2 = 0
        

        vis_s = 5000
        resolution = 1

        N = int(np.ceil(vis_s*2/resolution))
        if (N % 2) == 0:
           N = N + 1
        u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)

        idx_M = np.zeros(P.shape,dtype=int)
        c = 0
        for k in range(len(P)):
            for j in range(len(P)):
                c += 1
                idx_M[k,j] = c   

        for k in range(len(P)):
            for j in range(len(P)):
                counter2 += 1
                if j != k:  
                   if j > k:
                      #print(counter)
                      name = "g_"+str(k)+"_"+str(counter)+"_"+str(j)+"_"+str(P[k,j])+".p"
                      counter += 1
                      
                      pkl_file = open(name, 'rb')
                      g_pq_t = pickle.load(pkl_file)
                      g_pq = pickle.load(pkl_file)
                      g_pq_inv = pickle.load(pkl_file)
                      r_pq = pickle.load(pkl_file)
                      r_pq =  pickle.load(pkl_file)
                      g_kernal = pickle.load(pkl_file)
                      sigma_kernal = pickle.load(pkl_file)
                      B = pickle.load(pkl_file)
                      
                      ax = plt.subplot(14, 14,counter2)                      
                      if (k != 0) or (j != 1):
                            #print(k)
                            #print(j)
                            #plt.setp(ax, 'frame_on', False)
                            #ax.set_ylim([-0.1, 1.1])
                            #ax.set_xlabel('K={},L={}'.format(k, l), size=3)
                            #ax.set_xlim([-0.1, 4.1])
                            
                            ax.set_xticks([])
                            ax.set_yticks([])
                      else:
                         ax.set_xlabel(r"$u$"+" [rad"+r"$^{-1}$"+"]")
                         #print(k)
                         #print(j)

                      if (k==0):
                         ax.set_title(str(j))

                      
                      ax.plot(u,np.absolute(g_pq)/B,"r")
                      ax.plot(u,np.absolute(g_pq_t)/B,"b")      
                      #plt.subplot(14, 14, counter2)
                      ax = plt.subplot(14, 14, idx_M[j,k])
                      #plt.setp(ax, 'frame_on', False)
                      ax.set_ylim([0.9, 2.0])
                      
                    
                      if (k != 1) or (j != 2):
                         ax.set_xticks([])
                         ax.set_yticks([])
                      else:
                         ax.set_xticks([])
                         ax.yaxis.set_label_position("right")
                         ax.yaxis.tick_right()   
                         
    
                      if (k==0):
                         #ax.set_ylabel(str(j), rotation=0, size='large')
                         #ax.annotate(str(j),(0,10*j)) 
                         ax.annotate(str(j), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center') 

                      #ax.set_xlabel('K={},L={}'.format(k, l), size=3)
                      #ax.set_xlim([-0.1, 4.1])
                      
                      
                      ax.plot(u,np.absolute(g_pq**(-1))*B,"r")
                      ax.plot(u,np.absolute(g_pq_t**(-1))*B,"b")
                      ax.plot(u,np.absolute(g_pq_inv)*B,"g")     

                else:
                    if (k==0):  
                       ax = plt.subplot(14, 14,counter2)
                       plt.setp(ax, 'frame_on', False)
                       ax.set_xticks([])
                       ax.set_yticks([])
                       ax.set_title(str(j))
                       ax.annotate(str(0), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')  

                   #g_pq12 = pickle.load(pkl_file)
                   #r_pq12 = pickle.load(pkl_file)
                   #g_kernal = pickle.load(pkl_file)
                   #g_pq_inv12 = pickle.load(pkl_file)
                   #g_pq_t12 = pickle.load(pkl_file)
                   #sigma_b = pickle.load(pkl_file)
                   #delta_u = pickle.load(pkl_file)
                   #delta_l = pickle.load(pkl_file)
                   #s_size = pickle.load(pkl_file)
                   #siz = pickle.load(pkl_file)
                   #r = pickle.load(pkl_file)
                   #b0 = pickle.load(pkl_file)
                   #K1 = pickle.load(pkl_file)
                   #K2 = pickle.load(pkl_file)
                   #fr = pickle.load(pkl_file)
                   #P = pickle.load(pkl_file) 
                   #pkl_file.close()  
                   #plt.subplot(14, 14, counter2)
                   #plt.plot(x,y)   

                   
        #fig.tight_layout() 
        #plt.axis('tight')
        #plt.tight_layout()
        plt.show()

    def include_baseline(self,k=0,j=1):
      
      if (k==0) and (j==1):
         return False
      if (k==0) and (j==2):
         return False
      if (k==0) and (j==3):
         return False
      if (k==1) and (j==2):
         return False
      if (k==1) and (j==3):
         return False
      if (k==1) and (j==4):
         return False
      if (k==2) and (j==3):
         return False
      if (k==2) and (j==4):
         return False
      if (k==2) and (j==5):
         return False
      if (k==3) and (j==4):
         return False
      if (k==3) and (j==5):
         return False
      if (k==4) and (j==5):
         return False
      if (k==4) and (j==6):
         return False
      if (k==5) and (j==6):
         return False
      if (k==5) and (j==7):
         return False
      if (k==6) and (j==7):
         return False
      if (k==6) and (j==8):
         return False
      if (k==7) and (j==8):
         return False
      if (k==12) and (j==13):
         return False
      return True



                      
    def plt_imshow(self,data,P,l=r"Log(Peak Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$])",name_file="f",vmax=-1,vmin=-1):
        N = 20 # number of colors to extract from each of the base_cmaps below
        base_cmaps = ['YlGnBu','Greys','Purples','Reds','Oranges']

        n_base = len(base_cmaps)
        # we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
        colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2,0.8,N)) for name in base_cmaps])    
        cmap = mpl.colors.ListedColormap(colors)


        fig, ax = plt.subplots()
        if vmax==vmin:
           im = ax.imshow(data, cmap="jet")
        else:
           im = ax.imshow(data, cmap="jet",vmax=vmax,vmin=vmin)
        cbar = fig.colorbar(im)
        cbar.set_label(l,labelpad=10)

        min_val, max_val, diff = 0., 14., 1.

        #text portion
        ind_array = np.arange(min_val, max_val, diff)
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val, p, d in zip(x.flatten(), y.flatten(), P.flatten(), data.flatten()):
            c = str(int(p))
            
            if np.isnan(d): 
               ax.text(x_val, y_val, int(np.absolute(p)), va='center', ha='center',color="black")
            else:
               ax.text(x_val, y_val, int(np.absolute(p)), va='center', ha='center',color="white")
        
        ax = plt.gca()
        xticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        yticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlabel("Antenna $q$")
        ax.set_ylabel("Antenna $p$")
        plt.savefig(name_file+".png")
        plt.close()
        #plt.show()

    def c_max(self,m=np.array([])):
        m_new = m.flatten()
        m = m_new[~np.isnan(m_new)]
        #print(m)
        me = np.mean(m)
        s = np.std(m)
        #print(me)
        #print(s)
        return me + 0.5*s

    def c_min(self,m=np.array([])):
        m_new = m.flatten()
        m = m_new[~np.isnan(m_new)]
        #print(m)
        me = np.mean(m)
        s = np.std(m)
        #print(me)
        return me - 1*s


    def main_phi_plot(self,P=np.array([]),m=np.array([])):
        
        counter = 0
        a = np.unique(np.absolute(P)).astype(int)
        #print(a)
        values_real = np.zeros((len(a),3),dtype=float)#0-real,1-theory,2-theory_inv
        counter_array = np.zeros((len(a),3),dtype=float)
        
        amp_matrix = np.zeros((len(P),len(P),3),dtype=float)
        size_matrix = np.zeros((len(P),len(P),3),dtype=float)
        flux_matrix = np.zeros((len(P),len(P),3),dtype=float)
       
        for k in range(len(P)):
            for j in range(len(P)):
                if j == k:
                   amp_matrix[k,j,:] = np.nan
                   size_matrix[k,j,:] = np.nan
                   flux_matrix[k,j,:] = np.nan
                if j != k:  
                   if j > k:
                      name = "10_baseline_"+str(counter)+"_"+str(k)+"_"+str(j)+"_"+str(P[k,j])+".p"
                      counter += 1
                      pkl_file = open(name, 'rb')
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
                      #b0 = pickle.load(pkl_file)
                      K1 = pickle.load(pkl_file)
                      K2 = pickle.load(pkl_file)
                      #fr = pickle.load(pkl_file)
                      P = pickle.load(pkl_file) 
                      pkl_file.close()
                      B = 2*np.pi*(s_size*(np.pi/180))**2
                      #PROCESS THE INFORMATION AS A FUNCTION OF PHI
                      c1 = self.cut(self.img(np.absolute(g_pq12**(-1)*r_pq12)*B,delta_u,delta_u))
                      c2 = self.cut(self.img(np.absolute(g_pq_t12**(-1)*r_pq12)*B,delta_u,delta_u))
                      c3 = self.cut(self.img(np.absolute(g_pq_inv12*r_pq12)*B,delta_u,delta_u))
                      
                      

                      if (self.include_baseline(k,j)):
                         amp_matrix[k,j,0] = np.max(c1)
                      else:
                         amp_matrix[k,j,0] = np.nan
                      amp_matrix[k,j,1] = np.max(c2)
                      amp_matrix[k,j,2] = np.max(c3)
                      amp_matrix[j,k,:] = amp_matrix[k,j,:] 
                      
                      if (self.include_baseline(k,j)):
                         idx = (c1 >= (amp_matrix[k,j,0]/2))
                         integer_map = map(int, idx)
                         size_matrix[k,j,0] = np.sum(list(integer_map))*r #in arcseconds
                      else:
                         size_matrix[k,j,0] = np.nan
                      

                      idx = (c2 >= (amp_matrix[k,j,1]/2))
                      integer_map = map(int, idx)
                      size_matrix[k,j,1] = np.sum(list(integer_map))*r #in arcseconds

                      idx = (c3 >= ((amp_matrix[k,j,2]-np.min(c3))/2))
                      integer_map = map(int, idx)
                      size_matrix[k,j,2] = np.sum(list(integer_map))*r #in arcseconds
                      
                      size_matrix[j,k,:] = size_matrix[k,j,:]
                      
                      if (self.include_baseline(k,j)):
                         flux_matrix[k,j,0] =  amp_matrix[k,j,0]*(((size_matrix[k,j,0]/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
                      else:
                         flux_matrix[k,j,0] = np.nan       
                      flux_matrix[k,j,1] =  amp_matrix[k,j,1]*(((size_matrix[k,j,1]/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi       
                      flux_matrix[k,j,2] =  amp_matrix[k,j,2]*(((size_matrix[k,j,2]/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
                      
                      flux_matrix[j,k,:] = flux_matrix[k,j,:]

        size_matrix[amp_matrix>100]=np.nan
        flux_matrix[amp_matrix>100]=np.nan
        amp_matrix[amp_matrix>100]=np.nan

        fwhm = (s_size*(np.pi/180) * 2*np.sqrt(2*np.log(2)))*(180/np.pi)*3600
        #print(fwhm)
        #print(r)
        #print((2*r)/fwhm)
        
        P_new = np.zeros(P.shape,dtype=float)
        P_s = np.zeros(P.shape,dtype=float)
        P_new2 = np.zeros(P.shape,dtype=float)

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j == k:
                   P_new2[j,k] = np.nan
                if j > k:
                   P_new[k,j] = (np.mean(((P[k,:]**2+P[k,j]**2)/P[k,j]**2)**(-1)) + np.mean(((P[:,j]**2+P[k,j]**2)/P[k,j]**2)**(-1)))/2
                   P_s[k,j] = (np.std(((P[k,:]**2+P[k,j]**2)/P[k,j]**2)**(-1)) + np.std(((P[:,j]**2+P[k,j]**2)/P[k,j]**2)**(-1)))/2
                   P_s[j,k] = P_s[k,j]
                   P_new[j,k] = P_new[k,j]
                   P_new2[k,j] = (np.mean(((P[k,:]**2+P[k,j]**2)/P[k,j]**2)**(1)) + np.mean(((P[:,j]**2+P[k,j]**2)/P[k,j]**2)**(1)))/2
                   P_new2[j,k] = P_new2[k,j] 

        self.plt_imshow(np.absolute(P_s),P,l=r"std$(\hat{K}_{pq})$",name_file="s")
        self.plt_imshow(np.absolute(P),P,l=r"$\phi_{pq}$",name_file="phi")
        self.plt_imshow(P_new,P,l=r"$\hat{K}_{pq}$",name_file="k")
        #self.plt_imshow(P_new2,P,l=r"$Log(Log(\hat{K}_{pq}^{-1})$",vmax=10,vmin=0)
        self.plt_imshow(m,P,l="Category",name_file="cat")
        
        for k in range(3):

            self.plt_imshow(amp_matrix[:,:,k],P,l=r"c",name_file="pk"+str(k),vmin=1,vmax=2)
            self.plt_imshow(size_matrix[:,:,k]/fwhm,P,l=r"FWHM $/$ FWHM$_B$",name_file="fw"+str(k),vmin=0,vmax=1)
            self.plt_imshow(flux_matrix[:,:,k]/B,P,l=r"Flux [Jy]",name_file="f"+str(k),vmin=1,vmax=2)
        
        #plt.imshow(np.log(amp_matrix[:,:,0]),cmap="cool") 
        #plt.show()
        #plt.imshow(np.log(size_matrix[:,:,0]),cmap="cool")
        #plt.show()
        #plt.imshow(np.log(flux_matrix[:,:,0]/B),cmap="cool")             
                                            
                                            


                      #self.cut(self.img(np.absolute(r_pq12),delta_u,delta_u))
        #values_real = values_real/counter_array 
        

        #values_real_temp = values_real[:,0]
        #values_real_temp[a==1] = 0
        #values_real_temp[a==4] = 0
        #values_real_temp[a==12] = 0
        #values_real_temp[a==16] = 0

        #sp0 = UnivariateSpline(a[1:], values_real_temp[1:],k=5)
        #sp0.set_smoothing_factor(1)
        #sp1 = UnivariateSpline(a[1:], values_real[1:,1],k=5)
        #sp1.set_smoothing_factor(100)
        #sp2 = UnivariateSpline(a[1:], values_real[1:,2])
        #sp2.set_smoothing_factor(0.1)
        #print(sp(a))
        #print(values_real)

        #plt.semilogy(a,sp2(a),"g")
        #plt.semilogy(a,sp0(a),"r")
        #plt.semilogy(a,sp1(a),"b")
        #plt.plot(a,values_real[:,0],"rx")
        #plt.plot(a,values_real[:,1],"bx")
        #plt.plot(a,values_real[:,2],"gx")   
        #plt.show()
        #plt.close()

    def process_pickle_files_g2(self,P=np.array([])):
        counter = 0
        counter2 = 0
        

        vis_s = 5000
        resolution = 1

        N = int(np.ceil(vis_s*2/resolution))
        if (N % 2) == 0:
           N = N + 1
        u = np.linspace(-(N - 1) / 2 * resolution, (N - 1) / 2 * resolution, N)

        idx_M = np.zeros(P.shape,dtype=int)
        c = 0
        for k in range(len(P)):
            for j in range(len(P)):
                c += 1
                idx_M[k,j] = c   

        for k in range(len(P)):
            for j in range(len(P)):
                counter2 += 1
                if j != k:  
                   if j > k:
                      #print(counter)
                      name = "g_"+str(k)+"_"+str(counter)+"_"+str(j)+"_"+str(P[k,j])+".p"
                      counter += 1
                      
                      pkl_file = open(name, 'rb')
                      g_pq_t = pickle.load(pkl_file)
                      g_pq = pickle.load(pkl_file)
                      g_pq_inv = pickle.load(pkl_file)
                      r_pq = pickle.load(pkl_file)
                      r_pq =  pickle.load(pkl_file)
                      g_kernal = pickle.load(pkl_file)
                      sigma_kernal = pickle.load(pkl_file)
                      B = pickle.load(pkl_file)

                      ax = plt.subplot(14, 14,counter2)                      
                      #ax.set_ylim([0, 6.0])

                      if (k != 0) or (j != 1):
                            #print(k)
                            #print(j)
                            #plt.setp(ax, 'frame_on', False)
                            #ax.set_ylim([-0.1, 1.1])
                            #ax.set_xlabel('K={},L={}'.format(k, l), size=3)
                            #ax.set_xlim([-0.1, 4.1])
                            
                            ax.set_xticks([])
                            #ax.set_yticks([])
                      else:
                         ax.set_xticks([-5000,5000]) 
                         #ax.set_xticklabels([str(-5000), '', str(5000)]) 
                         ax.set_xlabel(r"$u$"+" [rad"+r"$^{-1}$"+"]")
                         #print(k)
                         #print(j)

                      plt_i_domain = True

                      if (k==0) and (j==1):
                         ax.set_ylim(0, 7)
                         plt_i_domain = False

                      if (k==0) and (j==2):
                         ax.set_ylim(0, 3)
                         plt_i_domain = False 
                      
                      if (k==0) and (j==3):
                         ax.set_ylim(0, 3)  
                         plt_i_domain = False
                      if (k==1) and (j==2):
                         ax.set_ylim(0, 4)
                         plt_i_domain = False
                      if (k==1) and (j==3):
                         ax.set_ylim(0, 2)
                         plt_i_domain = False 
                      if (k==1) and (j==4):
                         ax.set_ylim(0, 2)
                         plt_i_domain = False 
                      if (k==2) and (j==3):
                         ax.set_ylim(0, 4)
                         plt_i_domain = False  
                      if (k==2) and (j==4):
                         ax.set_ylim(0, 2) 
                         plt_i_domain = False
                      if (k==3) and (j==4):
                         ax.set_ylim(0, 4)
                         plt_i_domain = False
                      if (k==3) and (j==5):
                         ax.set_ylim(0, 2)
                         plt_i_domain = False
                      if (k==4) and (j==5):
                         ax.set_ylim(0, 4)  
                         plt_i_domain = False
                      if (k==4) and (j==6):
                         ax.set_ylim(0, 2)
                         plt_i_domain = False
                      if (k==5) and (j==6):
                         ax.set_ylim(0, 4)
                         plt_i_domain = False
                      if (k==5) and (j==7):
                         ax.set_ylim(0, 2)
                         plt_i_domain = False
                      if (k==6) and (j==7):
                         ax.set_ylim(0, 4)
                         plt_i_domain = False
                      if (k==6) and (j==8):
                         ax.set_ylim(0, 2)  
                         plt_i_domain = False
                      if (k==7) and (j==8):
                         ax.set_ylim(0, 3)  
                         plt_i_domain = False

                      if (k==12) and (j==13):
                         ax.set_ylim(0, 20)
                         plt_i_domain = False
                      
                      if (k==0):
                         ax.set_title(str(j))
                         plt_i_domain = False
                      
                      ax.plot(u,np.absolute(g_pq**(-1)*r_pq),"r")
                      ax.plot(u,np.absolute(g_pq_t**(-1)*r_pq),"b")
                      ax.plot(u,np.absolute(g_pq_inv*r_pq),"g")
                      #ax.plot(u,np.absolute(r_pq/B),"y")      
                      #plt.subplot(14, 14, counter2)

                   #if j < k:
                      #10_baseline_9_0_10_37.0.p
                      name = "14_10_baseline_"+str(counter-1)+"_"+str(k)+"_"+str(j)+"_"+str(P[k,j])+".p"
                      pkl_file = open(name, 'rb')
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
                      #b0 = pickle.load(pkl_file)
                      K1 = pickle.load(pkl_file)
                      K2 = pickle.load(pkl_file)
                      #fr = pickle.load(pkl_file)
                      P = pickle.load(pkl_file) 
                      pkl_file.close() 
                      
                      ax = plt.subplot(14, 14, idx_M[j,k])
                      #plt.setp(ax, 'frame_on', False)
                      #ax.set_ylim([0.9, 2.0])
                      
                    
                      if (k != 2) or (j != 3):
                         
                         #print(siz)
                         ax.set_xticks([])
                         #ax.yaxis.set_label_position("right")
                         ax.yaxis.tick_right()
                         
                      else:
                         #ax.set_xticks([])
                         ax.set_xticks([-siz,siz])
                         ax.set_xlabel(r"$l$"+" [deg]")  
                         ax.xaxis.set_label_position("top")
                         ax.xaxis.tick_top()
                         ax.yaxis.set_label_position("right")
                         ax.yaxis.tick_right()   
                         
    
                      if (k==0):
                         #ax.set_ylabel(str(j), rotation=0, size='large')
                         #ax.annotate(str(j),(0,10*j)) 
                         ax.annotate(str(j), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center') 

                      #ax.set_xlabel('K={},L={}'.format(k, l), size=3)
                      #ax.set_xlim([-0.1, 4.1])
                      
                      x = np.linspace(-1*siz,siz,len(g_pq12))
                      if plt_i_domain:
                         ax.plot(x,self.cut(self.img(np.absolute(g_pq12**(-1)*r_pq12)*B,delta_u,delta_u)),"r")
                      ax.plot(x,self.cut(self.img(np.absolute(g_pq_t12**(-1)*r_pq12)*B,delta_u,delta_u)),"b")
                      ax.plot(x,self.cut(self.img(np.absolute(g_pq_inv12*r_pq12)*B,delta_u,delta_u)),"g")
                      ax.plot(x,self.cut(self.img(np.absolute(r_pq12),delta_u,delta_u)),"y")   
                      #ax.plot(u,np.absolute(g_pq_t**(-1))*B,"b")
                      #ax.plot(u,np.absolute(g_pq_inv)*B,"g")     

                else:
                    if (k==0):  
                       ax = plt.subplot(14, 14,counter2)
                       plt.setp(ax, 'frame_on', False)
                       ax.set_xticks([])
                       ax.set_yticks([])
                       ax.set_title(str(j))
                       ax.annotate(str(0), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')  

                   #g_pq12 = pickle.load(pkl_file)
                   #r_pq12 = pickle.load(pkl_file)
                   #g_kernal = pickle.load(pkl_file)
                   #g_pq_inv12 = pickle.load(pkl_file)
                   #g_pq_t12 = pickle.load(pkl_file)
                   #sigma_b = pickle.load(pkl_file)
                   #delta_u = pickle.load(pkl_file)
                   #delta_l = pickle.load(pkl_file)
                   #s_size = pickle.load(pkl_file)
                   #siz = pickle.load(pkl_file)
                   #r = pickle.load(pkl_file)
                   #b0 = pickle.load(pkl_file)
                   #K1 = pickle.load(pkl_file)
                   #K2 = pickle.load(pkl_file)
                   #fr = pickle.load(pkl_file)
                   #P = pickle.load(pkl_file) 
                   #pkl_file.close()  
                   #plt.subplot(14, 14, counter2)
                   #plt.plot(x,y)   

                   
        #fig.tight_layout() 
        #plt.axis('tight')
        #plt.tight_layout()
        plt.subplots_adjust(wspace=1,hspace=0.3) 
        plt.show()

    def plot_curves(self,P=np.array([]),N=14):
                
          if (N == 5):
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=1)
          if (N == 6):
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=1)
          if (N == 7):
             P = np.delete(P,[1,3,5,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,8,11,12],axis=1)
          if (N == 8):
             P = np.delete(P,[1,3,5,7,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,11,12],axis=1)
          if (N == 9):
             P = np.delete(P,[1,3,5,7,12],axis=0)
             P = np.delete(P,[1,3,5,7,12],axis=1)
          if (N == 10):
             P = np.delete(P,[1,3,5,7],axis=0)
             P = np.delete(P,[1,3,5,7],axis=1)
          if (N == 11):
             P = np.delete(P,[3,5,7],axis=0)
             P = np.delete(P,[3,5,7],axis=1)
          if (N == 12):
             P = np.delete(P,[5,7],axis=0)
             P = np.delete(P,[5,7],axis=1)
          if (N == 13):
             P = np.delete(P,[7],axis=0)
             P = np.delete(P,[7],axis=1)
          if (N == 4):
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=1)
        
          #print(P)

          counter = 0

          final_matrix = np.zeros((N,N),dtype = float)
          amp_matrix = np.zeros((len(P),len(P),3),dtype=float)

          curves = [np.array([]),np.array([]),np.array([])]
          for i in range(P.shape[0]):
              for j in range(P.shape[1]):
                  if j > i:
                     name = str(N)+"_10_baseline_"+str(counter)+"_"+str(i)+"_"+str(j)+"_"+str(P[i,j])+".p"
                   
                     counter+=1

                     pkl_file = open(name, 'rb')
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
                     K1 = pickle.load(pkl_file)
                     K2 = pickle.load(pkl_file)
                     P = pickle.load(pkl_file) 
                     pkl_file.close()

                     B = 2*np.pi*(s_size*(np.pi/180))**2  
                     fwhm = (s_size*(np.pi/180) * 2*np.sqrt(2*np.log(2)))*(180/np.pi)*3600
                     x = np.linspace(-1*siz,siz,len(g_pq12))

                     c1 = self.cut(self.img(np.absolute(g_pq12**(-1)*r_pq12)*B,delta_u,delta_u))#red
                     c2 = self.cut(self.img(np.absolute(g_pq_t12**(-1)*r_pq12)*B,delta_u,delta_u))#blue
                     c3 = self.cut(self.img(np.absolute(g_pq_inv12*r_pq12)*B,delta_u,delta_u))#green
                     c4 = self.cut(self.img(np.absolute(r_pq12),delta_u,delta_u))#yellow
  
                     curves[0] = c1
                     curves[1] = c2
                     curves[2] = c3

                     plt.clf()
                     if np.max(c1) < 100:
                        plt.plot(x,c1,"r")
                     plt.plot(x,c2,"b")
                     plt.plot(x,c3,"g")
                     plt.plot(x,c4,"y")
                     plt.savefig(name+"ng")
                     
                                   


    def compute_division_matrix(self,P=np.array([]),N=14,peak_flux=2,peak_flux2=100):
                
          if (N == 5):
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=1)
          if (N == 6):
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=1)
          if (N == 7):
             P = np.delete(P,[1,3,5,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,8,11,12],axis=1)
          if (N == 8):
             P = np.delete(P,[1,3,5,7,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,11,12],axis=1)
          if (N == 9):
             P = np.delete(P,[1,3,5,7,12],axis=0)
             P = np.delete(P,[1,3,5,7,12],axis=1)
          if (N == 10):
             P = np.delete(P,[1,3,5,7],axis=0)
             P = np.delete(P,[1,3,5,7],axis=1)
          if (N == 11):
             P = np.delete(P,[3,5,7],axis=0)
             P = np.delete(P,[3,5,7],axis=1)
          if (N == 12):
             P = np.delete(P,[5,7],axis=0)
             P = np.delete(P,[5,7],axis=1)
          if (N == 13):
             P = np.delete(P,[7],axis=0)
             P = np.delete(P,[7],axis=1)
          if (N == 4):
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=1)
        
          #print(P)

          counter = 0

          final_matrix = np.zeros((N,N),dtype = float)
          amp_matrix = np.zeros((len(P),len(P),3),dtype=float)

          curves = [np.array([]),np.array([]),np.array([])]
          for i in range(P.shape[0]):
              for j in range(P.shape[1]):
                  if j > i:
                     name = str(N)+"_10_baseline_"+str(counter)+"_"+str(i)+"_"+str(j)+"_"+str(P[i,j])+".p"
                   
                     counter+=1

                     pkl_file = open(name, 'rb')
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
                     K1 = pickle.load(pkl_file)
                     K2 = pickle.load(pkl_file)
                     P = pickle.load(pkl_file) 
                     pkl_file.close()

                     B = 2*np.pi*(s_size*(np.pi/180))**2  
                     fwhm = (s_size*(np.pi/180) * 2*np.sqrt(2*np.log(2)))*(180/np.pi)*3600
                     x = np.linspace(-1*siz,siz,len(g_pq12))

                     c1 = self.cut(self.img(np.absolute(g_pq12**(-1)*r_pq12)*B,delta_u,delta_u))#red
                     c2 = self.cut(self.img(np.absolute(g_pq_t12**(-1)*r_pq12)*B,delta_u,delta_u))#blue
                     c3 = self.cut(self.img(np.absolute(g_pq_inv12*r_pq12)*B,delta_u,delta_u))#green
  
                     curves[0] = c1
                     curves[1] = c2
                     curves[2] = c3
                     
                     value = -1
                     
                     if (np.max(curves[0]) <= peak_flux and np.max(curves[1]) <= peak_flux and np.max(curves[2]) <= peak_flux):
                        value = 0
                     elif (np.max(curves[0]) <= peak_flux and np.max(curves[1]) <= peak_flux2 and  np.max(curves[1]) > peak_flux  and np.max(curves[2]) <= peak_flux):  
                        value = 2
                     elif (np.max(curves[1]) <= peak_flux and np.max(curves[0]) <= peak_flux2 and  np.max(curves[0]) > peak_flux  and np.max(curves[2]) <= peak_flux):  
                        value = 1
                     elif (np.max(curves[1]) <= peak_flux and np.max(curves[0]) > peak_flux2  and np.max(curves[2]) <= peak_flux):  
                        value = 5
                     elif (np.max(curves[1]) > peak_flux and np.max(curves[1]) <= peak_flux2 and np.max(curves[0]) <= peak_flux2 and np.max(curves[0]) > peak_flux  and np.max(curves[2]) <= peak_flux):  
                        value = 3
                     elif (np.max(curves[1]) > peak_flux and np.max(curves[1]) <= peak_flux2 and np.max(curves[0]) > peak_flux2  and np.max(curves[2]) <= peak_flux):  
                        value = 4 

                     final_matrix[i,j] = value
                     final_matrix[j,i] = value

                     amp_matrix[i,j,0] = np.max(c1)
                     amp_matrix[i,j,1] = np.max(c2)
                     amp_matrix[i,j,2] = np.max(c3)
                     amp_matrix[j,i,:] = amp_matrix[i,j,:]  

          return final_matrix,amp_matrix              
    
    def get_average_response(self,P=np.array([]),N=14,peak_flux=100):
                
          if (N == 5):
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=1)
          if (N == 6):
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=1)
          if (N == 7):
             P = np.delete(P,[1,3,5,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,8,11,12],axis=1)
          if (N == 8):
             P = np.delete(P,[1,3,5,7,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,11,12],axis=1)
          if (N == 9):
             P = np.delete(P,[1,3,5,7,12],axis=0)
             P = np.delete(P,[1,3,5,7,12],axis=1)
          if (N == 10):
             P = np.delete(P,[1,3,5,7],axis=0)
             P = np.delete(P,[1,3,5,7],axis=1)
          if (N == 11):
             P = np.delete(P,[3,5,7],axis=0)
             P = np.delete(P,[3,5,7],axis=1)
          if (N == 12):
             P = np.delete(P,[5,7],axis=0)
             P = np.delete(P,[5,7],axis=1)
          if (N == 13):
             P = np.delete(P,[7],axis=0)
             P = np.delete(P,[7],axis=1)
          if (N == 4):
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=1)
        
          #print(P)

          counter1 = 0
          counter2 = 0
          counter3 = 0

          average_curve1 = np.array([])
          average_curve2 = np.array([])
          average_curve3 = np.array([]) 

          curves = [np.array([]),np.array([]),np.array([])]

          max_peak = np.zeros((3,),dtype=float)
          width = np.zeros((3,),dtype=float)
          flux = np.zeros((3,),dtype=float)

          counter = 0
          
          for i in range(P.shape[0]):
              for j in range(P.shape[1]):
                  if j > i:
                     name = str(N)+"_10_baseline_"+str(counter)+"_"+str(i)+"_"+str(j)+"_"+str(P[i,j])+".p"
                   
                     counter+=1

                     pkl_file = open(name, 'rb')
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
                     K1 = pickle.load(pkl_file)
                     K2 = pickle.load(pkl_file)
                     P = pickle.load(pkl_file) 
                     pkl_file.close()

                     B = 2*np.pi*(s_size*(np.pi/180))**2  
                     fwhm = (s_size*(np.pi/180) * 2*np.sqrt(2*np.log(2)))*(180/np.pi)*3600
                     x = np.linspace(-1*siz,siz,len(g_pq12))

                     c1 = self.cut(self.img(np.absolute(g_pq12**(-1)*r_pq12)*B,delta_u,delta_u))#red
                     c2 = self.cut(self.img(np.absolute(g_pq_t12**(-1)*r_pq12)*B,delta_u,delta_u))#blue
                     c3 = self.cut(self.img(np.absolute(g_pq_inv12*r_pq12)*B,delta_u,delta_u))#green

                     #print(len()
                     #print(c2)
                     #print(c3)

  
                     if (np.max(c1) < peak_flux):
                        if (len(average_curve1) == 0):
                           average_curve1 = c1
                           counter1 = 1
                        else:
                           average_curve1 += c1
                           counter1 += 1

                     if (len(average_curve2) == 0):
                           average_curve2 = c2
                           counter2 = 1
                     else:
                           average_curve2 += c2
                           counter2 += 1

                     if (len(average_curve3) == 0):
                           average_curve3 = c3
                           counter3 = 1
                     else:
                           average_curve3 += c3
                           counter3 += 1

                     #print(average_curve1)
                     #print(average_curve2)
                     #print(average_curve3)

          average_curve1 /= (counter1*1.0)
          average_curve2 /= (counter2*1.0)
          average_curve3 /= (counter3*1.0)
          
          max_peak[0] = np.max(average_curve1)
          max_peak[1] = np.max(average_curve2)
          max_peak[2] = np.max(average_curve3)

          idx = (average_curve1 >= (max_peak[0]/2))
          integer_map = map(int, idx)
          width[0] = np.sum(list(integer_map))*r #in arcseconds

          idx = (average_curve2 >= (max_peak[1]/2))
          integer_map = map(int, idx)
          width[1] = np.sum(list(integer_map))*r #in arcseconds

          idx = (average_curve3 >= (max_peak[2]/2))
          integer_map = map(int, idx)
          width[2] = np.sum(list(integer_map))*r #in arcseconds

          

          flux[0] =  (max_peak[0]/B)*(((width[0]/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
          flux[1] =  (max_peak[1]/B)*(((width[1]/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
          flux[2] =  (max_peak[2]/B)*(((width[2]/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi
          
          width = width/fwhm 

          #print(flux)
          #print(width)
          
          return max_peak,width,flux              

    def plot_as_func_of_N(self,P=np.array([]),N=14,peak_flux=2,peak_flux2=100):
                
          if (N == 5):
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,4,5,6,7,8,11,12],axis=1)
          if (N == 6):
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,6,7,8,11,12],axis=1)
          if (N == 7):
             P = np.delete(P,[1,3,5,7,8,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,8,11,12],axis=1)
          if (N == 8):
             P = np.delete(P,[1,3,5,7,11,12],axis=0)
             P = np.delete(P,[1,3,5,7,11,12],axis=1)
          if (N == 9):
             P = np.delete(P,[1,3,5,7,12],axis=0)
             P = np.delete(P,[1,3,5,7,12],axis=1)
          if (N == 10):
             P = np.delete(P,[1,3,5,7],axis=0)
             P = np.delete(P,[1,3,5,7],axis=1)
          if (N == 11):
             P = np.delete(P,[3,5,7],axis=0)
             P = np.delete(P,[3,5,7],axis=1)
          if (N == 12):
             P = np.delete(P,[5,7],axis=0)
             P = np.delete(P,[5,7],axis=1)
          if (N == 13):
             P = np.delete(P,[7],axis=0)
             P = np.delete(P,[7],axis=1)
          if (N == 4):
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=0)
             P = np.delete(P,[1,2,3,4,5,6,7,8,11,12],axis=1)
        
          #print(P)

          counter = 0

          r_ampl = [np.array([]),np.array([]),np.array([])]
          r_size = [np.array([]),np.array([]),np.array([])]
          r_flux = [np.array([]),np.array([]),np.array([])]
          r_counter = np.array([0,0,0])

          r_ampl2 = [np.array([]),np.array([]),np.array([])]
          r_size2 = [np.array([]),np.array([]),np.array([])]
          r_flux2 = [np.array([]),np.array([]),np.array([])]
          r_counter2 = [np.array([]),np.array([]),np.array([])]

          r_phi = [np.array([]),np.array([]),np.array([])]
          r_phi2 = [np.array([]),np.array([]),np.array([])]
          curves = [np.array([]),np.array([]),np.array([])]


          for i in range(P.shape[0]):
              for j in range(P.shape[1]):
                  if j > i:
                     name = str(N)+"_10_baseline_"+str(counter)+"_"+str(i)+"_"+str(j)+"_"+str(P[i,j])+".p"
                   
                     counter+=1

                     pkl_file = open(name, 'rb')
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
                     K1 = pickle.load(pkl_file)
                     K2 = pickle.load(pkl_file)
                     P = pickle.load(pkl_file) 
                     pkl_file.close()

                     B = 2*np.pi*(s_size*(np.pi/180))**2  
                     fwhm = (s_size*(np.pi/180) * 2*np.sqrt(2*np.log(2)))*(180/np.pi)*3600
                     x = np.linspace(-1*siz,siz,len(g_pq12))

                     c1 = self.cut(self.img(np.absolute(g_pq12**(-1)*r_pq12)*B,delta_u,delta_u))
                     c2 = self.cut(self.img(np.absolute(g_pq_t12**(-1)*r_pq12)*B,delta_u,delta_u))
                     c3 = self.cut(self.img(np.absolute(g_pq_inv12*r_pq12)*B,delta_u,delta_u))
  
                     curves[0] = c1
                     curves[1] = c2
                     curves[2] = c3

                     store = False
                     if (np.max(curves[0]) < peak_flux and np.max(curves[1]) < peak_flux and np.max(curves[2]) < peak_flux):
                        store = True

                     for c in range(len(curves)):
                         if np.max(curves[c]) < peak_flux:
                            r_ampl[c] = np.append(r_ampl[c],[np.max(curves[c])])
                            idx = (curves[c] >= (np.max(curves[c])/2))
                            integer_map = map(int, idx)
                            r_size[c] = np.append(r_size[c],[np.sum(list(integer_map))*r]) #in arcseconds
                            r_flux[c] = np.append(r_flux[c],np.max(curves[c])*((((np.sum(list(integer_map))*r)/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi)
                            r_counter[c] += 1
                            r_phi[c] = np.append(r_phi[c],[P[i,j]])
                         else: 
                            if (np.max(curves[c]) < peak_flux2):
                               r_ampl2[c] = np.append(r_ampl2[c],[np.max(curves[c])])
                               idx = (curves[c] >= (np.max(curves[c])/2))
                               integer_map = map(int, idx)
                               r_size2[c] = np.append(r_size2[c],[np.sum(list(integer_map))*r]) #in arcseconds
                               r_flux2[c] = np.append(r_flux2[c],np.max(curves[c])*((((np.sum(list(integer_map))*r)/3600.0)*(np.pi/180) )/(2*np.sqrt(2*np.log(2))))**2*2*np.pi)
                               r_counter2[c] += 1
                               r_phi2[c] = np.append(r_phi2[c],[P[i,j]])

          return curves,x,self.cut(self.img(np.absolute(r_pq12),delta_u,delta_u)),counter,r_ampl,r_size,r_flux,r_counter,r_ampl2,r_size2,r_flux2,r_counter2,B,fwhm,r_phi,r_phi2              
    


    def set_axis_style(self,ax, labels):
      ax.xaxis.set_tick_params(direction='out')
      ax.xaxis.set_ticks_position('bottom')
      ax.set_xticks(np.arange(1, len(labels) + 1))
      ax.set_xticklabels(labels)
      ax.set_xlim(0.25, len(labels) + 0.75)
      ax.set_xlabel(r'$N$')

    def create_violin_plot(self,data=[],fc = "green",labels=['4','5','6','7','8','9','10','11','12','13','14'],yl=r"Amp. Bright. $\times B$ [Wm$^{-2}$Hz$^{-1}$sr$^{-1}$]",t=False):
        fig, ax = plt.subplots()
        # Create a plot
        parts = ax.violinplot(data,showmeans=True,showmedians=True)
   
        for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('green')
        parts['cmaxes'].set_color('magenta')

        for pc in parts['bodies']:
            pc.set_facecolor(fc)
            pc.set_edgecolor('black')
            pc.set_alpha(0.2)
        # Add title
        #ax.set_title('Violin Plot')
        # set style for the axes
        self.set_axis_style(ax, labels)
        ax.set_ylabel(yl)

        if t:
           N = np.array([4,5,6,7,8,9,10,11,12,13,14])
           x = np.array([1,2,3,4,5,6,7,8,9,10,11])
        
           plt.plot(x,(2.0*N-1)/N,"r")   
        plt.show() 
        

    def new_exp(self,P=np.array([]),b0=36,fr=1.45e9,K1=40,K2=100):

        b0_init = b0 #in m (based on the smallest baseline lenght of WSRT)
        freq = fr
        lam = (1.0*3*10**8)/freq
        b_len = b0_init*P[0,-1]

        fwhm = 1.02*lam/(b_len)#in radians
        sigma_kernal = (fwhm/(2*np.sqrt(2*np.log(2))))*K1 #in radians (40 for 14 antenna experiment), size of source 40 times the size of telescope resolution

        fwhm2 = 1.02*lam/(b0_init)#in radians
        sigma_kernal2 = (fwhm2/(2*np.sqrt(2*np.log(2)))) #in radians

        s_size = sigma_kernal*(180/np.pi) #in degrees
        #r = (s_size*3600)/10.0 #in arcseconds
        r = (sigma_kernal2*(180/np.pi)*3600)/K2 #in arcseconds
        siz = sigma_kernal2 * 3 * (180/np.pi)
   
        #print("beginning size")
        #print(s_size)
        sigma = s_size*(np.pi/180)#in radians
        B1 = 2*sigma**2*np.pi

        counter = 0

        for k in range(len(P)):
            for j in range(len(P)):
                if j > k:
                   print("***********************")
                   print(counter)
                   print("***********************")
                   print("Theory")
                   g_pq_t12,g_pq_inv12,delta_u,delta_v = self.theory_g(baseline=np.array([k,j]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
                   print("Simulation")
                   r_pq12,g_pq12,delta_u,delta_v,g_kernal,sigma_b,delta_l = self.extrapolation_function(baseline=np.array([k,j]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r,kernel=True,b0=b0_init,f=fr)
                   file_name = "baseline_"+str(counter)+"_"+str(k)+"_"+str(j)+"_"+str(P[k,j])+".p"
                   f = open(file_name, 'wb')
                   pickle.dump(g_pq12,f)
                   pickle.dump(r_pq12,f)
                   pickle.dump(g_kernal,f)
                   pickle.dump(g_pq_inv12,f)
                   pickle.dump(g_pq_t12,f)
                   pickle.dump(sigma_b,f)
                   pickle.dump(delta_u,f)
                   pickle.dump(delta_l,f)
                   pickle.dump(s_size,f)
                   pickle.dump(siz,f)
                   pickle.dump(r,f)
                   pickle.dump(b0,f)
                   counter+=1
                   pickle.dump(K1,f)
                   pickle.dump(K2,f)
                   pickle.dump(fr,f)
                   pickle.dump(P,f) 
                   f.close()  


if __name__ == "__main__":
   #print('hallo')   
   perform_exp_or_make_plots = False #if true make plots, if false perform experiments

   t = T_ghost()
   #t.process_pickle_file()
   P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(-18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])
   
   if perform_exp_or_make_plots:

      #t.anotherExpPicProcess(P=P)

      print("PLOTTING RESULTS") 

      print("FOR N=14 PLOTTING g AND g_inv")
      t.process_pickle_files_g(P=P)
      print("FOR N=14 PLOTTING F{g_inv} AND F{g_inv*r}")
      t.process_pickle_files_g2(P=P)
   
      print("FOR N=14 PLOTTING PEAK BRIGHT, FWHM AND FLUX FOR ALL PQ ---> ONLY SAVING FIGURES")
      m,amp_matrix = t.compute_division_matrix(P=P,N=14,peak_flux=2,peak_flux2=100)
      t.main_phi_plot(P=P,m=m)

      print("AS A FUNCTION OF N PLOTTING PEAK BRIGHT, FWHM AND FLUX AVEGARE") 
      n = np.array([4,5,6,7,8,9,10,11,12,13,14])

      peak = np.zeros((len(n),3),dtype=float)
      width = np.zeros((len(n),3),dtype=float)
      flux = np.zeros((len(n),3),dtype=float)

      for k in range(len(n)):
          peak[k,:],width[k,:],flux[k,:] = t.get_average_response(P=P,N=n[k],peak_flux=100)

      c = ['r','b','g']
      for i in range(3):

          plt.semilogy(n,peak[:,i],c[i])
          plt.semilogy(n,width[:,i],c[i]+"--")
          plt.semilogy(n,flux[:,i],c[i]+":")

      plt.xlabel(r"$N$")
      plt.ylabel(r"$c$ or FWHM/FWHM$_B$ or Flux [Jy]")
      plt.legend()
      plt.show()   

      print("PLOTTING DIST OF PEAK BRIGHT, FWHM AND FLUX AS FUNC OF N VIA VIOLIN PLOTS")
   
      n = np.array([4,5,6,7,8,9,10,11,12,13,14])
      final = np.zeros((len(n),3),dtype=float)
      final_count = np.zeros((len(n),),dtype=float)

      col = np.array(["r","b","g","y"])
      A1 = []
      A2 = []
      A3 = []

      S1 = []
      S2 = []
      S3 = []

      F1 = []
      F2 = []
      F3 = []

      P1 = []
      P2 = []
      P3 = []

      for k in range(len(n)):
          #print(k)
          #result1,result2,result3,l,c_final1,c_final2,c_final3,r,c,a1,a2,a3,s1,s2,s3,f1,f2,f3,B,FW = t.plot_as_func_of_N(P=P,N=n[k])
          curves,x,r,c,A,S,F,C,A_2,S_2,F_2,C_2,B,FW,PM,PM_2 = t.plot_as_func_of_N(P=P,N=n[k],peak_flux=100,peak_flux2=100)
        
          A1.append(A[0])
          A2.append(A[1])
          A3.append(A[2])
          #P1.append(PM[0])
          #P2.append(PM[1])
          #P3.append(PM[2]) 
          S1.append(S[0]/FW)
          S2.append(S[1]/FW)
          S3.append(S[2]/FW)
          F1.append(F[0]/B)
          F2.append(F[1]/B)
          F3.append(F[2]/B) 

      t.create_violin_plot(data=A1,fc='red',yl="c",t=False)  
      t.create_violin_plot(data=A2,fc='blue',yl="c",t=False)
      t.create_violin_plot(data=A3,fc='green',yl="c",t=True)

      t.create_violin_plot(data=S1,fc='red',yl=r"FWHM/FWHM$_B$")  
      t.create_violin_plot(data=S2,fc='blue',yl=r"FWHM/FWHM$_B$")
      t.create_violin_plot(data=S3,fc='green',yl=r"FWHM/FWHM$_B$")

      t.create_violin_plot(data=F1,fc='red',yl=r"Flux [Jy]")  
      t.create_violin_plot(data=F2,fc='blue',yl=r"Flux [Jy]")
      t.create_violin_plot(data=F3,fc='green',yl=r"Flux [Jy]")      

   else:
       print("PERFORM MAIN EXPERIMENT PER N, GENERATES ALL THE N_10_*.p FILES")
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 14)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 13)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 12)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 11)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 10)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 9)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 8)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 7)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 6)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 5)
      #  t.anotherExp(P,size_gauss=0.02,K1=30.0,K2=4.0, N = 4)
  
       print("PERFORM SECONDARY EXPERIMENT FOR N=14, GENERATES ALL THE g_*.p FILES")
       counter = 0
       for k in range(len(P)):
        for j in range(len(P)):
           if j > k:
              print(counter)
              g_pq_t,g_pq_inv,u,B = t.theory_g_linear(baseline=np.array([k,j]),true_sky_model=np.array([[1,0,0,0.02]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,vis_s=5000,resolution=1)
              r_pq,g_pq,u,g_kernal,sigma_kernal=t.extrapolation_function_linear(baseline=np.array([k,j]),true_sky_model=np.array([[1,0,0,0.02]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,vis_s=5000,resolution=1)
              plt.clf()
              plt.plot(u,np.absolute(g_pq_t**(-1)*r_pq),"b")
              plt.plot(u,np.absolute(g_pq**(-1)*r_pq),"r")
              name = "g_"+str(k)+"_"+str(counter)+"_"+str(j)+"_"+str(P[k,j])+".png"
              plt.savefig(name)
              counter = counter+1
              f = open(name[:-2], 'wb')
              pickle.dump(g_pq_t,f)
              pickle.dump(g_pq,f)
              pickle.dump(g_pq_inv,f)
              pickle.dump(r_pq,f)
              pickle.dump(r_pq,f)
              pickle.dump(g_kernal,f)
              pickle.dump(sigma_kernal,f)
              pickle.dump(B,f)
