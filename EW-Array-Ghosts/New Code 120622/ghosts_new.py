import numpy as np
import pylab as plt
import pickle
import sys

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
                g_stef, G = self.create_G_stef(R, M, 200, 1e-20, temp, no_auto=False) 
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
        print(P)

        a = np.unique(np.absolute(P)).astype(int)
        print(a)
        

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

        print(g_real.shape) 
        #plt.plot(g_real)
        #plt.show()
        

        print(Curves_theory)


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
        
        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   v = P[k,j]
                   result = np.where(a == v)[0][0]
                   values_real[result] += Amp_real[k,j]
                   values_t1[result] += Amp_theory[k,j]
                   values_t2[result] += Amp_theory2[k,j]
                   counter[result] += 1
        values_real = values_real/counter
        values_t1 = values_t1/counter
        values_t2 = values_t2/counter
        values_real[values_real>20000] = np.nan

        plt.semilogy(a,values_real,"ro")
        plt.semilogy(a,values_t1,"bs")
        plt.semilogy(a,values_t2,"gx")         
        plt.show()

        print("HALLO")
        plt.plot(Curves_real[9,10,:],"b")
        plt.show()

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   #print(k)
                   #print(j)
                   #print(np.max(Curves_real[k,j,:]))
                   #print(P[k,j])
                   plt.plot(Curves_theory[k,j,:],"--")
                   plt.plot(Curves_real[k,j,:])
                   plt.plot(Curves_theory2[k,j,:],".")
        plt.show()

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
        print(sigma_n**2-sigma_b**2)
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
   t = T_ghost()
   P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])
   t.new_exp(P=P)

   '''
   s1 = (0.009508775037750116/5)*40
   sb = 0.14263162556625175
   sb2 = 0.001901755
   
   #0.02852632511325035
   #0.00015847958396250194

   source,l,delta_l,c1 = t.gaussian(A=1,sigma=s1,image_s=3*sb,s=1,resolution=(sb/100)*3600)
   source_conv,l,delta_l,c2 = t.gaussian_conv_clean_beam(A=1,sigma=s1,sigma_b=sb2,image_s=3*sb,s=1,resolution=(sb/100)*3600)
   sigma_s,C,B,flux,peak_brightness = t.calc_information(source_conv=source_conv, delta_l=delta_l,sigma_b=sb2*(np.pi/180))
   B_old = B
   plt.plot(l,source*B,"b")
   plt.show()
   plt.plot(l,source_conv*C,"g")#kortse
   plt.show()
   #plt.plot(y,t.cut(t.img(g_kernal*r_pq12,delta_u,delta_v))*C,"r")#kortse

   plt.plot(l,source,"b")
   plt.plot(l,source_conv,"r")
   plt.show()

   '''

   
   
   '''
   half_theory = np.max(source_conv)/2.0
   idx = (source_conv >= half_theory)
   integer_map = map(int, idx)
   fwhm = np.sum(list(integer_map))*(delta_l * (180.0/np.pi)) #in degrees
   C = 2*np.sqrt(2*np.log(2))
   sigma_n = fwhm/C
   sigma_s = np.sqrt(sigma_n**2-0.5**2)
   print(sigma_n**2)
   print(sigma_s)
   sigma_s *= (np.pi/180) #converts to radians
   peak_brightness = 1/(2*np.pi*sigma_s**2) #brightness in W/(m^2 sr Hz) 
   print(peak_brightness)
   '''
   #P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])
      

   #P = np.array([[0,3,5],[-3,0,2],[-5,-2,0]])
   '''
   print(P)
   #SETTING UP THE SOURCE
   b0_init = 36 #in m (based on the smallest baseline lenght of WSRT)
   freq = 1.45e9
   lam = (1.0*3*10**8)/freq
   b_len = b0_init*P[0,-1]
   fwhm = 1.02*lam/(b_len)#in radians
   sigma_kernal = (fwhm/(2*np.sqrt(2*np.log(2))))*40 #in radians (40 for 14 antenna experiment)

   fwhm2 = 1.02*lam/(b0_init)#in radians
   sigma_kernal2 = (fwhm2/(2*np.sqrt(2*np.log(2)))) #in radians

   s_size = sigma_kernal*(180/np.pi) #in degrees
   #r = (s_size*3600)/10.0 #in arcseconds
   r = (sigma_kernal2*(180/np.pi)*3600)/100.0 #in arcseconds
   siz = sigma_kernal2 * 3 * (180/np.pi)
   
   print("beginning size")
   print(s_size)
   sigma = s_size*(np.pi/180)#in radians
   B1 = 2*sigma**2*np.pi

   #9,10 kortste baseline ...
   g_pq_t12,g_pq_inv12,delta_u,delta_v =t.theory_g(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
   r_pq12,g_pq12,delta_u,delta_v,g_kernal,sigma_b,delta_l =t.extrapolation_function(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r,kernel=True,b0=36,f=1.45e9)
   file_name = "baseline12_14.p"
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
   f.close()        


   print("sizes")
   print(sigma*(180/np.pi))
   print(sigma_b*(180/np.pi))
   print(siz)
   print(r/3600)

   #sigma_n =np.sqrt(sigma**2+sigma_b**2)
   #C = (sigma_n**2)/(sigma_b**2)
   y = np.linspace(-1*siz,siz,len(g_pq12))
   x = np.linspace(-1*delta_u*len(g_pq12)/2,1*delta_u*len(g_pq12)/2,len(g_pq12))

   plt.plot(x,t.cut(g_kernal),"y")
   plt.show()

   plt.plot(x,t.cut(r_pq12),"m")
   plt.show()

   source_conv = t.cut(t.img(g_kernal*r_pq12,delta_u,delta_v))
   source_conv2 = t.cut(t.img(r_pq12,delta_u,delta_v))
   plt.plot(y,source_conv/np.max(source_conv),"r")
   plt.plot(y,source_conv2/np.max(source_conv2),"b")
   plt.show()

   sigma_s,C,B,flux,peak_brightness = t.calc_information(source_conv=source_conv, delta_l=delta_l,sigma_b=sigma_b)
   B_old = B
   plt.plot(y,t.cut(t.img(r_pq12,delta_u,delta_v))*B,"g")#kortse
   plt.plot(y,t.cut(t.img(g_kernal*r_pq12,delta_u,delta_v))*C,"r")#kortse
   #plt.plot(y,t.cut(t.img(g_kernal*g_pq12**(-1)*r_pq12,delta_u,delta_v))*C,"b")#kortse
   plt.show()

   source_conv = t.cut(t.img(g_kernal*(g_pq12**(-1)*r_pq12),delta_u,delta_v))
   sigma_s,C,B,flux,peak_brightness = t.calc_information(source_conv=source_conv, delta_l=delta_l,sigma_b=sigma_b)
   plt.plot(y,t.cut(t.img(r_pq12,delta_u,delta_v))*B_old,"g")#kortse
   plt.plot(y,t.cut(t.img(g_kernal*(g_pq12**(-1)*r_pq12),delta_u,delta_v))*C,"r")#kortse
   #plt.plot(y,t.cut(t.img(g_kernal*(g_pq_t12**(-1)*r_pq12),delta_u,delta_v))*C,"y")#kortse
   plt.plot(y,t.cut(t.img(g_kernal,delta_u,delta_v)),"b")#kortse
   plt.show()
   print("answers")
   print(sigma_s*(180/np.pi))
   print(flux)
   print(peak_brightness)
   '''

   #max_value = np.max(source_conv)
   #half_theory = np.max(max_value)/2.0
   #idx = (source_conv >= half_theory)
   #integer_map = map(int, idx)
   #fwhm = np.sum(list(integer_map))*(delta_l * (180.0/np.pi)) #in degrees
   #C1 = 2*np.sqrt(2*np.log(2))
   #sigma_n = (fwhm/C1) * (np.pi/180.0)#in radians
   #sigma_s = np.sqrt(sigma_n**2-sigma_b**2)
   
   #print(sigma_n*(180.0/np.pi))
   #print(sigma_s*(180.0/np.pi))
   #C = (sigma_n**2)/(sigma_b**2)
   #B = (2*np.pi*sigma_s**2)
   #flux = max_value*C
   #print(flux)
   #peak_brightness = max_value/(2*np.pi*sigma_b**2)

   


   #print("r")
   #print(r)
   
   #print("siz")
   #print(siz)

   #sigma = s_size*(np.pi/180)
   #B1 = 2*sigma**2*np.pi

   #b_len = 144
   #f = 1.45*10**9

   #lam = 3*10**8/f

   #print(lam)

   #w = b_len/lam

   #b_len = 2700
   #f = 1.45*10**9

   #lam = 3*10**8/f

   #print(lam)

   #w = lam/b_len

   #print(w*180/np.pi)
                

   #t.rectangle(A=1,sigma=0.1,image_s=0.5,s=1,resolution=50)
   '''
   t.process_pickle_file()

    
   P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])
   
   s_size = 0.02
   r = (s_size*3600)/20.0
   print("r")
   print(r)
   siz = s_size * 3
   print("siz")
   print(siz)

   sigma = s_size*(np.pi/180)
   B1 = 2*sigma**2*np.pi

   b_len = 144
   f = 1.45*10**9

   lam = 3*10**8/f

   print(lam)

   w = b_len/lam

   b_len = 2700
   f = 1.45*10**9

   lam = 3*10**8/f

   print(lam)

   w = lam/b_len

   print(w*180/np.pi)

   #5,11
   #10,12

   g_pq_t12,g_pq_inv12,delta_u,delta_v =t.theory_g(baseline=np.array([10,12]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
   r_pq12,g_pq12,delta_u,delta_v,g_kernal =t.extrapolation_function(baseline=np.array([10,12]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
   x = np.linspace(-1*delta_u*len(g_pq12)/2,1*delta_u*len(g_pq12)/2,len(g_pq12))
   y = np.linspace(-1*siz,siz,len(g_pq12))
   
   plt.plot(x,t.cut(g_kernal),"b--")#kortse
   plt.show()

   plt.plot(y,t.cut(t.img(g_kernal,delta_u,delta_v)),"r--")#kortse
   plt.show()

   m = np.max(t.cut(t.img(r_pq12,delta_u,delta_v)))
   plt.plot(y,t.cut(t.img(g_kernal*g_pq12**(-1)*r_pq12,delta_u,delta_v)),"r--")#kortse
   plt.plot(y,t.cut(t.img(g_kernal*g_pq_t12**(-1)*r_pq12,delta_u,delta_v)),"y--")#kortse
   plt.plot(y,t.cut(t.img(g_kernal*r_pq12,delta_u,delta_v)),"b--")#kortse
   plt.plot(y,t.cut(t.img(g_kernal,delta_u,delta_v)),"m--")#kortse
   #plt.plot(y,t.cut(t.img(r_pq12,delta_u,delta_v))/m,"g--")#kortse
   plt.plot(y,t.cut(t.img(r_pq12,delta_u,delta_v)),"k--")#kortse
   plt.show()


   m = np.max(t.cut(g_pq_t12**(-1)*r_pq12))/2
   #plt.plot(x,t.cut(g_pq_t12**(-1)*r_pq12),"r--")
   plt.plot(x,t.cut(g_pq12**(-1)*r_pq12*g_kernal),"b--")#kortse
   #plt.plot(x,t.cut(g_pq12**(-1)*r_pq12),"r--")#kortse
   #plt.plot([-w,w],[m,m],"g")
   plt.show()

   m = np.max(t.cut(g_kernal))
   plt.plot(x,t.cut(g_pq_t12),"r--")
   plt.plot(x,t.cut(g_kernal)/m,"g--")
   plt.plot(x,t.cut(g_pq12),"b--")#kortse
   #plt.plot([-w,w],[m,m],"g")
   plt.show()

   plt.semilogy(x,t.cut(g_pq_t12),"r--")
   plt.semilogy(x,t.cut(g_pq12),"b--")#kortse
   plt.semilogy([-w,w],[m,m],"g")
   plt.show()

   m = np.max(t.cut(g_pq_t12**(-1)))/2
   plt.plot(x,t.cut(g_pq_t12**(-1)),"r--")
   plt.plot(x,t.cut(g_pq12)**(-1),"b--")#kortse
   plt.semilogy([-w,w],[m,m],"g")
   plt.show()
   '''
   

   '''
   P = np.array([[0,3,5],[-3,0,2],[-5,-2,0]])
   s_size = 1
   r = (s_size*3600)/20.0
   siz = s_size * 3

   sigma = s_size*(np.pi/180)
   B1 = 2*sigma**2*np.pi

   #b_len = 144
   #f = 1.45*10**9

   #lam = 3*10**8/f

   #w = b_len/lam


   g_pq_t12,g_pq_inv12,delta_u,delta_v =t.theory_g(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=8,s=1,resolution=200)
   r_pq12,g_pq12,delta_u,delta_v =t.extrapolation_function(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=8,s=1,resolution=200)
   g_pq_t01,g_pq_inv01,delta_u,delta_v =t.theory_g(baseline=np.array([0,1]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=8,s=1,resolution=200)
   r_pq01,g_pq01,delta_u,delta_v =t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=8,s=1,resolution=200)
   g_pq_t02,g_pq_inv02,delta_u,delta_v =t.theory_g(baseline=np.array([0,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=8,s=1,resolution=200)
   r_pq02,g_pq02,delta_u,delta_v =t.extrapolation_function(baseline=np.array([0,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=8,s=1,resolution=200)
   delta_l = 200 * (1.0 / 3600.0)
      


   curve = t.cut(t.img(r_pq12,delta_u,delta_v))
   x = np.linspace(-1*delta_u*len(g_pq12)/2,1*delta_u*len(g_pq12)/2,len(g_pq12))
   plt.plot(x,t.cut(g_pq_t12**(-1)*r_pq12),"r--")
   plt.plot(x,t.cut(g_pq12**(-1)*r_pq12),"b--")#kortse
   plt.plot(x,t.cut(g_pq_t02**(-1)*r_pq12),"r")
   plt.plot(x,t.cut(g_pq02**(-1)*r_pq12),"b")#langste
   plt.plot(x,t.cut(g_pq_t01**(-1)*r_pq12),"r.")#middel
   plt.plot(x,t.cut(g_pq01**(-1)*r_pq12),"b.")
      

   plt.plot(x,t.cut(r_pq12),"y--")
   plt.show()
   #plt.plot([-w/2,w/2],[0.5,0.5],"g")

   #plt.xlim([,)
   #plt.plot(t.cut(g_pq12),"g--")
   #plt.show()

   x = np.linspace(-1*delta_l*len(g_pq12)/2,1*delta_l*len(g_pq12)/2,len(g_pq12))

   plt.plot(x,t.cut(t.img(r_pq12,delta_u,delta_v)),"y--")
   plt.plot(x,t.cut(t.img(g_pq_t12**(-1)*r_pq12,delta_u,delta_v)),"r--") #kortste een
   plt.plot(x,t.cut(t.img(g_pq12**(-1)*r_pq12,delta_u,delta_v)),"b--") #kortste een
   plt.plot(x,t.cut(t.img(g_pq_t02**(-1)*r_pq12,delta_u,delta_v)),"r") #langste een
   plt.plot(x,t.cut(t.img(g_pq02**(-1)*r_pq12,delta_u,delta_v)),"b") #langste een
   plt.plot(x,t.cut(t.img(g_pq_t01**(-1)*r_pq12,delta_u,delta_v)),"r.") #middel een
   plt.plot(x,t.cut(t.img(g_pq01**(-1)*r_pq12,delta_u,delta_v)),"b.") #middel een
   
   plt.show()
   '''
   #plt.imshow(t.img(g_pq12**(-1)*r_pq12,delta_u,delta_v))
   #plt.show()

   #plt.imshow(t.img(r_pq12,delta_u,delta_v))
   #plt.show()

   #N_a = np.array([7,14])
   #s_size = np.array([0.001,0.1,1])
   #P = 4*np.array([[0,1,2,3,4],[-1,0,1,2,3],[-2,-1,0,1,2],[-3,-2,-1,0,1],[-4,-3,-2,-1,0]])
   #P = np.array([[0,3,5],[-3,0,2],[-5,-2,0]])
   
   #P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])
   
   #for k in range(len(N_a)):
   #    for i in range(len(s_size)):
   #        print("*****************")
   #        print(N_a[k])
   #        print(s_size[i])
   #        print("*****************")
   #        t.conductExp(P,N_a[k],s_size[i],i)

   #t.process_pickle_file()

   '''
   #print(t.c_func(0, 1, 1, 3, 5, 7, 9))
   s_size = 1
   r = (s_size*3600)/20.0
   siz = s_size * 3

   sigma = s_size*(np.pi/180)
   B1 = 2*sigma**2*np.pi

   P = np.array([[0,3,5],[-3,0,2],[-5,-2,0]])
   #P = np.array([[0,3,5,7],[-3,0,2,4],[-5,-2,0,2],[-7,-4,-2,0]])

   #P = 4*np.array([[0,1,2,3,4],[-1,0,1,2,3],[-2,-1,0,1,2],[-3,-2,-1,0,1],[-4,-3,-2,-1,0]])

   '''
   '''
   P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),
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
   '''
   '''
   g_pq_t12,g_pq_inv12,delta_u,delta_v =t.theory_g(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
   r_pq12,g_pq12,delta_u,delta_v =t.extrapolation_function(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)

   #g_pq_t02,g_pq_inv02,delta_u,delta_v =t.theory_g(baseline=np.array([0,4]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
   #r_pq02,g_pq02,delta_u,delta_v =t.extrapolation_function(baseline=np.array([0,4]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
 
   #g_pq_t01,g_pq_inv01,delta_u,delta_v =t.theory_g(baseline=np.array([0,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
   #r_pq01,g_pq01,delta_u,delta_v =t.extrapolation_function(baseline=np.array([0,2]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=siz,s=1,resolution=r)
 
   curve = t.cut(t.img(r_pq12,delta_u,delta_v))

   #f = open(file_name, 'wb')
   #pickle.dump(theta,f)
   #pickle.dump(result,f)
   #pickle.dump(result_prim,f)
   #pickle.dump(apparent,f)
   #pickle.dump(p_beam,f)
   #pickle.dump(l,f)
   #pickle.dump(m,f)
   #f.close()

   max_v = np.max(curve)
   min_v = np.min(curve)
   half = (max_v)/2.0

   idx = (curve >= half)

   integer_map = map(int, idx)

   integer_list = np.sum(list(integer_map))*r

   print(integer_list)

   #flux = max_v*(((integer_list/3600.0)*(np.pi/180))/(2*np.sqrt(2*np.log(2))))**2*2*np.pi

   print(flux)

   #peak = np.ones(((Phi.shape[0]**2-Phi.shape[0])*0.5,),dtype=float)
   
   #for i in range(P.shape[0]):
   #    for j in range(P.shape[1]):
   #        if (j > i):
   #           g_pq_t,g_pq_inv,delta_u,delta_v = t.theory_g(baseline=np.array([i,j]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=3,s=1,resolution=200)
   #           r_pq,g_pq,delta_u,delta_v =t.extrapolation_function(baseline=np.array([i,j]),true_sky_model=np.array([[1.0/B1,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,image_s=3,s=1,resolution=200)
   #           result_theory = t.cut(t.img(g_pq_t**(-1)*r_pq,delta_u,delta_v))
   #           result_real = t.cut(t.img(g_pq**(-1)*r_pq,delta_u,delta_v))
   #           plt.plot(P[i,j],np.max(result_theory),"x")
   #           plt.plot(P[i,j],np.max(result_real),"o")

   #plt.show()     


   plt.plot(t.cut(g_pq01),"r--")
   plt.plot(t.cut(g_pq02),"b--")
   plt.plot(t.cut(g_pq12),"g--")   
   #plt.show()

   plt.plot(t.cut(g_pq_t01),"r")
   plt.plot(t.cut(g_pq_t02),"b")
   plt.plot(t.cut(g_pq_t12),"g")   
   plt.show()


   #plt.plot(t.cut(t.img(g_pq01,delta_u,delta_v)))
   #plt.plot(t.cut(t.img(g_pq01,delta_u,delta_v))) 
   #plt.show()

   plt.plot(t.cut(t.img(r_pq01,delta_u,delta_v)),"y")
   plt.plot(t.cut(t.img(g_pq_t01**(-1)*r_pq01,delta_u,delta_v)),"r") #medium
   plt.plot(t.cut(t.img(g_pq_t02**(-1)*r_pq02,delta_u,delta_v)),"b") #longest
   plt.plot(t.cut(t.img(g_pq_t12**(-1)*r_pq12,delta_u,delta_v)),"g") #shortest 
   #plt.show()

   plt.plot(t.cut(t.img(r_pq01,delta_u,delta_v)),"y")
   plt.plot(t.cut(t.img(g_pq01**(-1)*r_pq01,delta_u,delta_v)),"r--") #medium
   plt.plot(t.cut(t.img(g_pq02**(-1)*r_pq02,delta_u,delta_v)),"b--") #longest
   plt.plot(t.cut(t.img(g_pq12**(-1)*r_pq12,delta_u,delta_v)),"g--") #shortest 
   plt.show()



   #plt.plot(r_old/B,"r")
   #plt.plot(r_new,"b")
   #plt.plot(g1**(-1)*r_old,"g")
   #h = g1*(B**(-1))
   #plt.plot(g1_inv,"r")
   #plt.plot(B**(-1)*(2-h),"m")
   #plt.plot(zz,"r")
   #plt.plot(zz2,"g")
   #f1 = zz
   #f2 = zz2
   
   #g1,B,g1_inv =t.theory_g(baseline=np.array([0,1]),true_sky_model=np.array([[1.0/B1,0,0,1]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100)
   #g2,r_old,r_new,zz,zz2=t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1.0/B1,0,0,1]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100)
   #f3 = zz

   #g1,B,g1_inv =t.theory_g(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,1]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100)
   #g2,r_old,r_new,zz,zz2=t.extrapolation_function(baseline=np.array([1,2]),true_sky_model=np.array([[1.0/B1,0,0,1]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100)
   #f4 = zz   

   #plt.plot(f2,"g")
   #plt.plot(f1,"b")#longest
   #plt.plot(f3,"r")#medium
   #plt.plot(f4,"y")#shortest
   #plt.plot((f1+f3+f4)/3,"k")
   #plt.show()

   #plt.show()
   #plt.plot((g1)**(-1),"r")
   #h = g1*B**(-1)
   #plt.plot(B**(-1)*(h**(2)-3*h+3),"g")
   #plt.plot(B**(-1)*(-1*h**(3)+4*h**(2)-6*h+4),"g")
   #m = np.max(g1**(-1))
   #n = np.min(g1**(-1))
   #plt.plot(g1**(-1),"r")
   #plt.plot(g2**(-1),"b")
   #plt.ylim(n,m)
   #plt.plot(g2,"b")
   #plt.plot((g1)**(-1),"g")
   #plt.plot(g2,"b")
   #plt.plot((g1)**(-1)-3/B,"g")
   #plt.plot((g2)**(-1),"m")
   #plt.plot(len(g1)/2,B+B/3,"bx")
   #plt.plot(len(g1)/2,B,"bx")
   plt.show()

   #point source case GT-1
   #t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,type_plot="GT-1")

   #point source case GTR-R
   #t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,type_plot="GTR-R")


   #extended source case GT-1
   #t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0,0.1],[0.2,1,0,0.2]]),cal_sky_model=np.array([[1,0,0,0.1]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=True,type_plot="GT-1")

   #extended source case GTR-R
   #t.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0,0.1],[0.2,1,0,0.2]]),cal_sky_model=np.array([[1,0,0,0.1]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),image_s=3,s=1,resolution=100,kernel=False,type_plot="GTR-R")
   '''

