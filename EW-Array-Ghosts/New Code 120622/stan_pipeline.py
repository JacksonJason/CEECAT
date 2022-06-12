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


class Pip():
    """
    This function initializes the theoretical ghost object
    """

    def __init__(self):
       pass

    
    def generate_uv_tracks(self, P=np.array([]),freq=1.45e9,b0 = 36,time_slots=5000, d = 90):
        lam = 3e8/freq #observational wavelenth
        H = np.linspace(-6,6,time_slots)*(np.pi/12) #Hour angle in radians
        delta = d*(np.pi/180) #Declination in radians
        
        u_m = np.zeros((len(P),len(P),len(H)),dtype=float)
        v_m = np.zeros((len(P),len(P),len(H)),dtype=float)

        for i in range(len(P)):
            for j in range(len(P)):
                if j > i:
                   u_m[i,j] = lam**(-1)*b0*P[i,j]*np.cos(H)
                   u_m[j,i] = -1*u_m[i,j]

                   v_m[i,j] = lam**(-1)*b0*P[i,j]*np.sin(H)*np.sin(delta) 
                   v_m[j,i] = -1*v_m[i,j]

        for i in range(len(P)):
            for j in range(len(P)):
                if j > i:
                   plt.plot(u_m[i,j,:],v_m[i,j,:],"r")
                   plt.plot(-u_m[i,j,:],-v_m[i,j,:],"b")
        plt.xlabel("$u$ [rad$^{-1}$]", fontsize=18)
        plt.ylabel("$v$ [rad$^{-1}$]", fontsize=18)
        plt.title("$uv$-Coverage of WSRT", fontsize=18)
        plt.show()
        return u_m,v_m

    #image_s is in degrees
    #resolution is in arcsesonds

    def gridding(self,u_m=np.array([]),v_m=np.array([]),D=np.array([]),image_s=3,s=1,resolution=8,w=1):
        
        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        print(delta_u)
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

        print(l_cor*(180.0/np.pi))
        
        uu, vv = np.meshgrid(u, v)
        u_dim = uu.shape[0]
        v_dim = uu.shape[1]
        #######################################################

        counter = np.zeros(uu.shape,dtype = int)

        grid_points = np.zeros(uu.shape,dtype = complex)
        
        for p in range(D.shape[0]):
            for q in range(D.shape[1]):
                #if p!=q:
                   for t in range(D.shape[2]):
                       idx_u = np.searchsorted(u,u_m[p,q,t])
                       idx_v = np.searchsorted(v,v_m[p,q,t])
                       if (idx_u != 0) and (idx_u != len(u)):
                          if (idx_v != 0) and (idx_v != len(v)):
                             grid_points[idx_u-w:idx_u+w,idx_v-w:idx_v+w] += D[p,q,t]
                             counter[idx_u-w:idx_u+w:,idx_v-w:idx_v+w] += 1

        grid_points[counter>0] = grid_points[counter>0]/counter[counter>0]
        fig, ax = plt.subplots() 
        im=ax.imshow(np.absolute(grid_points),extent=[u[0],u[-1],v[0],v[-1]])
        fig.colorbar(im, ax=ax) 
        plt.show() 

        return grid_points,l_cor,u  

    def img(self,image,psf,l_cor):
        #COMPUTING PSF FIRST
        #*******************
        zz_psf = psf
        zz_psf = np.roll(zz_psf, -int(zz_psf.shape[0]/2), axis=0)
        zz_psf = np.roll(zz_psf, -int(zz_psf.shape[0]/2), axis=1)

        zz_f_psf = np.fft.fft2(zz_psf)
        mx = np.max(np.absolute(zz_f_psf))
        zz_f_psf = zz_f_psf/mx
        zz_f_psf = np.roll(zz_f_psf, -int(zz_psf.shape[0]/2)-1, axis=0)
        zz_f_psf = np.roll(zz_f_psf, -int(zz_psf.shape[0]/2)-1, axis=1)
        #*******************

        #COMPUTING IMAGE
        #*******************
        zz = image
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=0)
        zz = np.roll(zz, -int(zz.shape[0]/2), axis=1)

        zz_f = np.fft.fft2(zz)
        zz_f = zz_f/mx
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2)-1, axis=0)
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2)-1, axis=1)
        #*******************

        fig, ax = plt.subplots() 
        im=ax.imshow(np.absolute(zz_f_psf),extent=[l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi),l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi)])
        fig.colorbar(im, ax=ax) 
        plt.show() 

        fig, ax = plt.subplots() 
        im=ax.imshow(np.absolute(zz_f),extent=[l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi),l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi)])
        fig.colorbar(im, ax=ax) 
        plt.show() 

        #fig, ax = plt.subplots()
        #print(image_s)
        #print(s)
        #im = ax.imshow(zz_f.real)
        #fig.colorbar(im, ax=ax) 
        #plt.show()
        return zz_f_psf,zz_f


    def create_vis_matrix(self,u_m=np.array([]),v_m=np.array([]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]])):   
        R = np.zeros(u_m.shape,dtype=complex)
        M = np.zeros(u_m.shape,dtype=complex) 
        for k in range(len(true_sky_model)):
            s = true_sky_model[k]
            if len(s) <= 3:
               R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
            else:
               sigma = s[3]*(np.pi/180)
               g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
               R += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal
        for k in range(len(cal_sky_model)):
            s = cal_sky_model[k]
            if len(s) <= 3:
               M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))
            else:
               sigma = s[3]*(np.pi/180)
               g_kernal = 2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(u_m**2+v_m**2))
               M += s[0]*np.exp(-2*np.pi*1j*(u_m*(s[1]*np.pi/180.0)+v_m*(s[2]*np.pi/180.0)))*g_kernal

        return R,M

    def cut(self,inp):
            return inp.real[inp.shape[0]/2,:]  

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

    def calibrate(self,R=np.array([]),M=np.array([])):
        G = np.zeros(R.shape,dtype=complex)
        g = np.zeros((R.shape[0],R.shape[2]),dtype=complex)
        temp = np.ones((R.shape[0],R.shape[1]), dtype=complex)

        for t in range(R.shape[2]):
            print(t)
            g[:,t],G[:,:,t] = self.create_G_stef(R[:,:,t],M[:,:,t],200,1e-20,temp,False)
        

        for k in range(R.shape[0]):
            plt.plot(np.absolute(g[k,:]))
        plt.show()

        fig, ax = plt.subplots() 
        im=ax.imshow(np.absolute(G[:,:,0]**(-1)))
        fig.colorbar(im, ax=ax) 
        plt.show()

        return g,G


if __name__ == "__main__":
   pO = Pip()

   #GEOMETRY MATRIX OF WSRT
   P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])

   u_m,v_m = pO.generate_uv_tracks(P=P)

   s_size = 0.1 #source size is in degrees
   B = 2*(s_size*(np.pi/180))**2*np.pi


   clean_beam = 0.0019017550075500233
   B2 = 2*(clean_beam*(np.pi/180))**2*np.pi
   
   R,M = pO.create_vis_matrix(u_m=u_m,v_m=v_m,true_sky_model=np.array([[1.0/B,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]))
   #print(M)
   g_psf,dl,u = pO.gridding(u_m=u_m,v_m=v_m,D=M,image_s=4,s=1,resolution=8)
   g_img,dl,u = pO.gridding(u_m=u_m,v_m=v_m,D=R,image_s=4,s=1,resolution=8)
   cut_img = pO.cut(g_img)
   print("Hallo")
   theory = np.exp(-2*np.pi**2*(s_size*(np.pi/180))**2*u**2)
   plt.plot(u,np.absolute(cut_img),"b")
   plt.plot(u,theory,"r")
   plt.show()
   
   psf,img = pO.img(g_img,g_psf,dl)   
   print(R[0,0,:])
   print(u_m[0,0,:])
   cut_img = pO.cut(img)
   theory = np.exp(-dl**2/(2*(s_size*(np.pi/180))**2))
   plt.plot(dl*(180.0/np.pi),(cut_img*B)/B2,"b")
   plt.plot(dl*(180.0/np.pi),theory,"r")
   plt.show()

   pO.calibrate(R,M)
   
   
   
   

