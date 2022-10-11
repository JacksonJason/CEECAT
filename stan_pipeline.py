import numpy as np
import pylab as plt
import pickle
import math
import matplotlib.colors as colors

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


    def include_100_baseline(self,p=0,q=0):
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


    def magic_baselin(self,p=0,q=1):

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


    def get_main_graphs(self,P=np.array([])):
        counter = 0
        counter2 = 0
        counter3 = 0
        

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

        new_x = []
        new_r = []

        B_old = 0 

        for k in range(len(P)):
            for j in range(len(P)):
                counter2 += 1
                if j != k:  
                   if j > k:
                      #print(counter)
                      name = (
                        "data/G/g_"
                        + str(k)
                        + "_"
                        + str(j)
                        + "_"
                        + str(P[k, j])
                        + ".p"
                    )
                      counter += 1
                      
                      if self.include_100_baseline(k,j):
                        counter3 += 1   
                        pkl_file = open(name, "rb")
                        g_pq_t = pickle.load(pkl_file)
                        g_pq = pickle.load(pkl_file)
                        g_pq_inv = pickle.load(pkl_file)
                        r_pq = pickle.load(pkl_file)
                        g_kernal = pickle.load(pkl_file)
                        sigma_kernal = pickle.load(pkl_file)
                        B = pickle.load(pkl_file)
                        B_old = B
                        '''
                        plt.plot(u,np.absolute(g_pq)/B,"r",label=r"$S_3$")
                        plt.plot(u,np.absolute(g_pq_t)/B,"b",label=r"$S_2$")
                        #plt.plot(u,np.absolute(g_pq_inv*r_pq),"g")
                        plt.title("Baseline "+str(k)+"-"+str(j))
                        plt.xlabel(r"$u$ [rad$^{-1}$]")
                        plt.legend()
                        plt.show()
                         
                        plt.plot(u,np.absolute(g_pq**(-1)*B),"r",label=r"$S_3$")
                        plt.plot(u,np.absolute(g_pq_t**(-1)*B),"b",label=r"$S_2$")
                        plt.plot(u,np.absolute(g_pq_inv)*B,"g",label=r"$S_1$")
                        plt.title("Baseline "+str(k)+"-"+str(j))
                        plt.ylim([0.9,2.1])
                        plt.xlabel(r"$u$ [rad$^{-1}$]")
                        plt.legend()
                        plt.show()

                        plt.plot(u,np.absolute(g_pq**(-1)*r_pq),"r",label=r"$S_3$")
                        plt.plot(u,np.absolute(g_pq_t**(-1)*r_pq),"b",label=r"$S_2$")
                        plt.plot(u,np.absolute(g_pq_inv*r_pq),"g",label=r"$S_1$")
                        if (k == 2) and (j == 3):
                           plt.ylim([0,3.6])
                        plt.xlabel(r"$u$ [rad$^{-1}$]")
                        plt.title("Baseline "+str(k)+"-"+str(j))
                        plt.legend()
                        plt.show()
                        '''
                   
                        #10_baseline_9_0_10_37.0.p
                        name = (
                        "data/10_baseline/14_10_baseline_"
                        + str(k)
                        + "_"
                        + str(j)
                        + "_"
                        + str(P[k, j])
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
                        P = pickle.load(pkl_file)
                        K1 = pickle.load(pkl_file)
                        K2 = pickle.load(pkl_file)
                        pkl_file.close()
                        #x = np.linspace(-1*siz,siz,len(g_pq12))
                        if len(new_x) == 0:
                           new_x = np.absolute(g_pq12**(-1)*r_pq12)#*B
                        else:
                           new_x += np.absolute(g_pq12**(-1)*r_pq12)#*B
                        if len(new_r) == 0:
                           new_r = r_pq12       
        #plt.imshow(self.img2(np.absolute(g_pq12**(-1)*r_pq12)*B/counter2,delta_u,delta_u),extent=[-siz,siz,-siz,siz],cbar="jet")
        N = g_pq12.shape[0]
        u = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
        v = np.linspace(-(N - 1) / 2 * delta_u, (N - 1) / 2 * delta_u, N)
        uu, vv = np.meshgrid(u, v)
        u_dim = uu.shape[0]
        v_dim = uu.shape[1]
        #######################################################
        f = np.zeros((u_dim,v_dim),dtype=complex)
        sigma = 0.0019017550075500233*(np.pi/180)
        #print("gaussian")
        #print(sigma)

        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                #f[i,j] = A*2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))
                f[i,j] = 1*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))
        B_new = 2*np.pi*sigma**2

        fig, ax = plt.subplots()
        psf = self.img2(f,1.0,1.0)
        #psf = self.img2(f,1.0,1.0)
        ma = np.max(psf)
        ma = 1/ma
        #print(ma)

        phi = np.linspace(0,2*np.pi,100)
        x = 0.04*np.cos(phi)
        y = 0.04*np.sin(phi)      

        x2 = 2*0.0019017550075500233*np.cos(phi)
        y2 = 2*0.0019017550075500233*np.sin(phi)      
 
        print("PLOTTING EXTRAPOLATION RESULTS")
        print("PLOTTING CLEAN BEAM")
        im=ax.imshow(self.img2(f,ma,1.0),extent=[-siz,siz,-siz,siz],cmap='jet')
        cbar=fig.colorbar(im, ax=ax) 
        cbar.set_label("Jy/beam",labelpad=10)
        ax.set_xlabel(r"$l$ [degrees]")
        ax.set_ylabel(r"$m$ [degrees]")
        #ax.plot(x2,y2,"k",lw=2.0)
        plt.show()   
        
        print("PLOTTING CORRECTED VIS")  
        fig, ax = plt.subplots()
        im=ax.imshow(self.img2(f*(new_x/counter3),ma,1),extent=[-siz,siz,-siz,siz],cmap='jet')
        cbar=fig.colorbar(im, ax=ax) 
        cbar.set_label("Jy/beam",labelpad=10)
        ax.set_xlabel(r"$l$ [degrees]")
        ax.set_ylabel(r"$m$ [degrees]")
        ax.plot(x,y,"r",lw=2.0)
        ax.plot(x2,y2,"k",lw=2.0)
        plt.show() 

        print("PLOTTING GAUSSIAN")
        fig, ax = plt.subplots()
        im=ax.imshow(self.img2(f*(new_r/B_old),ma,1),extent=[-siz,siz,-siz,siz],cmap='jet')
        cbar=fig.colorbar(im, ax=ax) 
        cbar.set_label("Jy/beam",labelpad=10)
        ax.set_xlabel(r"$l$ [degrees]")
        ax.set_ylabel(r"$m$ [degrees]")
        #ax.plot(x,y,"r",lw=2.0) 
        plt.show()        
        


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
                f[i,j] = A*2*np.pi*sigma**2*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))
                #f[i,j] = A*np.exp(-2*np.pi**2*sigma**2*(uu[i,j]**2+vv[i,j]**2))

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
    def point(self,A=1,sigma_b=1,image_s=3,s=1,resolution=100):
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
        sigma = sigma_b*(np.pi/180)
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
    b0 in m
    f in HZ
    """
    def extrapolation_function(self,baseline=np.array([0,1]),true_sky_model=np.array([[1,0,0],[0.2,1,0]]),cal_sky_model=np.array([[1,0,0]]),Phi=np.array([[0,3,5],[-3,0,2],[-5,-2,0]]),u=0,v=0):
                temp = np.ones(Phi.shape, dtype=complex) 
                ut = u
                vt = v
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
                r_pq = R[baseline[0],baseline[1]]
                m_pq = M[baseline[0],baseline[1]]
                g_pq = G[baseline[0],baseline[1]]
                return g_pq


    
    def generate_uv_tracks(self, P=np.array([]),freq=1.45e9,b0 = 36,time_slots=500, d = 90):
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

    def include_baseline(self,k=0,j=1):
      if (k == j):
         return False
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
 
      if (k==1) and (j==0):
         return False                   
      if (k==2) and (j==0):
         return False
      if (k==3) and (j==0):
         return False
      if (k==2) and (j==1):
         return False
      if (k==3) and (j==1):
         return False
      if (k==4) and (j==1):
         return False
      if (k==3) and (j==2):
         return False
      if (k==4) and (j==2):
         return False
      if (k==5) and (j==2):
         return False
      if (k==4) and (j==3):
         return False
      if (k==5) and (j==3):
         return False
      if (k==5) and (j==4):
         return False
      if (k==6) and (j==4):
         return False
      if (k==6) and (j==5):
         return False
      if (k==7) and (j==5):
         return False
      if (k==7) and (j==6):
         return False
      if (k==8) and (j==6):
         return False
      if (k==8) and (j==7):
         return False
      if (k==13) and (j==12):
         return False
      return True



    #image_s is in degrees
    #resolution is in arcsesonds

    def gridding(self,u_m=np.array([]),v_m=np.array([]),D=np.array([]),image_s=3,s=1,resolution=8,w=1,grid_all=True):
        
        # FFT SCALING
        ######################################################
        delta_u = 1 / (2 * s * image_s * (np.pi / 180))
        #print(delta_u)
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

        #print(l_cor*(180.0/np.pi))
        
        uu, vv = np.meshgrid(u, v)
        u_dim = uu.shape[0]
        v_dim = uu.shape[1]
        #######################################################

        counter = np.zeros(uu.shape,dtype = int)

        grid_points = np.zeros(uu.shape,dtype = complex)
        
        for p in range(D.shape[0]):
            for q in range(D.shape[1]):
                if p!=q:
                   if grid_all:
                      for t in range(D.shape[2]):
                           idx_u = np.searchsorted(u,u_m[p,q,t])
                           idx_v = np.searchsorted(v,v_m[p,q,t])
                           if (idx_u != 0) and (idx_u != len(u)):
                              if (idx_v != 0) and (idx_v != len(v)):
                                 grid_points[idx_u-w:idx_u+w,idx_v-w:idx_v+w] += D[p,q,t]
                                 counter[idx_u-w:idx_u+w:,idx_v-w:idx_v+w] += 1
 
                   else:
                      if self.include_100_baseline(p,q) or self.include_100_baseline(q,p):
                         for t in range(D.shape[2]):
                           idx_u = np.searchsorted(u,u_m[p,q,t])
                           idx_v = np.searchsorted(v,v_m[p,q,t])
                           if (idx_u != 0) and (idx_u != len(u)):
                              if (idx_v != 0) and (idx_v != len(v)):
                                 grid_points[idx_u-w:idx_u+w,idx_v-w:idx_v+w] += D[p,q,t]
                                 counter[idx_u-w:idx_u+w:,idx_v-w:idx_v+w] += 1
 
                       
        grid_points[counter>0] = grid_points[counter>0]/counter[counter>0]
        #fig, ax = plt.subplots() 
        #print("gridded_plot")
        #im=ax.imshow(np.absolute(grid_points),extent=[u[0],u[-1],v[0],v[-1]])
        #fig.colorbar(im, ax=ax) 
        #plt.show() 

        return grid_points,l_cor,u,delta_u  

    def img2(self,inp,delta_u,delta_v):
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
    
    def img(self,image,psf,l_cor,delta_u,sigma,add_circle=False):
        sigma = sigma*(np.pi/180)
        #COMPUTING PSF FIRST
        #*******************
        zz_psf = psf
        zz_psf = np.roll(zz_psf, -int(zz_psf.shape[0]/2), axis=0)
        zz_psf = np.roll(zz_psf, -int(zz_psf.shape[0]/2), axis=1)

        #zz_psf *= (2*np.pi*sigma**2)

        zz_f_psf = np.fft.fft2(zz_psf) #* (delta_u*delta_u)
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

        #zz *= (2*np.pi*sigma**2)

        zz_f = np.fft.fft2(zz) #* (delta_u*delta_u)
        zz_f = zz_f/mx
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2)-1, axis=0)
        zz_f = np.roll(zz_f, -int(zz.shape[0]/2)-1, axis=1)
        #*******************

        
        # colourmap from green to red, biased towards the blue end.
        # Try out different gammas > 1.0
        cmap = colors.LinearSegmentedColormap.from_list('nameofcolormap',['b','y','r'],gamma=0.35)
 
        fig, ax = plt.subplots() 
        im=ax.imshow(np.absolute(zz_f_psf),extent=[l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi),l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi)],cmap=cmap,interpolation='bicubic')
        cbar = fig.colorbar(im, ax=ax) 
        cbar.set_label("Jy/beam",labelpad=10)
        ax.set_xlabel(r"$l$ [degrees]")
        ax.set_ylabel(r"$m$ [degrees]")
        plt.show() 

        phi = np.linspace(0,2*np.pi,100)
        x = 0.04*np.cos(phi)
        y = 0.04*np.sin(phi)      

        x2 = 2*0.0019017550075500233*np.cos(phi)
        y2 = 2*0.0019017550075500233*np.sin(phi)      


        fig, ax = plt.subplots() 
        im=ax.imshow(np.absolute(zz_f),extent=[l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi),l_cor[0]*(180.0/np.pi),l_cor[-1]*(180.0/np.pi)],cmap='jet')
        cbar=fig.colorbar(im, ax=ax) 
        cbar.set_label("Jy/beam",labelpad=10)
        ax.set_xlabel(r"$l$ [degrees]")
        ax.set_ylabel(r"$m$ [degrees]")
        
        if add_circle:
           ax.plot(x,y,"r",lw=2.0)
           ax.plot(x2,y2,"k",lw=2.0)
        
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
            #print(t)
            g[:,t],G[:,:,t] = self.create_G_stef(R[:,:,t],M[:,:,t],200,1e-20,temp,False)
        

        #for k in range(R.shape[0]):
        #    plt.plot(np.absolute(g[k,:]))
        #plt.show()

        #fig, ax = plt.subplots() 
        #im=ax.imshow(np.absolute(G[:,:,0]))
        #fig.colorbar(im, ax=ax) 
        #plt.show()

        #plt.semilogy(g[:,0])
        #plt.show()

        return g,G

    def retrieve_pickle(self, name="baseline12_14.p"):
        pkl_file = open(name, 'rb')
        g_pq12=pickle.load(pkl_file)
        r_pq12=pickle.load(pkl_file)
        g_kernal=pickle.load(pkl_file)
        g_pq_inv12=pickle.load(pkl_file)
        g_pq_t12=pickle.load(pkl_file)
        sigma_b=pickle.load(pkl_file)
        delta_u=pickle.load(pkl_file)
        delta_l=pickle.load(pkl_file)
        s_size=pickle.load(pkl_file)
        siz=pickle.load(pkl_file)
        r=pickle.load(pkl_file)
        pkl_file.close()
        return g_pq12,r_pq12,g_kernal,g_pq_inv12,g_pq_t12,sigma_b,delta_u,delta_l,s_size,siz,r 


if __name__ == "__main__":
   pO = Pip()

   #GEOMETRY MATRIX OF WSRT
   
   resolution_bool = True # NB --- user needs to be able to toggle this --- if true produce at calculated resolution otherwise use pre-computed values corresponding to extrapolated case
   add_circle = False # Do not change

   P = 4 * np.array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.25, 9.75, 18.25, 18.75),(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.25, 8.75, 17.25, 17.75), (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7.25, 7.75, 16.25, 16.75), (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.25, 6.75, 15.25, 15.75), (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5.25, 5.75, 14.25, 14.75),(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.25, 4.75, 13.25, 13.75), (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.75, 12.25, 12.75), (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.25, 2.75, 11.25, 11.75),(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 1.25, 1.75, 10.25, 10.75),(-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.25, 0.75, 9.25, 9.75),(-9.25, -8.25, -7.25, -6.25, -5.25, -4.25, -3.25, -2.25, -1.25, -0.25, 0, 0.5, 9, 9.5),(-9.75, -8.75, -7.75, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, -0.5, 0, 8.5, 9),(-18.25, -17.25, -16.25, -15.25, -14.25, -13.25, -12.25, -11.25, -10.25, -9.25, -9, -8.5, 0, 0.5), (-18.75, -17.75, -16.75, -15.75, -14.75, -13.75, -12.75, -11.75, -10.75, -9.75,-9.5, -9, -0.5, 0)])

   pO.get_main_graphs(P=P)

   u_m,v_m = pO.generate_uv_tracks(P=P)

   #automatically generate parameters for imaging

   s_img = 1
   de_u = 1/(2*s_img*(np.pi/180))

   r_vis = 28000

   N = int(r_vis/de_u)

   de_l = (N*de_u)**(-1)

   r = ((1.0 / 3600.0) * (np.pi / 180.0))**(-1)*de_l

   #print(r)
   
   #s_size = 0.07607020030200093 #source size is in degrees
   s_size = 0.02 #source size is in degrees
   B = 2*(s_size*(np.pi/180))**2*np.pi

   f = 1.45e9
   lam = 3e8/1.45e9

   fwhm = 1.02*(lam/2700.0)

   #print(fwhm)

   C = 2*math.sqrt(2*math.log(2))

   beam = (fwhm/C)*(180/np.pi)

   #print(beam)

   clean_beam = 0.0019017550075500233
   B2 = 2*(clean_beam*(np.pi/180))**2*np.pi

   #print(s_size**2/(s_size**2+clean_beam**2))

   
   if resolution_bool: #use pre-computed values similar to other experiments
      s_img = 0.08
      r = 2.4
      add_circle = True

   print("PLOTTING SYNTHESIS")
   print("PLOTTING UV-COVERAGE")
   print("CREATING VISIBILITIES")
   R,M = pO.create_vis_matrix(u_m=u_m,v_m=v_m,true_sky_model=np.array([[1.0,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]))
   #print(M)
   print("GRIDDING PSF")
   g_psf,dl,u,du = pO.gridding(u_m=u_m,v_m=v_m,D=M,image_s=s_img,s=1,w=1,resolution=r)
   #print("image")
   print("CREATING AN IMAGE OF PSF")
   psf,img = pO.img(g_psf,g_psf,dl,du,clean_beam)
   print("GRIDDING PSF NOT ALL BASELINES")
   g_psf_2,dl,u,du = pO.gridding(u_m=u_m,v_m=v_m,D=M,image_s=s_img,s=1,resolution=r,grid_all=False)
   print("GRIDDING TRUE SKY_MODEL (1 Jy normalized)")
   g_img,dl,u,du = pO.gridding(u_m=u_m,v_m=v_m,D=R/B,image_s=s_img,s=1,resolution=r)
   print("CALIBRATE SYNTHESIS IMAGE")
   g,G = pO.calibrate(R,M)
   print("UNIT TEST: EXTRAPOLATION VERSUS STEFCAL")
   print("Single gain value: Stefcal")
   print(np.absolute(G[0,1,0]))
   g_pq = pO.extrapolation_function(baseline=np.array([0,1]),true_sky_model=np.array([[1.0,0,0,s_size]]),cal_sky_model=np.array([[1,0,0]]),Phi=P,u=u_m[0,1,0],v=v_m[0,1,0])
   print("single gain value: Extrapolation")
   print(np.absolute(g_pq))

   #plt.imshow(G/B)
   #plt.show()
   print("GRIDDING GAINS ONLY (normalized)")
   g_img2,dl,u,du = pO.gridding(u_m=u_m,v_m=v_m,D=G/B,image_s=s_img,s=1,resolution=r)
   print("GRIDDING CORRECTED VIS")
   g_img3,dl,u,du = pO.gridding(u_m=u_m,v_m=v_m,D=R*G**(-1),image_s=s_img,s=1,resolution=r,w=1,grid_all=False)
   
   print("PLOTTING GAUSSIAN SYNTHESIS IMAGE")
   psf,img = pO.img(g_img,g_psf,dl,du,clean_beam)
   print("PLOTTING GAINS SYNTHESIS IMAGE --- SHOWING GHOST")
   psf,img_c = pO.img(g_img2,g_psf,dl,du,clean_beam)
   print("PLOTTING CORRECTED VIS SYNTHESIS IMAGE")
   psf,img_c2 = pO.img(g_img3,g_psf_2,dl,du,clean_beam,add_circle=add_circle) 
   
   print("COMPARING CLEAN BEAM AND PSF")
   cut_psf = pO.cut(psf)
   plt.plot(dl*(180.0/np.pi),cut_psf/np.max(cut_psf),"b")
   theory = np.exp(-dl**2/(2*(clean_beam*(np.pi/180))**2))
   plt.plot(dl*(180.0/np.pi),theory,"r")
   plt.show()
   
   

