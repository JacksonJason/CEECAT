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

        plt.plot(Curves_theory[1,2,:],"b")
        print(P[9,10])
        plt.show()

        for k in range(P.shape[0]):
            for j in range(P.shape[1]):
                if j > k:
                   print(k)
                   print(j)
                   print(np.max(Curves_real[k,j,:]))
                   print(P[k,j])
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
        

if __name__ == "__main__":
   t = T_ghost()
   #t.process_pickle_file()
   
   g_pq12,r_pq12,g_kernal,g_pq_inv12,g_pq_t12,sigma_b,delta_u,delta_l,s_size,siz,r=t.retrieve_pickle()

   answer = t.cut(t.img(g_pq12**(-1)*r_pq12*g_kernal,delta_u,delta_u))
   x = np.linspace(-1*siz,siz,len(g_pq12))
   plt.plot(x,answer,"b")
   answer2 = t.cut(t.img(g_pq12**(-1)*r_pq12,delta_u,delta_u))
   answer3 = t.cut(t.img(g_pq_t12**(-1)*r_pq12,delta_u,delta_u))
   answer4 = t.cut(t.img(r_pq12,delta_u,delta_u))
   plt.plot(x,answer2,"r")
   plt.plot(x,answer3,"b")
   plt.plot(x,answer4,"g")
   plt.show()

   y = np.linspace(-1*delta_u*len(g_pq12)/2,1*delta_u*len(g_pq12)/2,len(g_pq12))
   vis1 = t.cut(g_pq12)
   vis2 = t.cut(g_pq_t12)
   plt.plot(y,vis1,"bx")
   plt.plot(y,vis2,"rx")
   plt.show()
   print(vis1)

   
   
   

