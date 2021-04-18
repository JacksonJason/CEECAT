import pickle 
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
   pkl_file = open('pickle_file.p', 'rb')

   data1 = pickle.load(pkl_file)/(2*np.pi) #r
   
   print "data1 = ",data1/(2*np.pi)
   print "len(data1) = ",len(data1)

   data2 = pickle.load(pkl_file)

   print "data2 = ",data2 #measured apparent
   print "data2.shape = ",data2.shape

   data3 = pickle.load(pkl_file)

   print "data3 = ",data3 #measured intrinsic
   print "data3.shape = ",data3.shape
   
   data4 = pickle.load(pkl_file)

   print "data4 = ",data4 #a-priori intrinsic
   print "data4.shape = ",data4.shape
   
   data5 = pickle.load(pkl_file)

   print "data5 = ",data5 #primary beam
   print "data5.shape = ",data5.shape
   
   data6 = pickle.load(pkl_file)

   print "data6 = ",data6
   print "data6.shape = ",data6.shape

   plt.semilogy(data1,data2[:,0])
   plt.hold("on")
   plt.semilogy(data1,data2[:,1])
   plt.semilogy(data1,data2[:,2])
   plt.semilogy(data1,data3[:,0])
   plt.semilogy(data1,data3[:,1])
   plt.semilogy(data1,data3[:,2])
   plt.semilogy(data1,data4[:,0])
   plt.semilogy(data1,data4[:,1])
   plt.semilogy(data1,1/data5[:,2],"m") #primary beam
   plt.semilogy(data1,data4[:,1]*(1./33),"r") #apparent a-priori-ghost
   plt.semilogy(data1,data4[:,1]*(1./33)*(1/data5[:,2]),"b") #apparent a-priori-ghost
   plt.show()
   pkl_file.close()
