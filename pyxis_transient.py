from pyrap.tables import table
from Pyxis.ModSupport import *
import mqt
import cal
import imager
import stefcal
import lsm
import ms
import std
import numpy as np
import pylab as plt
import pickle
import pyfits
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats            
from scipy import optimize
import os
from lofar import bdsm


#Setting Pyxis global variables
v.OUTDIR = '.'
v.BASELINE = ""
v.CALTECH = ""
v.POSNEG = ""
#v.MS = 'KAT7_1445_1x16_12h.ms'
v.DESTDIR_Template = "${OUTDIR>/}plots-${MS:BASE}"
v.OUTFILE_Template = "${DESTDIR>/}${MS:BASE}${_<CALTECH}${_<BASELINE}"
imager.DIRTY_IMAGE_Template = "${OUTFILE}.dirty.fits"
imager.RESTORED_IMAGE_Template = "${OUTFILE}.restored.fits"
imager.RESIDUAL_IMAGE_Template = "${OUTFILE}.residual.fits"
imager.MASK_IMAGE_Template = "${OUTFILE}.mask.fits"
imager.MODEL_IMAGE_Template = "${OUTFILE}.model.fits"
lsm.PYBDSM_OUTPUT_Template = "${OUTFILE}${_<POSNEG}.lsm.html"
v.PICKLENAME = ""
v.PICKLEFILE_Template = "${DESTDIR>/}Pickle/${MS:BASE}${_<PICKLENAME}.p"
v.GHOSTMAP_Template = "${DESTDIR>/}${MS:BASE}_GhostMap${_<BASELINE}.txt"
#v.LOG_Template = "${OUTDIR>/}log-${MS:BASE}"

class Mset(object):
  def __init__(self, ms_name):

    self.wave = 0
    self.na = 0
    self.nb = 0
    self.ns = 0
    self.pos =np.array([])
    self.uvw =np.array([])
    self.corr_data = np.array([])
    self.data = np.array([])
    self.A1 = np.array([])
    self.A2 = np.array([])
    self.ra0 = 0
    self.dec0 = 0
    v.MS = ms_name

  def extract(self):
      ms_v = II("${MS>/}")
      tl=table(ms_v,readonly=False)
      self.A1=tl.getcol("ANTENNA1")
      self.A2=tl.getcol("ANTENNA2")
      self.uvw=tl.getcol("UVW")
      self.corr_data=tl.getcol("CORRECTED_DATA")
      self.data = tl.getcol("DATA")
      self.model_data = tl.getcol("MODEL_DATA")
      ta=table(ms_v+"ANTENNA")
      tab=table(ms_v+"SPECTRAL_WINDOW",readonly=False)
      self.wave=3e8/tab.getcol("REF_FREQUENCY")[0]
      #print "self.wave = ",self.wave 
      self.pos = ta.getcol("POSITION")
      self.na=len(ta.getcol("NAME"))
      self.names = ta.getcol("NAME")
      temp = self.uvw[(self.A1==0)&(self.A2==0),0]
      if len(temp) == 0:
         self.auto = False
         self.nb=self.na*(self.na-1)/2
      else: 
         self.nb=self.na*(self.na-1)/2+self.na
         self.auto=True
      temp = self.uvw[(self.A1==0)&(self.A2==1),0] 
      self.ns=len(temp)
      tf = table(ms_v+"/FIELD")
      phase_centre = (tf.getcol("PHASE_DIR"))[0,0,:]
      self.ra0, self.dec0 = phase_centre[0], phase_centre[1] #Field centre in radians
      print "self.dec0 = ",self.dec0*(180/np.pi)
      tl.close()
      ta.close()
      tab.close()
      tf.close()

  def write(self,column):
      ms_v = II("${MS>/}")
      tl=table(ms_v, readonly=False)
      if column == "CORRECTED_DATA":
         tl.putcol(column ,self.corr_data)
      elif column == "DATA":
         tl.putcol(column ,self.data)
      else:
         tl.putcol(column ,self.model_data)
      tl.close()

class Sky_model(object):

  def __init__(self, ms, point_sources, point_sources_cal,on_time,noise):

        self.ms = ms
        self.point_sources = point_sources
        self.point_sources_cal = point_sources_cal
        self.point_sources[:,2] = self.point_sources[:,2]*(-1)
        self.point_sources_cal[:,2] = self.point_sources_cal[:,2]*(-1)
        self.on_time = on_time
        self.noise = noise
        #self.std_point = 0.0001
        
  def visibility(self,column):
        u=self.ms.uvw[:,0]
        v=self.ms.uvw[:,1]
        if column=="CORRECTED_DATA":
           vis=np.zeros((self.ms.corr_data.shape[0],),dtype=self.ms.data.dtype)
        elif  column=="DATA":
           vis=np.zeros((self.ms.data.shape[0],),dtype=self.ms.data.dtype)
        else:
           vis=np.zeros((self.ms.model_data.shape[0],),dtype=self.ms.data.dtype) 

        if column=="MODEL_DATA":
           for k in xrange(len(self.point_sources_cal)):
                  vis_temp = self.point_sources_cal[k,0]*np.exp((-2*np.pi*1j*(u*self.point_sources_cal[k,1]+v*self.point_sources_cal[k,2]))/self.ms.wave)
                  vis = vis + vis_temp
        else:
           for k in xrange(len(self.point_sources)):
                  if self.on_time[k] == 100:
                     print "k = ",k
                     vis_temp = self.point_sources[k,0]*np.exp((-2*np.pi*1j*(u*self.point_sources[k,1]+v*self.point_sources[k,2]))/self.ms.wave)
                     vis = vis + vis_temp

           if self.noise <> None:
              vis = vis + self.noise*np.random.randn(len(vis))
        
        if column=="CORRECTED_DATA":
           self.ms.corr_data[:,0,3]=vis
           self.ms.corr_data[:,0,0]=vis
        elif column == "DATA":
           self.ms.data[:,0,3]=vis
           self.ms.data[:,0,0]=vis
        else:
           self.ms.model_data[:,0,3]=vis
           self.ms.model_data[:,0,0]=vis
           
        self.ms.write(column)

  # converting from l and m coordinate system to ra and dec
  def lm2radec(self,l,m):#l and m in radians
      rad2deg = lambda val: val * 180./np.pi
      #ra0,dec0 = extract(MS) # phase centre in radians
      rho = np.sqrt(l**2+m**2)
      if rho==0:
         ra = self.ms.ra0
         dec = self.ms.dec0
      else:
         cc = np.arcsin(rho)
         ra = self.ms.ra0 - np.arctan2(l*np.sin(cc), rho*np.cos(self.ms.dec0)*np.cos(cc)-m*np.sin(self.ms.dec0)*np.sin(cc))
         dec = np.arcsin(np.cos(cc)*np.sin(self.ms.dec0) + m*np.sin(cc)*np.cos(self.ms.dec0)/rho)
      return rad2deg(ra), rad2deg(dec)

  # converting ra and dec to l and m coordiantes
  def radec2lm(self,ra_d,dec_d):# ra and dec in degrees
      rad2deg = lambda val: val * 180./np.pi
      deg2rad = lambda val: val * np.pi/180
      #ra0,dec0 = extract(MS) # phase center in radians
      ra_r, dec_r = deg2rad(ra_d), deg2rad(dec_d) # coordinates of the sources in radians
      l = np.cos(dec_r)* np.sin(ra_r - self.ms.ra0)
      m = np.sin(dec_r)*np.cos(self.ms.dec0) - np.cos(dec_r)*np.sin(self.ms.dec0)*np.cos(ra_r-self.ms.ra0)
      return rad2deg(l),rad2deg(m)

  # creating meqtrees skymodel    
  def meqskymodel(self,point_sources,antenna=""): 
      str_out = "#format: name ra_d dec_d i\n"
      for i in range(len(point_sources)):
          print "i = ",i
          amp, l ,m = point_sources[i,0], point_sources[i,1], point_sources[i,2] 
          #m = np.absolute(m) if m < 0 else -1*m # changing the signs since meqtrees has its own coordinate system
          ra_d, dec_d = self.lm2radec(l,m)
          print ra_d, dec_d
          l_t,m_t = self.radec2lm(ra_d,dec_d)
          print l_t, m_t
          name = "A"+ str(i)
          str_out += "%s %.10g %.10g %.4g\n"%(name, ra_d, dec_d,amp)
      #self.ms.p_wrapper.setting_BASELINE(antenna) 
      file_name = self.ms.p_wrapper.pyxis_to_string(v.GHOSTMAP)
      simmodel = open(file_name,"w")
      simmodel.write(str_out)
      simmodel.close()
      x.sh("tigger-convert $GHOSTMAP -t ASCII --format \"name ra_d dec_d i\" -f ${DESTDIR>/}${GHOSTMAP:BASE}.lsm.html")

class Calibration(object):
   def __init__(self, ms, antenna,point_sources,point_sources_cal,on_time):

        self.antenna = antenna
        self.ms = ms
        self.a_list = self.get_antenna(self.antenna,self.ms.names)
        self.point_sources = point_sources
        self.point_sources_cal = point_sources_cal
        self.on_time = on_time
        #self.a_names = self.ms.names[self.a_list]
        #print "a_list = ",self.a_list
        #v.CALTECH = cal_tech
        #self.cal_tech = cal_tech
        self.R = np.array([])
        self.M = np.array([])
        self.p = np.array([])
        self.q = np.array([])
        self.G = np.array([])
        self.g = np.array([])
        self.u_m = np.array([])
        self.v_m = np.array([])

   #Creating advanced pyxis imager settings
   def image_advanced_settings(self,antenna=True,img_nchan=1,img_chanstart=0,img_chanstep=1):
       #Creates baseline name and select string for image
       if antenna:
          antenna_str = "("
          #antenna_str = " && ("
          for k in range(len(self.a_list)):
              for i in range(k+1,len(self.a_list)):
                  antenna_str = antenna_str + "(ANTENNA1 = "+str(self.a_list[k])+" && ANTENNA2 = "+str(self.a_list[i])+") || "
                  #print "antenna_str = ",antenna_str
          antenna_str = antenna_str[0:len(antenna_str)-4]
          antenna_str = antenna_str+")"
       else:
         antenna_str = ""

       #strp=""" " """
       #strsel=((strp+"sumsqr(UVW[:2])<16.e6"+antenna_str+strp)).replace(" ","")
       strsel = antenna_str

       options = {}
       options["select"] = strsel
       options["img_nchan"] = img_nchan
       options["img_chanstart"] = img_chanstart
       options["img_chanstep"] = img_chanstep
       options["niter"] = 1000
       options["threshold"]= "0.5Jy"
       options["gain"] = 0.1
       imager.CLEAN_ALGORITHM = "csclean"        
       options["npix"] = 2048
       options["cellsize"]="40arcsec"
       options["stokes"]="I"
       #options["wprojplanes"]=128
       #options["operation"] = operation
       #options2 = {}
       #options2["operation"] = operation
       return options




   def get_antenna(self,ant,ant_names):
       if isinstance(ant[0],int) :
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
       d_list = list(xrange(self.ms.na))
       for k in range(len(self.a_list)):
           d_list.remove(self.a_list[k])
       return d_list

   def read_R(self,column):
       self.R = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       self.u_m = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       self.v_m = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       
       self.p = np.ones((self.ms.na,self.ms.na),dtype = int)
       self.p = np.cumsum(self.p,axis=0)-1
       self.q = self.p.transpose()

       for j in xrange(self.ms.na):
           for k in xrange(j+1,self.ms.na):
               #print "j = ",j
               #print "k = ",k
               if column == "CORRECTED_DATA":
                  r_jk = self.ms.corr_data[(self.ms.A1==j) & (self.ms.A2==k),0,0]
               else:
                  r_jk = self.ms.data[(self.ms.A1==j) & (self.ms.A2==k),0,0]
               
               #print "self.ms.A1 = ",self.ms.A1
               
               #print "antenna_b = ",(self.ms.A1==j) & (self.ms.A2==k)

               #print "r_jk = ",r_jk
               
               u_jk = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),0]
               v_jk = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),1]
               
               self.R[j,k,:] = r_jk
               self.R[k,j,:] = r_jk.conj()
               self.u_m[j,k,:] = u_jk 
               self.u_m[k,j,:] = -1*u_jk
               self.v_m[j,k,:] = v_jk
               self.v_m[k,j,:] = -1*v_jk                

           if self.ms.auto:
              self.R[j,j,:] = self.ms.corr_data[(self.ms.A1==j)&(self.ms.A2==j),0,0]
           else: 
              self.R[j,j,:] = 0  
              print "HELP2 ................."
       
       #NB NO AUTOCORRELATION HANDLING!
       #--- ADDING A TRANSIENT --- VERY HACKY  
       for k in xrange(len(self.point_sources)):
           if self.on_time[k] <> 100:
              vis_temp = self.point_sources[k,0]*np.exp((-2*np.pi*1j*(self.u_m*self.point_sources[k,1]+self.v_m*self.point_sources[k,2]))/self.ms.wave)
              time_steps = int(vis_temp.shape[2]*(self.on_time[k]/100.0))
              #print "time_steps = ",time_steps
              #print "vis_temp = ",vis_temp.shape[2]
              vis_temp[:,:,time_steps:] = 0
              #print "vis_temp = ",vis_temp[:,:,1]
              #self.R = self.R + vis_temp
       
       if self.antenna <> "all":
          d_list = self.calculate_delete_list()
          #print "d_list = ",d_list
          self.R = np.delete(self.R,d_list,axis = 0)
          self.R = np.delete(self.R,d_list,axis = 1)
          self.u_m = np.delete(self.u_m,d_list,axis = 0)
          self.u_m = np.delete(self.u_m,d_list,axis = 1)
          self.v_m = np.delete(self.v_m,d_list,axis = 0)
          self.v_m = np.delete(self.v_m,d_list,axis = 1)
          self.p = np.delete(self.p,d_list,axis = 0)
          self.p = np.delete(self.p,d_list,axis = 1)
          self.q = np.delete(self.q,d_list,axis = 0)
          self.q = np.delete(self.q,d_list,axis = 1)
       #print "self.p = ",self.p
       #print "self.q = ",self.q 
       #print "R[:,:,1300] = ",self.R[:,:,1300] 

   def read_M(self,column):
       self.M = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       self.u_m = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       self.v_m = np.zeros((self.ms.na,self.ms.na,self.ms.ns),dtype=complex)
       
       self.p = np.ones((self.ms.na,self.ms.na),dtype = int)
       self.p = np.cumsum(self.p,axis=0)-1
       self.q = self.p.transpose()

       for j in xrange(self.ms.na):
           for k in xrange(j+1,self.ms.na):
               #print "j = ",j
               #print "k = ",k
               m_jk = self.ms.model_data[(self.ms.A1==j) & (self.ms.A2==k),0,0]
               
               #print "self.ms.A1 = ",self.ms.A1
               
               #print "antenna_b = ",(self.ms.A1==j) & (self.ms.A2==k)

               #print "r_jk = ",r_jk
               
               u_jk = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),0]
               v_jk = self.ms.uvw[(self.ms.A1==j)&(self.ms.A2==k),1]
               
               self.M[j,k,:] = m_jk
               self.M[k,j,:] = m_jk.conj()
               self.u_m[j,k,:] = u_jk 
               self.u_m[k,j,:] = -1*u_jk
               self.v_m[j,k,:] = v_jk
               self.v_m[k,j,:] = -1*v_jk                

           if self.ms.auto:
              self.M[j,j,:] = self.ms.model_data[(self.ms.A1==j)&(self.ms.A2==j),0,0]
           else: 
              self.M[j,j,:] = 0
              print "HELP..........."

           
       if self.antenna <> "all":
          d_list = self.calculate_delete_list()
          #print "d_list = ",d_list
          self.M = np.delete(self.M,d_list,axis = 0)
          self.M = np.delete(self.M,d_list,axis = 1)
          self.u_m = np.delete(self.u_m,d_list,axis = 0)
          self.u_m = np.delete(self.u_m,d_list,axis = 1)
          self.v_m = np.delete(self.v_m,d_list,axis = 0)
          self.v_m = np.delete(self.v_m,d_list,axis = 1)
          self.p = np.delete(self.p,d_list,axis = 0)
          self.p = np.delete(self.p,d_list,axis = 1)
          self.q = np.delete(self.q,d_list,axis = 0)
          self.q = np.delete(self.q,d_list,axis = 1)
       #print "self.p = ",self.p
       #print "self.q = ",self.q 
       #print "M[:,:,200] = ",self.M[:,:,200] 
              
   '''Unpolarized direction independent phase-only calibration entails finding the G(theta) that minimizes ||R-GMG^H||. 
   This function evaluates R-G(theta)MG^H(theta).
   theta is a vactor containing the phase of the antenna gains.
   r is a vector containing a vecotrized R (observed visibilities), real and imaginary
   m is a vector containing a vecotrized M (predicted), real and imaginary   
   ''' 
   def err_func_theta(self,theta,r,m):
        Nm = len(r)/2
        N = len(theta)
        G = np.diag(np.exp(-1j*theta))
        R = np.reshape(r[0:Nm],(N,N))+np.reshape(r[Nm:],(N,N))*1j
        M = np.reshape(m[0:Nm],(N,N))+np.reshape(m[Nm:],(N,N))*1j
        T = np.dot(G,M)
        T = np.dot(T,G.conj())
        Y = R - T
        y_r = np.ravel(Y.real)
        y_i = np.ravel(Y.imag)
        y = np.hstack([y_r,y_i])
        return y

   '''This function finds argmin phase ||R-G(phase)MG(phase)^H|| using Levenberg-Marquardt. It uses the optimize.leastsq scipy to perform
   the actual minimization
   R is your observed visibilities matrx
   M is your predicted visibilities
   g the antenna gains
   G = gg^H''' 
   def create_G_LM_phase_only(self):
        N = self.R.shape[0]
        temp =np.ones((self.R.shape[0],self.R.shape[1]) ,dtype=complex)
        self.G = np.zeros(self.R.shape,dtype=complex)
        theta = np.zeros((self.R.shape[0],self.R.shape[2]),dtype=float)
      
        for t in xrange(self.R.shape[2]):
            print "t = ",t
            print "N = ",self.R.shape[2]
            theta_0 = np.zeros((N,))
            r_r = np.ravel(self.R[:,:,t].real)
            r_i = np.ravel(self.R[:,:,t].imag)
            r = np.hstack([r_r,r_i])
            m_r = np.ravel(self.M[:,:,t].real)
            m_i = np.ravel(self.M[:,:,t].imag)
            m = np.hstack([m_r,m_i])
            theta_lstsqr_temp = optimize.leastsq(self.err_func_theta, theta_0, args=(r, m))
            theta_lstsqr = theta_lstsqr_temp[0]          
           
            G_m = np.dot(np.diag(np.exp(-1j*theta_lstsqr)),temp)
            G_m = np.dot(G_m,np.diag(np.exp(-1j*theta_lstsqr)).conj())            

            theta[:,t] = theta_lstsqr       
            self.G[:,:,t] = G_m
         
        #return theta,G

   def create_G_phase_stef_it(self,R,M,imax=200,tau=1e-6,no_auto=True):
       N = R.shape[0]
       if no_auto:
          R = R - R*np.eye(R.shape[0])
          M = M - M*np.eye(M.shape[0])
       temp =np.ones((R.shape[0],R.shape[1]) ,dtype=complex)
       
       phase = np.zeros((R.shape[0],),dtype=float)
       phase_delta = np.zeros((R.shape[0],),dtype=float)
       m = np.zeros((R.shape[0],),dtype=float)  
       #model_inverse = ((np.sum(np.conj(M[:,p])*M[:,p])).real)**(-1) 
       g = np.exp(-1j*phase)
       for i in xrange(imax):
             #print "i_p = ",i
             for p in xrange(N):
                 if i == 0:
                    m[p] = ((np.sum(np.conj(M[:,p])*M[:,p])).real)**(-1)
                 z = g*M[:,p]
                 phase_delta[p] = (-1*np.conj(g[p])*np.sum(np.conj(R[:,p])*z)).imag*m[p]
             
             if (i % 2 == 0):
                phase_new = phase + phase_delta
             else:
                phase_new = phase + phase_delta/2.
             if (np.sqrt(np.sum(np.absolute(phase_new-phase)**2))/np.sqrt(np.sum(np.absolute(phase_new)**2)) <= tau):
                break
             else:
                phase = phase_new
                g = np.exp(-1j*phase)
       #print "Phase i = ",i
       G = np.dot(np.diag(g),temp)
       G = np.dot(G,np.diag(g.conj()))           
       return G

   def create_G_phase_stef(self):
       print "R.shape = ",self.R.shape
       print "M.shape = ",self.M.shape
       self.G = np.zeros(self.R.shape,dtype=complex)

       for t in range(self.ms.ns):
           print "t = ",t
           self.G[:,:,t] = self.create_G_phase_stef_it(self.R[:,:,t],self.M[:,:,t])

   def cal_G_eig(self):
        #U_n=np.zeros(self.R.shape,dtype=complex)
        #S_n=np.zeros(self.R.shape, dtype=complex)
        #V_n=np.zeros(self.R.shape, dtype=complex)
        D =np.zeros(self.R.shape, dtype=complex)
        Q=np.zeros(self.R.shape, dtype=complex)
        self.g=np.zeros((self.R.shape[0],self.ms.ns) , dtype=complex)
        self.G=np.zeros(self.R.shape ,dtype=complex)
        temp =np.ones((self.R.shape[0],self.R.shape[0]) ,dtype=complex)
        #print "self.R.shape = ",self.R.shape
        #print "self.R = ",self.R
        for t in range(self.ms.ns):
           #print "t=",t
           d,Q[:,:,t] = np.linalg.eigh(self.R[:,:,t])
           D[:,:,t] = np.diag(d)
           Q_H = Q[:,:,t].conj().transpose()
           #R_2 = np.dot(Q[:,:,t],np.dot(D[:,:,t],Q_H))
           abs_d=np.absolute(d)
           index=abs_d.argmax()
           if (d[index] > 0):
             self.g[:,t]=Q[:,index,t]*np.sqrt(d[index])
           else:
             self.g[:,t]=Q[:,index,t]*np.sqrt(np.absolute(d[index]))*1j

           self.G[:,:,t] = np.dot(np.diag(self.g[:,t]),temp)
           self.G[:,:,t]= np.dot (self.G[:,:,t] ,np.diag(self.g[:,t].conj()))

        self.G = self.G
        #print "G = ",self.G[:,:,200]

   def write_to_MS(self,column,type_w):
       if column == "CORECTED_DATA":
          t_data = self.ms.corr_data
       elif column == "DATA":
          t_data = self.ms.data
       else:
          t_data = self.ms.model_data

       if self.ms.auto:
          s = 0
       else:
          s = 1

       for j in xrange(self.R.shape[0]):
           for k in xrange(j+s,self.R.shape[0]):
               if type_w == "R":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.R[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.R[j,k,:]
               elif type_w == "M":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.M[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.M[j,k,:]
               elif type_w == "G":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]
               elif type_w == "GT":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)
               elif type_w == "GTR":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)*self.R[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)*self.R[j,k,:]
               elif type_w == "R-1":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.R[j,k,:]-1
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.R[j,k,:]-1
               elif type_w == "G-1":
                  #t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]-1.03-0.07*np.exp(-2*np.pi*1j*(self.u_m[j,k,:]/self.ms.wave)*(1*(np.pi)/180))
                  #t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]-1.03-0.07*np.exp(-2*np.pi*1j*(self.u_m[j,k,:]/self.ms.wave)*(1*(np.pi)/180))
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]-1
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]-1
               elif type_w == "GT-1":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)-1
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)-1
               elif type_w == "GTR-R":
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,0] = self.G[j,k,:]**(-1)*self.R[j,k,:]-self.R[j,k,:]
                  t_data[(self.ms.A1==self.p[j,k])&(self.ms.A2==self.q[j,k]),0,3] = self.G[j,k,:]**(-1)*self.R[j,k,:]-self.R[j,k,:]
                 
       if column == "CORRECTED_DATA":
          self.ms.corr_data = t_data
       elif column == "DATA":
          self.ms.data = t_data
       else:
          self.ms.model_data = t_data

       self.ms.write(column)
     
def apply_primary_beam(fits_file,primary_beam,number,type_img,r_str):
    number_str = str(int(number))

    len_n = len(number_str)
    iterats = 4 - len_n

    for k in xrange(iterats):
        number_str = "0"+number_str
    #if type_img == "NO_PRIM":
    #   dirc = "./NO_PRIM"
    #else:
    #   dirc = "./PRIM" 
    if type_img == "NO_PRIM":
       dirc = "./"+type_img
    else:
       dirc = "./"+type_img 

    ff = pyfits.open(fits_file)
    header_v = ff[0].header
    cellsize = np.absolute(header_v['cdelt1']) #pixel width in degrees
    data = ff[0].data
    data = data[0,0,::-1,:]
    npix = data.shape[0] #number of pixels in fits file
       
    ff_p = pyfits.open(primary_beam)
    header_v_p = ff_p[0].header
    cellsize_p = np.absolute(header_v_p['cdelt1']) #pixel width in degrees
    data_p = ff_p[0].data
    data_p = data_p[0,0,::-1,:]
    npix_p = data.shape[0] #number of pixels in fits file
    data_p[data_p>1.0]=1.0
    data_p[data_p<0.005]=1.0

    l = np.linspace(-npix_p/2.0*cellsize_p,npix_p/2.0*cellsize_p,npix_p)
    m = np.copy(l)
    ll,mm = np.meshgrid(l,m)

    max_v = np.amax(l)

    dd = np.sqrt(ll**2+mm**2)

    data_p[dd>10.0] = 1.0

    if npix==npix_p:
       data_new = data/data_p
       print "data_new = ",data_new
       print "data_new_max = ",np.amax(data_new)
       print "data_new_min = ",np.amin(data_new)

       plt.imshow(data,vmax=60.0,vmin=-1,cmap="hot",extent=[-max_v,max_v,-max_v,max_v])
       plt.contour(ll,mm,data_p[::-1,:],4,colors="y",fontsize="0.3",linestyles="dotted")
       plt.xlim([-max_v,max_v])
       plt.ylim([-max_v,max_v])
       plt.xlabel("$l$ [degrees]")
       plt.ylabel("$m$")
       plt.title("Position: "+number_str+" $r$ = "+r_str)
       plt.savefig(dirc+"A/"+type_img+number_str,bbox_inches="tight")  
       #plt.show()
       #plt.imshow(data_p,vmax=1.0,vmin=0,cmap="hot",extent=[-max_v,max_v,-max_v,max_v])
       #plt.hold("on")
       #l = np.linspace(-7,7,10)
       #m = np.copy(l)
       #plt.plot(l,m,"y:")
       #plt.xlim([-max_v,max_v])
       #plt.ylim([-max_v,max_v])
       #plt.show()
       plt.imshow(data_new,vmax=20.0,vmin=-1,cmap="hot",extent=[-max_v,max_v,-max_v,max_v])
       #plt.hold("on")
       #l = np.linspace(-7,7,10)
       #m = np.copy(l)
       #plt.plot(l,m,"y:")
       plt.contour(ll,mm,data_p[::-1,:],4,colors="y",fontsize="0.3",linestyles="dotted")
       plt.xlim([-max_v,max_v])
       plt.ylim([-max_v,max_v])
       plt.xlabel("$l$ [degrees]")
       plt.ylabel("$m$")
       #plt.show()
       plt.title("Position: "+number_str+" $r$ = "+r_str)
       plt.savefig(dirc+"B/"+type_img+number_str,bbox_inches="tight")  
       #plt.savefig("foo2.png",bbox_inches="tight")  
    ff[0].data[0,0,::-1,:] = data_new
    
    fits_file = dirc+"_fits/"+type_img+number_str+"_applied.fits" 
    if os.path.isfile(fits_file):
       os.system("rm "+fits_file)
    ff.writeto(fits_file)
    
    ff.close()
    ff_p.close()

def create_new_primary_beam(fits_file,primary_beam,new_prim):

    ff = pyfits.open(fits_file)
    header_v = ff[0].header
    cellsize = np.absolute(header_v['cdelt1']) #pixel width in degrees
    data = ff[0].data
    data = data[0,0,::-1,:]
    npix = data.shape[0] #number of pixels in fits file
       
    ff_p = pyfits.open(primary_beam)
    header_v_p = ff_p[0].header
    cellsize_p = np.absolute(header_v_p['cdelt1']) #pixel width in degrees
    data_p = ff_p[0].data
    data_p = data_p[0,0,::-1,:]
    npix_p = data.shape[0] #number of pixels in fits file

    ff[0].data[0,0,::-1,:] = data_p
    
    fits_file = new_prim
    if os.path.isfile(fits_file):
       os.system("rm "+fits_file)
    ff.writeto(fits_file)
    ff.close()
    ff_p.close()

def create_primary_beam_cont(primary_beam):

    ff_p = pyfits.open(primary_beam)
    header_v_p = ff_p[0].header
    cellsize_p = np.absolute(header_v_p['cdelt1']) #pixel width in degrees
    data_p = ff_p[0].data
    data_p = data_p[0,0,::-1,:]
    npix_p = data_p.shape[0] #number of pixels in fits file
    data_p[data_p>1.0]=0.0
    data_p[data_p<0.005]=0.0

    l = np.linspace(-npix_p/2.0*cellsize_p,npix_p/2.0*cellsize_p,npix_p)
    m = np.copy(l)
    ll,mm = np.meshgrid(l,m)

    max_v = np.amax(l)

    dd = np.sqrt(ll**2+mm**2)

    data_p[dd>10.0] = 0.0

    ff_p[0].data[0,0,::-1,:] = data_p
    
    fits_file = "prim_contour.fits" 
    if os.path.isfile(fits_file):
       os.system("rm "+fits_file)
    ff_p.writeto(fits_file)
    
    ff_p.close()
 

def extract_values_from_fits(fits_file,mask,window=20,pix_deg="PIX",clip=1):
    ff = pyfits.open(fits_file)
    header_v = ff[0].header
    cellsize = np.absolute(header_v['cdelt1']) #pixel width in degrees
    data = ff[0].data
    data = data[0,0,::-1,:]
    npix = data.shape[0] #number of pixels in fits file
    cpix = npix/2.0

    if clip==1:
       data[data>1.0]=1.0
       #plt.imshow(data,extent=[-10,10,-10,10])
       #plt.show()
       l = np.linspace(-npix/2.0*cellsize,npix/2.0*cellsize,npix)
       m = np.copy(l)
       ll,mm = np.meshgrid(l,m)
       max_v = np.amax(l)
       dd = np.sqrt(ll**2+mm**2)
       data[dd>10.0] = 1.0

    values = np.zeros((mask.shape[0],3))

    if pix_deg == "PIX":
       w = window
    else:
       w = int(window/cellsize)+1

    for s in xrange(mask.shape[0]):
        l = mask[s,0]
        m = mask[s,1]
        source = True
        pix_x = int(cpix + l/cellsize)
        pix_y = int(cpix - m/cellsize)

        if pix_x < 0 or pix_x > npix:
           source = False
        if pix_y < 0 or pix_y > npix:
           source = False
        if source:
           if pix_x - w < 0:
              x_1 = 0
           else:
              x_1 = pix_x - w
           if pix_x + w + 1> npix-1:
              x_2 = npix-1
           else:
              x_2 = pix_x + w + 1

           if pix_y - w < 0:
              y_1 = 0
           else:
              y_1 = pix_y - w
           if pix_y + w + 1> npix-1:
              y_2 = npix-1
           else:
              y_2 = pix_y + w + 1

           #print "y_1 = ",y_1
           #print "y_2 = ",y_2

           data_temp = data[y_1:y_2,x_1:x_2]
           #data[y_1:y_2,x_1:x_2] = -100
           #data_temp = data[0:1000,:]

           #plt.imshow(data_temp)
           #plt.show()

           min_v = np.amin(data_temp)

           max_v = np.amax(data_temp)

           mean_v = np.mean(data_temp)

        else:
           if clip == 1:
              min_v = 1
              max_v = 1
              mean_v = 1
           else:
              min_v = 0
              max_v = 0
              mean_v = 0
        
        values[s,0] = mean_v
        values[s,1] = min_v
        values[s,2] = max_v

    #plt.imshow(data,extent=[-10,10,-10,10])
    #plt.show()

    ff.close()
    return values

def video_experiment(max_v,num,type_exp="PRIM"):

    l = np.linspace(-max_v,max_v,num)
    m = np.copy(l)

    ll,mm = np.meshgrid(l,m)
 
    #ms_object = Mset("LOFAR.MS")
  
    #ms_object.extract()

    number = 0
    for k in xrange(len(l)):
        for j in xrange(len(m)):
            ms_object = Mset("LOFAR.MS")
  
            ms_object.extract()
            #The l and m coordinates of the experiment
            print "number = ",number
            print "l = ",ll[k,j]
            print "m = ",mm[k,j]
            print "k = ",k
            print "j = ",j
            
            x_3C61 = -1.29697746*(np.pi/180.0)
            y_3C61 = -3.44496443*(np.pi/180.0)
            x_trans = ll[k,j]*(np.pi/180.0)
            y_trans = mm[k,j]*(np.pi/180.0)
            #x_trans = 2*(np.pi/180.0)
            #y_trans = 2*(np.pi/180.0)
            point_sources = np.array([(88.7,x_3C61,y_3C61),(60,x_trans,y_trans)])
            point_sources_cal = np.array([(88.7,x_3C61,y_3C61)])
            on_time = np.array([100,100])
            
            mask_sources = point_sources[:,1:]*(180/np.pi)
            mask_cal = point_sources_cal[:,1:]*(180/np.pi)        

            prim_fits = "Lofar_primary_beam.fits"

            primary_beam = extract_values_from_fits(prim_fits,mask_sources,window=20,pix_deg="PIX",clip=1)
            primary_beam_cal = extract_values_from_fits(prim_fits,mask_cal,window=20,pix_deg="PIX",clip=1)
           
            print "primary_beam = ",primary_beam
            print "point_sources = ",point_sources
 
            if type_exp <> "PRIM":
               primary_beam = np.ones(np.shape(primary_beam))
               primary_beam_cal = np.ones(np.shape(primary_beam_cal))

            point_sources[:,0] = point_sources[:,0]*primary_beam[:,0]
            point_sources_cal[:,0] = point_sources_cal[:,0]*primary_beam_cal[:,0]

            print "point_sources_after = ",point_sources[:,0]
        
            #CREATE VISIBILITIES
            s = Sky_model(ms_object,point_sources,point_sources_cal,on_time,0.5)
            s.visibility("CORRECTED_DATA")
            s.visibility("MODEL_DATA")
            
            #LOAD VISIBILITIES FROM MS
            antenna = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
            c = Calibration(ms_object,antenna,point_sources,point_sources_cal,on_time)
            c.read_R(column="CORRECTED_DATA") 
            c.read_M(column="MODEL_DATA")
            c.write_to_MS(column="CORRECTED_DATA",type_w="R")
            options = c.image_advanced_settings(antenna=True,img_nchan=1,img_chanstart=0,img_chanstep=1)
            #v.CALTECH=str(number)
            #imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
            #v.CALTECH=""
            
            #CALIBRATE AND IMAGE
            c.create_G_LM_phase_only()
            c.write_to_MS(column="CORRECTED_DATA",type_w="GTR")
            v.CALTECH="GT"
            imager.make_image(column="CORRECTED_DATA",dirty=False,restore=options)
            v.CALTECH = ""

            #MAKING PNG IMAGES
            fits_file = "./plots-LOFAR/LOFAR_GT.restored.fits"
            apply_primary_beam(fits_file,prim_fits,number,type_exp)
            number = number + 1

def set_pybdsm_options(thresh_isl = 3.0, thresh_pix = 5.0, src_ra_dec = None, src_radius_pix = 8):
    options = {}
    options["thresh_isl"] = thresh_isl
    options["thresh_pix"] = thresh_pix
    options["src_ra_dec"] = src_ra_dec
    options["src_radius_pix"] = src_radius_pix
    options["group_by_isl"] = True
    return options

def pybdsm_search(input_image,options):
    img = bdsm.process_image(input_image, thresh_isl=options["thresh_isl"],thresh_pix=options["thresh_pix"],src_ra_dec=options["src_ra_dec"],src_radius_pix=options["src_radius_pix"],group_by_isl=options["group_by_isl"])
    img.write_catalog(format='ascii', catalog_type='gaul',clobber=True)

def process_gaul_file(gaul_file):
    f = open(gaul_file, "r")
    f_lines = f.readlines()
    f_lines = f_lines[6:]

    src_results = np.zeros((len(f_lines),10))
    counter = 0
    for line in f_lines:
        #print "line = ",line.split(" ") 
        s_line = line.split(" ")

        s_line_filt = [s for s in s_line if s <> '']
        #print "s_line_filt = ",s_line_filt
        #print "s_line[4] = ",s_line_filt[4]    
        #print "s_line[5] = ",s_line_filt[5]    
        #print "s_line[6] = ",s_line_filt[6]    
        #print "s_line[7] = ",s_line_filt[7]    
        #print "s_line[8] = ",s_line_filt[8]    
        #print "s_line[9] = ",s_line_filt[9]    
        src_results[counter,0] = float(s_line_filt[4]) #ra_d
        src_results[counter,1] = float(s_line_filt[5]) #E_RA
        src_results[counter,2] = float(s_line_filt[6]) #dec_d
        src_results[counter,3] = float(s_line_filt[7]) #E_DEC
        src_results[counter,4] = float(s_line_filt[8]) #Total_flux
        src_results[counter,5] = float(s_line_filt[9]) #E_Total_flux
        src_results[counter,6] = float(s_line_filt[10]) #Peak_flux
        src_results[counter,7] = float(s_line_filt[11]) #E_Peak_flux
        src_results[counter,8] = float(s_line_filt[1]) #Isl_id
        src_results[counter,9] = float(s_line_filt[2]) #Src_id
        counter = counter + 1
    #print src_results
    return src_results

def get_flux(src_gaul_extract, src_ra_dec=None):
    #print len(src_ra_dec)    
    #x = src_ra_dec[0]
    #print x[0]
    flux = np.zeros((len(src_ra_dec),4))
    src_id = int(src_gaul_extract[0,8])
    for k in xrange(len(src_ra_dec)):
        flux[k,0] = np.sum(src_gaul_extract[src_gaul_extract[:,8].astype(int)==src_id,4])
        flux[k,1] = np.sum(src_gaul_extract[src_gaul_extract[:,8].astype(int)==src_id,5])
        flux[k,2] = np.sum(src_gaul_extract[src_gaul_extract[:,8].astype(int)==src_id,6])
        flux[k,3] = np.sum(src_gaul_extract[src_gaul_extract[:,8].astype(int)==src_id,7])
        src_id = src_id + 1        

        #dist = np.sqrt((src_ra_dec[k][0] - src_gaul_extract[:,0])**2 + (src_ra_dec[k][1] - src_gaul_extract[:,2])**2)
        #print dist 
        #ind = np.argmin(dist)
        #print "dis<cut = ",dist<cut
        #if dist[ind] < cut:
        #   flux[k,0] = np.sum(src_gaul_extract[dist<cut,4])
        #   flux[k,1] = np.sum(src_gaul_extract[dist<cut,5])
        #   flux[k,2] = np.sum(src_gaul_extract[dist<cut,6])
        #   flux[k,3] = np.sum(src_gaul_extract[dist<cut,7])
           #print "ind = ",ind
    #print "flux = ",flux
    return flux

#l and m in degrees
def convert_to_world_coordinates(l,m,p_180 = True,start_value=19.85):
    
    ra = np.arctan2(m,l)*180/np.pi
    ra[ra<0] = ra[ra<0]+360
    ra = start_value*15-ra
    ra[ra<0] = ra[ra<0]+360

    if p_180:
       ra = ra + 180
       ra[ra>360] = ra[ra>360]-360

    l_rad = l*np.pi/180
    m_rad = m*np.pi/180

    d_rad = np.sqrt(l_rad**2+m_rad**2)
    dec = np.arcsin(d_rad)*180/np.pi
    dec = 90-dec
    return ra,dec

def video_experiment_spiral(max_v,num,type_exp="PRIM",file_name="spiral.p",algo="STEF",skip=False,source_extr="PYBDSM"):

    SOURCE_POS_PYBDSM = [(35.50060358+180,86.31644),(63.5+180,84.070111),((343.250545139+180)-360,86.36294)]

    theta = np.linspace(0,2*max_v*np.pi,num)
    a = 1.0/(2*np.pi)

    l = -1*a*theta*np.cos(theta)
    m = -1*a*theta*np.sin(theta)

    if skip:

       l = np.array([4.4770401,4.36475859])
       m = np.array([0.64042245,3.521221])

    print "skip = ",skip

    result_prim = np.zeros((len(l),3))
    result = np.zeros((len(l),3))
    apparent = np.zeros((len(l),2))
    p_beam = np.zeros((len(l),3))

    number = 0
    for k in xrange(len(l)):
          ms_object = Mset("LOFAR2.MS")
  
          ms_object.extract()
          #The l and m coordinates of the experiment
          print "number = ",number
          print "l = ",l[k]
          print "m = ",m[k]
          print "k = ",k
          print "j = "
           
          x_3C61 = -1.29697746*(np.pi/180.0)
          y_3C61 = -3.44496443*(np.pi/180.0)
          x_trans = l[k]*(np.pi/180.0)
          y_trans = m[k]*(np.pi/180.0)
          dx = x_trans - x_3C61
          dy = y_trans - y_3C61
          x_ghost = x_3C61 - dx
          y_ghost = y_3C61 - dy

          #x_trans = 2*(np.pi/180.0)
          #y_trans = 2*(np.pi/180.0)
          point_sources = np.array([(88.7,x_3C61,y_3C61),(60,x_trans,y_trans)])
          point_sources_cal = np.array([(88.7,x_3C61,y_3C61)])
          point_sources_all = np.array([(88.7,x_3C61,y_3C61),(60,x_trans,y_trans),(1,x_ghost,y_ghost)])
          on_time = np.array([100,100])
            
          mask_sources = point_sources[:,1:]*(180/np.pi)
          mask_cal = point_sources_cal[:,1:]*(180/np.pi)        
          mask_all = point_sources_all[:,1:]*(180/np.pi)        
          
          print "mask_all = ",mask_all[:,0]
          print "mask_all = ",mask_all[:,1]
 
          ra,dec = convert_to_world_coordinates(mask_all[:,0],mask_all[:,1],p_180=False) 

          print "ra = ",ra
          print "dec = ",dec
          for s in xrange(len(SOURCE_POS_PYBDSM)):
              print "s = ",s
              SOURCE_POS_PYBDSM[s] = (ra[s],dec[s]) 
          
          print "SOURCE_POS_PYBDSM = ",SOURCE_POS_PYBDSM
 
          prim_fits = "Lofar_primary_beam_new.fits"

          primary_beam = extract_values_from_fits(prim_fits,mask_sources,window=20,pix_deg="PIX",clip=1)
          primary_beam_cal = extract_values_from_fits(prim_fits,mask_cal,window=20,pix_deg="PIX",clip=1)
          primary_beam_all = extract_values_from_fits(prim_fits,mask_all,window=20,pix_deg="PIX",clip=1)
           
          print "primary_beam = ",primary_beam
          print "point_sources = ",point_sources
 
          if type_exp <> "PRIM":
             primary_beam = np.ones(np.shape(primary_beam))
             primary_beam_cal = np.ones(np.shape(primary_beam_cal))
             primary_beam_all = np.ones(np.shape(primary_beam_cal))

          point_sources[:,0] = point_sources[:,0]*primary_beam[:,0]
          point_sources_cal[:,0] = point_sources_cal[:,0]*primary_beam_cal[:,0]
          point_sources_all[:,0] = point_sources_all[:,0]*primary_beam_all[:,0]

          print "point_sources_after = ",point_sources[:,0]
        
          #CREATE VISIBILITIES
          s = Sky_model(ms_object,point_sources,point_sources_cal,on_time,0.1)
          s.visibility("CORRECTED_DATA")
          s.visibility("MODEL_DATA")
            
          #LOAD VISIBILITIES FROM MS
          antenna = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
          c = Calibration(ms_object,antenna,point_sources,point_sources_cal,on_time)
          c.read_R(column="CORRECTED_DATA") 
          c.read_M(column="MODEL_DATA")
          c.write_to_MS(column="CORRECTED_DATA",type_w="R")
          options = c.image_advanced_settings(antenna=True,img_nchan=1,img_chanstart=0,img_chanstep=1)
          #v.CALTECH=str(number)
          #imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
          #v.CALTECH=""
          
          #CALIBRATE AND IMAGE
          #print "before cal"
          if algo == "STEF":
             c.create_G_phase_stef()
          else:
             c.create_G_LM_phase_only()
          #print "after cal"
          c.write_to_MS(column="CORRECTED_DATA",type_w="GTR")
          v.CALTECH="GT"
          imager.make_image(column="CORRECTED_DATA",dirty=False,restore=options)
          v.CALTECH = ""
          #print "--- HALLO1 ---"
          #print "skip = ",skip
          if not skip:
             #MAKING PNG IMAGES
             #print "--- HALLO ---"
             fits_file = "./plots-LOFAR2/LOFAR2_GT.restored.fits"
             apply_primary_beam(fits_file,prim_fits,number,type_exp+"SPIRAL","{:.2f}".format(np.sqrt(l[k]**2+m[k]**2)))
             number = number + 1
         
          #READ_VALUES 
          print "SOURCE_POS_PYBDSM = ",SOURCE_POS_PYBDSM 
          fits_file = "./plots-LOFAR2/LOFAR2_GT.restored.fits"
          if source_extr == "PYBDSM":
             pybdsm_opts = set_pybdsm_options(src_ra_dec=SOURCE_POS_PYBDSM)
             pybdsm_search(fits_file,pybdsm_opts)
             src_gaul_extract = process_gaul_file("."+fits_file.strip(".fits")+".pybdsm.gaul")
             pybdsm_values = get_flux(src_gaul_extract,src_ra_dec = SOURCE_POS_PYBDSM) 
             values_cal = pybdsm_values[:,0]
          else: 
             mask_all = point_sources_all[:,1:]*(180/np.pi)        
             fits_file = "./plots-LOFAR2/LOFAR2_GT.restored.fits"
             values_cal = extract_values_from_fits(fits_file,mask_all,window=20,pix_deg="PIX",clip=0)
             values_cal = values_cal[:,2]
          result_prim[k,:] = values_cal/primary_beam_all[:,0]
          result[k,:] = values_cal
          apparent[k,:] = point_sources[:,0]
          p_beam[k,:] = primary_beam_all[:,0]

          print "result_prim = ",result_prim
          print "result = ",result
          print "apparent = ",apparent
          print "p_beam = ",p_beam

    f = open(file_name, 'wb')
    pickle.dump(theta,f)
    pickle.dump(result,f)
    pickle.dump(result_prim,f)
    pickle.dump(apparent,f)
    pickle.dump(p_beam,f)
    pickle.dump(l,f)
    pickle.dump(m,f)
    f.close()

def experiment_source_suppression(num,file_name="suppression.p",algo="STEF",ant=7,source_extr="PYBDSM"):

    l = 0 
    m = 0
    
    SOURCE_POS_PYBDSM = [(35.50060358+180,86.31644),(63.5+180,84.070111),((343.250545139+180)-360,86.36294)]

    A_2 = np.linspace(10,61,num)

    result = np.zeros((len(A_2),3))

    number = 0
    for k in xrange(len(A_2)):
          ms_object = Mset("LOFAR3.MS")
  
          ms_object.extract()
          print "number = ",number
          print "file_name = ",file_name 
          x_3C61 = -1.29697746*(np.pi/180.0)
          y_3C61 = -3.44496443*(np.pi/180.0)
          x_trans = l*(np.pi/180.0)
          y_trans = m*(np.pi/180.0)
          dx = x_trans - x_3C61
          dy = y_trans - y_3C61
          x_ghost = x_3C61 - dx
          y_ghost = y_3C61 - dy

          #x_trans = 2*(np.pi/180.0)
          #y_trans = 2*(np.pi/180.0)
          point_sources = np.array([(48.52,x_3C61,y_3C61),(A_2[k],x_trans,y_trans)])
          point_sources_cal = np.array([(48.52,x_3C61,y_3C61)])
          point_sources_all = np.array([(48.52,x_3C61,y_3C61),(A_2[k],x_trans,y_trans),(1,x_ghost,y_ghost)])
          on_time = np.array([100,100])
            
          mask_sources = point_sources[:,1:]*(180/np.pi)
          mask_cal = point_sources_cal[:,1:]*(180/np.pi)        
          mask_all = point_sources_all[:,1:]*(180/np.pi)        
          
          ra,dec = convert_to_world_coordinates(mask_all[:,0],mask_all[:,1],p_180=False) 

          #print "ra = ",ra
          #print "dec = ",dec
          for s in xrange(len(SOURCE_POS_PYBDSM)):
              #print "s = ",s
              SOURCE_POS_PYBDSM[s] = (ra[s],dec[s]) 
          
          print "SOURCE_POS_PYBDSM = ",SOURCE_POS_PYBDSM

          #CREATE VISIBILITIES
          s = Sky_model(ms_object,point_sources,point_sources_cal,on_time,0.1)
          s.visibility("CORRECTED_DATA")
          s.visibility("MODEL_DATA")
            
          #LOAD VISIBILITIES FROM MS
          antenna = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
          c = Calibration(ms_object,antenna[0:ant],point_sources,point_sources_cal,on_time)
          c.read_R(column="CORRECTED_DATA") 
          c.read_M(column="MODEL_DATA")
          c.write_to_MS(column="CORRECTED_DATA",type_w="R")
          options = c.image_advanced_settings(antenna=True,img_nchan=1,img_chanstart=0,img_chanstep=1)
          #v.CALTECH=str(number)
          #imager.make_image(column="CORRECTED_DATA",dirty=options,restore=False)
          #v.CALTECH=""
          
          #CALIBRATE AND IMAGE
          print "before cal"
          if algo == "STEF":
             c.create_G_phase_stef()
          else:
             c.create_G_LM_phase_only()
          print "after cal"
          c.write_to_MS(column="CORRECTED_DATA",type_w="GTR")
          v.CALTECH="GT"
          imager.make_image(column="CORRECTED_DATA",dirty=False,restore=options)
          v.CALTECH = ""

          #MAKING PNG IMAGES
          fits_file = "./plots-LOFAR3/LOFAR3_GT.restored.fits"
          #apply_primary_beam(fits_file,prim_fits,number,type_exp+"SPIRAL","{:.2f}".format(np.sqrt(l[k]**2+m[k]**2)))
          number = number + 1
        

          print "SOURCE_POS_PYBDSM = ",SOURCE_POS_PYBDSM 
          #READ_VALUES 
          fits_file = "./plots-LOFAR3/LOFAR3_GT.restored.fits"
          if source_extr == "PYBDSM":
             pybdsm_opts = set_pybdsm_options(src_ra_dec=SOURCE_POS_PYBDSM)
             pybdsm_search(fits_file,pybdsm_opts)
             src_gaul_extract = process_gaul_file("."+fits_file.strip(".fits")+".pybdsm.gaul")
             pybdsm_values = get_flux(src_gaul_extract,src_ra_dec = SOURCE_POS_PYBDSM) 
             values_cal = pybdsm_values[:,0]
          else: 
             mask_all = point_sources_all[:,1:]*(180/np.pi)        
             values_cal = extract_values_from_fits(fits_file,mask_all,window=20,pix_deg="PIX",clip=0)
             values_cal = values_cal[:,2]
          result[k,:] = values_cal
          
          print "result = ",result

    f = open(file_name, 'wb')
    pickle.dump(result,f)
    pickle.dump(A_2,f)
    pickle.dump(l,f)
    pickle.dump(m,f)
    f.close()  

def plot_suppression():
    f = open("suppression_7.p",'rb')
    result_7 = pickle.load(f)
    A_2_7 = pickle.load(f)
    l = pickle.load(f)
    m = pickle.load(f)
    f.close()
    
    f = open("suppression_14.p",'rb')
    result_14 = pickle.load(f)
    A_2_14 = pickle.load(f)
    l = pickle.load(f)
    m = pickle.load(f)
    f.close()

    f = open("suppression_25.p",'rb')
    result_27 = pickle.load(f)
    A_2_27 = pickle.load(f)
    l = pickle.load(f)
    m = pickle.load(f)
    f.close()

    f = open("suppression_33.p",'rb')
    result_33 = pickle.load(f)
    A_2_33 = pickle.load(f)
    l = pickle.load(f)
    m = pickle.load(f)
    f.close()
    
    f = open("spiral_lm_mini_200.p",'rb')
    theta = pickle.load(f)
    result = pickle.load(f)
    result_prim = pickle.load(f)
    apparent = pickle.load(f)
    p_beam = pickle.load(f)
    l = pickle.load(f)
    m = pickle.load(f)
    f.close()

    plt.plot(A_2_7,(A_2_7-result_7[:,1])/A_2_7*100,"rx")
    plt.plot(A_2_14,(A_2_14-result_14[:,1])/A_2_14*100,"bx")
    plt.plot(A_2_27,(A_2_27-result_27[:,1])/A_2_27*100,"gx")
    plt.plot(A_2_33,(A_2_33-result_33[:,1])/A_2_33*100,"kx")
    plt.plot(apparent[:,1],(apparent[:,1]-result[:,1])/apparent[:,1]*100,"yx")   
    plt.plot(apparent[:,1],(result[:,2])/apparent[:,1]*100,"cx")   
    sup_ghost = (np.absolute(result[:,1]-apparent[:,1])/apparent[:,1])*100

    print "l = ", l[(sup_ghost>20)&(np.sqrt(l**2+m**2)>2)]
    print "m = ", m[(sup_ghost>20)&(np.sqrt(l**2+m**2)>2)]
    print "result[:,1] = ",result[(sup_ghost>20)&(np.sqrt(l**2+m**2)>2),1]
    print "apparent[:,1] = ",apparent[(sup_ghost>20)&(np.sqrt(l**2+m**2)>2),1]


    plt.show()

def plot_spiral(file_name="spiral_stef_mini_300.p"):
    f = open(file_name,'rb')
    theta = pickle.load(f)
    result = pickle.load(f)
    result_prim = pickle.load(f)
    apparent = pickle.load(f)
    p_beam = pickle.load(f)
    l = pickle.load(f)
    m = pickle.load(f)
    f.close()

    x_3C61 = -1.29697746*(np.pi/180.0)
    y_3C61 = -3.44496443*(np.pi/180.0)
    x_trans = l*(np.pi/180.0)
    y_trans = m*(np.pi/180.0)
    dx = x_trans - x_3C61
    dy = y_trans - y_3C61
    x_ghost = x_3C61 - dx
    y_ghost = y_3C61 - dy

    l_ghost = x_ghost*(180./np.pi)
    m_ghost = y_ghost*(180./np.pi)

    d_ghost = np.sqrt(l_ghost**2 + m_ghost**2)

    interval_mask = np.zeros(theta.shape)
    interval_mask[d_ghost>9.0] = 1

    a = 1/(2*np.pi)

    result_ghost = result[:,2]
    result_ghost[d_ghost>9.0] = 0
    result_ghost_prim = result_prim[:,2]
    result_ghost_prim[d_ghost>9.0] = 0

    p_beam_inv = 1./p_beam
    p_beam_inv[d_ghost>9.0]=0
    ghost_t = apparent[:,1]*1./32
    ghost_t[d_ghost>9.0] = 0 
    ghost_t_prim = ghost_t*p_beam_inv[:,2]

    r = a*theta
    #ghost_t = ghost_t[r>2.54]
   
    #result[d_ghost>9.0,:] = 0
    #result_prim[d_ghost>9.0,:]= 0
    #apparent[d_ghost>9.0,:]= 0
    #p_beam[d_ghost>9.0,:]=0
    #p_beam_inv = 1./p_beam
    #p_beam_inv[d_ghost>9.0]=0

    #PLOT INTRINSIC
    #MODELLED
    plt.semilogy(a*theta,np.ones(theta.shape)*88.7,"b:",lw=1.5)
    plt.hold('on')
    #UNMODELLED
    plt.semilogy(a*theta,np.ones(theta.shape)*60,"b--",lw=1.5)
    #GHOST
    plt.semilogy(r[r>2.54],ghost_t_prim[r>2.54],"b",label="Intrinsic",lw=1.5)
     
    #PLOT APPARENT
    #MODELLED
    plt.semilogy(a*theta,apparent[:,0],"r:",lw=1.5)
    #UNMODELLED
    plt.semilogy(a*theta,apparent[:,1],"r--",lw=1.5)
    #GHOST
    plt.semilogy(r[r>2.54],ghost_t[r>2.54],"r",label="Apparent",lw=1.5)
        
    #PLOT MEASURED
    #MODELLED
    plt.semilogy(a*theta,result[:,0],"g:",lw=1.5)
    #UNMODELLED
    plt.semilogy(a*theta,result[:,1],"g--",lw=1.5)
    #GHOST
    plt.semilogy(a*theta,result_ghost,"g",label="Measured",lw=1.5)

    #PLOT PRIMARY BEAM
    #MODELLED
    plt.semilogy(a*theta,result_prim[:,0],"k:",lw=1.5)
    #UNMODELLED 
    plt.semilogy(a*theta,result_prim[:,1],"k--",lw=1.5)
    #GHOST
    plt.semilogy(a*theta,result_ghost_prim,"k",label="Primary beam",lw=1.5)
    
    plt.semilogy(a*theta,p_beam_inv[:,2],"m",label="Gain",lw=1.5)

    #plt.semilogy(a*theta[::4],np.ones(theta[::4].shape)*88.7,"bx",lw=1.5)
    #plt.semilogy(a*theta[::4],np.ones(theta[::4].shape)*60,"rx",lw=1.5)

    #for k in xrange(3):
    #    if k == 0:
    #       c = "b"
    #       label_v = "3C61.1"
    #    elif k == 1:
    #       c = "r"
    #       label_v = "S"
    #    elif k == 2:
    #       c = "k"
    #       label_v = "GHOST"
    #    if k <> 2:    
    #       plt.semilogy(a*theta,result[:,k],c+"--",label=label_v,lw=1.5)
    #       plt.hold('on')
    #       plt.semilogy(a*theta,result_prim[:,k],c,label=label_v+" P",lw=1.5)
    #       plt.semilogy(a*theta,apparent[:,k],c+":",label=label_v+" A",lw=1.5)
    
    #plt.semilogy(a*theta,p_beam_inv[:,2],"c",label="AMP",lw=1.5)
    #plt.semilogy(r[r>2.54],ghost_t[r>2.54],"k:",lw=1.5)
    #plt.semilogy(r[r>2.54],ghost_t_prim[r>2.54],"k-.",lw=1.5)
    #plt.semilogy(a*theta[:],result_ghost[:],"k--",lw=1.5)
    #plt.semilogy(a*theta[:],result_ghost_prim[:],"k",lw=1.5)
    #plt.semilogy(a*theta,apparent[:,1]-result[:,2],"c",lw=2.0)
    #plt.semilogy(a*theta,apparent[:,1]*1./32,"k:",lw=2.0)
    #plt.semilogy(a*theta,interval_mask,"m",lw=2.0)

    #w a default vline at x=1 that spans the yrange
    
    #p = plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)
    p = plt.axvspan(0, 2.54, facecolor='r', alpha=0.1)
    p = plt.axvspan(2.54, 9, facecolor='b', alpha=0.1)
    p = plt.axvspan(1.63, 1.74, facecolor='y', alpha=0.2)
    p = plt.axvspan(2.54, 2.89, facecolor='y', alpha=0.2)
    p = plt.axvspan(3.47, 3.93, facecolor='y', alpha=0.2)
    p = plt.axvspan(4.43, 4.98, facecolor='y', alpha=0.2)
    p = plt.axvspan(5.42, 5.97, facecolor='y', alpha=0.2)
    p = plt.axvspan(6.4, 7, facecolor='y', alpha=0.2)
    p = plt.axvspan(7.41, 7.98, facecolor='y', alpha=0.2)
    p = plt.axvspan(8.4, 9, facecolor='y', alpha=0.2)
    l = plt.axvline(x=2.54,color="k",lw=2.0,ls="dashed")   
    l = plt.axvline(x=0.32,color="c",lw=2.0,ls="dashed")   
    l = plt.axvline(x=3.08,color="y",lw=2.0,ls="dashed")   
    l = plt.axvline(x=3.48,color="g",lw=2.0,ls="dashed")   
    plt.xlim([0,9])
    plt.ylabel("Jy") 
    plt.title("Ghost flux as a function of $r$")
    plt.xlabel("$r$ [degrees]")
    

    plt.legend(loc=3,prop={'size':10})          
    plt.show()

    """
    plt.semilogy(a*theta[::4],np.ones(theta[::4].shape)*88.7,"bx",lw=1.5)
    plt.semilogy(a*theta[::4],np.ones(theta[::4].shape)*60,"rx",lw=1.5)

    for k in xrange(3):
        if k == 0:
           c = "b"
           label_v = "3C61.1"
        elif k == 1:
           c = "r"
           label_v = "S"
        elif k == 2:
           c = "k"
           label_v = "GHOST"
        if k <> 2:    
           plt.semilogy(a*theta,result[:,k],c+"--",label=label_v,lw=1.5)
           plt.hold('on')
           plt.semilogy(a*theta,result_prim[:,k],c,label=label_v+" P",lw=1.5)
           plt.semilogy(a*theta,apparent[:,k],c+":",label=label_v+" A",lw=1.5)
    
    plt.semilogy(a*theta,p_beam_inv[:,2],"c",label="AMP",lw=1.5)
    plt.semilogy(r[r>2.54],ghost_t[r>2.54],"k:",lw=1.5)
    plt.semilogy(r[r>2.54],ghost_t_prim[r>2.54],"k-.",lw=1.5)
    plt.semilogy(a*theta[:],result_ghost[:],"k--",lw=1.5)
    plt.semilogy(a*theta[:],result_ghost_prim[:],"k",lw=1.5)
    #plt.semilogy(a*theta,apparent[:,1]-result[:,2],"c",lw=2.0)
    #plt.semilogy(a*theta,apparent[:,1]*1./32,"k:",lw=2.0)
    #plt.semilogy(a*theta,interval_mask,"m",lw=2.0)

    #w a default vline at x=1 that spans the yrange
    
    #p = plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)
    p = plt.axvspan(0, 2.54, facecolor='r', alpha=0.1)
    p = plt.axvspan(2.54, 9, facecolor='b', alpha=0.1)
    p = plt.axvspan(1.63, 1.74, facecolor='y', alpha=0.2)
    p = plt.axvspan(2.54, 2.89, facecolor='y', alpha=0.2)
    p = plt.axvspan(3.47, 3.93, facecolor='y', alpha=0.2)
    p = plt.axvspan(4.43, 4.98, facecolor='y', alpha=0.2)
    p = plt.axvspan(5.42, 5.97, facecolor='y', alpha=0.2)
    p = plt.axvspan(6.4, 7, facecolor='y', alpha=0.2)
    p = plt.axvspan(7.41, 7.98, facecolor='y', alpha=0.2)
    p = plt.axvspan(8.4, 9, facecolor='y', alpha=0.2)
    l = plt.axvline(x=2.54,color="k",lw=2.0,ls="dashed")   
    plt.xlim([0,9])
    plt.ylabel("Jy") 
    plt.title("Ghost flux as a function of $r$")
    plt.legend()          
    plt.show()
    """
    #plt.semilogy(np.arange(len(theta)),result[:,k],c,label=label_v,lw=2.0)
    
    """
    for k in xrange(3):
        if k == 0:
           c = "b"
           label_v = "3C61.1"
        elif k == 1:
           c = "r"
           label_v = "S"
        elif k == 2:
           c = "k"
           label_v = "GHOST"
        plt.semilogy(np.arange(len(theta)),result[:,k],c,label=label_v,lw=2.0)
        plt.hold('on')
        plt.semilogy(np.arange(len(theta)),result_prim[:,k],c+"--",label=label_v+" P",lw=2.0)
        if k <> 2:
           plt.semilogy(np.arange(len(theta)),apparent[:,k],c+":",label=label_v+" A",lw=2.0)
    #plt.semilogy(np.arange(len(theta)),1.0/p_beam[:,2],"c",label="AMP",lw=2.0)
    #plt.semilogy(np.arange(len(theta)),interval_mask,"m",lw=2.0)

    plt.xlabel("$r$ [degrees]")
    plt.ylabel("Jy")
    plt.title("Ghost flux as a function of $r$")
    plt.legend()
    plt.show()

    #for k in xrange(3):
    #    if k == 0:
    #       c = "b"
    #       label_v = "3C61.1"
    #    elif k == 1:
    #       c = "r"
    #       label_v = "S"
    #    elif k == 2:
    #       c = "k"
    #       label_v = "GHOST"
    #    plt.plot(a*theta,result[:,k],c,label=label_v,lw=2.0)
    #    plt.hold('on')
    #    plt.plot(a*theta,result_prim[:,k],c+"--",label=label_v+" P",lw=2.0)
    #    if k <> 2:    
    #       plt.plot(a*theta,apparent[:,k],c+":",label=label_v+" A",lw=2.0)

       
    #plt.semilogy(a*theta,1.0/p_beam[:,2],"c",label="AMP",lw=2.0)
    #plt.plot(a*theta,interval_mask,"m",lw=2.0)
    #plt.show()
    """
    """
    for k in xrange(3):
        if k == 0:
           c = "b"
           label_v = "3C61.1"
        elif k == 1:
           c = "r"
           label_v = "S"
        elif k == 2:
           c = "k"
           label_v = "GHOST"
        plt.plot(a*theta,p_beam[:,k],c,label=label_v,lw=2.0)
        plt.hold('on')
    plt.xlabel("r [degrees]")
    plt.ylabel("Suppression factor") 
    plt.title("Primary beam as a function of $r$")
    plt.legend()
    plt.show()

    sup_ghost = (np.absolute(result[:,1]-apparent[:,1])/apparent[:,1])*100

    plt.plot(a*theta,sup_ghost)

    plt.plot(a*theta,np.ones(sup_ghost.shape)*3,"r")

    plt.show()
    """
    app = apparent[:,1]
    ret = result[:,1]
    ret2 = result_ghost

    #m = 1-1./33.
    #plt.plot(app[app<60],ret[app<60],"x")
    #plt.plot(app[app<60],m*app[app<60],"rx")
    #plt.plot(app[app<60],app[app<60]-ret2[app<60],"gx")
    #plt.show()
    
    sup_ghost = (np.absolute(ret-app)/app)*100
    anti_ghost = (np.absolute(ret2)/app)*100
    r = a*theta

    r_anti = r[d_ghost<9.0]
    anti_ghost = anti_ghost[d_ghost<9.0]

    r_s = r[r<=2.5]
    r_b = r[r>2.5]

    sup_s = sup_ghost[r<=2.5]
    sup_b = sup_ghost[r>2.5]

    plt.plot(r_s[::1],sup_s[::1],"bx")

    sup_mean1 = np.mean(sup_ghost[r<=2.5])
    sup_std1 = np.std(sup_ghost[r<=2.5])
    
    plt.plot(r[r<=2.5],np.ones(r[r<=2.5].shape)*sup_mean1,"b")        
    plt.fill_between(x=r[r<=2.5],y1=(sup_mean1-sup_std1)*np.ones(r[r<=2.5].shape),y2=(sup_mean1+sup_std1)*np.ones(r[r<=2.5].shape),alpha=0.2)
    
    plt.plot(r_b[::3],sup_b[::3],"rx")

    sup_mean2 = np.mean(sup_ghost[r>2.5])
    sup_std2 = np.std(sup_ghost[r>2.5])
    
    plt.plot(r[r>2.5],np.ones(r[r>2.5].shape)*sup_mean2,"r")        
    plt.fill_between(x=r[r>2.5],y1=(sup_mean2-sup_std2)*np.ones(r[r>2.5].shape),y2=(sup_mean2+sup_std2)*np.ones(r[r>2.5].shape),alpha=0.2,color="r")

    #plt.plot(r[::3],anti_ghost[::3],"rx")

    plt.plot(r[r>2.5],np.ones(sup_ghost[r>2.5].shape)*100./32,"k")
    l = plt.axvline(x=2.54,color="k",lw=2.0,ls="dashed")   

    #print "result = ",result[0,0]
    plt.xlabel("$r$ [degrees]")
    plt.ylabel("% of $A_2$")
    plt.show()
    #plt.semilogy(a*theta,1.0/p_beam[:,2],"c",label="AMP",lw=2.0)
    #plt.semilogy(a*theta,p_beam[:,1]/p_beam[:,2],"b",label="AMP",lw=2.0)
    #plt.show()

    r_s = r_anti[r_anti<=2.5]
    r_b = r_anti[r_anti>2.5]

    sup_s = anti_ghost[r_anti<=2.5]
    sup_b = anti_ghost[r_anti>2.5]

    plt.plot(r_s[::1],sup_s[::1],"bx")

    sup_mean1 = np.mean(anti_ghost[r_anti<=2.5])
    sup_std1 = np.std(anti_ghost[r_anti<=2.5])
    
    plt.plot(r_anti[r_anti<=2.5],np.ones(r_anti[r_anti<=2.5].shape)*sup_mean1,"b")        
    plt.fill_between(x=r_anti[r_anti<=2.5],y1=(sup_mean1-sup_std1)*np.ones(r_anti[r_anti<=2.5].shape),y2=(sup_mean1+sup_std1)*np.ones(r_anti[r_anti<=2.5].shape),alpha=0.2)
    
    plt.plot(r_b[::3],sup_b[::3],"rx")

    sup_mean2 = np.mean(sup_ghost[r>2.5])
    sup_std2 = np.std(sup_ghost[r>2.5])
    
    plt.plot(r_anti[r_anti>2.5],np.ones(r_anti[r_anti>2.5].shape)*sup_mean2,"r")        
    plt.fill_between(x=r_anti[r_anti>2.5],y1=(sup_mean2-sup_std2)*np.ones(r_anti[r_anti>2.5].shape),y2=(sup_mean2+sup_std2)*np.ones(r_anti[r_anti>2.5].shape),alpha=0.2,color="r")

    #plt.plot(r[::3],anti_ghost[::3],"rx")

    plt.plot(r_anti[r_anti>2.5],np.ones(anti_ghost[r_anti>2.5].shape)*100./32,"k")
    l = plt.axvline(x=2.54,color="k",lw=2.0,ls="dashed")   
    p = plt.axvspan(1.63, 1.74, facecolor='y', alpha=0.2)
    p = plt.axvspan(2.54, 2.89, facecolor='y', alpha=0.2)
    p = plt.axvspan(3.47, 3.93, facecolor='y', alpha=0.2)
    p = plt.axvspan(4.43, 4.98, facecolor='y', alpha=0.2)
    p = plt.axvspan(5.42, 5.97, facecolor='y', alpha=0.2)
    p = plt.axvspan(6.4, 7, facecolor='y', alpha=0.2)
    p = plt.axvspan(7.41, 7.98, facecolor='y', alpha=0.2)
    p = plt.axvspan(8.4, 9, facecolor='y', alpha=0.2)

    #print "result = ",result[0,0]
    plt.xlabel("$r$ [degrees]")
    plt.ylabel("% of $A_2$")
    plt.ylim([0,80])
    plt.show() 
    

def burn ():
       
        #new_prim = "prim_contour_new.fits"
        #fits_file = "PRIMSPIRAL0000_applied.fits"
        #primary_beam = "prim_contour.fits"
 
        #create_new_primary_beam(fits_file,primary_beam,new_prim)
        
        #plot_suppression()
        #video_experiment_spiral(9,300,type_exp="PRIM",file_name="spiral_stef_mini_300_pybdsm.p")
        #video_experiment_spiral(9,200,type_exp="PRIM",algo="LM",file_name="spiral_lm_mini_200.p",source_extr="PIX")
        #prim_fits = "Lofar_primary_beam.fits"
        #create_primary_beam_cont(prim_fits)
        #experiment_source_suppression(100,file_name="suppression_33.p",algo="STEF",ant=33,source_extr="PIX")
        #experiment_source_suppression(100,file_name="suppression_25.p",algo="STEF",ant=25,source_extr="PIX")
        #experiment_source_suppression(100,file_name="suppression_33_pybdsm.p",algo="STEF",ant=33)
        #experiment_source_suppression(100,file_name="suppression_25_pybdsm.p",algo="STEF",ant=25)

        #experiment_source_suppression(100,file_name="suppression_27_pybdsm.p",algo="STEF",ant=27)
        #experiment_source_suppression(100,file_name="suppression_33_pybdsm.p",algo="STEF",ant=33)
        plot_spiral(file_name="spiral_lm_mini_200.p")
        """
        ms_object = Mset("LOFAR.MS")

        ms_object.extract()

        #point_sources = np.array([(88,0,0),(10,(1*np.pi)/180,(0*np.pi)/180),(10,(-1*np.pi)/180,(0*np.pi)/180)])
        #point_sources_cal = np.array([(88,0,0),(40,(1*np.pi)/180,(0*np.pi)/180)])
        #on_time = np.array([100,70,70])

        #The l and m coordinates of the experiment
        x_3C61 = -1.29697746*(np.pi/180.0)
        y_3C61 = -3.44496443*(np.pi/180.0)
        x_ghost_old = -4.44218777*(np.pi/180.0) #directly from ADAM's measurement
        y_ghost_old = -3.91218349*(np.pi/180.0)
        #x_trans = 1.90553958*(np.pi/180.0)
        #y_trans = -3.09505509*(np.pi/180.0)
        x_trans = 0.5*(np.pi/180.0)
        y_trans = 0.5*(np.pi/180.0)
        x_extra = -5.16028135*(np.pi/180.0)
        y_extra = 4.31160524*(np.pi/180.0)
        x_ghost = -4.4994945*(np.pi/180.0) #inforcing symmetry
        y_ghost = -3.79487377*(np.pi/180.0)


        point_sources = np.array([(88.7,x_3C61,y_3C61),(60,x_trans,y_trans)])
        point_sources_cal = np.array([(88.7,x_3C61,y_3C61)])
        #point_sources = np.array([(88.7,x_3C61,y_3C61),(11,x_ghost,y_ghost),(5,x_trans,y_trans),(30.1,x_extra,y_extra)])
        #point_sources_cal = np.array([(88.7,x_3C61,y_3C61),(50,x_trans,y_trans),(30.1,x_extra,y_extra)])
        on_time = np.array([100,100])
        
        #point_sources = np.array([(88.7,x_3C61,y_3C61),(11,x_ghost,y_ghost),(5,x_trans,y_trans),(42.6,x_extra,y_extra)])
        #point_sources_cal = np.array([(88.7,x_3C61,y_3C61),(50,x_trans,y_trans),(42.6,x_extra,y_extra)])

        mask_sources = point_sources[:,1:]*(180/np.pi)
        mask_cal = point_sources_cal[:,1:]*(180/np.pi)        

        fits_file = "Lofar_primary_beam.fits"

        primary_beam = extract_values_from_fits(fits_file,mask_sources,window=20,pix_deg="PIX",clip=1)
        primary_beam_cal = extract_values_from_fits(fits_file,mask_cal,window=20,pix_deg="PIX",clip=1)

        #primary_beam = np.array([0.8,0.1,0.8,0.6])
        #primary_beam_cal = np.array([0.8,0.8,0.6])

        print "primary_beam = ",primary_beam
 
        #point_sources[:,0] = point_sources[:,0]*primary_beam
        #point_sources_cal[:,0] = point_sources_cal[:,0]*primary_beam_cal
        point_sources[:,0] = point_sources[:,0]*primary_beam[:,0]
        point_sources_cal[:,0] = point_sources_cal[:,0]*primary_beam_cal[:,0]

        print "point_sources = ",point_sources
        print "point_sources_cal = ",point_sources_cal
        
        s = Sky_model(ms_object,point_sources,point_sources_cal,on_time,0.5)
        s.visibility("CORRECTED_DATA")
        s.visibility("MODEL_DATA")
                
        #imager.make_image(column="CORRECTED_DATA",dirty=True,restore=False)
        #v.CALTECH="MODEL"
        #print "im = ",II("${imager.DIRTY_IMAGE}")
        #imager.make_image(column="MODEL_DATA",dirty=True,restore=False)
        #v.CALTECH = ""
        antenna = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS106LBA', 'RS205LBA', 'RS208LBA', 'RS306LBA', 'RS307LBA', 'RS406LBA', 'RS503LBA', 'RS508LBA', 'RS509LBA']
        c = Calibration(ms_object,antenna,point_sources,point_sources_cal,on_time)
        c.read_R(column="CORRECTED_DATA") 
        c.read_M(column="MODEL_DATA")
        c.write_to_MS(column="CORRECTED_DATA",type_w="R")
        options = c.image_advanced_settings(antenna=True,img_nchan=1,img_chanstart=0,img_chanstep=1)

        imager.make_image(column="CORRECTED_DATA",dirty=options,restore=options)
        v.CALTECH="MODEL"
        print "im = ",II("${imager.DIRTY_IMAGE}")
        imager.make_image(column="MODEL_DATA",dirty=options,restore=False)
        v.CALTECH = ""
        c.create_G_LM_phase_only()
        c.write_to_MS(column="CORRECTED_DATA",type_w="GTR")
        v.CALTECH="GT"
        imager.make_image(column="CORRECTED_DATA",dirty=options,restore=options)
        v.CALTECH = ""
        
        fits_file = "./plots-LOFAR/LOFAR.dirty.fits"  
        values_before = extract_values_from_fits(fits_file,mask_sources,window=20,pix_deg="PIX",clip=0)
        fits_file = "./plots-LOFAR/LOFAR_GT.dirty.fits"
        values_cal = extract_values_from_fits(fits_file,mask_sources,window=20,pix_deg="PIX",clip=0)

        print "values_before = ",values_before[:,2]
        print "values_cal = ",values_cal[:,2] 

        #print "values_before_corrected = ",values_before[:,2]/primary_beam
        #print "values_before_cal = ",values_cal[:,2]/primary_beam
        print "values_before_corrected = ",values_before[:,2]/primary_beam[:,0]
        print "values_before_cal = ",values_cal[:,2]/primary_beam[:,0]
           
        prim_fits = "Lofar_primary_beam.fits"
        apply_primary_beam(fits_file,prim_fits,10,"PRIM")
        """ 

