#import pyrap.tables
import numpy as np
import pylab as plt
import pickle
import sys

class T_ghost():
    def __init__(self,
		point_sources = np.array([]),
                antenna = "",
                MS=""):
        #self.p_wrapper = Pyxis_helper()
        self.antenna = antenna
        self.A_1 = point_sources[0,0]
        self.A_2 = point_sources[1,0]
        self.l_0 = point_sources[1,1]
        self.m_0 = point_sources[1,2]
        #v.MS = MS
        #v.PICKLENAME = "antnames"
        #file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        #self.ant_names = pickle.load(open(file_name,"rb"))
        self.ant_names = pickle.load(open("KAT7_1445_1x16_12h_antnames.p", "rb"))

        #print "ant_names = ",self.ant_names
        #self.a_list = self.get_antenna(self.antenna,self.ant_names)
        #print "a_list = ",self.a_list

        #v.PICKLENAME = "phi_m"
        #file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        #self.phi_m = pickle.load(open(file_name,"rb"))
        #self.phi_m =  pickle.load(open(MS[2:-4]+"_phi_m.p","rb"))
        self.phi_m = pickle.load(open("KAT7_1445_1x16_12h_phi_m.p", "rb"))

        #v.PICKLENAME = "b_m"
        #file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        #self.b_m = pickle.load(open(file_name,"rb"))
        #self.b_m = pickle.load(open(MS[2:-4]+"_b_m.p","rb"))
        self.b_m = pickle.load(open("KAT7_1445_1x16_12h_b_m.p", "rb"))

        #v.PICKLENAME = "theta_m"
        #file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        #self.theta_m = pickle.load(open(file_name,"rb"))
        #self.theta_m = pickle.load(open(MS[2:-4]+"_theta_m.p","rb"))
        self.theta_m = pickle.load(open("KAT7_1445_1x16_12h_theta_m.p","rb"))

        #v.PICKLENAME = "sin_delta"
        #file_name = self.p_wrapper.pyxis_to_string(v.PICKLEFILE)
        #self.sin_delta = pickle.load(open(file_name,"rb"))
        #self.sin_delta = pickle.load(open(MS[2:-4]+"_sin_delta.p","rb"))
        self.sin_delta = pickle.load(open("KAT7_1445_1x16_12h_sin_delta.p","rb"))

        #print "phi_m = ",self.phi_m
        #print "b_m = ",self.b_m

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
        d_list = list(xrange(len(self.ant_names)))
        for k in range(len(self.a_list)):
            d_list.remove(self.a_list[k])
        return d_list

    def plot_visibilities_pq(self,baseline,u=None,v=None,resolution=0,image_s=0,s=0):
        u_temp = u
        v_temp = v
        u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s)

        uu,vv = np.meshgrid(u,v)

        fig,axs = plt.subplots(2,2)

        cs = axs[1,0].contourf(uu,vv,V_G_pq.real)
        axs[1,0].set_title("Real---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[1,0].set_xlabel("$u$ [1/rad]")
        axs[1,0].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[1,0],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[1,0].plot(u_temp,v_temp,'k')
        cs = axs[1,1].contourf(uu,vv,V_G_pq.imag)
        axs[1,1].set_title("Imaginary---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[1,1].set_xlabel("$u$ [1/rad]")
        axs[1,1].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[1,1],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[1,1].plot(u_temp,v_temp,'k')
        cs = axs[0,0].contourf(uu,vv,V_R_pq.real)
        axs[0,0].set_title("Real---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[0,0].set_xlabel("$u$ [1/rad]")
        axs[0,0].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[0,0],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[0,0].plot(u_temp,v_temp,'k')
        cs = axs[0,1].contourf(uu,vv,V_R_pq.imag)
        axs[0,1].set_title("Imaginary---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
        axs[0,1].set_xlabel("$u$ [1/rad]")
        axs[0,1].set_ylabel("$v$ [1/rad]")
        fig.colorbar(cs,ax=axs[0,1],use_gridspec=True,shrink=0.9) #ax.set_title("extend = %s" % extend)
        if u_temp <> None:
           axs[0,1].plot(u_temp,v_temp,'k')
        plt.tight_layout()
        plt.show()

        if u_temp <> None:
           u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,u=u_temp,v=v_temp,resolution=resolution,image_s=image_s,s=s)
           #baseline_new = [0,0]
           #baseline_new[0] = baseline[1]
           #baseline_new[1] = baseline[0]
           #u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,u=u_temp,v=v_temp,resolution=resolution,image_s=image_s,s=s)
           #V_R_pq = (V_R_pq + V_R_qp)/2

           #V_G_pq = (V_G_pq + V_G_qp)/2
           fig,axs = plt.subplots(2,2)
           print "V_G_pq = ",V_G_pq.real
           axs[1,0].set_title("Real---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[1,0].set_xlabel("Timeslot [n]")
           axs[1,0].set_ylim([0.8,1.2])
           axs[1,0].plot(V_G_pq.real,'k')


           axs[1,1].set_title("Imaginary---$g_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[1,1].set_xlabel("Timeslot [n]")
           axs[1,1].set_ylim([-0.3,0.2])
           axs[1,1].plot(V_G_pq.imag,'k')


           axs[0,0].set_title("Real---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[0,0].set_xlabel("Timeslot [n]")
           axs[0,0].set_ylim([0.8,1.2])
           axs[0,0].plot(V_R_pq.real,'k')


           axs[0,1].set_title("Imaginary---$r_{%s,%s}(X_{%s,%s}^{-1}(u,v))$"%(str(baseline[0]),str(baseline[1]),str(baseline[0]),str(baseline[1])))
           axs[0,1].set_xlabel("Timeslot [n]")
           axs[0,1].plot(V_R_pq.imag,'k')
           axs[0,1].set_ylim([-0.3,0.2])
           plt.tight_layout()
           plt.show()

    def create_mask(self,baseline,plot_v = False):

        point_sources = np.array([(1,0,0)])
        point_sources = np.append(point_sources,[(1,self.l_0,-1*self.m_0)],axis=0)
        point_sources = np.append(point_sources,[(1,-1*self.l_0,1*self.m_0)],axis=0)

        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline,self.ant_names)
        #print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        #print "d_list = ",d_list

        phi = self.phi_m[b_list[0],b_list[1]]
        delta_b = self.b_m[b_list[0],b_list[1]]
        theta = self.theta_m[b_list[0],b_list[1]]


        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################
        if plot_v == True:
           plt.plot(0,0,"rx")
           plt.plot(self.l_0*(180/np.pi),self.m_0*(180/np.pi),"rx")
           plt.plot(-1*self.l_0*(180/np.pi),-1*self.m_0*(180/np.pi),"rx")
        for j in xrange(theta_new.shape[0]):
            for k in xrange(j+1,theta_new.shape[0]):
                if not np.allclose(phi_new[j,k],phi):
                   l_cordinate = phi_new[j,k]/phi*(np.cos(theta_new[j,k]-theta)*self.l_0+self.sin_delta*np.sin(theta_new[j,k]-theta)*self.m_0)
                   m_cordinate = phi_new[j,k]/phi*(np.cos(theta_new[j,k]-theta)*self.m_0-self.sin_delta**(-1)*np.sin(theta_new[j,k]-theta)*self.l_0)
                   if plot_v == True:
                      plt.plot(l_cordinate*(180/np.pi),m_cordinate*(180/np.pi),"rx")
                      plt.plot(-1*l_cordinate*(180/np.pi),-1*m_cordinate*(180/np.pi),"gx")
                   point_sources = np.append(point_sources,[(1,l_cordinate,-1*m_cordinate)],axis=0)
                   point_sources = np.append(point_sources,[(1,-1*l_cordinate,1*m_cordinate)],axis=0)

        return point_sources


    #window is in degrees, l,m,point_sources in radians, point_sources[k,:] ---> kth point source
    def extract_flux(self,image,l,m,window,point_sources,plot):
        window = window*(np.pi/180)
        point_sources_real = np.copy(point_sources)
        point_sources_imag = np.copy(point_sources)
        for k in range(len(point_sources)):
            l_0 = point_sources[k,1]
            m_0 = point_sources[k,2]*(-1)

            l_max = l_0 + window/2.0
            l_min = l_0 - window/2.0
            m_max = m_0 + window/2.0
            m_min = m_0 - window/2.0

            m_rev = m[::-1]

            #ll,mm = np.meshgrid(l,m)

            image_sub = image[:,(l<l_max)&(l>l_min)]
            #ll_sub = ll[:,(l<l_max)&(l>l_min)]
            #mm_sub = mm[:,(l<l_max)&(l>l_min)]

            if image_sub.size <> 0:
               image_sub = image_sub[(m_rev<m_max)&(m_rev>m_min),:]
               #ll_sub = ll_sub[(m_rev<m_max)&(m_rev>m_min),:]
               #mm_sub = mm_sub[(m_rev<m_max)&(m_rev>m_min),:]

            #PLOTTING SUBSET IMAGE
            if plot:
               l_new = l[(l<l_max)&(l>l_min)]
               if l_new.size <> 0:
                  m_new = m[(m<m_max)&(m>m_min)]
                  if m_new.size <> 0:
                     l_cor = l_new*(180/np.pi)
                     m_cor = m_new*(180/np.pi)

                     # plt.contourf(ll_sub*(180/np.pi),mm_sub*(180/np.pi),image_sub.real)
                     # plt.show()

                     #fig = plt.figure()
                     #cs = plt.imshow(mm*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                     #fig.colorbar(cs)
                     #plt.show()
                     #fig = plt.figure()
                     #cs = plt.imshow(ll*(180/np.pi),interpolation = "bicubic", cmap = "jet")
                     #fig.colorbar(cs)
                     #plt.show()

                     fig = plt.figure()
                     cs = plt.imshow(image_sub.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
                     #plt.plot(l_0*(180/np.pi),m_0*(180/np.pi),"rx")
                     fig.colorbar(cs)
                     plt.title("REAL")
                     plt.show()
                     fig = plt.figure()
                     cs = plt.imshow(image_sub.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],l_cor[-1],m_cor[0],m_cor[-1]])
                     fig.colorbar(cs)
                     plt.title("IMAG")
                     plt.show()
            #print "image_sub = ",image_sub
            if image_sub.size <> 0:
               max_v_r = np.amax(image_sub.real)
               max_v_i = np.amax(image_sub.imag)
               min_v_r = np.amin(image_sub.real)
               min_v_i = np.amin(image_sub.imag)
               if np.absolute(max_v_r) > np.absolute(min_v_r):
                  point_sources_real[k,0] = max_v_r
               else:
                  point_sources_real[k,0] = min_v_r
               if np.absolute(max_v_i) > np.absolute(min_v_i):
                  point_sources_imag[k,0] = max_v_i
               else:
                  point_sources_imag[k,0] = min_v_i

            else:
              point_sources_real[k,0] = 0
              point_sources_imag[k,0] = 0

        return point_sources_real,point_sources_imag

    def determine_flux_pos_pq(self,baseline,i_max = 1,number = 10,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="G-1",window=0.2,window_c=0.2):
        l = np.linspace(np.absolute(i_max)*(-1),np.absolute(i_max),number)
        m = np.copy(l)
        ll,mm = np.meshgrid(l,m)
        mask = self.create_mask(baseline)

        l_c_min = window_c*(-1)
        l_c_max = window_c
        m_c_min = window_c*(-1)
        m_c_max = window_c

        result = np.zeros((len(mask),ll.shape[0],ll.shape[1]))

        for i in xrange(ll.shape[0]):
            for j in xrange(ll.shape[1]):
                print "i = ",i
                print "j = ",j

                critical_region = False

                #print "ll[i,j] =", ll[i,j]
                #print "mm[i,j] =", mm[i,j]

                #print "l_c_min = ",l_c_min
                #print "l_c_max = ",l_c_max
                #print "m_c_min = ",m_c_min
                #print "m_c_max = ",m_c_max


                if (ll[i,j] < l_c_max) and (ll[i,j] > l_c_min) and (mm[i,j] < m_c_max) and (mm[i,j] > m_c_min):
                   print "Critical reg"
                   critical_region = True

                if not critical_region:
                   self.l_0 = ll[i,j]*(np.pi/180)
                   self.m_0 = mm[i,j]*(np.pi/180)
                   image,l_v,m_v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
                   mask = self.create_mask(baseline)
                   point_real,point_imag = self.extract_flux(image,l_v,m_v,window,mask,False)
                   result[:,i,j] = point_real[:,0]
                else:
                   result[:,i,j] = 0

        return result,l,m

    def determine_flux_A2(self,A_2_min = 0.001, A_2_max = 0.5,number = 20,resolution=250,image_s=3,s=2,sigma_t=0.05,type_w_t="GT-1",window=0.2):
        A_2_v = np.linspace(A_2_min,A_2_max,number)
        four = np.array([(1,0,0),(1,self.l_0,-1*self.m_0),(1,-1*self.l_0,self.m_0),(1,2*self.l_0,-2*self.m_0)])

        result = np.zeros((len(four),len(A_2_v)))

        for i in xrange(len(A_2_v)):
            print "i = ",i
            print "A_2 = ",A_2_v[i]
            self.A_2 = A_2_v[i]
            image,l_v,m_v = self.sky_2D(resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
            point_real,point_imag = self.extract_flux(image,l_v,m_v,window,four,False)
            result[:,i] = point_real[:,0]
        return result,A_2_v

    def determine_top_k_ghosts(self,k=10,resolution=150,image_s=3,s=2,sigma_t=0.05,type_w_t="GT-1",window=0.2):
        baseline_ind = np.array([])
        len_ant = len(self.a_list)
        baseline = [0,0]

        for i in xrange(len_ant):
            for j in xrange(i+1,len_ant):

                baseline[0] = i
                baseline[1] = j
                print "baseline = ",baseline
                image,l_v,m_v = self.sky_pq_2D(baseline,resolution,image_s,s,sigma = sigma_t, type_w=type_w_t, avg_v=False, plot=False)
                mask = self.create_mask(baseline)
                mask = mask[3:,:]
                mask_temp = np.copy(mask)
                mask_temp[:,1] = mask_temp[:,1]*(180/np.pi)
                mask_temp[:,2] = mask_temp[:,2]*(180/np.pi)
                print "mask = ",mask_temp
                point_real,point_imag = self.extract_flux(image,l_v,m_v,window,mask,False)
                mask_temp = np.copy(point_real)
                mask_temp[:,1] = mask_temp[:,1]*(180/np.pi)
                mask_temp[:,2] = mask_temp[:,2]*(180/np.pi)
                print "point_real = ",mask_temp
                baseline_t = np.zeros((len(mask),2))
                baseline_t[:,0] = i
                baseline_t[:,1] = j

                if (i == 0) and (j == 1):
                   baseline_result = np.copy(baseline_t)
                   point_result = np.copy(point_real)
                else:
                   baseline_result = np.concatenate((baseline_result,baseline_t))
                   point_result = np.concatenate((point_result,point_real))
                print "baseline_result = ",baseline_result
                #print "point_result = ",point_result
        s_index = np.argsort(np.absolute(point_result[:,0]))
        #s_index = s_index[::-1]
        print "s_index = ",s_index
        print "s_index.shape = ",s_index.shape
        point_result = point_result[s_index,:]
        point_result = point_result[::-1,:]
        baseline_result = baseline_result[s_index,:]
        baseline_result = baseline_result[::-1,:]
        print "point_result_fin = ",point_result
        return point_result[0:k,:],baseline_result[0:k,:]

    def runall ():

        p_wrapper = Pyxis_helper()

        ms_object = Mset("KAT7_1445_1x16_12h.ms")

        ms_object.extract()

        #CREATE LINEAR TRANSFORMATION MATRICES
        #*************************************
        #e = Ellipse(ms_object)
        #e.calculate_baseline_trans()
        #e.create_phi_c()
        #*************************************

        point_sources = np.array([(1,0,0),(0.2,(1*np.pi)/180,(0*np.pi)/180)])

        s = Sky_model(ms_object,point_sources)
        #point_sources = np.array([(0.2,0,0),(-0.01,(-1*np.pi/180),(0*np.pi/180)),(0.012,(1*np.pi/180),(-1*np.pi/180)),(-0.1,(-1*np.pi/180),(-1*np.pi/180))])
        #s.meqskymodel(point_sources,antenna=[4,5])

        s.visibility("CORRECTED_DATA")

        c_eig = Calibration(s.ms,"all","eig_cal",s.total_flux) #0A sum of weight must be positive exception..., AB ---> value error
        c_eig.read_R("CORRECTED_DATA")
        c_eig.cal_G_eig()
        c_eig.write_to_MS("CORRECTED_DATA","GTR")

        p_wrapper.pybdsm_search_pq([4,5])

    def create_G_stef(self, N, R, M, temp, imax, tau):
       '''This function finds argmin G ||R-GMG^H|| using StEFCal.
        R is your observed visibilities matrx.
        M is your predicted visibilities.
        imax maximum amount of iterations.
        tau stopping criteria.
        g the antenna gains.
        G = gg^H.'''
       g_temp = np.ones((N,),dtype=complex)
       for k in xrange(imax):
           g_old = np.copy(g_temp)
           for p in xrange(N):
               z = g_old*M[:,p]
               g_temp[p] = np.sum(np.conj(R[:,p])*z)/(np.sum(np.conj(z)*z))
               if (t == 0):
                   if (k == 0):
                       if (p == 0):
                           print "R = ",R[:,:]
                           print "M = ",M[:,:]
                           print "z = ",z
                           print "g_temp[0] = ",g_temp[0]

           if  (k%2 == 0):
               if (np.sqrt(np.sum(np.absolute(g_temp-g_old)**2))/np.sqrt(np.sum(np.absolute(g_temp)**2)) <= tau):
                   break
               else:
                   g_temp = (g_temp + g_old)/2

       G_m = np.dot(np.diag(g_temp),temp)
       G_m = np.dot(G_m,np.diag(g_temp.conj()))

       g = g_temp
       G = G_m

       return g,G

    # resolution --- arcsecond, image_s --- degrees
    def visibilities_pq_2D(self,baseline,u=None,v=None,resolution=0,image_s=0,s=0):
        #SELECTING ONLY SPECIFIC INTERFEROMETERS
        #####################################################
        b_list = self.get_antenna(baseline,self.ant_names)
        #print "b_list = ",b_list
        d_list = self.calculate_delete_list()
        #print "d_list = ",d_list

        phi = self.phi_m[b_list[0],b_list[1]]
        delta_b = self.b_m[b_list[0],b_list[1]]
        theta = self.theta_m[b_list[0],b_list[1]]


        p = np.ones(self.phi_m.shape,dtype = int)
        p = np.cumsum(p,axis=0)-1
        q = p.transpose()

        if d_list == np.array([]):
            p_new = p
            q_new = q
            phi_new = self.phi_m
        else:
            p_new = np.delete(p,d_list,axis = 0)
            p_new = np.delete(p_new,d_list,axis = 1)
            q_new = np.delete(q,d_list,axis = 0)
            q_new = np.delete(q_new,d_list,axis = 1)

            phi_new = np.delete(self.phi_m,d_list,axis = 0)
            phi_new = np.delete(phi_new,d_list,axis = 1)

            b_new = np.delete(self.b_m,d_list,axis = 0)
            b_new = np.delete(b_new,d_list,axis = 1)

            theta_new = np.delete(self.theta_m,d_list,axis = 0)
            theta_new = np.delete(theta_new,d_list,axis = 1)
        #####################################################

        #print "theta_new = ",theta_new
        #print "b_new = ",b_new
        #print "phi_new = ",phi_new
        #print "delta_sin = ",self.sin_delta

        #print "phi = ",phi
        #print "delta_b = ",delta_b
        #print "theta = ",theta*(180/np.pi)

        if u <> None:
           u_dim1 = len(u)
           u_dim2 = 1
           uu = u
           vv = v
           l_cor = None
           m_cor = None
        else:
           # FFT SCALING
           ######################################################
           delta_u = 1/(2*s*image_s*(np.pi/180))
           delta_v = delta_u
           delta_l = resolution*(1.0/3600.0)*(np.pi/180.0)
           delta_m = delta_l
           N = int(np.ceil(1/(delta_l*delta_u)))+1

           if (N % 2) == 0:
              N = N + 1

           delta_l_new = 1/((N-1)*delta_u)
           delta_m_new = delta_l_new
           u = np.linspace(-(N-1)/2*delta_u,(N-1)/2*delta_u,N)
           v = np.linspace(-(N-1)/2*delta_v,(N-1)/2*delta_v,N)
           l_cor = np.linspace(-1/(2*delta_u),1/(2*delta_u),N)
           m_cor = np.linspace(-1/(2*delta_v),1/(2*delta_v),N)
           uu,vv = np.meshgrid(u,v)
           u_dim1 = uu.shape[0]
           u_dim2 = uu.shape[1]
           #######################################################

        #DO CALIBRATION
        #######################################################

        V_R_pq = np.zeros(uu.shape,dtype=complex)
        V_G_pq = np.zeros(uu.shape,dtype=complex)
        temp = np.ones(phi_new.shape ,dtype=complex)

        for i in xrange(u_dim1):
            for j in xrange(u_dim2):
                if u_dim2 <> 1:
                   u_t = uu[i,j]
                   v_t = vv[i,j]
                else:
                   u_t = uu[i]
                   v_t = vv[i]

                #BASELINE CORRECTION (Single operation)
                #####################################################
                #ADDITION
                v_t = v_t - delta_b
                #SCALING
                u_t = u_t/phi
                v_t = v_t/(self.sin_delta*phi)

                #ROTATION (Clockwise)
                u_t_r = u_t*np.cos(theta) + v_t*np.sin(theta)
                v_t_r = -1*u_t*np.sin(theta) + v_t*np.cos(theta)
                #u_t_r = u_t
                #v_t_r = v_t
                #NON BASELINE TRANSFORMATION (NxN) operations
                #####################################################
                #ROTATION (Anti-clockwise)
                u_t_m = u_t_r*np.cos(theta_new) - v_t_r*np.sin(theta_new)
                v_t_m = u_t_r*np.sin(theta_new) + v_t_r*np.cos(theta_new)
                #u_t_m = u_t_r
                #v_t_m = v_t_r
                #SCALING
                u_t_m = phi_new*u_t_m
                v_t_m = phi_new*self.sin_delta*v_t_m
                #ADDITION
                v_t_m = v_t_m + b_new

                #print "u_t_m = ",u_t_m
                #print "v_t_m = ",v_t_m

                R = self.A_1 + self.A_2*np.exp(-2*1j*np.pi*(u_t_m*self.l_0+v_t_m*self.m_0))

                d,Q = np.linalg.eigh(R)
                D = np.diag(d)
                Q_H = Q.conj().transpose()
                abs_d=np.absolute(d)
                index=abs_d.argmax()

                if len(sys.argv) == 2 and sys.argv[1] == "stefcal":
                    N = R.shape[0]
                    # imax = 20
                    # tau = 1e-6

                    M = self.A_1*np.ones(R.shape,dtype=complex)

                    g, G = self.create_G_stef(N, R, M, temp, 20, 1e-6)
                    # g_temp = np.ones((N,),dtype=complex)
                    # for k in xrange(imax):
                    #     g_old = np.copy(g_temp)
                    #     for p in xrange(N):
                    #
                    #         z = g_old*M[:,p]
                    #         g_temp[p] = np.sum(np.conj(R[:,p])*z)/(np.sum(np.conj(z)*z))
                    #         if (t == 0):
                    #             if (k == 0):
                    #                 if (p == 0):
                    #                     print "R = ",R[:,:]
                    #                     print "M = ",M[:,:]
                    #                     print "z = ",z
                    #                     print "g_temp[0] = ",g_temp[0]
                    #
                    #     if  (k%2 == 0):
                    #         if (np.sqrt(np.sum(np.absolute(g_temp-g_old)**2))/np.sqrt(np.sum(np.absolute(g_temp)**2)) <= tau):
                    #             break
                    #         else:
                    #             g_temp = (g_temp + g_old)/2
                    #
                    # G_m = np.dot(np.diag(g_temp),temp)
                    # G_m = np.dot(G_m,np.diag(g_temp.conj()))
                    #
                    # g = g_temp
                    # G = G_m
                else:
                    # g_0 = np.ones((2*N,))
                    # g_0[N:] = 0
                    # r_r = np.ravel(R[:,:,t].real)
                    # r_i = np.ravel(R[:,:,t].imag)
                    # r = np.hstack([r_r,r_i])
                    # m_r = np.ravel(M[:,:,t].real)
                    # m_i = np.ravel(M[:,:,t].imag)
                    # m = np.hstack([m_r,m_i])
                    # g_lstsqr_temp = optimize.leastsq(err_func, g_0, args=(r, m))
                    # g_lstsqr = g_lstsqr_temp[0]
                    #
                    # G_m = np.dot(np.diag(g_lstsqr[0:N]+1j*g_lstsqr[N:]),temp)
                    # G_m = np.dot(G_m,np.diag((g_lstsqr[0:N]+1j*g_lstsqr[N:]).conj()))
                    #
                    # g[:,t] = g_lstsqr[0:N]+1j*g_lstsqr[N:]
                    # G[:,:,t] = G_m
                    if (d[index] >= 0):
                       g=Q[:,index]*np.sqrt(d[index])
                    else:
                       g=Q[:,index]*np.sqrt(np.absolute(d[index]))*1j
                    G = np.dot(np.diag(g),temp)
                    G = np.dot(G,np.diag(g.conj()))
                if self.antenna == "all":
                   if u_dim2 <> 1:
                      V_R_pq[i,j] = R[b_list[0],b_list[1]]
                      V_G_pq[i,j] = G[b_list[0],b_list[1]]
                   else:
                      V_R_pq[i] = R[b_list[0],b_list[1]]
                      V_G_pq[i] = G[b_list[0],b_list[1]]
                else:
                   for k in xrange(p_new.shape[0]):
                       for l in xrange(p_new.shape[1]):
                           if (p_new[k,l] == b_list[0]) and (q_new[k,l] == b_list[1]):
                              if u_dim2 <> 1:
                                 V_R_pq[i,j] = R[k,l]
                                 V_G_pq[i,j] = G[k,l]
                              else:
                                 V_R_pq[i] = R[k,l]+R[l,k]
                                 V_G_pq[i] = G[k,l]+G[l,k]

        #print "V_G_pq = ",V_G_pq
        #if u_dim2 <> 1:
#   p  lt.contourf(uu,vv,V_G_pq)
        #else:
        #   plt.plot(V_G_pq)

        #plt.show()

        return u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor

    def vis_function(self,type_w,avg_v,V_G_pq,V_G_qp,V_R_pq):
        if type_w == "R":
           vis = V_R_pq
        elif type_w == "RT":
           vis = V_R_pq**(-1)
        elif type_w == "R-1":
           vis = V_R_pq - 1
        elif type_w == "RT-1":
           vis = V_R_pq**(-1)-1
        elif type_w == "G":
           if avg_v:
              vis = (V_G_pq+V_G_qp)/2
           else:
              vis = V_G_pq
        elif type_w == "G-1":
           if avg_v:
              vis = (V_G_pq+V_G_qp)/2-1
           else:
              vis = V_G_pq-1
        elif type_w == "GT":
           if avg_v:
              vis = (V_G_pq**(-1)+V_G_qp**(-1))/2
           else:
              vis = V_G_pq**(-1)
        elif type_w == "GT-1":
           if avg_v:
              vis = (V_G_pq**(-1)+V_G_qp**(-1))/2-1
           else:
              vis = V_G_pq**(-1)-1
        elif type_w == "GTR-R":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq-V_R_pq
           else:
              vis = V_G_pq**(-1)*V_R_pq - V_R_pq
        elif type_w == "GTR":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq
           else:
              vis = V_G_pq**(-1)*V_R_pq
        elif type_w == "GTR-1":
           if avg_v:
              vis = ((V_G_pq**(-1)+V_G_qp**(-1))/2)*V_R_pq-1
           else:
              vis = V_G_pq**(-1)*V_R_pq-1
        return vis

    def plt_circle_grid(self,grid_m):
        plt.hold('on')
        rad = np.arange(0.5,0.5+grid_m,0.5)
        x = np.linspace(0,1,500)
        y = np.linspace(0,1,500)

        x_c = np.cos(2*np.pi*x)
        y_c = np.sin(2*np.pi*y)
        for k in range(len(rad)):
            plt.plot(rad[k]*x_c,rad[k]*y_c,"k")

    # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_2D(self,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=True,mask=False):

        ant_len = len(self.a_list)
        counter = 0
        baseline = [0,0]

        for k in xrange(ant_len):
            for j in xrange(k+1,ant_len):
                baseline[0] = self.a_list[k]
                baseline[1] = self.a_list[j]
                counter = counter + 1
                print "counter = ",counter
                print "baseline = ",baseline
                if avg_v:
                   baseline_new = [0,0]
                   baseline_new[0] = baseline[1]
                   baseline_new[1] = baseline[0]
                   u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s)
                else:
                   V_G_qp = 0

                u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s)

                if (k==0) and (j==1):
                   vis = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)
                else:
                   vis = vis + self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)

        vis = vis/counter

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)

        N = l_cor.shape[0]

        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]

        if sigma <> None:

           uu,vv = np.meshgrid(u,v)

           sigma = (np.pi/180) * sigma

           g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))

           vis = vis*g_kernal

           vis = np.roll(vis,-1*(N-1)/2,axis = 0)
           vis = np.roll(vis,-1*(N-1)/2,axis = 1)

           image = np.fft.fft2(vis)*(delta_u*delta_v)
        else:

           image = np.fft.fft2(vis)/N**2


        #ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]
        #image = image[::-1,:]

        #image = (image/1)*100

        if plot:

           l_cor = l_cor*(180/np.pi)
           m_cor = m_cor*(180/np.pi)

           fig = plt.figure()
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           #print "amax = ",np.amax(image.real)
           #print "amax = ",np.amax(np.absolute(image))

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
              self.create_mask_all(plot_v=True)

           #self.create_mask(baseline,plot_v = True)

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()

           fig = plt.figure()
           cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
              self.create_mask_all(plot_v=True)
           #self.create_mask(baseline,plot_v = True)

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()

        return image,l_old,m_old

 # sigma --- degrees, resolution --- arcsecond, image_s --- degrees
    def sky_pq_2D(self,baseline,resolution,image_s,s,sigma = None,type_w="G-1",avg_v=False,plot=False,mask=False):

        if avg_v:
           baseline_new = [0,0]
           baseline_new[0] = baseline[1]
           baseline_new[1] = baseline[0]
           u,v,V_G_qp,V_R_qp,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline_new,resolution=resolution,image_s=image_s,s=s)
        else:
           V_G_qp = 0

        u,v,V_G_pq,V_R_pq,phi,delta_b,theta,l_cor,m_cor = self.visibilities_pq_2D(baseline,resolution=resolution,image_s=image_s,s=s)

        l_old = np.copy(l_cor)
        m_old = np.copy(m_cor)

        N = l_cor.shape[0]

        vis = self.vis_function(type_w,avg_v,V_G_pq,V_G_qp,V_R_pq)

        #vis = V_G_pq-1

        delta_u = u[1]-u[0]
        delta_v = v[1]-v[0]

        if sigma <> None:

           uu,vv = np.meshgrid(u,v)

           sigma = (np.pi/180) * sigma

           g_kernal = (2*np.pi*sigma**2)*np.exp(-2*np.pi**2*sigma**2*(uu**2+vv**2))

           vis = vis*g_kernal

           vis = np.roll(vis,-1*(N-1)/2,axis = 0)
           vis = np.roll(vis,-1*(N-1)/2,axis = 1)

           image = np.fft.fft2(vis)*(delta_u*delta_v)
        else:

           image = np.fft.fft2(vis)/N**2


        #ll,mm = np.meshgrid(l_cor,m_cor)

        image = np.roll(image,1*(N-1)/2,axis = 0)
        image = np.roll(image,1*(N-1)/2,axis = 1)

        image = image[:,::-1]
        #image = image[::-1,:]

        #image = (image/0.1)*100

        if plot:
	   print("1")
           l_cor = l_cor*(180/np.pi)
           m_cor = m_cor*(180/np.pi)

           fig = plt.figure()
           cs = plt.imshow(image.real,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           fig.colorbar(cs)
           self.plt_circle_grid(image_s)

           #print "amax = ",np.amax(image.real)
           #print "amax = ",np.amax(np.absolute(image))

           plt.xlim([-image_s,image_s])
           plt.ylim([-image_s,image_s])

           if mask:
             p = self.create_mask(baseline,plot_v = True)

           for k in xrange(len(p)):
               plt.plot(p[k,1]*(180/np.pi),p[k,2]*(180/np.pi),"kv")

           plt.xlabel("$l$ [degrees]")
           plt.ylabel("$m$ [degrees]")
           plt.show()

           #fig = plt.figure()
           #cs = plt.imshow(image.imag,interpolation = "bicubic", cmap = "jet", extent = [l_cor[0],-1*l_cor[0],m_cor[0],-1*m_cor[0]])
           #fig.colorbar(cs)
           #self.plt_circle_grid(image_s)

           #plt.xlim([-image_s,image_s])
           #plt.ylim([-image_s,image_s])

           #if mask:
           #   self.create_mask(baseline,plot_v = True)

           #plt.xlabel("$l$ [degrees]")
           #plt.ylabel("$m$ [degrees]")
           #plt.show()

        return image,l_old,m_old


if  __name__=="__main__":
       point_sources = np.array([(1,0,0),(0.2,(1*np.pi)/180,(0*np.pi)/180)]) #creates your two point sources
       t = T_ghost(point_sources,"all","KAT7_1445_1x16_12h.ms") #creates a T_ghost object instance
       image,l_v,m_v = t.sky_pq_2D([3,5],250,3,2,sigma = 0.05,type_w="G-1",avg_v=False,plot=True,mask=True)
