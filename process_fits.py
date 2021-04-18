import numpy as np
from pyrap.tables import table
import aplpy
import pylab as plt
from matplotlib.colors import LinearSegmentedColormap


# plot primary beam and spirals
def plot_fits(fits_file="PRIMSPIRAL0007_applied.fits",number=7,max_value=205,c="r"):
    cdict_own = {'red':   ((0.0, 0.0, 0.0),
                   (0.3, 0.0, 1.0),
                   (0.7, 1.0, 0.3),
                   (1.0, 1.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.3, 0.0, 1.0),
                   (0.7, 1.0, 0.0),  
                   (1.0, 0.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.3),
                   (0.3, 1.0, 1.0),
                   (0.7, 1.0, 0.0),  
                   (1.0, 0.0, 1.0))
                 }


    a = 1/(2*np.pi)

    theta = np.linspace(0,2*9*np.pi,200) 

    l = -1 * a * theta*np.cos(theta)
    m = -1 * a * theta*np.sin(theta)

    l = l[number]
    m = m[number]

    r = np.sqrt(l**2 + m**2)

    r_str = "{:.2f}".format(r)

    x_3C61 = -1.29697746
    y_3C61 = -3.44496443
    #x_trans = l[k]*(np.pi/180.0)
    #y_trans = m[k]*(np.pi/180.0)
    dx = l - x_3C61
    dy = m - y_3C61
    x_ghost = x_3C61 - dx
    y_ghost = y_3C61 - dy
    
    ra,dec = convert_to_world_coordinates(np.array([l]),np.array([m]),p_180=False)
    ra_ghost,dec_ghost = convert_to_world_coordinates(np.array([x_ghost]),np.array([y_ghost]),p_180=False)
    ra_3C61,dec_3C61 = convert_to_world_coordinates(np.array([x_3C61]),np.array([y_3C61]),p_180=False)
    
    #plt.plot(l,m)
    #plt.show()

    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict_own)
    plt.register_cmap(cmap=blue_red1)

    gc = aplpy.FITSFigure(fits_file)
    #gc.show_grayscale(vmax=1.0)
    gc.show_colorscale(vmin=0.0,vmax=max_value,cmap="cubehelix")
    gc.show_contour('prim_contour_new.fits',colors="yellow",alpha=0.5,linewidths=0.8,linestyles="dashed")
    #gc.show_colorscale(vmin=0.0,vmax=214,cmap=blue_red1)
    #gc.show_circles(np.array([ra[0],ra_3C61[0],ra_ghost[0]]),np.array([dec[0],dec_3C61[0],dec_ghost[0]]) ,np.array([0.5,0.5,0.5]))
    gc.show_circles([ra[0]],[dec[0]],[0.25],color=c,alpha=0.4)
    gc.show_circles([ra_3C61[0]],[dec_3C61[0]],[0.25],color=c,alpha=0.4)
    gc.show_circles([ra_ghost[0]],[dec_ghost[0]],[0.25],color=c,alpha=0.4)
    gc.add_label(ra[0], dec[0]+0.75, 'U', color="w", alpha=0.4)
    gc.add_label(ra_3C61[0], dec_3C61[0]+0.75, 'M', color="w",alpha=0.4)
    gc.add_label(ra_ghost[0], dec_ghost[0]+0.75, 'G', color="w",alpha=0.4)
    gc.add_grid()
    gc.grid.set_color('white') 
    gc.grid.set_alpha(0.2)
    gc.add_colorbar()
    gc.grid.set_xspacing(35)
    gc.ticks.set_xspacing(35)
    gc.colorbar.set_axis_label_text('Flux (Jy/beam)')
    gc.set_title("$r$ = "+r_str)
    plt.show()
    return ra[0],dec[0],ra_ghost[0],dec_ghost[0]


# plot primary beam and spirals
def plot_fits_adam(U,M,G,fits_file="PRIMSPIRAL0007_applied.fits",c="w"):
   
    gc = aplpy.FITSFigure(fits_file)
    #gc.show_grayscale(vmax=1.0)
    gc.show_colorscale(vmin=0.0,vmax=16.1,cmap="cubehelix")
    #gc.show_contour('prim_contour_new.fits',colors="yellow",alpha=0.5,linewidths=0.8,linestyles="dashed")
    #gc.show_colorscale(vmin=0.0,vmax=214,cmap=blue_red1)
    #gc.show_circles(np.array([ra[0],ra_3C61[0],ra_ghost[0]]),np.array([dec[0],dec_3C61[0],dec_ghost[0]]) ,np.array([0.5,0.5,0.5]))
    gc.show_circles([U[0]],[U[1]],[0.25],color=c,alpha=0.4)
    gc.show_circles([M[0]],[M[1]],[0.25],color=c,alpha=0.4)
    gc.show_circles([G[0]],[G[1]],[0.25],color=c,alpha=0.4)
    gc.add_label(U[0], U[1]+0.4, 'U', color="w", alpha=0.4)
    gc.add_label(M[0], M[1]+0.4, '3C61.1', color="w",alpha=0.4)
    gc.add_label(G[0], G[1]+0.4, 'G', color="w",alpha=0.4)
    gc.add_grid()
    gc.grid.set_color('white') 
    gc.grid.set_alpha(0.2)
    gc.add_colorbar()
    gc.grid.set_xspacing(35)
    gc.ticks.set_xspacing(35)
    gc.colorbar.set_axis_label_text('Flux (Jy/beam)')
    gc.recenter(30.57220833,86.30264722,radius=3)
    #gc.set_title("$r$ = "+r_str)
    plt.show()
    

#l and m in degrees
def convert_to_world_coordinates(l,m,p_180 = True,start_value=19.85):
    #start_value = 19.85
    #start_value = 7
    #start_value = 19.00
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


# plot primary beam and spirals
def plot_pbeam2(ra1,dec1,rag1,decg1,ra2,dec2,rag2,decg2,ra3,dec3,rag3,decg3):
    
    a = 1/(2*np.pi)

    theta = np.linspace(0,2*9*np.pi,1000) 

    l = -1 * a * theta*np.cos(theta)
    m = -1 * a * theta*np.sin(theta)

    x_3C61 = -1.29697746
    y_3C61 = -3.44496443
    #x_trans = l[k]*(np.pi/180.0)
    #y_trans = m[k]*(np.pi/180.0)
    dx = l - x_3C61
    dy = m - y_3C61
    x_ghost = x_3C61 - dx
    y_ghost = y_3C61 - dy

    #plt.plot(l,m)
    #plt.show()

    ra,dec = convert_to_world_coordinates(l,m,p_180=False)
    ra_ghost,dec_ghost = convert_to_world_coordinates(x_ghost,y_ghost,p_180=False)
    ra_3C61,dec_3C61 = convert_to_world_coordinates(np.array([x_3C61]),np.array([y_3C61]),p_180=False)
    
    line_list = np.zeros((2,len(dec)))
    line_list[0,:] = ra
    line_list[1,:] = dec 

    line_list2 = np.zeros((2,len(dec_ghost)))
    line_list2[0,:] = ra_ghost
    line_list2[1,:] = dec_ghost 
      
    #gc = aplpy.FITSFigure('LOFAR2_GT.restored.fits')
    gc = aplpy.FITSFigure('prim_contour_new.fits')
    gc.show_colorscale(vmin=0.0,vmax=1.0,cmap="gist_heat")
    #gc.show_grayscale(vmin=0.0,vmax=1.0)
    gc.show_lines([line_list],linewidth=1.5,color="r",alpha=0.8)
    gc.show_lines([line_list2],linewidth=1.5,color="b",alpha=0.8)
    gc.add_label(ra_3C61[0]+1, dec_3C61[0]+0.5, 'Modelled Source', color="g")
    gc.add_label(line_list[0,0]+270, line_list[1,0]+1.0, 'Unmodelled Source Trajectory', color="r")
    gc.add_label(line_list2[0,0]+1, line_list2[1,0]+0.5, 'Ghost Trajectory', color="b")
    gc.show_markers([ra_3C61[0]], [dec_3C61[0]], marker="x",linewidth=1.0,edgecolor="g")

    gc.show_circles([ra1],[dec1],[0.25],color="c")
    gc.show_circles([rag1],[decg1],[0.25],color="c")

    gc.show_circles([ra2],[dec2],[0.25],color="y")
    gc.show_circles([rag2],[decg2],[0.25],color="y")

    gc.show_circles([ra3],[dec3],[0.25],color="g")
    gc.show_circles([rag3],[decg3],[0.25],color="g")
       
    gc.show_contour('prim_contour_new.fits',colors="yellow",alpha=0.75,linewidths=0.8,linestyles="dashed")
    gc.add_grid()
    gc.grid.set_alpha(0.5)
    #gc.grid.set_xspacing(30)
    gc.tick_labels.set_font(size='small')
    #gc.ticks.set_xspacing(10)
    gc.add_colorbar()
    #gc.colorbar.set_width(0.1)
    #gc.colorbar.set_pad(0.03)
    gc.colorbar.set_ticks([0,0.2,0.4,0.6,0.8,1])
    gc.grid.set_xspacing(35)
    gc.ticks.set_xspacing(35)
    plt.show()

# plot primary beam and spirals
def plot_pbeam(pos=[7,68,77],c=["c","y","g"]):
    
    a = 1/(2*np.pi)

    theta = np.linspace(0,2*9*np.pi,200) 

    l = -1 * a * theta*np.cos(theta)
    m = -1 * a * theta*np.sin(theta)

    x_3C61 = -1.29697746
    y_3C61 = -3.44496443
    #x_trans = l[k]*(np.pi/180.0)
    #y_trans = m[k]*(np.pi/180.0)
    dx = l - x_3C61
    dy = m - y_3C61
    x_ghost = x_3C61 - dx
    y_ghost = y_3C61 - dy



    #plt.plot(l,m)
    #plt.show()

    ra = np.arctan2(m,l)*180/np.pi
    ra_ghost = np.arctan2(y_ghost,x_ghost)*180/np.pi
    ra_3C61 = np.arctan2(y_3C61,x_3C61)*180/np.pi

    ra[ra<0] = ra[ra<0]+360
    ra_ghost[ra_ghost<0] = ra_ghost[ra_ghost<0]+360
    if ra_3C61<0:
       ra_3C61 = ra_3C61 + 360

    ra = 7*15-ra 
    ra_ghost = 7*15-ra_ghost
    ra_3C61 = 7*15 - ra_3C61

    ra[ra<0] = ra[ra<0]+360
    ra_ghost[ra_ghost<0] = ra_ghost[ra_ghost<0]+360
    if ra_3C61<0:
       ra_3C61 = ra_3C61 + 360

    l_rad = l*np.pi/180
    m_rad = m*np.pi/180

    x_ghost_rad = x_ghost*np.pi/180 
    y_ghost_rad = y_ghost*np.pi/180 

    x_3C61_rad = x_3C61*np.pi/180 
    y_3C61_rad = y_3C61*np.pi/180

    d_rad = np.sqrt(l_rad**2+m_rad**2)
    d_ghost_rad = np.sqrt(x_ghost_rad**2+y_ghost_rad**2)    
    d_3C61_rad = np.sqrt(x_3C61_rad**2+y_3C61_rad**2) 

    dec = np.arcsin(d_rad)*180/np.pi
    dec_ghost = np.arcsin(d_ghost_rad)*180/np.pi
    dec_3C61 = np.arcsin(d_3C61_rad)*180/np.pi

    dec = 90-dec  
    dec_ghost = 90-dec_ghost
    dec_3C61 = 90 - dec_3C61   

    #plt.plot(ra,dec)
    #plt.show()

    line_list = np.zeros((2,len(dec)))
    line_list[0,:] = ra
    line_list[1,:] = dec 

    line_list2 = np.zeros((2,len(dec_ghost)))
    line_list2[0,:] = ra_ghost
    line_list2[1,:] = dec_ghost 
      
    #LOFAR2_GT.restored
    #gc = aplpy.FITSFigure('LOFAR2_GT.restored.fits')
    gc = aplpy.FITSFigure('prim_contour_new.fits')
    gc.show_grayscale(vmax=1.0)
    gc.show_lines([line_list],linewidth=1.5,color="r",alpha=0.8)
    gc.show_lines([line_list2],linewidth=1.5,color="b",alpha=0.8)
    gc.add_label(ra_3C61+1, dec_3C61+0.5, 'Modelled Source', color="g")
    gc.add_label(line_list[0,0]+270, line_list[1,0]+1.0, 'Unmodelled Source Trajectory', color="r")
    gc.add_label(line_list2[0,0]+1, line_list2[1,0]+0.5, 'Ghost Trajectory', color="b")
    gc.show_markers([ra_3C61], [dec_3C61], marker="x",linewidth=1.0,edgecolor="g")
    
    gc.show_circles([line_list[0,pos[0]]],[line_list[1,pos[0]]],[0.25],color=c[0],alpha=0.4)
    gc.show_circles([line_list2[0,pos[0]]],[line_list2[1,pos[0]]],[0.25],color=c[0],alpha=0.4)
   
    gc.show_contour('prim_contour_new.fits',colors="yellow",alpha=0.75,linewidths=0.8,linestyles="dashed")
    gc.add_grid()
    gc.grid.set_alpha(0.3)
    #gc.grid.set_xspacing(30)
    gc.tick_labels.set_font(size='small')
    gc.ticks.set_xspacing(20)
    gc.add_colorbar()
    #gc.colorbar.set_width(0.1)
    #gc.colorbar.set_pad(0.03)
    gc.colorbar.set_ticks([0,0.2,0.4,0.6,0.8,1])
    gc.grid.set_xspacing(35)
    gc.ticks.set_xspacing(35)
    plt.show()




if __name__=="__main__":
          
          #plot_fits_adam(fits_file="L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.dppp.dppp.prepeel.2500clip.img.CASACORRECT_InModel.fits")
          #plot_fits_adam(fits_file="L40032_SAP003_SB240_uv.MS.NEW_Feb13_1CHNL.dppp.dppp.prepeel.2500clip.img.CASACORRECT.fits")
          U1 = np.array([1.5397083333,85.45])
          M = np.array([35.5722083333,86.3026472222])
          G1 = np.array([72.5887916667,85.6])
          U2= np.array([340.77675,88.1185805556])
          G2= np.array([49.5287083333,83.3833333333])
          plot_fits_adam(U1,M,G1,fits_file="L43341_frame_3355.NoGhost.fits")
          plot_fits_adam(U2,M,G2,fits_file="L43341_frame_3367.GhostVisible.fits")
          #plot_fits_adam(fits_file="L43341_frame_3368.NoGhost.fits")
 
           
          #plot_pbeam2()
          #ra1,dec1,rag1,decg1 = plot_fits(c="c")
          #ra2,dec2,rag2,decg2 = plot_fits(fits_file="PRIMSPIRAL0068_applied.fits",number=68,max_value=205,c="y")
          #ra3,dec3,rag3,decg3 = plot_fits(fits_file="PRIMSPIRAL0077_applied.fits",number=77,max_value=205,c="g")
          #plot_pbeam2(ra1,dec1,rag1,decg1,ra2,dec2,rag2,decg2,ra3,dec3,rag3,decg3)


 
