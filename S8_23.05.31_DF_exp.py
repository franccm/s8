'''s8_210316_dj.py is from s8_210315.py
   imports organized
   with 2 new user inputs at start of execution:
      - ask whether to execute interactive graphics function
      - ask what thresholding method to use
'''


# Standard imports
import numpy as np #best friend
import os #selecting files
import time #timing
import sys #input arguements
import psutil #memory watching
from tqdm import tqdm # progress bar
import tkinter as tk #for gui to select variables
from tkinter import filedialog #to open manager to select folders
import time

# Reading/writing files
import pickle as pkl #writing output file
from pims import TiffStack #reading tif data
from tifffile import imsave #saving tif images
import csv #saving csv files

# Thresholding and cropping
from skimage import morphology #multicrop
from skimage.filters import threshold_triangle, threshold_yen,threshold_otsu, gaussian # adaptive_thresholding
from skimage.measure import label, regionprops # multicrop and three tracker for both
from scipy import stats #adaptive_thresholding but commented out

# Filtering
from scipy.signal import savgol_filter # new function for sav filter
#from math import factorial #used in defined savitzky_golay function
#from skimage.restoration import(denoise_tv_chambolle) #used in savitzky_golay but commented out

# Graphics
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from matplotlib.backends.backend_pdf import PdfPages 
from mpl_toolkits.mplot3d import Axes3D 




# Interactive Graphics
#import pandas as pd 
#from bokeh.plotting import figure, output_file, show
#from bokeh.layouts import row 
#import plotly.express as px 
#import plotly.graph_objects as go 
#import dash_html_components as html 
#from plotly.subplots import make_subplots 

# Not used
#from skimage import util, filters, img_as_uint
#import trackpy as tp 
#import matplotlib.style 
#import math 
#import scipy
#import pims 
#import skimage 
#import matplotlib.patches as mpatches 
#from pims import Frame 
#from PIL import Image 
#import dash 
#import dash_core_components as dcc 
#from dash.dependencies import Input, Output 

#######################################################################################
t_start = time.time()
start_ram = psutil.virtual_memory()[3]/1000000000

def adaptive_thresholding(yval, var_otsu, var_gate, value_inside):

    #tau = np.divide(np.multiply(np.shape(yval)[1],np.shape(yval)[2]),50000) #probably should have a check for image dims...
    #print(tau)
    if var_otsu:
        tau = 0.95
    else:
        tau = 0
    #tau = np.std(yval[0], axis = 0)
    #tau = np.mean(tau)
    #tau = 100*np.divide(tau, data_max)
    #print(tau)
   #---------------------------------------
   #Blur and Noise Gating
   #-------------------------------------------
    yval = np.array(yval, dtype=np.int32)
    #mode filter
    #-----------------------------------------
    #mode = stats.mode(yval, axis = 0)
    #nf = np.clip(yval,a_min = 0, a_max = mode[0][0])
    #rf = stats.mode(nf, axis = 0)
    #yval = yval - np.multiply(rf[0],0.3)
    #---------------------------------------
    #mean filter
    mean_floor = np.mean(yval, axis = 0)
    nf = np.clip(yval, a_min = 0, a_max = mean_floor)
    rf = np.mean(nf, axis = 0)
    rf = np.array(rf, dtype=np.uint32)
    if var_gate:
        yval = yval - np.multiply(rf,1.4)
    else:
        yval = yval
    yval[yval < 0] =  0
    yval = np.array(yval, dtype=np.int16)
    #imsave(raw_path + '/' +filename + '_floor.tif', yval) #save the mask
    #print(nf)
    #yval = yval - np.median(yval, axis = 0) - np.std(yval, axis = 0)
   #---------------------------------------------------------
    ##yval = np.array(yval)
    #imsave(raw_path +"/" + filename + "_filtered.tif", yval)
    print('tau: {0}'.format(str(tau)))
    blurred = gaussian(yval, sigma = (0,tau, tau)) #blur the image
    if value_inside in ['Otsu']:
       gauss_thresh = threshold_otsu(blurred[:]) #find the threshold
    elif value_inside in ['Yen']:
       gauss_thresh = threshold_yen(blurred[:])
    elif value_inside in ['Triangle']:
       gauss_thresh = threshold_triangle(blurred[:])

    binary_gauss = blurred > gauss_thresh #convert to binary
    #binary_gauss_image = np.asarray(binary_gauss, dtype = 'uint8')#convert to boolean #IS THIS USED?
    #imsave(raw_path + '/' +filename + '_binary.tif', binary_gauss_image) #save the mask
    binary_gauss = np.asarray(binary_gauss, dtype = bool) #convert to boolean
    return binary_gauss
def masker(binary_gauss, yval):
    mask = np.multiply(binary_gauss,yval) #make the mask
    m_file = temp_full_filename.replace('.tif','_mask.tif')
    imsave(m_file, mask) #save the mask 
    return mask
def masker_sum(mask):
    sum_mask = np.sum(mask,axis=0,dtype='uint64') #create the summed mask
    #imsave(raw_path + "/" + filename + '_sum_mask.tif', sum_mask)  
    return sum_mask
def particle_sum(binary_gauss):
    sum_particle = np.sum(binary_gauss, axis=0,dtype='uint32') #render the binary #MADE uint64 INCASE DATA FILE IS 32BIT 
    sum_particle[sum_particle != 0] = 255 #scale the binary for viewing #THIS FORCES 8BIT IMAGE, IS THAT OK?
    #imsave(raw_path + "/" + filename + '_sum_particle.tif',sum_particle)
    return sum_particle
def multicrop(sum_particle, binary_gauss, mask):
    label_sum = label(sum_particle)
    label_sum = morphology.remove_small_objects(label_sum,min_size = 10, in_place = True, connectivity = 2)# slice filtering        
    boxes = [region.bbox for region in regionprops(label_sum)]
    boxes = np.asarray(boxes, dtype = int)
    #print(len(boxes))
    #print(label_sum)
    label_sum = np.asarray(label_sum, dtype='uint8') #DONT THINK THIS IS USED AFTER HERE, SO uint8 CAN STAY?
    cropped_mask = {x: mask[:,boxes[x][0]:boxes[x][2],boxes[x][1]:boxes[x][3]] for x in range(len(boxes))}
	
    #[imsave(raw_path +"/" + filename + "_cropped" + str(key) +".tif",cropped_mask[key]) for key in cropped_mask]
    #print(boxes)
    #return np.array(list(cropped_mask[2]))
    return cropped_mask, boxes
    #---------------------------------------------------------------------------------------------------------------
# Three Tracker
#----------------------------------------------------------------------------------------------------------------    
def three_tracker(tal, k, cropped_mask, boxes, raw_path):
    #print("ram before flab:",psutil.virtual_memory()[3]/1000000000-start_ram)
    #intensity_image = cropped_mask
    #imsave(raw_path + '/' +filename + '_intensity.tif', intensity_image) #save the mask
    cropped_binary = cropped_mask.astype(bool).astype(int)
    #cropped_binary = adaptive_thresholding(cropped_mask, 0)
   #problem area
    #cropped_binary[cropped_binary != 0] = 155    
    sum_particle = np.sum(cropped_binary, axis=0,dtype='uint32') #render the binary 
    sum_particle[sum_particle != 0] = 255 #scale the binary for viewing
    sum_mask = np.sum(cropped_mask,axis=0,dtype='uint32') #create the summed mask
    f_lab = label(cropped_binary)
    #print("ram after flab:",psutil.virtual_memory()[3]/1000000000-start_ram) 
    #f_lab = morphology.remove_small_objects(f_lab,min_size = 5000, in_place = True, connectivity = 3)# voxel filtering   
    point = [region.centroid for region in regionprops(f_lab)]
    f_coords = [region.coords for region in regionprops(f_lab)]    
    f_box = [region.bbox for region in regionprops(f_lab)] 
    point = np.asarray(point, dtype = int)
    point = point[:,(1,2)]
    sum_label = label(sum_particle)
    sum_coords = [region.coords for region in regionprops(sum_label)]  
    sum_coords = np.asarray(sum_coords) 
    label_sum = label(sum_particle)
    sites = np.max(label_sum)     
    events = {}  
    serial = {}
    print("time series analysis:")  
    #The splitting code"
    sum_box = np.array([region.bbox for region in regionprops(label_sum)])
    #print(sum_box)
    #print(len(sum_box))    
    obj = {}
    obj = [f_lab[:, sum_box[x,0]:sum_box[x,2], sum_box[x,1]:sum_box[x,3]] for x in range(len(sum_box))] #do the subslicing
    #obj = f_lab
    obj[0] = np.asarray(obj[0], dtype = 'uint8')
    #imsave(raw_path + "/" + filename + "_obj.tif", obj[0])
    #rint(boxes[k-1])
    
    
    for i in tqdm(range(len(point))): 
        t0 = time.time()
        binary_region = f_lab == i+1           
        t1 = time.time()
        #print(t1-t0)
        binary_region[binary_region > 1 ] = 1
        t2 = time.time()
        #print(t2-t1)      
        spacers = np.zeros(((2*len(binary_region)))*len(binary_region[0,:])*len(binary_region[0,0,:]))      
        spacers = spacers.reshape((2*len(binary_region)),len(binary_region[0,:]),len(binary_region[0,0,:]))
        spacers[0::2] = binary_region
        space_lab = label(spacers)
        #space_lab = morphology.remove_small_objects(space_lab,min_size = 500, in_place = True, connectivity = 2)# slice filtering        
        space_lab = space_lab[0::2]   
        particle_mask = np.zeros_like(f_lab)
        particle_mask[f_box[i][0]:f_box[i][3],f_box[i][1]:f_box[i][4],f_box[i][2]:f_box[i][5]] = np.multiply(space_lab[f_box[i][0]:f_box[i][3],f_box[i][1]:f_box[i][4],f_box[i][2]:f_box[i][5]], binary_region[f_box[i][0]:f_box[i][3],f_box[i][1]:f_box[i][4],f_box[i][2]:f_box[i][5]])
        t3 = time.time()
        #print(t3-t2)
        p_amp = [region.mean_intensity for region in regionprops(particle_mask, intensity_image = cropped_mask)]
        #print(p_amp)
        t4 = time.time()
        #print(t4-t3)
        p_area = [region.area for region in regionprops(particle_mask)]  
        t5 = time.time()
        #print(np.max(p_area))
        #print(t5-t4)
        p_cent = [region.centroid for region in regionprops(particle_mask)]
        #p_cent = list(p_cent)
        #print(p_cent[0])
        t6 = time.time()
        #print(t6-t5)
        p_region = [region.coords for region in regionprops(particle_mask)] 
        t7 = time.time()
        #print(t7-t6)
        p_cent = np.asarray(p_cent)
        t8 = time.time()
        #print(t8-t7)
        p_coords = np.vstack(p_region)
        t9 = time.time()
        #print(t9-t8)
        p_region = np.asarray(p_region, dtype=object)
        t10 = time.time()
        #print(t10-t9)
        first_frame  = p_cent[0,0]
        t11 = time.time()
        #print(t11-t10)
        area_freq = np.unique(p_cent[:,0],return_counts = True)
        t12 = time.time()
        #print(t12-t11)
        divergence = False
        convergence = False
        if(np.max(area_freq[1])>1):
           if(np.diff(area_freq[1]).any()>0):
              divergence = True  
           if(np.diff(area_freq[1]).any()<0):
              convergence = True   
        split_status = np.array([divergence,convergence])
        wave = False
        #print(np.diff(p_cent[:,1]))
        if((np.max(p_cent[:,1])-np.min(p_cent[:,1]) or np.max(p_cent[:,2])-np.min(p_cent[:,2]))>5):
          wave = True
          #print(len(regionprops(particle_mask)))
        #imsave(filename + str(i) +"_particle_test", np.asarray(particle_mask,dtype=np.uint8))       
        #for site in range(0,len(sum_coords)):
            #index_match = (point[i] == sum_coords[site]).all(axis = 1).any()                 
             #print("index match: ", index_match) 
        events[tal +i] = np.array([p_area,np.max(p_area),p_cent,len(regionprops(particle_mask)),p_amp,np.max(p_amp)-np.min(p_amp),first_frame,p_region[0],
        p_coords,split_status,area_freq,np.diff(area_freq[1]),wave,np.tile(boxes[k],(len(regionprops(particle_mask)),1)),np.tile(boxes[k],(len(p_coords),1)),
        np.tile(boxes[k],(len(p_region[0]),1))],dtype=object)  
		
    t13 = time.time()
    #events.clear()
    #print(t13-t12)
    #pkl.dump( events, open( raw_path + "/" +filename + "_raw.p", "wb" ) )
    return events, len(point)



def graphics(sites,binary_gauss,sum_particle,sum_mask,mask):
    events = np.array(list(sites.values()),dtype=object)
    areas = np.concatenate(events[:,0]).ravel()
    amps = np.concatenate(events[:,4]).ravel()
    amps = amps.astype(int)
    
    cents = np.concatenate(events[:,2]).ravel()
    cents = cents.reshape(((len(cents)//3),3))
    
    boxes = np.concatenate(events[:,13]).ravel()
    boxes = boxes.reshape(((len(boxes)//4),4))
    
    box_offset = np.concatenate(events[:,14]).ravel()
    box_offset = box_offset.reshape(((len(box_offset)//4),4)) 
     
    cents[:,1] = cents[:,1] + boxes[:,0]
    cents[:,2] = cents[:,2] + boxes[:,1]
    
    cors = np.array(events[:,8])
    coords = np.concatenate(cors).ravel()
    coords = coords.reshape(((len(coords)//3),3))
    
    coords[:,1] = coords[:,1] + box_offset[:,0]
    coords[:,2] = coords[:,2] + box_offset[:,1]
    
   # print("coords: ", coords)
    #print("len: ", len(coords))
    #print("cents: ", cents)
    #print("len: ", len(box_offset))

    starts = np.concatenate(events[:,7]).ravel()
    starts = starts.reshape(((len(starts)//3),3))
    #print(starts)
    start_offset = np.concatenate(events[:,15]).ravel()
    start_offset = start_offset.reshape(((len(start_offset)//4),4)) 
     
    starts[:,1] = starts[:,1] + start_offset[:,0]
    starts[:,2] = starts[:,2] + start_offset[:,1]

    # create color maps so each event gets a distinct color
    event_colors = cm.tab20b(np.linspace(.15, 0.6, len(events)))
    area_colors = []
    coords_colors = []
    coorz_colors = []
    starts_colors = []
    for i in range(len(events)):
        for x in events[i][0]:
            area_colors.append(event_colors[i])
        for x in events[i][8]:
            coords_colors.append(event_colors[i])
        for x in events[i][8]:
            coorz_colors.append(event_colors[i,0])
        for x in events[i][7]:
            starts_colors.append(event_colors[i])

    
    #print(amps)
    
    #print(areas)
    mpl.style.use('tableau-colorblind10')    
    with PdfPages(raw_path + "/" + filename +"_signal_report.pdf") as pdf:    
         mask_page1 = plt.figure(figsize=(16, 9), dpi=300)
         plt.rcParams.update({'font.size': 12})
         plt.subplot(1,3,1)
         plt.scatter(np.random.uniform(0.95,1.05,size=len(events[:,1])),events[:,1],edgecolors='black', c = event_colors) 
         plt.xlim(0.9,1.1)
         plt.xticks([])
         plt.ylabel('maximal area (um^2)')
         plt.subplot(1,3,2)
         plt.title("Signal Descriptors")
         plt.scatter(np.random.uniform(0.95,1.05,size=len(events[:,3])),events[:,3],edgecolors='black', c = event_colors)
         plt.xlim(0.9,1.1)
         plt.xticks([])
         plt.ylabel('duration (s)')
         plt.subplot(1,3,3)
         plt.scatter(np.random.uniform(0.95,1.05,size=len(events[:,5])),events[:,5],edgecolors='black', c = event_colors)
         plt.xlim(0.9,1.1)
         plt.xticks([])
         plt.ylabel('maximal intensity (delta f)')
         #txt="Figure 1. Event parameter strip charts. Maximal area (u^2), total duration (s), and maximal intensity (f) are plotted as points."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         mask_page2 = plt.figure(figsize=(11, 8.5), dpi=300)        
         im2 = plt.imshow(sum_mask,cmap='magma', interpolation='gaussian')
         plt.colorbar(im2)
         plt.title("Signal Time Lapse")
         #txt="Figure 2. Event masks. The filtered and summed image is shown overlayed with its particle mask."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig() 
          #cb.remove()
         mask_page3 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.rcParams.update({'font.size': 12})
         plt.subplot(1,1,1)
         im1 = plt.imshow(sum_particle, cmap = 'inferno')
         cb = plt.colorbar(im1)
         plt.title("Signal Locations")
         #txt="Figure 3. The particle mask."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         mask_page4 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.subplot(2,1,1)
         plt.scatter(cents[:,0],areas,edgecolors='black',linewidths=0.2, c = area_colors)  
         #[plt.scatter(events[key][2][:,0],events[key][0],edgecolors='black',linewidths=0.2) for key in events]
         plt.title("Signal Time Course ")
         plt.ylabel('area (um^2)')
         plt.xlabel('time (s)')
         #plt.ylim(0,np.max(events[:,1])+10)
         plt.xlim(0,len(binary_gauss))
         plt.subplot(2,1,2)
         plt.scatter(cents[:,0],amps,edgecolors='black',linewidths=0.2, c = area_colors)  
         plt.ylabel('intensity (f)')
         plt.xlabel('time (s)')
         #plt.ylim(0,300)
         plt.xlim(0,len(binary_gauss))
         #txt="Figure 4. Area and Intensity versus time plots. The time course of event area and intensity are plotted as colored lines."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         mask_page5 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.rcParams.update({'font.size': 12})
         ax_1 = mask_page5.add_subplot(111, projection='3d',rasterized=True)
         ax_1.scatter3D(cents[:,2],cents[:,0],cents[:,1],c = area_colors) #here is the modified line
         ax_1.set_xlabel('x')
         ax_1.set_ylabel('time (s)')
         ax_1.set_zlabel('y')
         ax_1.set_title("Signal Centers Over Time")
         ax_1.set_zlim(0,len(binary_gauss[0,1]))
         ax_1.set_xlim(0,len(binary_gauss[0,0]))
         #ax_1.set_facecolor('w')
         plt.gca().invert_zaxis()
         #ax_1.grid(b=None)
         #txt="Figure 5. 3D plot of events centroids over time. The center of each event over time is plotted as 3D lines."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         mask_page6 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.rcParams.update({'font.size': 12})
         ax_2 = mask_page6.add_subplot(111, projection='3d', rasterized=True)
         #ax_2.plot3D(coords[:,2],coords[:,0],coords[:,1],linewidth = 0.1) #here si the modified line
         ax_2.scatter3D(coords[:,2],coords[:,0],coords[:,1], c = coords_colors) #here is the modified line
         ax_2.set_xlabel('x')
         ax_2.set_ylabel('time (s)')
         ax_2.set_zlabel('y')
         ax_2.set_title("Signals Over Time")
         ax_2.set_zlim(0,len(binary_gauss[0,1]))
         ax_2.set_xlim(0,len(binary_gauss[0,0]))
         #ax_1.set_facecolor('w')
         plt.gca().invert_zaxis()
         #ax_1.grid(b=None)
         #txt="Figure 5. 3D plot of events centroids over time. The center of each event over time is plotted as 3D lines."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         mask_page7 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.rcParams.update({'font.size': 12})
         plt.subplot(1,1,1)
         plt.scatter(starts[:,2],starts[:,1],edgecolors='black',c = starts_colors) 
         plt.ylim(0,len(binary_gauss[0,1]))
         plt.xlim(0,len(binary_gauss[0,0]))
         plt.gca().invert_yaxis()
         plt.title("Signal Origination Sites")
         #txt="Figure 6. Event origination sites. The coordinates of the first frame of each event is shown as a colored scatter plot."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         #mask_page8 = plt.figure(figsize=(11, 8.5), dpi=300)
         #[plt.scatter(events[key][8][:,2],events[key][8][:,1]) for key in events]
         #plt.ylim(0,len(binary_gauss[0,1]))
         #plt.xlim(0,len(binary_gauss[0,0]))
         #plt.gca().invert_yaxis()
         #txt=""
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         #pdf.savefig()
         #------------------------------------------
         # untouched half)         
         #mask_page9 = plt.figure(figsize=(11, 8.5), dpi=300)
         #for key in events:
         #    spectrum = np.real(np.fft.fft(events[key][4]))
         #    frequencies = np.real(np.fft.fftfreq(len(spectrum)))
         #    plt.plot(abs(frequencies),spectrum)
         #    plt.ylim(0,1000)
         #    plt.xlim(0,0.5)
         #plt.ylabel('Intensity')
         #plt.xlabel('Frequency')
         #plt.title("Signal Intensity Fourier Frequencies")
         #pdf.savefig() 
         #for key in events:
         #    if(events[key][9].any()== True):
         #       mask_page10 = plt.figure(figsize=(11, 8.5), dpi=300)
         #       plt.rcParams.update({'font.size': 12})
         #       ax_3 = mask_page10.add_subplot(111, projection='3d')  
         #       ax_3.plot3D(events[key][8][:,2], events[key][8][:,0],events[key][8][:,1])
         #       ax_3.set_xlabel('x')
         #       ax_3.set_ylabel('time (s)')
         #       ax_3.set_zlabel('y')
         #       ax_3.set_title("Divergent/Convergent Signals")
         #       ax_3.set_zlim(0,len(binary_gauss[0,1]))
         #       ax_3.set_xlim(0,len(binary_gauss[0,0]))
         #       plt.gca().invert_zaxis()
        # pdf.savefig()  
        # for key in events:
         #    if(events[key][12] == True):
         #       mask_page11 = plt.figure(figsize=(11, 8.5), dpi=300)
         #       plt.rcParams.update({'font.size': 12})
         #       ax_4 = mask_page11.add_subplot(111, projection='3d')  
         #       ax_4.plot3D(events[key][8][:,2], events[key][8][:,0],events[key][8][:,1])
         #       ax_4.set_xlabel('x')
         #       ax_4.set_ylabel('time (s)')
         #       ax_4.set_zlabel('y')
         #       ax_4.set_title("Propagating Waves")
         #       ax_4.set_zlim(0,len(binary_gauss[0,1]))
         #       ax_4.set_xlim(0,len(binary_gauss[0,0]))
         #       plt.gca().invert_zaxis()
         #pdf.savefig()      
         #mask_page13 = plt.figure(figsize=(11, 8.5), dpi=300)
         #plt.rcParams.update({'font.size': 12})
         #ax_5 = mask_page13.add_subplot(111, projection='3d')  
         #ax_5.scatter3D(events[:,1], events[:,3],events[:,5],c = range(len(events[:,1])),s=80) 
         #ax_5.set_xlabel('maximal area (um^2)')
         #ax_5.set_ylabel('duration (s)')
         #ax_5.set_zlabel('maximal intensity (f)')
         #ax_5.set_zlim(0,len(binary_gauss[0,1]))
         #ax_5.set_title("signal scatter")    
         #pdf.savefig()      
         plt.close('all')
    return
def graphics_inter(sites, binary_gauss, sum_particle, sum_mask, mask):
    events = np.array(list(sites.values()),dtype=object)
    areas = np.concatenate(events[:,0]).ravel()
    amps = np.concatenate(events[:,4]).ravel()
    amps = amps.astype(int)
    
    cents = np.concatenate(events[:,2]).ravel()
    cents = cents.reshape(((len(cents)//3),3))
    
    boxes = np.concatenate(events[:,13]).ravel()
    boxes = boxes.reshape(((len(boxes)//4),4))
    
    box_offset = np.concatenate(events[:,14]).ravel()
    box_offset = box_offset.reshape(((len(box_offset)//4),4)) 
     
    cents[:,1] = cents[:,1] + boxes[:,0]
    cents[:,2] = cents[:,2] + boxes[:,1]
    
    cors = np.array(events[:,8])
    coords = np.concatenate(cors).ravel()
    coords = coords.reshape(((len(coords)//3),3))
    
    coords[:,1] = coords[:,1] + box_offset[:,0]
    coords[:,2] = coords[:,2] + box_offset[:,1]
    
   # print("coords: ", coords)
    #print("len: ", len(coords))
    #print("cents: ", cents)
    #print("len: ", len(box_offset))

    starts = np.concatenate(events[:,7]).ravel()
    starts = starts.reshape(((len(starts)//3),3))
    #print(starts)
    start_offset = np.concatenate(events[:,15]).ravel()
    start_offset = start_offset.reshape(((len(start_offset)//4),4)) 
     
    starts[:,1] = starts[:,1] + start_offset[:,0]
    starts[:,2] = starts[:,2] + start_offset[:,1]

    # create color maps so each event gets a distinct color
    event_colors = cm.tab20c(np.linspace(0, 1, len(events)))
    area_colors = []
    coords_colors = []
    coorz_colors = []
    starts_colors = []
    for i in range(len(events)):
        for x in events[i][0]:
            area_colors.append(event_colors[i])
        for x in events[i][8]:
            coords_colors.append(event_colors[i])
        for x in events[i][8]:
            coorz_colors.append(event_colors[i,0])
        for x in events[i][7]:
            starts_colors.append(event_colors[i])
            
    # normalize area_colors based on intensity
    max_amp = max(amps) 
    min_amp = min(amps) - 5000
    norm_amps = (amps - min_amp) / (max_amp - min_amp)
    temp = np.array(area_colors).T
    temp = [temp[0],temp[1],temp[2],norm_amps]
    area_colors_norm = np.array(temp).T
    
    # normalize area_colors based on intensity
    max_dur = max(amps) 
    min_dur = min(amps) - 5000
    norm_durs = (amps - min_dur) / (max_dur - min_dur)
    temp = np.array(area_colors).T
    temp = [temp[0],temp[1],temp[2],norm_durs]
    duration_colors_norm = np.array(temp).T

    
    #print(amps)
    #print(areas)
    mpl.style.use('tableau-colorblind10')    
    pdf_file = temp_full_filename.replace('.tif','_report.pdf')
    with PdfPages(pdf_file) as pdf:    
         #mask_page1 = plt.figure(figsize=(16, 9), dpi=300)
         #plt.rcParams.update({'font.size': 12})
         #plt.subplot(1,3,1)
         #plt.scatter(np.random.uniform(0.95,1.05,size=len(events[:,1])),events[:,1],edgecolors='black', c = event_colors) 
         #plt.xlim(0.9,1.1)
         #plt.xticks([])
         #plt.ylabel('maximal area (pixel^2)')
         #plt.subplot(1,3,2)
         #plt.title("Signal Descriptors")
         #plt.scatter(np.random.uniform(0.95,1.05,size=len(events[:,3])),events[:,3],edgecolors='black', c = event_colors)
         #plt.xlim(0.9,1.1)
         #plt.xticks([])
         #plt.ylabel('duration (frames)')
         #plt.subplot(1,3,3)
         #plt.scatter(np.random.uniform(0.95,1.05,size=len(events[:,5])),events[:,5],edgecolors='black', c = event_colors)
         #plt.xlim(0.9,1.1)
         #plt.xticks([])
         #plt.ylabel('maximal intensity (delta f)')
         #txt="Figure 1. Event parameter strip charts. Maximal area (u^2), total duration (s), and maximal intensity (f) are plotted as points."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         #pdf.savefig()
         mask_page2 = plt.figure(figsize=(11, 8.5), dpi=300)        
         im2 = plt.imshow(sum_mask,cmap='magma', interpolation='gaussian')
         #plt.colorbar(im2)
         #plt.title("Signal Time Lapse")
         #txt="Figure 2. Event masks. The filtered and summed image is shown overlayed with its particle mask."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig() 
          #cb.remove()
         mask_page3 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.rcParams.update({'font.size': 12})
         plt.subplot(1,1,1)
         im1 = plt.imshow(sum_particle, cmap = 'inferno')
         cb = plt.colorbar(im1)
         plt.title("Signal Locations")
         #txt="Figure 3. The particle mask."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         mask_page4 = plt.figure(figsize=(11, 8.5), dpi=300)
         plt.subplot(2,1,1)
         plt.scatter(cents[:,0],areas,edgecolors='black',linewidths=0.2, c = area_colors)  
         #[plt.scatter(events[key][2][:,0],events[key][0],edgecolors='black',linewidths=0.2) for key in events]
         plt.title("Signal Time Course ")
         plt.ylabel('area (pixel^2)')
         plt.xlabel('time (frames)')
         #plt.ylim(0,np.max(events[:,1])+10)
         plt.xlim(0,len(binary_gauss))
         plt.subplot(2,1,2)
         plt.scatter(cents[:,0],amps,edgecolors='black',linewidths=0.2, c = area_colors)  
         plt.ylabel('intensity (f)')
         plt.xlabel('time (frames)')
         #plt.ylim(0,300)
         plt.xlim(0,len(binary_gauss))
         #txt="Figure 4. Area and Intensity versus time plots. The time course of event area and intensity are plotted as colored lines."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         pdf.savefig()
         #mask_page5 = plt.figure(figsize=(11, 8.5), dpi=300)
         #plt.rcParams.update({'font.size': 12})
         #ax_1 = mask_page5.add_subplot(111, projection='3d',rasterized=True)
         #ax_1.scatter3D(cents[:,2],cents[:,0],cents[:,1],c = area_colors) #here is the modified line
         #ax_1.set_xlabel('x')
         #ax_1.set_ylabel('time (frames)')
         #ax_1.set_zlabel('y')
         #ax_1.set_title("Signal Centers Over Time")
         #ax_1.set_zlim(0,len(binary_gauss[0,1]))
         #ax_1.set_xlim(0,len(binary_gauss[0,0]))
         #ax_1.set_facecolor('w')
         #plt.gca().invert_zaxis()
         #ax_1.grid(b=None)
         #txt="Figure 5. 3D plot of events centroids over time. The center of each event over time is plotted as 3D lines."
         #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
         #pdf.savefig()
                
         plt.close('all')

    return

def pkl_to_csv(file_name, save_files):
    
    output_data_csv = file_name.replace('_raw.p','_data.csv')
    output_coords_csv = file_name.replace('_raw.p','_coords.csv')
    rois = pkl.load( open (file_name, "rb") )  
    rois = np.array(list(rois.values()),dtype=object)

    column_titles_data = ['ROI_ID','t_centroids [frame]','x_centroid [pixel]','y_centroid [pixel]','Area [pixel^2]','Amplitude',
                          'Starting Frame','Duration [frames]','Max Total Area','Max Amplitude','Converge/Diverge']
    column_titles_coords = ['ROI_ID', 't [frame]', 'x [pixel]', 'y [pixel]']

    all_rows = []
    all_coords = []
    
    all_rows.append(column_titles_data)
    all_coords.append(column_titles_coords)
    
    for i in range(len(rois)):
        
        for j in range(len(rois[i][2])):
                        
            idx = str(i)
            t = str(int(rois[i][2][j][0]))
            box_offset = rois[i][13][j]
            y_c = str(int(rois[i][2][j][1]) + int(box_offset[0]))
            x_c = str(int(rois[i][2][j][2]) + int(box_offset[1]))
            area = str(rois[i][0][j])
            amp = str(int(rois[i][4][j]))
            if j == 0:
                sf = str(rois[i][6])
                duration = str(int(rois[i][3]))
                con_div = '0'
                if rois[i][9][0] != False or rois[i][9][1] != False:
                    con_div = '1'
                max_amp = str(int(max([rois[i][4][x] for x in range(len(rois[i][2]))])))
                max_area = 0
                uniq_t = set([rois[i][2][x][0] for x in range(len(rois[i][2]))])
                for y in uniq_t:
                    areas = [rois[i][0][x] for x in range(len(rois[i][0])) if rois[i][2][x][0] == y]
                    sum_areas = np.sum(areas)
                    if sum_areas > max_area:
                        max_area = sum_areas
                max_area = str(max_area)

                csv_row = [idx, t, x_c, y_c, area, amp, sf, duration, max_area, max_amp, con_div]
            else:
                csv_row = [idx, t, x_c, y_c, area, amp]
            all_rows.append(csv_row)
        
        if save_files[1] == True:
            x_offset = rois[i][14][:,1]
            y_offset = rois[i][14][:,0]
            roi_ts = rois[i][8][:,0]
            roi_xs = rois[i][8][:,2] + x_offset
            roi_ys = rois[i][8][:,1] + y_offset
            roi_coords = [[str(i),roi_ts[j],roi_xs[j],roi_ys[j]] for j in range(len(roi_ts))]
            all_coords.extend(roi_coords)
    if save_files[0] == True:
        with open(output_data_csv, 'w', newline='') as csv_file:
            csvwriter = csv.writer(csv_file,delimiter=',')
            csvwriter.writerows(all_rows)
    
    if save_files[1] == True:
        with open(output_coords_csv, 'w', newline='') as csv_file:
            csvwriter = csv.writer(csv_file,delimiter=',')
            csvwriter.writerows(all_coords)
    
def get_s8_inputs():
    root = tk.Tk()
    root.title('S8 Menu        ')
    
    def help_menu():
        os.system('gedit Info/help.txt')


    menubar = tk.Menu(root)
    f = tk.Menu(menubar, tearoff=1)
    #f.add_command(label='Help',command=help_menu)
    f.add_command(label='Quit',command=sys.exit)
    menubar.add_cascade(label='File',menu=f)
    root.config(menu=menubar)

    #root.geometry('400x200')
    print('\n') 
    def get_input_directory():
        global input_path
        input_path = filedialog.askdirectory(title='Select INPUT directory')
        print('Input directory set as: {0}'.format(input_path))
        return input_path
    def get_output_directory():
        global output_path
        output_path = filedialog.askdirectory(title='Select OUTPUT directory')
        print('Output directory set as: {0}'.format(output_path))


    tk.Label(root, text='(1) INPUT directory').grid(row=0, sticky=tk.W)
    tk.Button(root, text='Click here', 
              command=get_input_directory).grid(row=1, sticky=tk.N, pady=6)
    tk.Label(root, text='(2) OUTPUT directory').grid(row=2, sticky=tk.W)
    tk.Button(root, text='Click here', 
              command=get_output_directory).grid(row=3, sticky=tk.N, pady=6)

    tk.Label(root, text="(3) Use spatial smoothing:").grid(row=4, sticky=tk.W)
    var_otsu = tk.BooleanVar(value=True)
    tk.Radiobutton(root, text='Yes', variable=var_otsu, value=True).grid(row=5,column=0,sticky=tk.W)
    tk.Radiobutton(root, text='No', variable=var_otsu, value=False).grid(row=5,column=1,sticky=tk.W)
    
    tk.Label(root, text="(4) Use temporal smoothing:").grid(row=9, sticky=tk.W)
    var_savgol = tk.BooleanVar(value=True)
    tk.Radiobutton(root, text='Yes', variable=var_savgol, value=True).grid(row=10,column=0, sticky=tk.W)
    tk.Radiobutton(root, text='No', variable=var_savgol, value=False).grid(row=10,column=1,sticky=tk.W)

    tk.Label(root, text="(5) Use noise gate:").grid(row=12, sticky=tk.W)
    var_gate = tk.BooleanVar(value=True)
    tk.Radiobutton(root, text='Yes', variable=var_gate, value=True).grid(row=13, column=0, sticky=tk.W)
    tk.Radiobutton(root, text='No', variable=var_gate, value=False).grid(row=13, column=1, sticky=tk.W)

    tk.Label(root, text="(6) Adaptive Threshold Method:").grid(row=14,sticky=tk.W)
    options_list = ["Otsu","Triangle","Yen"]
    value_inside = tk.StringVar(root)
    value_inside.set("Otsu")
    tk.OptionMenu(root,value_inside,*options_list).grid(row=15,column=0,sticky=tk.W)



    tk.Label(root, text="(7) Select csv files to save").grid(row=16, sticky=tk.W)
    var1 = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text='ROI Data file', variable=var1).grid(row=17,column=0, sticky=tk.W)
    var2 = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Coordinate file", variable=var2).grid(row=18,column=0, sticky=tk.W)

    #tk.Label(root, text="(8) Click submit").grid(row=15, sticky=tk.W)
    tk.Button(root, text='Submit', command=root.destroy).grid(row=19, sticky=tk.N, pady=6)
    tk.mainloop()
    try:
        return input_path, output_path, var_otsu.get(), var_savgol.get(),  var_gate.get(), value_inside.get(), var1.get(), var2.get()
    except:
        print('\n')
        print('NO DIRECTORY SELECTED, TRY AGAIN')
        print('\n')
        print('Terminal will close in:')
        for i in range(5,0,-1):
            print(i)
            time.sleep(1)
        print('Goodbye')
        sys.exit()
    
print('Welcome to the S8 signal processing code!')
folder_path, output_path, var_otsu, var_savgol, var_gate, value_inside, save_csv1, save_csv2 = get_s8_inputs()
raw_path = output_path
save_csvs = [save_csv1, save_csv2]
print(save_csvs)

#print("folder path:", folder_path)
#print("raw path:", output_path)
for filename in os.listdir(folder_path):

    # Only select tif files in folder_path directory
    if filename.endswith('.tif') and os.path.isfile('{0}/{1}'.format(folder_path,filename)):
        print("\n" + filename)
        temp_full_filename = output_path + '/' + filename
        p_file = temp_full_filename.replace('.tif', '_raw.p')
        if os.path.isfile(p_file):
            continue

        images = TiffStack(folder_path +"/" +filename)
        yval = np.array(images)
        data_type = type(yval[0][0][0])
        print("val_inside:", value_inside)
        #print("image ram:", psutil.virtual_memory()[3]/1000000000-start_ram)
        #--------------------------------------------------------------
        #Noise Filtering Module
        #----------------------------------------------------------------
        #yval = np.apply_along_axis(savitzky_golay,0,yval)
        
        if var_savgol:
            #win_size = 2 * int(len(yval) / 5) + 1
            win_size = 49
            yval = savgol_filter(yval,win_size,3,axis=0, mode ='mirror')
        yval = yval-np.amin(yval, axis = 0)
        #yval[yval < 0] =  0
        yval = np.array(yval, dtype=data_type)  
    
        imsave(raw_path + '/' + filename + '_filtered_new.tif', yval) #save the filtered image
        #print("filtered ram:",psutil.virtual_memory()[3]/1000000000-start_ram)
        #-------------------------------------------------------------------------------
        #Binarize
        #--------------------------------------------------------------------------------
        binary_gauss = adaptive_thresholding(yval,var_otsu, var_gate, value_inside)

        #print("binary ram:",psutil.virtual_memory()[3]/1000000000-start_ram)
        #-------------------------------------------------------------------------------
        #Masking
        #----------------------------------------------------------------------------------
        mask = masker(binary_gauss, yval)
        
        #print("mask ram:",psutil.virtual_memory()[3]/1000000000-start_ram)
        #----------------------------------------------------------------------------------
        # Summing
        #-------------------------------------------------------------------------------------
        sum_particle = particle_sum(binary_gauss)
        sum_mask = masker_sum(mask)
        #---------------------------------------------------------------------------------------------
        # Splitting
        #----------------------------------------------------------------------------------
        cropped_values = multicrop(sum_particle,binary_gauss,mask)
         
        #----------------------------------------------------------------------------------------
        #Event Tracking
        #-----------------------------------------------------------------------------------------------
        sites = {}
        tal = 0
        for k,v in cropped_values[0].items():
            trk = three_tracker(tal, k, v,cropped_values[1], raw_path)
            sites.update(trk[0])
            tal += trk[1]
            
        #print("events ram:",psutil.virtual_memory()[3]/1000000000-start_ram)       
        #print("time series analysis time:", time.time() -t_start)
        #--------------------------------------------------
        #Write Dictionary
        #------------------------------------------------------------------------
        
        with open(p_file, "wb") as handle:
            pkl.dump(sites, handle, protocol=pkl.HIGHEST_PROTOCOL)
        #------------------------------------------------------------------------
        #Read Dictionary and make plots
        #-------------------------------------------------------------------
        #myDicts = pkl.load( open (raw_path + "/" +filename +"_raw.p", "rb") )  
        myDicts = pkl.load( open (p_file, 'rb'))
        if len(myDicts) != 0:
            
            graphics(myDicts, binary_gauss, sum_particle, sum_mask, mask)
            
            #if save_csvs[0] == True or save_csvs[1] == True:
            #    pkl_to_csv(p_file,  save_csvs)
                
        else:
            print('No sites found')
        
        print("graphics time:",time.time() - t_start) 
        
if save_csvs[0] == True or save_csvs[1] == True:
    for f in os.listdir(output_path):
        full_f = output_path + '/' + f
        if full_f.endswith('.p'):            
            pkl_to_csv(full_f,  save_csvs)
      
print('\n')
input('S8 has finished!  Press enter to close terminal')
