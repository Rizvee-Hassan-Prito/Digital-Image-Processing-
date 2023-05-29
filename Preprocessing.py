# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 05:27:09 2023

@author: User
"""


"""
colors=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
# Plot a slice of the image using matplotlib
for j in (colors):
    for i in range(0,130,10):
        plt.imshow(nii_data[:,:,i], cmap=j)
        plt.show()
"""

import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

#%%

### Liver and Liver Mask Images extracting and Augmenting ###
### CLAHE is implemented while extracting ###


#################################################################
"""
for j in range(0,131): #(46,131)
    path_name="LiTS17/segmentations/segmentation-"+str(j)+".nii"
     # Load nii file
    nii_img = nib.load(path_name)
     
     # Get the image data as a numpy array
    nii_data = nii_img.get_fdata()
    

    for i in range(0,75,1):
        print("j=",j,"i=",i)
        
        if (nii_data[:,:,i] == 0).all():
            print("The image is entirely black.")
            

        else:
            print("The image is not entirely black.")
            
            path="LiTS17/volume_pt1/volume-"+str(j)+".nii"
            nii_img2 = nib.load(path)
            nii_data2 = nii_img2.get_fdata()
            image = cv2.normalize(nii_data2[:,:,i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(17,17))
            nii_data2 = clahe.apply(image)
            
            ##################Livers##############
            folder_name="LiTS17/Liver"
            nii_name2="volume-"+str(j)+".nii_"+str(i)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(nii_data2, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name2}.png', transparent=True)
            plt.clf()
            plt.close()
            ################################
            
            rotated_img = np.rot90(nii_data2, k=1)
            nii_name2="volume-"+str(j)+".nii_"+str(i)+"(2)"
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(nii_data2, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name2}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(rotated_img, k=1)
            nii_name2="volume-"+str(j)+".nii_"+str(i)+"(3)"
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name2}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(rotated_img, k=1)
            nii_name2="volume-"+str(j)+".nii_"+str(i)+"(4)"
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name2}.png', transparent=True)
            plt.clf()
            plt.close()
            

            
            
            #######################Liver Masks#####################################
            folder_name="LiTS17/Liver mask"
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(nii_data[:,:,i], cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            ##########################
            
            
            folder_name="LiTS17/Liver mask"
            rotated_img = np.rot90(nii_data[:,:,i], k=1)
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)+"(2)"
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            folder_name="LiTS17/Liver mask"
            rotated_img = np.rot90(rotated_img, k=1)
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)+"(3)"
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            folder_name="LiTS17/Liver mask"
            rotated_img = np.rot90(rotated_img, k=1)
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)+"(4)"
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()

            

"""
#############################################
#%%

### Tumor Mask Images extracting from Liver Mask and Augmenting ###

#################################################################
"""
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

for j in range(46,131): #(46,131)
    path_name="segmentations/segmentation-"+str(j)+".nii"
     # Load nii file
    nii_img = nib.load(path_name)
     
     # Get the image data as a numpy array
    nii_data = nii_img.get_fdata()
    
    
    fig = plt.figure(figsize=(35, 20))
    
    
    for i in range(0,75,1):
        print("i=",i,"j=",j)
        
        
        new_mask = np.zeros_like(nii_data[:,:,i])
        segmented_mask=nii_data[:,:,i]
        
        print(np.unique(segmented_mask))
        
        # Set the threshold for the desired segment
        threshold = 1

        # Iterate through each pixel in the segmented mask image and create the new mask
        for m in range(segmented_mask.shape[0]):
            for n in range(segmented_mask.shape[1]):
                if segmented_mask[m,n] >threshold:
                    new_mask[m,n] = 1
        
        
        if(len(np.unique(segmented_mask))==3):
            print("yes")
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(new_mask, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Tumor Liver"
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(new_mask, k=1)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Tumor Liver"
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)+"(2)"
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(rotated_img, k=1)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Tumor Liver"
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)+"(3)"
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(rotated_img, k=1)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Tumor Liver"
            nii_name="segmentaions-"+str(j)+".nii"+"_"+str(i)+"(4)"
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
"""
#################################################################


#%%


### Liver with Tumor Images extracting and Augmenting ###
### CLAHE is implemented while extracting ###


################################################################################
"""
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

for j in range(0,131): #(46,131)
    path_name="segmentations/segmentation-"+str(j)+".nii"
    
    # Load nii file
    nii_img = nib.load(path_name)
     # Get the image data as a numpy array
    nii_data = nii_img.get_fdata()

    
    fig = plt.figure(figsize=(35, 20))
    
    
    for i in range(0,75,1):
        print("i=",i,"j=",j)
        
        
        new_mask = np.zeros_like(nii_data[:,:,i])
        segmented_mask=nii_data[:,:,i]
        
        print(np.unique(segmented_mask))
        
        # Set the threshold for the desired segment
        threshold = 1

        # Iterate through each pixel in the segmented mask image and create the new mask
        for m in range(segmented_mask.shape[0]):
            for n in range(segmented_mask.shape[1]):
                if segmented_mask[m,n] >threshold:
                    new_mask[m,n] = 1
        
        
        if(len(np.unique(segmented_mask))==3):
            print("yes")
            
            path_name2="volume_pt1/volume-"+str(j)+".nii"
            # Load nii file
            nii_img2 = nib.load(path_name2)
             # Get the image data as a numpy array
            nii_data2 = nii_img2.get_fdata()
            
            nii_data2 = cv2.normalize(nii_data2[:,:,i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            clahe = cv2.createCLAHE(clipLimit=90.0, tileGridSize=(17,17))
            nii_data2 = clahe.apply(nii_data2)
            
            segmented_mask = segmented_mask.astype(np.uint8)
            masked = cv2.bitwise_and(nii_data2, nii_data2, mask=segmented_mask)
            
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(masked, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Segmented Liver with Tumor"
            nii_name="volume-"+str(j)+".nii"+"_"+str(i)
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(masked, k=1)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Segmented Liver with Tumor"
            nii_name="volume-"+str(j)+".nii"+"_"+str(i)+"(2)"
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(rotated_img, k=1)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Segmented Liver with Tumor"
            nii_name="volume-"+str(j)+".nii"+"_"+str(i)+"(3)"
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            rotated_img = np.rot90(rotated_img, k=1)
            fig, ax = plt.subplots(figsize=(35, 20))
            ax.imshow(rotated_img, cmap='gray')
            ax.axis('off')
            plt.axis('off')
            folder_name="Segmented Liver with Tumor"
            nii_name="volume-"+str(j)+".nii"+"_"+str(i)+"(4)"
            plt.savefig(f'{folder_name}/{nii_name}.png', transparent=True)
            plt.clf()
            plt.close()
            
            
        elif (len(np.unique(segmented_mask))==2):
            print("no")
   """
################################################################################