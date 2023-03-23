#!/usr/bin/env python
# coding: utf-8

# # import mods

# In[1]:


import cv2
import numpy as np
import os


# # Calculate the main color of each image and extract the background (for xai OCT)

# In[2]:


def Extractbackground(newpath,savepath):
    img=cv2.imread(newpath)

    #Duplicate original image to overwrite
    img_temp = img.copy()

    #Compute Primary Color
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]

    # subtract background
    ooo=cv2.subtract(img,img_temp)

    #Reduce noisy blood vessels
    ooo1=ooo.copy()
    unique, counts = np.unique(ooo1.reshape(-1, 3), axis=0, return_counts=True)
    ooo1[:,:,0], ooo1[:,:,1], ooo1[:,:,2] = [16.60023585, 13.96658805, 11.85154612]
    # subtract background
    ooo=cv2.subtract(img,img_temp)
    cv2.imwrite(savepath,ooo)    


# # Subtract the values of the three-dimensional matrix of the two images and synthesize a new image (for FAG)

# In[3]:


def synthesis(backpath,froontpath,savepath):
    #import image
    img1=cv2.imread(backpath)
    img2=cv2.imread(froontpath)
    # subtract background    
    newimg=cv2.subtract(img1,img2)
    cv2.imwrite(savepath,newimg)

