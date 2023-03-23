#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import required mods
import os
import pandas as pd
from fuzzywuzzy import fuzz
import scipy.stats
import numpy as np
from scipy.spatial.distance import hamming
import Levenshtein as LV
import Levenshtein
import os.path
import shutil
import re
import statistics
from glob import glob
# get file path
cpath=input('imagefilepath')     
file1 = glob(f'{cpath}/*.csv') 
for file in file1:
    CK=[]
    DK=[]
    ss=[]
    tt=[]
    L_ratio=[]
    L_distance=[]
    H_distance=[]
    W_Distance=[]
    W_Distance_Chang=[]
    A_ratio=[]
    A_partial_ratio=[]
    D_distance=[]
    K1_distance=[]
    K2_distance=[]
    K3_distance=[]
    K4_distance=[]
    K5_distance=[]
    K6_distance=[]
    K7_distance=[]
    K8_distance=[]
    K9_distance=[]    
    df=pd.read_csv(file)#read CSV
# Get image sorting results
    for i in range(df.shape[0]):
        CC=re.findall(r"\d+",df['x34_rank'][i])
        DD=re.findall(r'\d+',df['t_rank'][i])
        CK.append(CC)
        DK.append(DD)
        basenamess = os.path.basename(file)
        bks = os.path.splitext(basenamess)[0]
# The class that converts the sorting result into a module operation
    df['x34']=CK
    df['t']=DK
    for s in CK:
        results = list(map(int, s))
        ss.append(results)
    for t in DK:
        results = list(map(int, t))
        tt.append(results)
    df['AAA']=ss
    df['BBB']=tt
    for n in range(df.shape[0]):      
        GG=tuple(ss[n])
        GF=tuple(tt[n])
        D1=[str(GG)]
        D2=[str(GF)]
        # levenshtein Distance
        ratio = Levenshtein.ratio(GG,GF)#距離比 =（編輯距離）/（對齊長度）
        dist = Levenshtein.distance(GG,GF)#編輯次數
        L_ratio.append(ratio)
        L_distance.append(dist)


        # Hamming Distance
        hamming_distance = hamming(GG, GF)*len(GG)
        H_distance.append(hamming_distance)

        #Wasserstein Distance
        P = np.array(GG)
        Q = np.array(GF)

        dists=[i for i in range(len(P))]
        Wasserstein_Distance=scipy.stats.wasserstein_distance(dists,dists,P,Q)
        W_Distance.append(dist)

        #Variation Wasserstein Distance *
        c=tuple(abs(GG[i]-GF[i]) for i in range (0,len(GG)))
        l=0
        if range(len(c))==range(0, 4): 
            for discrepancy in c:
                if discrepancy in range(0,1):
                    l=l+discrepancy*1
                if discrepancy in range(1,3):
                    l=l+discrepancy*1
                if discrepancy in range(3,5):
                    l=l+discrepancy*2
            W_Distance_Chang.append(l)
        if range(len(c))==range(0, 8):
            for discrepancy in c:
                if discrepancy in range(0,3):
                    l=l+discrepancy*2
                if discrepancy in range(3,5):
                    l=l+discrepancy*2
                if discrepancy in range(5,9):
                    l=l+discrepancy*3
            W_Distance_Chang.append(l)
        if range(len(c))==range(0, 12):
            for discrepancy in c:
                # print(discrepancy)
                if discrepancy in range(0,4):
                    l=l+discrepancy*3
                if discrepancy in range(4,8):
                    l=l+discrepancy*3
                if discrepancy in range(8,12):
                    l=l+discrepancy*5
            W_Distance_Chang.append(l)

        # Approximate string Distance
        Approximate_string_distance = fuzz.ratio(GG,GF)
        partial_ratio = fuzz.partial_ratio(GG,GF)   
        A_ratio.append(Approximate_string_distance)
        A_partial_ratio.append(partial_ratio)

        # Dice coefficient distance        
        c=bigram_sequence(D1)
        d=bigram_sequence(D2) 
        Dice=dice(c, d)
        D_distance.append(Dice)

    df['L_ratio']=L_ratio
    df['L_distance']=L_distance
    df['H_distance']=H_distance
    df['W_Distance']=W_Distance
    df['W_Distance_Chang']=W_Distance_Chang
    df['A_ratio']=A_ratio
    df['A_partial_ratio']=A_partial_ratio
    df['D_distance']=D_distance
    aa=df.drop(columns=['x34_rank','t_rank','x34','t'])
    
    aa.to_csv(f'{fin}\{bks}_a.csv',index=False)

