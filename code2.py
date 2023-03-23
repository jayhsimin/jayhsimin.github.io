#!/usr/bin/env python
# coding: utf-8

# # Import required mods

# In[1]:


import cv2
import numpy as np
import os
import shutil
import re
import math
import pandas as pd
from glob import glob        


# # Import image data，Then create [4, 8, 12] equal division & absolute path combined with tangent and bevel cut.

# In[2]:


opath=input('imagefilepath')
r4=os.path.join(opath,'4r')
s4=os.path.join(opath,'4s')
r8=os.path.join(opath,'8r')
s8=os.path.join(opath,'8s')
r12=os.path.join(opath,'12r')
s12=os.path.join(opath,'12s')
llfile=[r4,s4,r8,s8,r12,s12]


# # Create a storage folder for output files

# In[3]:


for llfiles in llfile:
    if not os.path.isdir(llfiles):
        os.mkdir(llfiles) 
        
colorfile=os.path.join(opath,'rgbvalue_file')
if not os.path.isdir(colorfile):
    os.mkdir(colorfile)  
    
s_path=os.path.join(opath,'String_sorted_file')
if not os.path.isdir(s_path):
    os.mkdir(s_path)   


# # Cut the image to create a new image for image masking, and pre-set all absolute coordinates (the mask image is calculated with a size of 400*400)

# In[4]:

slice straight cut and cross cut
#cross cut to 4 slice
fr1=np.asarray([[0,0],[0,400],[400,400],[400,0],[200,200]],dtype=np.int32)
fr2=np.asarray([[200,200],[400,0],[0,0],[0,400],[400,400]],dtype=np.int32)
fr3=np.asarray([[200,200],[400,400],[400,0],[0,0],[0,400]],dtype=np.int32)
fr4=np.asarray([[200,200],[0,400],[400,400],[400,0],[0,0]],dtype=np.int32)
#straight cut to 4 slice
fs1=np.asarray([[0,0],[200,0],[200,200],[400,200],[400,400],[0,400]],dtype=np.int32)
fs2=np.asarray([[200,0],[400,0],[400,400],[0,400],[0,200],[200,200]],dtype=np.int32)
fs3=np.asarray([[200,200],[0,200],[0,0],[400,0],[400,400],[200,400]],dtype=np.int32)
fs4=np.asarray([[200,200],[400,200],[400,0],[0,0],[0,400],[200,400]],dtype=np.int32)
#cross cut to 8 slice
er1=np.asarray([[0,0],[0,400],[400,400],[400,0],[200,0],[200,200]],dtype=np.int32)
er2=np.asarray([[0,200],[0,400],[400,400],[400,0],[0,0],[200,200]],dtype=np.int32)
er3=np.asarray([[0,400],[400,400],[400,0],[0,0],[0,200],[200,200]],dtype=np.int32)
er4=np.asarray([[200,400],[400,400],[400,0],[0,0],[0,400],[200,200]],dtype=np.int32)
er5=np.asarray([[400,400],[400,0],[0,0],[0,400],[200,400],[200,200]],dtype=np.int32)
er6=np.asarray([[400,200],[400,0],[0,0],[0,400],[400,400],[200,200]],dtype=np.int32)
er7=np.asarray([[400,0],[0,0],[0,400],[400,400],[400,200],[200,200]],dtype=np.int32)
er8=np.asarray([[200,0],[0,0],[0,400],[400,400],[400,0],[200,200]],dtype=np.int32)
#straight cut to 8 slice
es1=np.asarray([[133,0],[133,133],[0,133],[0,400],[400,400],[400,0]],dtype=np.int32)
es2=np.asarray([[0,133],[133,133],[133,267],[0,267],[0,400],[400,400],[400,0],[0,0]],dtype=np.int32)
es3=np.asarray([[0,267],[133,267],[133,400],[400,400],[400,0],[0,0]],dtype=np.int32)
es4=np.asarray([[267,0],[267,133],[133,133],[133,0],[0,0],[0,400],[400,400],[400,0]],dtype=np.int32)
es5=np.asarray([[267,400],[267,267],[133,267],[133,400],[0,400],[0,0],[400,0],[400,400]],dtype=np.int32)
es6=np.asarray([[400,133],[267,133],[267,0],[0,0],[0,400],[400,400]],dtype=np.int32)
es7=np.asarray([[400,267],[267,267],[267,133],[400,133],[400,0],[0,0],[0,400],[400,400]],dtype=np.int32)
es8=np.asarray([[400,267],[267,267],[267,400],[0,400],[0,0],[400,0]],dtype=np.int32)
#cross cut to 12 slice
tr1=np.asarray([[200,200],[133,0],[0,0],[0,400],[400,400],[400,0],[267,0]],dtype=np.int32)
tr2=np.asarray([[133,0],[200,200],[0,0],[0,400],[400,400],[400,0],[133,0]],dtype=np.int32)
tr3=np.asarray([[0,0],[200,200],[0,133],[0,400],[400,400],[400,0]],dtype=np.int32)
tr4=np.asarray([[0,133],[200,200],[0,267],[0,400],[400,400],[400,0],[0,0]],dtype=np.int32)
tr5=np.asarray([[0,267],[200,200],[0,400],[400,400],[400,0],[400,0],[0,0]],dtype=np.int32)
tr6=np.asarray([[0,400],[200,200],[133,400],[400,400],[400,0],[0,0]],dtype=np.int32)
tr7=np.asarray([[133,400],[200,200],[267,400],[400,400],[400,0],[0,0],[0,400]],dtype=np.int32)
tr8=np.asarray([[267,400],[200,200],[400,400],[400,0],[0,0],[0,400]],dtype=np.int32)
tr9=np.asarray([[400,400],[200,200],[400,267],[400,0],[0,0],[0,400]],dtype=np.int32)
tr10=np.asarray([[400,267],[200,200],[400,133],[400,0],[0,0],[0,400],[400,400]],dtype=np.int32)
tr11=np.asarray([[400,133],[200,200],[400,0],[0,0],[0,400],[400,400]],dtype=np.int32)
tr12=np.asarray([[400,0],[200,200],[267,0],[0,0],[0,400],[400,400]],dtype=np.int32)
#straight cut to 12 slice
ts1=np.asarray([[100,0],[100,100],[0,100],[0,400],[400,400],[400,0]],dtype=np.int32)
ts2=np.asarray([[0,100],[100,100],[100,200],[0,200],[0,400],[400,400],[400,0],[0,0]],dtype=np.int32)
ts3=np.asarray([[0,200],[100,200],[100,300],[0,300],[0,400],[400,400],[400,0],[0,0]],dtype=np.int32)
ts4=np.asarray([[0,300],[100,300],[100,400],[400,400],[400,0],[0,0]],dtype=np.int32)
ts5=np.asarray([[200,0],[200,100],[100,100],[100,0],[0,0],[0,400],[400,400],[400,0]],dtype=np.int32)
ts6=np.asarray([[200,400],[200,300],[100,300],[100,400],[0,400],[0,0],[400,0],[400,400]],dtype=np.int32)
ts7=np.asarray([[300,0],[300,100],[200,100],[200,0],[0,0],[0,400],[400,400],[400,0]],dtype=np.int32)
ts8=np.asarray([[300,400],[300,300],[200,300],[200,400],[0,400],[0,0],[400,0],[400,400]],dtype=np.int32)
ts9=np.asarray([[400,100],[300,100],[300,0],[0,0],[0,400],[400,400]],dtype=np.int32)
ts10=np.asarray([[400,200],[300,200],[300,100],[400,100],[400,0],[0,0],[0,400],[400,400]],dtype=np.int32)
ts11=np.asarray([[400,300],[300,300],[300,200],[400,200],[400,0],[0,0],[0,400],[400,400]],dtype=np.int32)
ts12=np.asarray([[400,300],[300,300],[300,400],[0,400],[0,0],[400,0]],dtype=np.int32)

#Put all mask images into a list
fr=[fr1,fr2,fr3,fr4]
fs=[fs1,fs2,fs3,fs4]
er=[er1,er2,er3,er4,er5,er6,er7,er8]
es=[es1,es2,es3,es4,es5,es6,es7,es8]
tr=[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,tr11,tr12]
ts=[ts1,ts2,ts3,ts4,ts5,ts6,ts7,ts8,ts9,ts10,ts11,ts12]


# # Mask the segmented image

# In[5]:


li=['r','s']
ls=["4","8","12"]
for i in li:
    for j in ls:
        a=f'{i}{j}'        
        if a =='r4':
            b=fr
            c=r4
        if a =='r8':
            b=er 
            c=r8
        if a =='r12':
            b=tr
            c=r12
        if a =='s4':
            b=fs 
            c=s4
        if a =='s8':
            b=es   
            c=s8
        if a =='s12':
            b=ts
            c=s12
        file_names = os.listdir(opath)#遊歷資料夾
        for s in range(1,int(j)+1):   
            for file_name in file_names:
                newpath=os.path.join(opath, file_name)
                if file_name.split('.')[-1] == 'bmp': 
                    basename = os.path.basename(newpath)
                    qq = os.path.splitext(basename)[0]#取得每張影像之檔名
                    img = cv2.imread(newpath)#讀取圖像
                    reimg=cv2.resize(img,(400,400))#將圖像縮放成400*400之大小
                    pt = b[s-1].reshape((-1, 1, 2))
                    test=cv2.fillPoly(reimg,[pt],(0,0,0))#在需處裡的圖像上像將所設定的圖像遮罩座標繪制多邊形進行遮罩
                    cv2.imwrite(f'{c}\{s}_{qq}.bmp', test)


# # Drop the same side images into the same folder

# In[6]:


fanfile=[r4,s4,r8,s8,r12,s12]
for ff in fanfile:
    for root, dirs, files in os.walk(ff):
        for item in files:
            os.path.join(root, item)
            if item.split('.')[-1] == 'bmp':
                slice_num = item.split('_')[0]
                subdir_path = os.path.join(ff, slice_num) 
            #create folder
            if not os.path.exists(subdir_path): 
                os.makedirs(subdir_path)  
    for i in range(len(files)):
        if os.path.isfile(ff):
            pass
        if os.path.isdir(ff):
            old_file_path = os.path.join(ff, files[i])   # Specify the path to move: specific to the folder name
            file_name_number = re.findall(r"\d+", files[i])[0]   # Extract the information to use for matching in filenames
           
            #complete match information
            new_file_path = os.path.join(ff, file_name_number)  # Specify the path to move to: specific to the folder name
            # print("new_file_path:", new_file_path)
            shutil.move(old_file_path, new_file_path)  # move file


# # Calculate the average R, G, B of each pixel in the image

# In[7]:


four=[r4,s4,r8,s8,r12,s12]
for fours in four:
    basename = os.path.basename(fours)
    if '4r' in basename or '4s' in basename:
        a=4
    if '8r' in basename or '8s' in basename:
        a=8
    if '12r' in basename or '12s' in basename:
        a=12
    #Calculate the average pixel of each image R, G, B
    for i in range(1,a+1):
        path=f'{fours}\{i}'
        img_csv=f'{fours}\{i}.csv'
        file_names = os.listdir(path)
        per_image_Rmean = []
        per_image_Gmean = []
        per_image_Bmean = []
        #Create a CSV and set the label name for each column
        with open ( img_csv, 'a') as f:
            f.write(f'id{i}'+','+f'Q{i}_R'+','+f'Q{i}_G'+','+f'Q{i}_B'+','+f'TW{i}'+'\n')

            for file_name in file_names:
                newpath=os.path.join(path, file_name)
                if file_name.split('.')[-1] == 'bmp':                
                    AA=file_name.split('_')[1]
                    BB=file_name.split('_')[2]
                    CC=file_name.split('_')[3]
                    DD=file_name.split('_')[4]
                    f_name=os.path.join(AA+'_'+BB+'_'+CC+'_'+DD)
                    img = cv2.imread(newpath, 1)
                    #Average R, G, B value of each pixel in the image
                    bb=np.mean(img[:,:,0])
                    gg=np.mean(img[:,:,1])
                    rr=np.mean(img[:,:,2])
                    weight=float(rr)*10+float(gg)+float(bb)*5
                    #Print the calculated value to csv
                    f.write( str(f_name)+','+str(rr)+ ',' +str(gg)+ ',' +str(bb)+ ',' +str(weight)+'\n')
                    
                #Clear all VALUE of RBG
                per_image_Rmean.clear()
                per_image_Gmean.clear()
                per_image_Bmean.clear()  
                
    files = glob(f'{fours}\*.csv')    
    #Merge all CSV
    df = pd.concat((pd.read_csv(file, dtype={'id': str}) for file in files), axis='columns',ignore_index=False)
    kk=[]
    for i in range(2,a+1):
        aa=f'id{i}'
        kk.append(aa)
    pp=df.drop(kk,axis=1)
    basename = os.path.basename(fours)
    pp.to_csv(f'{colorfile}/{basename}.csv',index=False)


# # Sort the total value of RGB in each image and print it to csv

# In[8]:


files = glob(f'{colorfile}\*.csv')    
for file in files:
    df = pd.read_csv(file)
    basename = os.path.basename(file)
    filee = os.path.splitext(basename)[0]
    newname=filee+'_s'
    if '4r'  in filee or '4s' in filee:
        pd.DataFrame(df,columns=['id1','TW1','TW2','TW3','TW4'])#4
        rank_list = []
        for ID in df['id1']:
            temp_Q_list = []
            temp_num_list = []
            temp_str = ''
            lis=['TW1','TW2','TW3','TW4']
            for Q in lis:
                temp_Q_list.append(Q)
                temp_num_list.append(df[df['id1']==ID][Q].reset_index(drop=True)[0])
                temp_df = pd.DataFrame({'Q': temp_Q_list, 'num': temp_num_list})
                temp_df = temp_df.sort_values(by = 'num', ascending = False)        
            for string in temp_df['Q']:
                temp_str += string
            rank_list.append(temp_str)
        df['t_rank'] = rank_list
        df.to_csv(f'{s_path}/{newname}.csv',index=False)    
    elif '8r'  in filee or '8s' in filee:
        pd.DataFrame(df,columns=['id1','TW1','TW2','TW3','TW4','TW5','TW6','TW7','TW8'])#8
        rank_list = []
        for ID in df['id1']:
            temp_Q_list = []
            temp_num_list = []
            temp_str = ''
            lis=['TW1','TW2','TW3','TW4','TW5','TW6','TW7','TW8']
            for Q in lis:
                temp_Q_list.append(Q)
                temp_num_list.append(df[df['id1']==ID][Q].reset_index(drop=True)[0])
                temp_df = pd.DataFrame({'Q': temp_Q_list, 'num': temp_num_list})
                temp_df = temp_df.sort_values(by = 'num', ascending = False)        
            for string in temp_df['Q']:
                temp_str += string
            rank_list.append(temp_str)
        df['t_rank'] = rank_list
        df.to_csv(f'{s_path}/{newname}.csv',index=False)  
    elif '12r'  in filee or '12s' in filee:
        pd.DataFrame(df,columns=['id1','TW1','TW2','TW3','TW4','TW5','TW6','TW7','TW8','TW9','TW10','TW11','TW12'])#12
        rank_list = []
        for ID in df['id1']:
            temp_Q_list = []
            temp_num_list = []
            temp_str = ''
            lis=['TW1','TW2','TW3','TW4','TW5','TW6','TW7','TW8','TW9','TW10','TW11','TW12']
            for Q in lis:
                temp_Q_list.append(Q)
                temp_num_list.append(df[df['id1']==ID][Q].reset_index(drop=True)[0])
                temp_df = pd.DataFrame({'Q': temp_Q_list, 'num': temp_num_list})
                temp_df = temp_df.sort_values(by = 'num', ascending = False)        
            for string in temp_df['Q']:
                temp_str += string
            rank_list.append(temp_str)
        df['t_rank'] = rank_list
        df.to_csv(f'{s_path}/{newname}.csv',index=False)        

