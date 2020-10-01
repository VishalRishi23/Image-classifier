#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import cv2 as cv

train=pd.read_csv('train.csv')
os.chdir('Train Images')
X=train['Image'].values
Xtrain=[]

for i in X:
    im = cv.imread(i)
    Xtrain.append(im)
Xtrain=np.array(Xtrain)


# In[21]:


os.chdir('C:\Windows\System32\ML_PATH')


# In[ ]:

#Compute the SIFT feature descriptors
sift=cv.xfeatures2d.SIFT_create()
a,des_cluster=sift.detectAndCompute(Xtrain[0],None)
des_hist=[des_cluster]
for i in range(1,Xtrain.shape[0]):
    kp,des=sift.detectAndCompute(Xtrain[i],None)
    if type(des) is np.ndarray:
        des_cluster=np.vstack((des_cluster,des))
        des_hist.append(des)
    else:
        print(i)
des_hist=np.array(des_hist)


# In[30]:

#Clustering
from sklearn.cluster import KMeans
cluster=KMeans(800)
cluster.fit(des_cluster)


# In[32]:


bov=[]
for i in range(des_hist.shape[0]):
    hist=[0 for m in range(100)]
    for j in range(des_hist[i].shape[0]):
        a=int(cluster.predict(des_hist[i][j].reshape(1,-1)))
        hist[a]+=1
    
    bov.append(hist)
bov=np.array(bov)  #bov is an array with shape (no. of images,no. of clusters)

