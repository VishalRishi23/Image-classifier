#!/usr/bin/env python
# coding: utf-8

# In[20]:


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
    im = cv.resize(im,(96,96),interpolation=cv.INTER_LINEAR)
    Xtrain.append(new)
Xtrain=np.array(Xtrain)    
Xtest,Xtrain=Xtrain[:983],Xtrain[983:]
train[train['Class']=='Food']=0
train[train['Class']=='misc']=1
train[train['Class']=='Attire']=2
train[train['Class']=='Decorationandsignage']=3
ytrain=train['Class'].values
ytest,ytrain=ytrain[:983],ytrain[983:]
os.chdir('C:\Windows\System32\ML_PATH')


# In[21]:


Xtrain.shape


# In[22]:


Xtrain=Xtrain.astype('float32')
Xtest=Xtest.astype('float32')
ytrain=ytrain.astype('float32')
ytest=ytest.astype('float32')
m,s=Xtrain.mean(),Xtrain.std()
def standardize(data):
    mean,std=data.mean(),data.std()
    data=(data-mean)/std
    return data   
Xtrain=standardize(Xtrain)
Xtest=(Xtest-m)/s
Xtrain,Xval=Xtrain[:4000],Xtrain[4000:]
ytrain,yval=ytrain[:4000],ytrain[4000:]


# In[23]:


from sklearn.utils import class_weight
a=class_weight.compute_class_weight('balanced',np.array([0.,1.,2.,3.]),ytrain)


# In[11]:


from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rotation_range=40,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,brightness_range=(0.5,1.5),width_shift_range=0.1,height_shift_range=0.1)
datagen.fit(Xtrain)


# In[24]:

#MobileNet network with pretrained weights
base_model=tf.keras.applications.MobileNetV2(input_shape=[96,96,3],include_top=False,weights='imagenet')


# In[25]:


base_model.trainable=False


# In[26]:


avg=keras.layers.GlobalAveragePooling2D()
d1=keras.layers.Dense(1024,activation='elu',kernel_initializer='he_normal')
bn1=keras.layers.BatchNormalization()
dr1=keras.layers.Dropout(0.50)
d2=keras.layers.Dense(512,activation='elu',kernel_initializer='he_normal')
bn2=keras.layers.BatchNormalization()
dr2=keras.layers.Dropout(0.25)
d3=keras.layers.Dense(256,activation='elu',kernel_initializer='he_normal')
bn3=keras.layers.BatchNormalization()
dr3=keras.layers.Dropout(0.25)
d4=keras.layers.Dense(128,activation='elu',kernel_initializer='he_normal')
bn4=keras.layers.BatchNormalization()
output=keras.layers.Dense(4,activation='softmax')
model=tf.keras.Sequential([base_model,avg,output])
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=1e-3,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])


# In[27]:


n_epochs = 20
batch_size=32
def exp(epoch):
    return  0.005*0.1**(epoch/60)
lr=keras.callbacks.LearningRateScheduler(exp)
history = model.fit(Xtrain, ytrain,batch_size=batch_size, epochs=n_epochs, validation_data=[Xval,yval],class_weight=a)          


# In[9]:


history = model.fit_generator(datagen.flow(Xtrain, ytrain,batch_size=batch_size), epochs=40,initial_epoch=20,validation_data=[Xval,yval],class_weight=a) 


# In[ ]:


model.evaluate(Xtrain,ytrain)


# In[ ]:





# In[ ]:




