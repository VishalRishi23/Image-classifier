#!/usr/bin/env python
# coding: utf-8

# In[15]:

#Import the libraries
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


# In[ ]:





# In[18]:


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


# In[19]:

#Dealing with class imbalance
from sklearn.utils import class_weight
a=class_weight.compute_class_weight('balanced',np.array([0.,1.,2.,3.]),ytrain)


# In[5]:

#Data augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rotation_range=40,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,brightness_range=(0.5,1.5),width_shift_range=0.1,height_shift_range=0.1)
datagen.fit(Xtrain)


# In[23]:

#Custom ResNet 34 network
from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,padding="SAME", use_bias=False)
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
          DefaultConv2D(filters, strides=strides),
          keras.layers.BatchNormalization(),
          self.activation,
          DefaultConv2D(filters),
            
          keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
              DefaultConv2D(filters, kernel_size=1, strides=strides),
              keras.layers.BatchNormalization()]
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


# In[24]:


model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,input_shape=[96, 96, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1024,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(4, activation="softmax"))    
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[8]:


model.load_weights('Image_Classifier.h5')
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[10]:

#Learning rate range test
from keras_lr_finder import LRFinder
lr_finder=LRFinder(model)
outputs=lr_finder.find_generator(datagen.flow(Xtrain,ytrain,batch_size=32,shuffle=True),0.00001,10,validation_data=[Xval,yval],class_weight=a,epochs=4)
lr_finder.plot_loss()


# In[11]:


print(np.amin(lr_finder.losses),lr_finder.lrs[np.argmin(lr_finder.losses)]) # p=0.25 (dropout)


# In[14]:


lr_finder=LRFinder(model)
outputs=lr_finder.find_generator(datagen.flow(Xtrain,ytrain,batch_size=32,shuffle=True),0.00001,10,validation_data=[Xval,yval],class_weight=a,epochs=4)
lr_finder.plot_loss()


# In[15]:


print(np.amin(lr_finder.losses),lr_finder.lrs[np.argmin(lr_finder.losses)]) # p=0.0 (No dropout)


# In[18]:


lr_finder=LRFinder(model)
outputs=lr_finder.find_generator(datagen.flow(Xtrain,ytrain,batch_size=32,shuffle=True),0.00001,10,validation_data=[Xval,yval],class_weight=a,epochs=4)
lr_finder.plot_loss()


# In[19]:


print(np.amin(lr_finder.losses),lr_finder.lrs[np.argmin(lr_finder.losses)]) # p=0.30 (Dropout)


# In[22]:


lr_finder=LRFinder(model)
outputs=lr_finder.find_generator(datagen.flow(Xtrain,ytrain,batch_size=32,shuffle=True),0.00001,10,validation_data=[Xval,yval],class_weight=a,epochs=4)
lr_finder.plot_loss()


# In[23]:


print(np.amin(lr_finder.losses),lr_finder.lrs[np.argmin(lr_finder.losses)]) # p=0.25, wd=0.1 (Dropout and weight decay)


# In[26]:


lr_finder=LRFinder(model)
outputs=lr_finder.find_generator(datagen.flow(Xtrain,ytrain,batch_size=32,shuffle=True),0.00001,10,validation_data=[Xval,yval],class_weight=a,epochs=4)
lr_finder.plot_loss()


# In[27]:


print(np.amin(lr_finder.losses),lr_finder.lrs[np.argmin(lr_finder.losses)]) # p=0.25, wd=0.01 (Dropout and weight decay)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


n_epochs = 20
batch_size=32
def exp(epoch):
    return  0.005*0.1**(epoch/60)
lr=keras.callbacks.LearningRateScheduler(exp)
history = model.fit(Xtrain, ytrain,batch_size=batch_size, epochs=n_epochs, validation_data=[Xval,yval],class_weight=a)          


# In[ ]:


history = model.fit_generator(datagen.flow(Xtrain, ytrain,batch_size=32), epochs=100,initial_epoch=0, validation_data=[Xval,yval],class_weight=a)          


# In[12]:


model.save_weights('Image_Classifier.h5')


# In[ ]:




