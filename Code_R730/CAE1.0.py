#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #for GPU usage
#import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras import callbacks, optimizers


import DataPreprocessing as DP
from matplotlib import pyplot as plt
from astropy.io import fits
#from skimage import exposure
#------------------------------------------------------------------------#
import time
Tstart = time.time() #Timer start


# In[5]:


import tensorflow


# In[6]:


import tensorflow


# In[ ]:


kinds = ['boss_cv','boss_da+ms','boss_db','boss_db+ms','boss_dq','boss_dz','fgkm','hotstars','wd','wdsb2','yso','hotstars_m']
flux_cv, spectrum_cv = DP.Preprocessing7('/home/njl/ML/optical/'+kinds[0]+'/'+'*.fit')
flux_dams, spectrum_dams = DP.Preprocessing7('/home/njl/ML/optical/'+kinds[1]+'/'+'*.fit')
flux_db, spectrum_db = DP.Preprocessing7('/home/njl/ML/optical/'+kinds[2]+'/'+'*.fit')

flux_db.pop(324)
spectrum_db.pop(324)


# In[115]:


size = (2,4096)
X_train = spectrum_cv+spectrum_dams+spectrum_db
X_train = np.stack(X_train)
X_train = X_train.reshape(size[0],len(X_train), size[1]) #change the shape to NHWC for CAE input
print(X_train.shape) #print information of training samples


# In[113]:


#input_img = Input(shape=(2,4096))
input_spe = Input(shape=(1214,4096))


# In[1]:


# encoded = Dense(units=4096, activation='relu', name='embedding')(input_spe)
# decoded = Dense(4096, activation='sigmoid')(encoded) 
# autoencoder = Model(input_spe, decoded)
# optimizer_adam = optimizers.Adam(lr=0.001)
# autoencoder.compile(optimizer=optimizer_adam, loss='categorical_crossentropy')####loss function
# print(autoencoder.summary())
## END ##


# In[131]:


x = Conv1D(4096, 8, activation='relu', padding='same')(input_spe)#(1214,4096)
x = MaxPooling1D(2, padding='same')(x)#(607,4096)
x = Conv1D(2048, 8, activation='relu', padding='same')(x)#(607,2048)
x = MaxPooling1D(2, padding='same')(x) #(304,2048) 
x = Conv1D(1024, 8, activation='relu', padding='same')(x)#(304,1024)
x = MaxPooling1D(2, padding='same')(x) #(152,1024) 

x = Flatten()(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
encoded = Dense(units=64, activation='relu', name='embedding')(x)
x = Dense(units=128, activation='relu')(encoded)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=512, activation='relu')(x)
x = Reshape((256, 2))(x)


x = Conv1D(1024, 8, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(2048, 8, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(4096, 8, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(4096, 835, activation='sigmoid')(x)
print(decoded.shape)


# In[132]:


autoencoder = Model(input_spe, decoded)
optimizer_adam = optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=optimizer_adam, loss='categorical_crossentropy')####loss function
print(autoencoder.summary())


# In[133]:


Nepochs = 10 #number of epochs for CAE training
tosavemodel = True #if save the trained CAE model
plot_reconstruction = True #if plot the reconstruction comparison
savename = 'AE_reconstruction' #setup if "tosavemodel=True" or "plot_reconstuction=True"


# In[ ]:


Tprocess0 = time.time()
print('\n', '## DATE PREPARATION RUNTIME:', Tprocess0-Tstart) #Timer

## MAIN ##
#training
autoencoder.fit(X_train, X_train,
                epochs=Nepochs,
                shuffle=True)

Tprocess1 = time.time()
print('\n', '## AE TRAINING RUNTIME:', Tprocess1-Tprocess0) #Timer


# In[21]:


if tosavemodel:
    #restore the model
    autoencoder.save(savename + '.h5')

if plot_reconstruction:
    #plot the results
    decoded_imgs = autoencoder.predict(X_train)


# In[ ]:




