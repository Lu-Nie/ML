#!/usr/bin/env python
# coding: utf-8

# In[3]:


import loadspec as ls
import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 
from astropy.io import fits
from astropy.table import Table
from laspec import normalization as norm
import os
import glob


# In[4]:


#Data pre-processing boss_cv
#############norm
def Preprocessing1(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    samples = []
    for input_file in files:
        hdu_list = fits.open(input_file, memmap=True)
        data = Table(hdu_list[1].data)
        flux = np.array(data['flux'])
        wavelength = np.array(10**data['loglam'])
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
        num_end = 0
        num_head = 0
        for i in range(len(wavelength)):
            if wavelength[i] <= 9500:
                num_end += 1
            if wavelength[i] <= 3500:
                num_head += 1
        data_spec = [wavelength[num_head:num_end], flux_norm[num_head:num_end]]
        samples.append(data_spec)
    return samples


# In[5]:


################mastar
def Preprocessing2(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    samples = []
    for input_file in files:
        spec = ls.rdspec(input_file)
        flux = np.array(spec.flux)
        wavelength = np.array(spec.wave)
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
        num_end = 0
        num_head = 0
        for i in range(len(wavelength)):
            if wavelength[i] <= 9500:
                num_end += 1
            if wavelength[i] <= 3500:
                num_head += 1
        data_spec = [wavelength[num_head:num_end], flux_norm[num_head:num_end]]
        samples.append(data_spec)
    return samples


# In[6]:


#Data pre-processing boss_cv
#############norm_SN
def Preprocessing3(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    #spectrum's S/N >= 15
    Better_files = []
    for input_file in files:
        spec = ls.rdspec(input_file)
        if spec.snr[0] >= 15:
            Better_files.append(input_file)
    #read and normalize spectrum
    samples = []
    for input_file in Better_files:
        hdu_list = fits.open(input_file, memmap=True)
        data = Table(hdu_list[1].data)
        flux = np.array(data['flux'])
        wavelength = np.array(10**data['loglam'])
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
        num_end = 0
        num_head = 0
        for i in range(len(wavelength)):
            if wavelength[i] <= 9500:
                num_end += 1
            if wavelength[i] <= 3500:
                num_head += 1
        data_spec = [wavelength[num_head:num_end], flux_norm[num_head:num_end]]
        samples.append(data_spec)
    return samples


# In[7]:


################mastar_SN
def Preprocessing4(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    #spectrum's S/N >= 15
    Better_files = []
    for input_file in files:
        spec = ls.rdspec(input_file)
        if spec.snr[0] >= 15:
            Better_files.append(input_file)
    #read and normalize spectrum
    samples = []
    for input_file in Better_files:
        spec = ls.rdspec(input_file)
        flux = np.array(spec.flux)
        wavelength = np.array(spec.wave)
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
        num_end = 0
        num_head = 0
        for i in range(len(wavelength)):
            if wavelength[i] <= 9500:
                num_end += 1
            if wavelength[i] <= 3500:
                num_head += 1
        data_spec = [wavelength[num_head:num_end], flux_norm[num_head:num_end]]
        samples.append(data_spec)
    return samples


# In[8]:


#Data pre-processing(without cutting) boss_cv
#############norm
def Preprocessing5(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    samples = []
    for input_file in files:
        hdu_list = fits.open(input_file, memmap=True)
        data = Table(hdu_list[1].data)
        flux = np.array(data['flux'])
        wavelength = np.array(10**data['loglam'])
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
#         num_end = 0
#         num_head = 0
#         for i in range(len(wavelength)):
#             if wavelength[i] <= 9500:
#                 num_end += 1
#             if wavelength[i] <= 3500:
#                 num_head += 1
        data_spec = [wavelength[0:4096], flux_norm[0:4096]]
        samples.append(data_spec)
    return samples


# In[9]:


################mastar
def Preprocessing6(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    samples = []
    for input_file in files:
        spec = ls.rdspec(input_file)
        flux = np.array(spec.flux)
        wavelength = np.array(spec.wave)
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
#         num_end = 0
#         num_head = 0
#         for i in range(len(wavelength)):
#             if wavelength[i] <= 9500:
#                 num_end += 1
#             if wavelength[i] <= 3500:
#                 num_head += 1
        data_spec = [wavelength, flux_norm]
        samples.append(data_spec)
    return samples



#############norm
def Preprocessing7(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    fluxes = []
    samples = []
    for input_file in files:
        hdu_list = fits.open(input_file, memmap=True)
        data = Table(hdu_list[1].data)
        flux = np.array(data['flux'])
        wavelength = np.array(10**data['loglam'])
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
#         num_end = 0
#         num_head = 0
#         for i in range(len(wavelength)):
#             if wavelength[i] <= 9500:
#                 num_end += 1
#             if wavelength[i] <= 3500:
#                 num_head += 1
        data_spec = [wavelength, flux_norm]
        flux = np.array(flux_norm)
        fluxes.append(flux)
        samples.append(data_spec)
    return fluxes, samples


# In[9]:


################mastar
def Preprocessing8(path):
    files = []
    for filename in glob.glob(path):
        files.append(filename)
    fluxes = []
    samples = []
    for input_file in files:
        spec = ls.rdspec(input_file)
        flux = np.array(spec.flux)
        wavelength = np.array(spec.wave)
        flux_norm, flux_cont = norm.normalize_spectrum(wavelength, flux, (3500., 9500.), 100.)
#         num_end = 0
#         num_head = 0
#         for i in range(len(wavelength)):
#             if wavelength[i] <= 9500:
#                 num_end += 1
#             if wavelength[i] <= 3500:
#                 num_head += 1
        data_spec = [wavelength, flux_norm]
        flux = np.array(flux_norm)
        fluxes.append([flux])
        samples.append(data_spec)
    return fluxes, samples

