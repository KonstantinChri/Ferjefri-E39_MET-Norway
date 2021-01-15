#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from Spec_functions import Heave_to_WelchSpec1D

# This program convert raw heave data from thredds.met.no/thredds/
# to spectra and then save it to a netcdf file

# Provide raw wave data from thredds.met.no/thredds/:
file_in = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_C_Sulafjorden_raw_wave.nc'
ds_raw = xr.open_dataset(file_in)

# Provide the frequency grid:
freq_resolution = 0.01
freq_new = np.arange(0.04,1,freq_resolution)

# Using Welch method to estimate the spectra:
ds_welch = Heave_to_WelchSpec1D(ds_raw.heave,
                                 freq_resolution=freq_resolution,
                                 sample_frequency=2, 
                                 detrend_str='constant', window_str='hann')

# save spectra data to netcdf 
file_out = file_in.split('/')[-1].replace('raw','SPEC')
ds_welch.to_netcdf(file_out)

