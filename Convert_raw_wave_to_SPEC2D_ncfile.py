#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from Spec_functions import Directional_Spectra

# This program converts raw wave data from thredds.met.no/thredds/
# to directional spectra and then saves it to a netcdf file

# Provide raw wave data from thredds.met.no/thredds/:
file_in = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_C_Sulafjorden_raw_wave.nc'
ds_raw = xr.open_dataset(file_in)

# Estimate the directional spectra:
ds = Directional_Spectra(raw_data = ds_raw, freq_resolution= 0.01,n_direction= 72,sample_frequency= 2,direction_units='rad')    

# save spectra data to netcdf 
file_out = file_in.split('/')[-1].replace('raw','SPEC') # nc-file to save spectra
ds.to_netcdf(file_out)

