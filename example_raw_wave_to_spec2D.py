#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from Spec_functions import Directional_Spectra

time0='2020-01-09T00:00'
ds_raw = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_Svinoy_raw_wave.nc')

ds = Directional_Spectra(raw_data = ds_raw, freq_resolution= 0.01,n_direction= 72,sample_frequency= 1,direction_units='rad')

Hs_heave =  4*np.std(ds_raw.heave,axis=1)
Hs_spec = 4*(np.sum(ds.SPEC.integrate("frequency"),axis=1))**0.5

plt.figure(0)
Hs_spec.plot(color='grey',label='from 2Dspec')
Hs_heave.plot(color='black',label='from heave')
plt.plot(Hs_heave.time.loc[time0], Hs_heave.loc[time0], marker='o',color='red')
plt.ylabel('Hs[m]')
plt.grid()
plt.legend()

plt.figure(1)
ds.SPEC.loc[time0].plot.pcolormesh()
