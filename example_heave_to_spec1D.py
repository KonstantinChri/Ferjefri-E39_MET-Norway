#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from Spec_functions import Heave_to_RawSpec1D, Heave_to_WelchSpec1D

# Provide raw wave data from thredds.met.no/thredds/
ds_raw = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_Svinoy_raw_wave.nc')
#ds_raw = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2017/11/201711_E39_F_Vartdalsfjorden_raw_wave.nc')
#ds_raw = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_C_Sulafjorden_raw_wave.nc')

# Provide the frequency grid
freq_resolution = 0.01
freq_new = np.arange(0.04,1,freq_resolution)

# Use a resampling method
ds = Heave_to_RawSpec1D(ds_raw.heave,freq_new,sample_frequency=1, detrend = True, window=True)
# Using Welch method
ds_welch = Heave_to_WelchSpec1D(ds_raw.heave,
                                 freq_resolution=freq_resolution,
                                 sample_frequency=1, 
                                 detrend_str='constant', window_str='hann')

# different estimations of Hs for plotting
Hs_heave =  4*np.std(ds_raw.heave,axis=1)
Hs_raw =  4*(ds.SPEC_raw.integrate("freq_raw"))**0.5
Hs_welch =  4*(ds_welch.SPEC.integrate("freq"))**0.5

# maximum Hs
max_Hs = Hs_heave.where(Hs_heave==Hs_heave.max(),drop=True).squeeze()


# Plot Spec1D for time with  max-Hs
plt.figure()
plt.plot(ds.freq_raw, ds.SPEC_raw.loc[max_Hs.time],color='grey',label='Raw-Spec, Hm0[m]:'+str(Hs_raw.loc[max_Hs.time].values.round(2))+',Hs_heave[m]:'+str(Hs_heave.loc[max_Hs.time].values.round(2)))
plt.plot(ds_welch.freq, ds_welch.SPEC.loc[max_Hs.time],color='green',label='Welch-Spec., Hm0[m]:'+str(Hs_welch.loc[max_Hs.time].values.round(2)))
plt.grid()
plt.legend()
plt.title('time:'+str(max_Hs.time.values)[:19])
plt.xlabel('freq.[Hz]')
plt.ylabel('SPEC.[m**2 s]')
plt.show()

# plot Hs time series
plt.figure()
Hs_heave.plot(marker='.',label='Hs_heave')
Hs_welch.plot(marker='.',label='Hs_welch')
plt.xlabel('time')
plt.ylabel('[m]')
plt.title('Max-Hs at time:'+str(max_Hs.time.values)[:19])
plt.grid()
plt.legend()
plt.show()
