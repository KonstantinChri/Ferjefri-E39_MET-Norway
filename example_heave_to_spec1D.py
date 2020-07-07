#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from Spec_functions import heave_to_spec1D


# Provide raw wave data from thredds.met.no/thredds/
ds_raw = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_Svinoy_raw_wave.nc')
#ds_raw = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2017/11/201711_E39_F_Vartdalsfjorden_raw_wave.nc')
#ds_raw = xr.open_dataset('Data/201711_201804_E39_F_Vartdalsfjorden_raw_wave.nc')


# Provide desired frequencies
freq_new = np.arange(0.04,0.50,0.01)
# Call heave_to_spec1D function
ds = heave_to_spec1D(ds_raw.heave,freq_new,detrend = True)

# different estimations of Hs for plotting
Hs_heave =  4*np.std(ds_raw.heave,axis=1)
Hs_raw =  4*(ds.SPEC_raw.integrate("freq_raw"))**0.5
Hs_inter =  4*(ds.SPEC_new.integrate("freq_new"))**0.5

# time step for maximum Hs
time_step = np.where(Hs_heave==Hs_heave.max())[0].tolist()
#max_Hs = Hs_heave.where(Hs_heave==Hs_heave.max(),drop=True).squeeze()

# Plot Spec1D for given time step
plt.figure()
plt.plot(ds.freq_raw, ds.SPEC_raw[time_step[0],:],color='grey',label='Raw-Spec, Hm0[m]:'+str(Hs_raw.values[time_step[0]].round(2))+',4*std(heave)[m]:'+str(Hs_heave.values[time_step[0]].round(2)))
plt.plot(ds.freq_new, ds.SPEC_new[time_step[0],:],color='red',label='Interpol.-Raw-Spec., Hm0[m]:'+str(Hs_inter.values[time_step[0]].round(2)))
plt.grid()
plt.legend()
plt.title('time:'+str(time_step[0]))
plt.xlabel('freq.[Hz]')
plt.ylabel('SPEC.[m**2 s]')
plt.show()

# plot Hs time series
plt.figure()
Hs_heave.plot(marker='.',label='4*std(heave)')
Hs_inter.plot(marker='.',label='4*sqrt(m0_interp)') 
plt.xlabel('time')
plt.ylabel('Hm0[m]')
plt.title('Max-Hs at time:'+str(time_step[0]))
plt.grid()
plt.legend()
plt.show()
