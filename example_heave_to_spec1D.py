#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from Spec_functions import heave_to_spec1D


# Provide raw wave data from thredds.met.no/thredds/
ds = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_Svinoy_raw_wave.nc')
#ds = xr.open_dataset('https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2020/01/202001_E39_F_Vartdalsfjorden_raw_wave.nc')

# Provide desired frequencies
freq_new = np.arange(0.04,0.50,0.01)
# Call heave_to_spec1D function
SPEC_new, SPEC_raw, freq_raw = heave_to_spec1D(ds.heave,freq_new,detrend='True')

# different estimations of Hs for plotting
Hs_heave =  4*np.std(ds.heave,axis=1)
Hs_heave_spec0 =  4*np.trapz(SPEC_raw*np.diff(np.hstack((0, freq_raw))))**0.5
Hs_inter = 4*np.trapz(SPEC_new*np.diff(np.hstack((0, freq_new))),axis=1)**0.5

# time step for maximum Hs
time_step = np.where(Hs_heave==Hs_heave.max())[0].tolist()

# Plot Spec1D for given time step
plt.figure()
plt.plot(freq_raw, SPEC_raw[time_step[0],:],color='grey',label='Raw-Spec, Hm0[m]:'+str(Hs_heave_spec0[time_step[0]].round(2))+',4*std(heave)[m]:'+str(Hs_heave.values[time_step[0]].round(2)))
plt.plot(freq_new, SPEC_new[time_step[0],:],color='red',label='Interpol.-Raw-Spec., Hm0[m]:'+str(Hs_inter[time_step[0]].round(2)))
plt.grid()
plt.legend()
plt.title('Time:'+str(time_step[0]))
plt.xlabel('freq.[Hz]')
plt.ylabel('SPEC.[m**2 s]')
plt.show()

# plot Hs time series
plt.figure()
plt.plot(Hs_heave,'.',label='4*std(heave)')
plt.plot(Hs_inter,label='4*sqrt(m0_interp)') 
plt.plot(time_step[0],Hs_inter[time_step[0]],label='Max-Hs',marker='o',color='r') # plot the maximum Hs
plt.xlabel('time')
plt.ylabel('Hm0[m]')
plt.title('Max-Hs at time:'+str(time_step[0]))
plt.grid()
plt.legend()
plt.show()
