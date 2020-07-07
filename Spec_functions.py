#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import spectres as spec
import xarray as xr


def heave_to_spec1D(heave,freq_new, detrend):
    ''' -estimates SPEC_raw[time,freq_raw] from heave[time,samples]
        -resample SPEC_raw[time,freq_raw] to SPEC_new[time,freq_new]
         freq_new: 1D array - new frequency grid
    '''
    n = np.zeros(heave.shape)
    index_nan = []
    freq_raw = np.fft.rfftfreq(heave.shape[1],d=1)
    if detrend == True:
        for i in range(heave.shape[0]):
            if np.isnan(heave[i,:]).all() ==True:
                index_nan.append(i) # time index for nan-heave
                n[i,:] = -999#heave[i,:]
            else:
                n[i,:] = signal.detrend(heave[i,:]) # detrend heave data that contains no nan
    else:
        n = heave
    # fft: heave to SPEC_raw
    SPEC_raw = np.abs(np.fft.rfft(n,axis=1))**2 / len(freq_raw) 
    # resample SPEC_raw to a new frequency grid 
    SPEC_new = spec.spectres(freq_new, freq_raw, SPEC_raw, spec_errs=None, fill=None, verbose=True) # resample
    SPEC_raw[index_nan,:] = np.nan #  time_index where heave is nan
    SPEC_new[index_nan,:] = np.nan #  time_index where heave is nan    
    # create xarray output
    ds = xr.Dataset({'SPEC_raw': xr.DataArray(SPEC_raw,
                                dims   = ['time','freq_raw'],
                                coords = {'time': heave.time,'freq_raw':freq_raw},
                                attrs  = {'units': 'm**2 s'}),
                     'SPEC_new': xr.DataArray(SPEC_new, 
                                dims   = ['time','freq_new'],
                                coords = {'time': heave.time,'freq_new':freq_new},
                                attrs  = {'units': 'm**2 s'})})
    return ds  
