#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import spectres as spec
import xarray as xr


def heave_to_spec1D(heave,freq_new,sample_frequency, detrend, window):
    """
    Function for estimation of frequency spectra from sea surface elevation 
    (heave)
    Parameters:
    ----------
    heave : 2D array containing surface eleavation [time,samples]
    freq_new : 1D array containing the desired frequency grid
    sample_frequency: value in Hz e.g. for Svinøy is 1 Hz, A-F is 2 Hz
    detrend : bool (if True, it uses a linear detrend)
    window  : bool (if True, it uses hanning window in fft)
   Returns
   -------
    ds : xr.Dataset
        freq_raw: Array of frequencies before resampling
        SPEC_raw[time,freq_raw]: Array of spectra before resampling
        SPEC_new[time,freq_new]: Array of spectra after resampling
    """
    n = np.zeros(heave.shape)
    index_nan = []
    freq_raw = np.fft.rfftfreq(heave.shape[1],d=1./sample_frequency)
    if detrend == True:
        for i in range(heave.shape[0]):
            if np.isnan(heave[i,:]).all() ==True:
                index_nan.append(i) # time index for nan-heave
                n[i,:] = -999#heave[i,:]
            else:
                n[i,:] = signal.detrend(heave[i,:]) # detrend heave data that contains no nan
    else:
        n = heave
    if window==True:
        # fft - hanning window
        SPEC_raw =   np.abs(np.fft.rfft(n*np.hanning(n.shape[1]),axis=1))**2  /  ((sum(np.hanning(n.shape[1])**2)/n.shape[1])*sample_frequency*len(freq_raw))
    else:
        # fft - without window
        SPEC_raw =   np.abs(np.fft.rfft(n,axis=1))**2  /  (sample_frequency*len(freq_raw))
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






