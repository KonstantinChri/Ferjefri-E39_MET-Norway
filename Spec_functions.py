#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import xarray as xr

def Heave_to_RawSpec1D(heave,freq_new,sample_frequency, detrend, window):
    """
    Function for estimation of frequency spectra from sea surface elevation 
    (heave)
    Parameters:
    ----------
    heave : Array containing surface eleavation [time,samples]
    freq_new : Array containing the desired frequency grid
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
    SPEC_raw[index_nan,:] = np.nan #  time_index where heave is nan
    # create xarray output
    ds = xr.Dataset({'SPEC_raw': xr.DataArray(SPEC_raw,
                                dims   = ['time','freq_raw'],
                                coords = {'time': heave.time,'freq_raw':freq_raw},
                                attrs  = {'units': 'm**2 s'})})
    return ds  


def Heave_to_WelchSpec1D(heave,freq_resolution,sample_frequency, detrend_str, window_str):
    """
    Function for estimation of frequency spectra from sea surface elevation
    using the Welch method
    Parameters:
    ----------
    heave : Array containing surface eleavation [time,samples]
    freq_resolution : desired resolution of frequencies in Hz, common used 0.01Hz
    sample_frequency: value in Hz e.g. for Svinøy is 1 Hz, A-F is 2 Hz
    detrend_str : str e.g. 'constant'
    window_str  : str e.g. 'hann'
   Returns
   -------
    ds : xr.Dataset
        freq: Array of frequencies
        SPEC[time,freq_raw]: Spectra array
    """
    #sampling_time = 600#(heave.time[1]-heave.time[0])/np.timedelta64(1, 's') # in seconds
    # sampling frequency len(heave.samples/sampling_time)
    freq, SPEC =  signal.welch(heave, fs=sample_frequency, window=window_str,
                               nperseg=sample_frequency/freq_resolution, 
                               noverlap=0.5*(sample_frequency/freq_resolution), 
                          nfft=None, detrend=detrend_str, return_onesided=True, 
                          scaling='density', axis=-1)
    

    ds = xr.Dataset({'SPEC': xr.DataArray(SPEC, 
                                dims   = ['time','freq'],
                                coords = {'time': heave.time,'freq':freq},
                                attrs  = {'units': 'm**2 s'})})
    return ds  


