#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import xarray as xr
from urllib.request import urlretrieve
import pandas as pd
import os
from nco import Nco


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

def Heave_to_Coherence(heave1,heave2,freq_resolution,sample_frequency, detrend_str, window_str):
    """
    Function for estimation of coherence from 2 sea surface elevation time series
    Parameters:
    ----------
    heave1 : Array containing surface eleavation 1 [time,samples]
    heave1 : Array containing surface eleavation 2 [time,samples]
    freq_resolution : desired resolution of frequencies in Hz, common used 0.01Hz
    sample_frequency: value in Hz e.g. for Svinøy is 1 Hz, A-F is 2 Hz
    detrend_str : str e.g. 'constant'
    window_str  : str e.g. 'hann'
   Returns
   -------
    ds : xr.Dataset
        freq: Array of frequencies
        Coh[time,freq_raw]: Coherence array
    """
    ds0 = xr.merge([heave1.rename('heave1'),heave2.rename('heave2')],compat='override', join='inner')
    freq, Coh = signal.coherence(ds0.heave1,ds0.heave2,
                                  fs = sample_frequency,
                                  nperseg=sample_frequency/freq_resolution,
                                  noverlap=0.5*(sample_frequency/freq_resolution),
                                   nfft=None, detrend=detrend_str, window=window_str,
                                   axis=-1)

    ds = xr.Dataset({'Coh': xr.DataArray(Coh,
                                dims   = ['time','freq'],
                                coords = {'time': ds0.time,'freq':freq},
                                attrs  = {'units': '-'})})
    return ds


def Heave_to_CSD(heave1,heave2,freq_resolution,sample_frequency, detrend_str, window_str):
    """
    Function for estimation of Cross Spectral Density from sea surface elevation
    Parameters:
    ----------
    heave : Array containing surface eleavation 1 [time,samples]
    freq_resolution : desired resolution of frequencies in Hz, common used 0.01Hz
    sample_frequency: value in Hz e.g. for Svinøy is 1 Hz, A-F is 2 Hz
    detrend_str : str e.g. 'constant'
    window_str  : str e.g. 'hann'
   Returns
   -------
    ds : xr.Dataset
        freq: Array of frequencies
        CSD[time,freq_raw]: Cross Spectral Density array
    """
    ds0 = xr.merge([heave1.rename('heave1'),heave2.rename('heave2')],compat='override', join='inner')
    freq, CSD = signal.csd(ds0.heave1,ds0.heave2,
                                  fs = sample_frequency,
                                  nperseg=sample_frequency/freq_resolution,
                                  noverlap=0.5*(sample_frequency/freq_resolution),
                                   nfft=None, detrend=detrend_str, window=window_str,
                                   axis=-1)

    ds = xr.Dataset({'CSD': xr.DataArray(CSD,
                                dims   = ['time','freq'],
                                coords = {'time': ds0.time,'freq':freq},
                                attrs  = {'units': 'm**2 s'})})
    return ds



def download_E39_obs(start_date,end_date,buoy,fjord,variable):
    """
    Download E39 observations from https://thredds.met.no/ in netcdf format.
    """
    nco = Nco()
    date_list = pd.date_range(start=start_date , end=end_date, freq='MS')
    outfile = date_list.strftime('%Y%m')[0]+'_'+date_list.strftime('%Y%m')[-1]+'_E39_'+buoy+'_'+fjord+'_'+variable+'.nc'

    if os.path.exists(outfile):
        os.remove(outfile)
        print(outfile, 'already exists, so it will be deleted and create a new....')

    else:
        print("....")


    infile = [None] *len(date_list)
    # extract point and create temp files
    for i in range(len(date_list)):
        #infile[i] = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/'+date_list.strftime('%Y')[i] + '/' + date_list.strftime('%m')[i] + '/'+ date_list.strftime('%Y')[i] +date_list.strftime('%m')[i] + '_E39_'+buoy +'_'+fjord+'_'+variable +'.nc'
        url = 'https://thredds.met.no/thredds/fileServer/obs/buoy-svv-e39/'+date_list.strftime('%Y')[i] + '/' + date_list.strftime('%m')[i] + '/'+ date_list.strftime('%Y')[i] +date_list.strftime('%m')[i] + '_E39_'+buoy +'_'+fjord+'_'+variable +'.nc'
        infile[i] = date_list.strftime('%Y')[i] +date_list.strftime('%m')[i] + '_E39_'+buoy +'_'+fjord+'_'+variable +'.nc'
        urlretrieve(url,infile[i])
    print(infile)
    #merge temp files
    nco.ncrcat(input=infile, output=outfile)

    #remove temp files
    for i in range(len(date_list)):
        os.remove(infile[i])

    return
