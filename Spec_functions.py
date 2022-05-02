#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import xarray as xr
from urllib.request import urlretrieve
import pandas as pd
import os
from nco import Nco
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker




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
                                coords = {'time': ds0.time,'frequency':freq},
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
                                dims   = ['time','frequency'],
                                coords = {'time': ds0.time,'frequency':freq},
                                attrs  = {'units': 'm**2 s'})})
    ds['frequency'].attrs["units"] = 'Hz'
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

def MEM(n_direction,a1, b1, a2, b2, direction_units):
    """ The function generates a Maximum entropy directional density matrix
    n_direction: number of directions.
    a1 b1 a2 b2: are the fourier coefficients size(time,frequency) """

    n_freq = len(a1.frequency)
    n_time = len(a1.time)
    direction0 = np.linspace(0,2*np.pi,n_direction) # range 0...2pi, nautical dir.
    direction = np.mod(3*np.pi/2-direction0,2*np.pi)  #Rotate to oceanographic dir.
    d = xr.DataArray(np.zeros((n_time, n_freq, n_direction)),
    dims=["time", "frequency", "direction"],
    coords={"time": a1.time, "frequency": a1.frequency, "direction": direction})
    direction  = xr.DataArray(direction,dims=["direction"])

    for nn in range(0,n_freq):
        c1 = a1[:,nn]+1j*b1[:,nn]
        c2 = a2[:,nn]+1j*b2[:,nn]
        f1 = ( c1 - c2*np.conj(c1) )/(1-np.abs(c1)**2)
        f2 = c2 -c1*f1
        s1 = 1 - f1*np.conj(c1) - f2*np.conj(c2)
        den= 1 - f1*np.exp(-1j*direction) - f2*np.exp(-2j*direction)
        if direction_units == 'rad':
            d[:,nn,:]  = np.real(s1/(np.abs(den)**2 *2*np.pi))
            d['direction'] = direction0 # naut. dir.
        elif direction_units == 'deg':
            d[:,nn,:]  = np.real(s1/(np.abs(den)**2 *2*180))
            d['direction'] = np.round(np.linspace(0,360,n_direction),0) # naut. dir.

    d.attrs["units"] = '1/'+direction_units #In general,  d is dimensionless
    d['direction'].attrs["units"] = direction_units
    d['frequency'].attrs["units"] = 'Hz'

    #print(d.integrate('direction')) # integral of D should be close to 1
    return d


def Directional_Spectra(raw_data, freq_resolution, n_direction , sample_frequency, direction_units):
    """
    Function for estimation of directional spectra(frequency,direction)
    from heave, pitch and roll (tested for Wavescan buoys)
    Parameters:
    ----------
    raw_data : Xarray dataset containing variables:
        heave(time,samples) in meters, pitch(time,samples) in degress,
        roll(time,samples) in degress and compass(time,samples) in degrees
    freq_resolution : desired resolution of frequencies in Hz, common used 0.01Hz
    sample_frequency: value in Hz e.g., for Svinøy buoy is 1 Hz, A-F is 2 Hz
    Returns
    -------
    ds : xr.Dataset
        frequency: Array of frequencies [Hz]
        direction: Array of directions [degrees]
        SPEC[time,frequency,direction]: Spectra array [m² /Hz*deg or m²/Hz*rad]
        and integrated parameters Hm0, Tp, pdir
    """
    R = np.deg2rad(raw_data['roll'])
    P = np.deg2rad(raw_data['pitch'])
    Z = raw_data['heave']
    A = np.deg2rad(raw_data['compass'])

    # Estimate slopes (east-west and north/south):
    Zy = ((np.sin(P)/np.cos(P))*np.sin(A)) + (np.sin(R)/(np.cos(P)*np.cos(R))*np.cos(A))
    Zx = ((np.sin(P)/np.cos(P))*np.cos(A)) - (np.sin(R)/(np.cos(P)*np.cos(R))*np.sin(A))

    del R, P, A, raw_data
    # More details about the methodology can be found:
    # Longuet-Higgins et al (1963) , Lygre and Krogstad (1986),
    # P.N. Ananth et al. (1992), F. P. Brissette et al. (1994), Steele et al. (1998)
    # Nondirectional and Directional Wave Data Analysis Procedures,  NDBC Technical Document 96-01

    # Estimate Quadrature Spectra Q:
    Qzx =  np.imag(Heave_to_CSD(Z , Zx ,
                                     freq_resolution=freq_resolution,
                                     sample_frequency=sample_frequency,
                                     detrend_str=False, window_str='hann')['CSD'])
    Qzy =  np.imag(Heave_to_CSD(Z , Zy,
                                     freq_resolution=freq_resolution,
                                     sample_frequency=sample_frequency,
                                     detrend_str=False, window_str='hann')['CSD'])
   #Estimate Co- Spectra C:
    Czz =  np.real(Heave_to_CSD(Z , Z,
                                      freq_resolution=freq_resolution,
                                      sample_frequency=sample_frequency,
                                      detrend_str=False, window_str='hann')['CSD'])

    Cyy =  np.real(Heave_to_CSD(Zy , Zy,
                                     freq_resolution=freq_resolution,
                                     sample_frequency=sample_frequency,
                                     detrend_str=False, window_str='hann')['CSD'])

    Cxx =  np.real(Heave_to_CSD(Zx , Zx,
                                     freq_resolution=freq_resolution,
                                     sample_frequency=sample_frequency,
                                     detrend_str=False, window_str='hann')['CSD'])

    Cxy =  np.real(Heave_to_CSD(Zx , Zy,
                                      freq_resolution=freq_resolution,
                                      sample_frequency=sample_frequency,
                                      detrend_str=False, window_str='hann')['CSD'])


    del Zx, Zy
    # Estimate Wavenumber using cross spectra
    k = np.sqrt( (Cxx+Cyy)/Czz )

    #Estimate first 4 Fourier Coefficients:
    a1 = Qzy/(k*Czz)
    b1 = Qzx/(k*Czz)

    a2 = (Cyy - Cxx)/(Czz*k*k)
    b2 = 2*Cxy/(Czz*k*k)

    del Qzy, Qzx, Cxx, Cyy, Cxy
    # Use Max. Entropy Method:
    D = MEM(n_direction = n_direction, a1 = a1,b1 = b1,a2 = a2,b2 = b2, direction_units = direction_units)
    # Estimate directional spectra
    SPEC2D = Czz * D
    del D
    # Create Dataset
    ds = xr.Dataset({'SPEC': xr.DataArray(SPEC2D,
                                dims   = ['time','frequency','direction'],
                                coords = {'time': SPEC2D.time,'frequency':SPEC2D.frequency, 'direction':SPEC2D.direction},
                                attrs  = {'units': 'm²/Hz/'+direction_units, 'standard_name' : 'directional_spectrum'})})
    # Estimate integrated parameters
    ds['Hm0'] = (4*(SPEC2D.integrate("frequency").integrate("direction"))**0.5).assign_attrs(units='m', standard_name = 'significant_wave_height_from_spectrum')
    ds['Hs'] = (4*np.std(Z,axis=1)).assign_attrs(units='m', standard_name = 'significant_wave_height_from_heave')
    fp = SPEC2D.integrate('direction').idxmax(dim='frequency')
    ds['Tp'] = (1/fp).assign_attrs(units='s', standard_name = 'peak_wave_period')

    if direction_units == 'rad':
        pdir = np.rad2deg(SPEC2D.integrate('frequency').idxmax(dim='direction'))
    elif direction_units == 'deg':
        pdir = SPEC2D.integrate('frequency').idxmax(dim='direction')
    ds['pdir'] = pdir.assign_attrs(units='deg', standard_name = 'peak_wave_direction')

    ds['h_SPEC'] = (Czz).assign_attrs(units='m²/Hz', standard_name = 'frequency_spectrum_from_heave')
    # m0 = ds['h_SPEC'].integrate('frequency')
    # m1 = (ds['frequency']*ds['h_SPEC']).integrate('frequency')
    # m2 = (ds['frequency']*ds['frequency']*ds['h_SPEC']).integrate('frequency')

    # ds['kurtosis'] = ((Z**4).mean('samples')/((Z**2).mean('samples'))**2) -1  # degree of peakedness
    # ds['skewness'] = ((Z**3).mean('samples')/((Z**2).mean('samples'))**(3/2))  # degree of asymmetry
    # ds['spec_width'] = (((m0*m2/(m1**2)) - 1)**0.5).assign_attrs(standard_name = 'spectral_width')
    return ds


def Spectral_Partition(ds, beta):
    ds['cp'] = 9.81/(2*np.pi*ds['frequency'])
    ds['A']  = beta*(ds['WindSpeed']/ds['cp'])*np.cos(ds['direction']-np.rad2deg(ds['WindDirection']))
    ds['SPEC_swell'] = ds['SPEC'].where(ds['A']<=1,0)
    ds['SPEC_windsea'] = ds['SPEC'].where(ds['A']>1,0)
    # Estimate integrated parameters
    part = ['swell','windsea']
    for k in range(len(part)):
        ds['Hm0_'+part[k]] = (4*(ds['SPEC_'+part[k]].integrate("frequency").integrate("direction"))**0.5).assign_attrs(units='m', standard_name = 'significant_wave_height_from_spectrum_'+part[k])
        ds['Tp_'+part[k]] = (1/ds['SPEC_'+part[k]].integrate('direction').idxmax(dim='frequency')).assign_attrs(units='s', standard_name = 'peak_wave_period_'+part[k])
        ds['pdir_'+part[k]] = np.rad2deg(ds['SPEC_'+part[k]].integrate('frequency').idxmax(dim='direction'))

    return ds


def plot_Directional_Spectra(ds,plot_type,vmin, vmax,cmap,  filter_factor, fig_title, SPEC_units, add_par,add_h_Spec,log_scale, fig_format):
    """
    Plot Directional Spectra[frequency, direction] in polar coordinates
    SPEC: xarray with coordinates frequency in Hz and direction in degrees (nautical convection)
    fig_title: title string format
    fig_format: format of the figure e.g., png, pdf, eps
    """
    if SPEC_units == 'm²/Hz/deg':
        theta = np.deg2rad(ds.SPEC['direction'])
    else:
        theta = ds.SPEC['direction']


    SPEC = ds.SPEC.where(ds.SPEC>ds.SPEC.max()*filter_factor,np.nan) # set to nan values < SPEC.max()*filter_factor
    fig = plt.figure(figsize=(7,5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    if plot_type == 'pcolormesh':
        if log_scale == True:
            cs = ax.pcolormesh(theta, SPEC['frequency'], SPEC, cmap=cmap ,shading='auto', norm=colors.LogNorm(vmin=vmin,vmax=vmax))
        else:
            cs = ax.pcolormesh(theta, SPEC['frequency'], SPEC,vmin=vmin,vmax=vmax, cmap=cmap ,shading='auto')
    elif plot_type == 'contourf':
        if log_scale == True:
            cs = ax.contourf(theta, SPEC['frequency'], SPEC, cmap=cmap, norm=colors.LogNorm(vmin=vmin,vmax=vmax))
        else:
            cs = ax.contourf(theta, SPEC['frequency'], SPEC,vmin=vmin,vmax=vmax, cmap=cmap)

    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Adding text on the plot.
    if add_par == True:
        plt.gcf().text(0.01, 0.05,
                       '$H_{m0}$:'+str(ds.Hm0.values.round(1))+' m' +'\n'+
                       '$T_{p}$:'+str(ds.Tp.values.round(1))+' s'+'\n'+
                       '$\u03B8_p$:'+str(ds.pdir.values.round(1))+'$^o$'+'\n'+
                       # '---------------'+'\n'+
                       # '$H_{m0,swell}$:'+str(ds.Hm0_swell.values.round(1))+' m' +'\n'+
                       # '$T_{p,swell}$:'+str(ds.Tp_swell.values.round(1))+' s'+'\n'+
                       # '$\u03B8_{p,swell}$:'+str(ds.pdir_swell.values.round(1))+'$^o$'+'\n'+
                       # '---------------'+'\n'+
                       # '$H_{m0,wind}$:'+str(ds.Hm0_windsea.values.round(1))+' m' +'\n'+
                       # '$T_{p,wind}$:'+str(ds.Tp_windsea.values.round(1))+' s'+'\n'+
                       # '$\u03B8_{p,wind}$:'+str(ds.pdir_windsea.values.round(1))+'$^o$'+'\n'+
                       '---------------'+'\n'+
                       '$U_w$:'+str(ds.WindSpeed.values.round(1))+' $m/s$'+'\n'+
                       '$\u03B8_{w}$:'+str(ds.WindDirection.values.round(1))+'$^o$'+'\n'+
                       '---------------'+'\n'+
                       '$U_c$:'+str(ds.currSp001.values.round(1))+' $m/s$'+'\n'+
                       '$\u03B8_{c}$:'+str(ds.currDir001.values.round(1))+'$^o$'+'\n'
                       ,fontsize=14)
    else:
        pass

    if add_h_Spec == True:
        plt.axes([.84, .09, .12, .3], facecolor= None)
        ds.h_SPEC.plot()
        plt.ylabel('[m²/Hz]')
        plt.xlabel('[Hz]')
        fig.colorbar(cs,shrink=0.5,orientation='horizontal', label=SPEC_units)
    else:
        fig.colorbar(cs, label=SPEC_units)
    ax.set_title(fig_title, fontsize=16)
    fig.savefig(fig_title+'_2DSPEC.'+fig_format, dpi=300)
    plt.close()
