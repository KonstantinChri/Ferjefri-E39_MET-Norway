#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import xarray as xr

buoy = 'B'
fjord = 'Sulafjorden' # _E39_C_Sulafjorden_ / 
variable = 'aquadopp' # wave / wind / aquadopp
start_date = '2016-10-01 00:00:00'
end_date = '2020-07-30 00:00:00'
dir_for_download = '/home/konstantinosc/PhD/Paper_Norkyst/Data/Raw_Spec/all_files/'+buoy+'/'+variable+'/'


dates = pd.period_range(start_date,end_date, freq="M")
for i in range(len(dates)):
    filein = dates[i].strftime('%Y') + '/' + dates[i].strftime('%m') + '/'+ dates[i].strftime('%Y') + dates[i].strftime('%m') + '_E39_'+buoy +'_'+fjord+'_'+variable +'.nc'
    url_thredds = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/'+filein
    print(url_thredds)
    ds0 = xr.open_dataset(url_thredds)
    if i==0:
        ds=ds0
    else:
        ds = xr.concat([ds,ds0],dim='time')

ds.to_netcdf(dir_for_download+dates[0].strftime('%Y')+dates[0].strftime('%m')+
             '_'+dates[-1].strftime('%Y')+dates[-1].strftime('%m')+'_E39_'+buoy +
             '_'+fjord+'_'+variable +'.nc')

