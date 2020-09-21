#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from urllib.request import urlretrieve
import pandas as pd

buoy = 'B'
fjord = 'Sulafjorden' # Sulafjorden / Breisundet etc
variable = 'aquadopp' # wave / wind / aquadopp
start_date = '2016-10-01 00:00:00'
end_date = '2020-07-30 00:00:00'
dir_for_download = '.../'+buoy+'/'+variable+'/'


dates = pd.period_range(start_date,end_date, freq="M")
for i in range(len(dates)):
    filein = dates[i].strftime('%Y') + '/' + dates[i].strftime('%m') + '/'+ dates[i].strftime('%Y') + dates[i].strftime('%m') + '_E39_'+buoy +'_'+fjord+'_'+variable +'.nc'
    url_thredds = 'https://thredds.met.no/thredds/fileServer/obs/buoy-svv-e39/'+filein
    print(url_thredds)
    urlretrieve(url_thredds,dir_for_download+filein.split('/')[2])

#Merge nc files
#ncrcat *.nc -O  2016_2020_E39_B_Sulafjorden_aquadopp.nc
