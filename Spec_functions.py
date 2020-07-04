#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import spectres as spec


def heave_to_spec1D(heave,freq_new, detrend):
    ''' estimates SPEC(freq_new) from heave 
    Heave[time,samples]
    freq_new: 1D array
    '''
    freq_raw = np.fft.rfftfreq(heave.shape[1],d=1)
    if detrend=='True':
        n = signal.detrend(heave,axis=1) # detrend heave data
    else:
        n=  heave
    SPEC_raw = np.abs(np.fft.rfft(n,axis=1))**2 / len(freq_raw) # fft
    SPEC_new = spec.spectres(freq_new, freq_raw, SPEC_raw, spec_errs=None, fill=None, verbose=True) # resample 
    return SPEC_new, SPEC_raw, freq_raw  

