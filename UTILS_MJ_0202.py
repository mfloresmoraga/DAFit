#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import glob
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d



### MODELS ###

def aumentar_resolucion(model):

    # nos vamos a quedar con la misma zona de observado
    wave = model["wave"]
    flux = model["flux"]
    teff = model["teff"]
    logg = model["logg"]
    mask = (wave > 4600) & (wave < 6700)
    
    wave_short = wave[mask]
    flux_short = flux[mask]
    idx = np.argsort(wave_short)
    wave_short = wave_short[idx]
    flux_short = flux_short[idx]

    #plt.plot(wave_short, flux_short, c='k', label='Original')
    #plt.show()

    dw_ = np.diff(wave_short)

    #print('BINSIZE MODELS BEFORE CHANGE IN RES',dw_)

    #hacemos una interpolacion lineal para igualar la resolucion
 
    wave_new = np.linspace(4600, 6700, 80100)
    flux_new = np.interp(wave_new, wave_short, flux_short)

    dw = np.diff(wave_new)              # paso en longitud de onda
    #0.04 

    #for j,i, in zip(dw_, dw):
    #    print(j,i)

    return wave_new, flux_new#, wave_short, flux_short


#Para convolucionar el espectro hay que pasarle las regiones de la linea, no es para todo el espectro

    
def convolve_spectrum(wave, flux, fwhm):

    # R = wave / delta_wave = 6562.79 / 

    # fwhm = wave / Ralpha ----> FWHM = 6562.79 / 18 500 = 
    

    dw = np.median(np.diff(wave))               # paso en longitud de onda

    sigma_lambda = fwhm / 2.3548     # sigma en Angstrom, ese 2.3548 =  2sqrt(2*ln(2))
    sigma_pix = sigma_lambda / dw    # sigma en pixeles


    flux_conv = gaussian_filter1d(flux, sigma_pix)


    return  wave, flux_conv
    

### OBSERVATIONS ###

