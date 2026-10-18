#!/usr/bin/env python
# coding: utf-8

import os
import sys
import io
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
import glob
import UTILS_MJ_0202 as UTILS_MJ
import matplotlib.pyplot as plt
import astropy.constants as cte
import astropy.units as un
import time
from datetime import datetime
from scipy.interpolate import UnivariateSpline

c_kms = cte.c.to(un.km / un.s).value

info = pd.read_csv('info_stars_P116_centrales')

full_names = info['nombre_1'].tolist() #obsblock_nombre
obs_blocks = info['OB'].tolist()
targets = info['target'] #nombre
dates = info['date']
vel_helio = info['vhelio'].tolist() #solo 0s
central_Ha = info['central_1'].tolist()
central_Hb = info['central_2'].tolist()

balmer_ab = ["Ha", "Hb"]
step = 1
rv_test   = np.arange(-55, 130, step)
teff_test = np.arange(5000, 27000, 250)
logg_test = np.arange(7, 8.75, 0.25)

Ha = 6562.79
Hb = 4861.35


lineas = {
    "Ha": {"nplot":r"H$_{\alpha}$", "rest": Ha, "fixed_window": 135, "lim_inf": 6400, "lim_sup": 7800},
    "Hb": {"nplot":r"H$_{\beta}$" , "rest": Hb, "fixed_window": 150, "lim_inf": 4635, "lim_sup": 5050}}

def informacion_star(date):

    v_helio = 0
    centrales = None
    global central_Ha
    global central_Hb

    for n, ca, cb in zip(dates, central_Ha, central_Hb):
        if n == date:
            v_helio = 0
            centrales = [ca, cb]
            break

    if v_helio is None or centrales is None:
        print(f"AVISO: SKIP.")
        return None

    return date, v_helio, centrales

def load_koester_models(model_dir): 
    """
    Carga y lee los modelos de Koester. Guarda sus Teff, logg, wave, flux y 
    nombre de archivo en un diccionario.
    """
    all_models = []
    for model_path in glob.glob(f"{model_dir}/*"):

        try:
            teff, logg = None, None

            with open(model_path, "r") as f:
                for line in f:
               
                    if line.startswith("# teff"):
                        teff = float(line.split("=")[1].split()[0])
                  
                    if line.startswith("# logg"):
                        logg = float(line.split("=")[1].split()[0])
                       
                    if teff is not None and logg is not None:
                        break
                        
            if teff is None or logg is None:
         
                continue

            df = pd.read_csv(model_path, sep=r"\s+", comment='#', names=["wavelength","flux"])
            wave_model = df['wavelength'].values
            flux_model = df['flux'].values

            mask_wave = (wave_model >= 4600) & (wave_model <= 7800)

            if teff >= 5000 and teff <= 27000:
                if logg >= 7 and logg <= 8.75:
                    all_models.append({
                           'teff': teff,
                           'logg': logg,
                           'wave': wave_model[mask_wave],
                           'flux': flux_model[mask_wave],
                           'file': os.path.basename(model_path)})
            
        except Exception as e:
            print(f"Advertencia: Fallo la carga de {model_path}: {e}")
            continue

    return all_models

def aumentar_resolucion_models(all_models):
    """
    Interpola modelos de Koester a una malla más fina de Teff y logg
    antes de aumentar la resolución y normalizar.
    """

    modelos_aumentados = []
    for model in all_models:
        teff = model["teff"]
        logg = model["logg"]
        wave_new, flux_new = UTILS_MJ.aumentar_resolucion(model)
        
        modelo_aumentado = {"teff": teff, "logg": logg, "wave_new": wave_new, "flux_new": flux_new}

        modelos_aumentados.append(modelo_aumentado)

    return modelos_aumentados


def find_spec(full_name, linea):
    espectro_list = glob.glob(f'/Users/mariajesusfloresmoraga/Desktop/IFMR_4/P116/normalization/{linea}_2006_ifmr_corr/norm_{full_name}_{linea}.csv')

    espectro =  espectro_list[0]
    spec = pd.read_csv(espectro)

    spec.columns = spec.columns.str.strip()

    if 'wave' in spec.columns and 'norm_flux' in spec.columns:
        wave_norm = spec['wave']
        flux_norm = spec['norm_flux']


    return wave_norm, flux_norm
        
def convolution(models, lineas, linea, min_wave_search):

    model_grid = {}

    for model in models:

        teff = model["teff"]
        logg = model["logg"]
        wave_new = model["wave_new"]
        flux_new = model["flux_new"]
      
        for nombre, datos in lineas.items():
            
            if nombre == linea:
                rest = datos["rest"]
                lim_inf_ = datos["lim_inf"]
                lim_sup_ = datos["lim_sup"]
        
      
        lim_inf, lim_sup = lim_inf_, lim_sup_
      
        mask_region = (wave_new >= lim_inf) & (wave_new <= lim_sup)

      
        fwhm = rest / 18500

      
        wave_conv, flux_conv = UTILS_MJ.convolve_spectrum(wave_new[mask_region], flux_new[mask_region], fwhm = fwhm)



        model_grid[(teff, logg)] = (wave_conv, flux_conv)

    return model_grid


def calculate_chi2_vectorized(rv_array, wave_common, interp_flux_norm, 
                              wave_masked_obs, flux_masked_obs, eflux_masked_obs):


    rv_array = np.array(rv_array)
    wave_masked_obs = np.array(wave_masked_obs)
    flux_masked_obs = np.array(flux_masked_obs)   
    eflux_masked_obs = np.array(eflux_masked_obs)

    rv_grid = rv_array[:, np.newaxis] 

    wave_query = wave_masked_obs / (1 + rv_grid / c_kms)
 
    flux_shifted_interp = np.interp(wave_query, wave_common, interp_flux_norm)
    
    diff = flux_masked_obs - flux_shifted_interp

    chi2_val = np.sum((diff / eflux_masked_obs)**2, axis=1)
    
    chi2_r_val = chi2_val / len(flux_masked_obs)
    
    return chi2_r_val


def norm_model(teff, logg, wave_common, interp_flux, line_center_obs, lineas, linea, fixed_window1, fixed_window2, s=0, k=3, plot=True):
    
    res = model_continuum(wave_common, interp_flux, line_center_obs, lineas, linea, fixed_window1, fixed_window2)
    
    line_center_obs =  res["line_center_obs"]
    wave_region, flux_region = res["wave_region"], res["flux_region"]
    wave_line, flux_line = res["wave_line"], res["flux_line"]
    wave_line_cont, flux_line_cont = res["wave_line_cont"], res["flux_line_cont"] 

    orden = np.argsort(wave_line_cont)
    wave_line_cont = wave_line_cont[orden]
    flux_line_cont = flux_line_cont[orden]

    unique_x, unique_idx = np.unique(wave_line_cont, return_index=True)
    wave_line_cont = wave_line_cont[unique_idx]
    flux_line_cont = flux_line_cont[unique_idx]


    spline = UnivariateSpline(wave_line_cont, flux_line_cont, s=s, k=k)
  
        
    new_x = wave_region

    new_f = spline(new_x)

    #fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))
        
    norm = flux_region / new_f

    if plot == True:

       fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))

       ax1.plot(wave_region, flux_region, c='k')
       ax1.plot(new_x, new_f, c='orange')

       ax2.plot(new_x, norm, c='k')
       ax2.axhline(1, c='orange')

       plt.close()
       #plt.savefig(f'/Volumes/KINGSTON/lineas/models_treatment/normalizados/{star}/{teff}-{logg}.pdf', dpi=100)
       
       #plt.show()


    return new_x, norm, line_center_obs

def model_continuum(wave_common, interp_flux, line_center_obs, lineas, linea, fixed_window1, fixed_window2):

    line0 = line_center_obs - fixed_window1
    line1 = line_center_obs + fixed_window2 # will be the same??

    for nombre, datos in lineas.items():
            
            if nombre == linea:
                rest = datos["rest"]
                lim_inf_ = datos["lim_inf"]
                lim_sup_ = datos["lim_sup"]

    lim_inf, lim_sup = lim_inf_, lim_sup_

    line_cont1 = (wave_common >= lim_inf) & (wave_common < line0)
    line_spec = (wave_common >= line0 ) & (wave_common <= line1)
    line_cont2 = (wave_common > line1 ) & (wave_common <= lim_sup)

    min_region = min(line_cont1)
    max_region = max(line_cont2)

    mask_region = (wave_common >= lim_inf) & (wave_common <= lim_sup) 
    wave_region = wave_common[mask_region]
    flux_region = interp_flux[mask_region]

    mask_cont = line_cont1 | line_cont2

    wave_line = np.array(wave_common[line_spec])
    flux_line = np.array(interp_flux[line_spec])
    
    wave_line_cont = np.array(wave_common[mask_cont])
    flux_line_cont = np.array(interp_flux[mask_cont])

    wave_line_cont1 = np.array(wave_common[line_cont1])
    flux_line_cont1 = np.array(interp_flux[line_cont1])
    
    wave_line_cont2 = np.array(wave_common[line_cont2])
    flux_line_cont2 = np.array(interp_flux[line_cont2])

    return {
    "line_center_obs": line_center_obs,    
    "wave_region": wave_region,
    "flux_region": flux_region,
    "wave_line": wave_line,
    "flux_line": flux_line,
    "wave_line_cont1": wave_line_cont1,
    "flux_line_cont1": flux_line_cont1,
    "wave_line_cont2": wave_line_cont2,
    "flux_line_cont2": flux_line_cont2,
    "wave_line_cont": wave_line_cont,
    "flux_line_cont": flux_line_cont}

def get_model_interp(
    fixed_window1, 
    fixed_window2, 
    line_center_obs, 
    model_grid, 
    lineas, 
    linea, 
    teff_val, 
    logg_val, 
    plot=False,
    target_wave=None):


    teff_unique = sorted(set(k[0] for k in model_grid.keys()))
    logg_unique = sorted(set(k[1] for k in model_grid.keys()))

    teff_low = max(t for t in teff_unique if t <= teff_val)
    teff_high = min(t for t in teff_unique if t >= teff_val)
    logg_low = max(g for g in logg_unique if g <= logg_val)
    logg_high = min(g for g in logg_unique if g >= logg_val)

    teff_on_node = np.isclose(teff_low, teff_high)
    logg_on_node = np.isclose(logg_low, logg_high)

    if teff_on_node and logg_on_node:
        wave_model_base, interp_flux = model_grid[(teff_low, logg_low)] 
        
    else:

        if teff_on_node:
            tx = 0.0
        else:
            tx = (teff_val - teff_low) / (teff_high - teff_low)

        if logg_on_node:
            ty = 0.0
        else:
            ty = (logg_val - logg_low) / (logg_high - logg_low)

        m00 = model_grid.get((teff_low, logg_low))
        m01 = model_grid.get((teff_low, logg_high))
        m10 = model_grid.get((teff_high, logg_low))
        m11 = model_grid.get((teff_high, logg_high))

        corners = [m00, m01, m10, m11]

        if None in corners:
            raise ValueError(f"Missing models for interpolation corners: T={teff_val}, logg={logg_val}.")
        
   

        wmin = max(m[0].min() for m in corners)
        wmax = min(m[0].max() for m in corners)
        n_points = min(len(m[0]) for m in corners)
        wave_model_base = np.linspace(wmin, wmax, n_points) 
        
        def interp_flux_model(model): 
            w, f = model
            return np.interp(wave_model_base, w, f) 

        flux00 = interp_flux_model(m00)
        flux01 = interp_flux_model(m01)
        flux10 = interp_flux_model(m10)
        flux11 = interp_flux_model(m11)

        f_Tval_Glow = (1.0 - tx) * flux00 + tx * flux10
        f_Tval_Ghigh = (1.0 - tx) * flux01 + tx * flux11
        
        interp_flux = (1.0 - ty) * f_Tval_Glow + ty * f_Tval_Ghigh

    wave_norm, interp_flux_norm, min_flux_wavelength = norm_model(
        teff_val, logg_val, wave_model_base, interp_flux, line_center_obs, lineas, linea, fixed_window1, fixed_window2, s=0, k=1, plot=True
    )
    

    if target_wave is not None:

        
        final_interp_flux = np.interp(
            target_wave, 
            wave_norm, # Base de longitud de onda (L=12000)
            interp_flux_norm, # Base de flujo (L=12000)
            left=np.nan, 
            right=np.nan
        )
        final_interp_wave = target_wave

    
        return final_interp_wave, final_interp_flux, min_flux_wavelength
        
    else:
        
        return wave_norm, interp_flux_norm, min_flux_wavelength


def safe_get_model(datos_por_linea, linea, model_grid_cache, teff, logg):

    info = datos_por_linea
    mg = model_grid_cache

    if mg is None:
        return None, None
        
    try:

        wm, fm, _ = get_model_interp(
            info["fixed_window"], info["fixed_window"],
            info["line_center_obs"], mg,
            lineas, linea, teff, logg,
            plot=True,
            target_wave=None # <--- Importante: No alineamos aquí
        )
        
        if wm is None or fm is None or np.all(np.isnan(fm)):
            return None, None
            
        return wm, fm
        
    except:
        return None, None

def calcular_CHI2_tensor(teff_test, logg_test, rv_test, w_change, linea, line_center_obs, model_grid, wave, 
                         flux, eflux_norm, cut=False):


    fixed_window = None
    for n, datos in lineas.items():
        if n == linea:
            fixed_window = datos["fixed_window"]
            break
            
    if fixed_window is None:
        raise ValueError(f"No se encontró 'fixed_window' para la línea: {linea}")

    fit_window = w_change

    print(f'Usando {w_change}')

    chi2_tensor = np.zeros((len(teff_test), len(logg_test), len(rv_test)))


    mask = (wave >= line_center_obs - fit_window) & (wave <= line_center_obs + fit_window)
    flux_masked_obs = flux[mask].astype(np.float64)  
    eflux_masked_obs = eflux_norm[mask].astype(np.float64) 
    wave_masked_obs = wave[mask].astype(np.float64) 

    expected_rv_len = len(rv_test) 

    for i, teff_val in enumerate(teff_test):
        for j, logg_val in enumerate(logg_test):
            
           
            chi2_slice = np.full(expected_rv_len, np.inf)
   

            try:
                target_wave = wave_masked_obs 
                wave_common, interp_flux_norm, min_flux_wavelength = get_model_interp(
                    fixed_window, fixed_window, line_center_obs, model_grid, lineas, linea, teff_val, logg_val, 
                    plot=False, target_wave=target_wave)
      

                if interp_flux_norm is None or np.all(np.isnan(interp_flux_norm)):
                    print('pass')
                    pass 
                    
                else:
 
                    chi2_val = calculate_chi2_vectorized(rv_test, wave_common, interp_flux_norm, 
                                             wave_masked_obs, flux_masked_obs, eflux_masked_obs)
         
                    if isinstance(chi2_val, np.ndarray) and len(chi2_val) == expected_rv_len:
                        chi2_val[np.isnan(chi2_val)] = np.inf
                        chi2_val[np.isinf(chi2_val)] = np.inf
                        chi2_slice = chi2_val 
                    else:
                  
                        print(f"Error interno en calcular_CHI2: Resultado inválido para T={teff_val}, logg={logg_val}.")

            except Exception as e:

                print(f"Error inesperado en CHI2 para T={teff_val}, logg={logg_val}: {e}")
          

            chi2_tensor[i, j, :] = chi2_slice       

            
    return chi2_tensor, flux_masked_obs, eflux_masked_obs, wave_masked_obs, model_grid



#give_model('Ha', 10000, 7.0, 0, [Ha-15,Ha+15])


#def guardar_resultados_rv(resultado, nombre_star, linea,
#                           base_dir="/Users/mariajesusfloresmoraga/Desktop/IFMR_3/P116/boots2206_500"):
#    """
#    Guarda resultados de RV en archivo separado por línea (Ha / Hb)
#    """
#
#    nombre_star_path = nombre_star.replace('/',':')
#    outdir = os.path.join(base_dir, nombre_star_path, linea)
#    os.makedirs(outdir, exist_ok=True)
#
#    outfile = os.path.join(outdir, f"resultado_rvs_{linea}.txt")
#    
#    np.savetxt(outfile, resultado, fmt="%.8f", header="rv_km_s", comments='')
#
#    return outfile




