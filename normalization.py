#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline


# In[2]:


Ha = 6562.79
Hb = 4861.35


# In[3]:


lineas = {
    "Ha": {"nplot":r"H$_{\alpha}$", "rest": Ha, "fixed_window": 65, "lim_inf": 6400, "lim_sup": 6640},
    "Hb": {"nplot":r"H$_{\beta}$", "rest": Hb, "fixed_window": 55, "lim_inf": 4787, "lim_sup": 5050}
   }


# In[4]:


#espectros = glob.glob('/Users/mariajesusfloresmoraga/Desktop/IFMR_3/P116/cleaned_comb/wds/*/*.csv')
espectros_corr = glob.glob('/Users/mariajesusfloresmoraga/Desktop/IFMR_4/P116/cleaned_corr_comp/wds/*/*.csv')


# In[5]:


def plotting_classic(espectro):
    print('espectro',espectro)
    spec = pd.read_csv(espectro, sep=',')
    
    name = espectro.split('/')[8]
    print(name)

    out_dir = '/Users/mariajesusfloresmoraga/Desktop/IFMR_4/P116/normalization/classic_plots/'
    os.makedirs(out_dir, exist_ok=True)

    wave = spec["wave"]
    flux = spec["flux"]
    
    # Creamos las máscaras para separar los brazos
    mask_blue = (wave < 5590)
    mask_red = (wave > 5690)

    plt.figure(figsize=(8, 5))
    plt.title(f'Gaia DR3 {name}', fontsize=18)

    plt.plot(wave[mask_blue], flux[mask_blue], color='k', lw=0.8)
    plt.plot(wave[mask_red], flux[mask_red], color='k', lw=0.8)
    plt.annotate((r'H$_{{\beta}}$'), xy = (0.15, 0.9), xycoords = 'axes fraction',  color='b', fontsize=15)
    plt.axvline(Hb, c='b', ls='--')
    plt.annotate((r'H$_{{\alpha}}$'), xy = (0.85, 0.65), xycoords = 'axes fraction',  color='r', fontsize=15)
    plt.axvline(Ha, c='r', ls='--')
    plt.xlabel('Wavelength (Å)', fontsize=18)
    plt.ylabel('Relative Flux', fontsize=18)
    plt.tick_params(axis='both',
                    which='major',
                    labelsize=12,
                    length=6,
                    width=1.5)
    plt.tight_layout()
    plt.xlim(6250,6600)
    #
    print('outdir', out_dir)
   
    #plt.savefig(os.path.join(out_dir, f'{name}.png'))
    plt.show()


# In[6]:


for spec in espectros_corr:
    plotting_classic(spec)


# In[7]:


def continuum_regions(name, spec, line, model=True):

    mini_window = 2

    rest = None
    fixed_window = None
    nplot = None

    for nombre, datos in lineas.items():
    
        if nombre == line:
            nplot = datos["nplot"]
            rest = datos["rest"]
            fixed_window = datos["fixed_window"]
            lim_inf = datos["lim_inf"]
            lim_sup = datos["lim_sup"]
        
    line0_ref = rest - mini_window
    line1_ref = rest + mini_window

    line_ref = (spec["wave"] >= line0_ref) & (spec["wave"] <= line1_ref)
    wave_line_ref = np.array(spec["wave"][line_ref])
    flux_line_ref = np.array(spec["flux"][line_ref])

    idx_min = np.argmin(flux_line_ref)
    min_wave_search = wave_line_ref[idx_min]       
    min_flux_search = flux_line_ref[idx_min] 

    line0 = min_wave_search - fixed_window
    line1 = min_wave_search + fixed_window

    line_cont1 = (spec["wave"] >= lim_inf) & (spec["wave"] < line0) 
    line_spec = (spec["wave"] >= line0) & (spec["wave"] <= line1)
    line_cont2 = (spec["wave"] > line1) & (spec["wave"] <= lim_sup)

    min_region = min(line_cont1)
    max_region = max(line_cont2)

    mask_region = (spec["wave"] >= lim_inf) & (spec["wave"] <= lim_sup) 
    wave_region = spec["wave"][mask_region]
    flux_region = spec["flux"][mask_region]

    mask_cont = line_cont1 | line_cont2

    wave_line = np.array(spec["wave"][line_spec])
    flux_line = np.array(spec["flux"][line_spec])
    
    wave_line_cont = np.array(spec["wave"][mask_cont])
    flux_line_cont = np.array(spec["flux"][mask_cont])

    wave_line_cont1 = np.array(spec["wave"][line_cont1])
    flux_line_cont1 = np.array(spec["flux"][line_cont1])

    wave_line_cont2 = np.array(spec["wave"][line_cont2])
    flux_line_cont2 = np.array(spec["flux"][line_cont2])

    return {
    "min_wave_search": min_wave_search,    
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


# In[8]:


def normalization_spec(name, spec, linea, path, s, k): #linea es string

    if name == 'HE0204-3821_B':
        pass

    out_dir = path
    print(name)
    os.makedirs(out_dir, exist_ok=True)
    
    res = continuum_regions(name, spec, linea)

    min_wave_search =  res["min_wave_search"]

    wave_region, flux_region = res["wave_region"], res["flux_region"]
    wave_line, flux_line = res["wave_line"], res["flux_line"]
    wave_line_cont, flux_line_cont = res["wave_line_cont"], res["flux_line_cont"] 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10), sharex = True)

    ax1.plot(wave, flux, c='gray', lw=.2)
    ax1.plot(wave_line, flux_line, c='k', lw=.2, label=f'line')
    ax1.set_xlim(min(wave_region), max(wave_region))
    ax1.set_ylim(min(flux_region), max(flux_region))
    ax1.set_ylabel('Relative Flux', fontsize=18)
    ax1.tick_params(axis='both',
                    which='major',
                    labelsize=15,
                    length=6,
                    width=1.5)
    #
    orden = np.argsort(wave_line_cont)
    wave_line_cont = wave_line_cont[orden]
    flux_line_cont = flux_line_cont[orden]
#
    ax2.axhline(1, c='orange')
    ax2.set_ylabel('Normalized Flux', fontsize=15)
    ax2.set_xlabel('Wavelength (A)', fontsize=15)
#
    fig.suptitle(f'{name} - {linea}', y=0.95, fontsize=15)
    fig.subplots_adjust(hspace=0.05)
#
    spline = UnivariateSpline(wave_line_cont, flux_line_cont, s=s, k=k)
        
    new_x = wave_region

    new_f = spline(new_x)
    ax1.plot(new_x, new_f, c='orange')
  
        
        
    norm = flux_region / new_f

    ax2.plot(new_x, norm, lw=.2, c='k')
    ax2.tick_params(axis='both',
                    which='major',
                    labelsize=15,
                    length=6,
                    width=1.5)
    ax2.axvline(4783)
    
    plt.savefig(f'{path}/norm_corr_{name}_{linea}_{s}.png', dpi=150)

    plt.close()

    df_norm = pd.DataFrame({
    "wave": new_x,
    "norm_flux": norm})
     
    csv_path = f"{path}/norm_{name}_{linea}.csv"
    print(f'resultados saved in {csv_path}')
    df_norm.to_csv(csv_path, index=False)

    return min_wave_search, new_x, norm


# In[13]:


nombres = []
dates = []
centrals = []

linea = 'Ha'
path = '/Users/mariajesusfloresmoraga/Desktop/IFMR_4/P116/normalization/Ha_2006_ifmr_corr'

out_dir = path

#import TEST_geryonfast_all as TEST

for espectro in espectros_corr:

    name = espectro.split('/')[8]
    
    spec = pd.read_csv(espectro, sep=',')
    
    wave = spec["wave"]
    flux = spec["flux"]


    s=70000000

    print(name)
    
    try:

        min_wave_search, wave, flux = normalization_spec(name, spec, linea, path, s, k=1)

        nombres.append(name)
        dates.append(name.split('_')[2])
        centrals.append(min_wave_search)
    
    except:
        print('pass')
        pass


df = pd.DataFrame({
    "nombre": nombres,
    "OBS-DATE": dates,
    "central": centrals
})



df.to_csv(f"{out_dir}/valores_centrales_corr_{linea}_500_1.csv", index=False)


# In[10]:


from pathlib import Path

# Definimos la ruta base absoluta exacta
ruta_base = Path("/Users/mariajesusfloresmoraga/Desktop/IFMR_3/P116/normalization")

# Buscamos los archivos con el patrón correcto dentro de sus respectivas carpetas
files_Ha = [str(f) for f in ruta_base.glob("Ha_2006_ifmr/norm_cleaned_*_Ha.csv")]
files_Hb = [str(f) for f in ruta_base.glob("Hb_2006_ifmr/norm_cleaned_*_Hb.csv")]

print(f"Encontrados: {len(files_Ha)} archivos de Ha y {len(files_Hb)} de Hb")


# In[11]:


for file_Ha in files_Ha:
#file_Ha = files_Ha[13]
     print(file_Ha)
     nombre = file_Ha.split('/')[8].replace('norm_cleaned_', '').replace('_Ha.csv', '')
     spec_Ha = pd.read_csv(file_Ha)
     wave = spec_Ha['wave']
     flux = spec_Ha['norm_flux']
     plt.plot(wave, flux, lw=.5, c='k')
     plt.axvline(Ha,c='r',ls='--',label='Ha')
     plt.title(f'{nombre}', fontsize=14)
     plt.xlabel('Wavelength (A)', fontsize=14)
     plt.ylabel('Norm Flux', fontsize=14)
     #plt.xlim(6467.75 - 10, 6467.75 + 10)
     plt.legend()
     #plt.savefig(f'{nombre}_Ha_zoom.png',dpi=200)
     plt.show()


# In[12]:


for file_Hb in files_Hb:
#file_Ha = files_Ha[13]

     nombre = file_Hb.split('/')[8].replace('norm_cleaned_', '').replace('_Hb.csv', '')
     spec_Hb = pd.read_csv(file_Hb)
     wave = spec_Hb['wave']
     flux = spec_Hb['norm_flux']
     plt.plot(wave, flux, lw=.5, c='k')
     plt.axvline(Hb,c='b',ls='--',label='Hb')
     plt.title(f'{nombre}', fontsize=14)
     plt.xlabel('Wavelength (A)', fontsize=14)
     plt.ylabel('Norm Flux', fontsize=14)
     #plt.xlim(Hb - 10, Hb + 10)
     plt.legend()
     #plt.savefig(f'{nombre}_Ha_zoom.png',dpi=200)
     plt.show()


# In[ ]:




