#!/usr/bin/env python
# coding: utf-8

# In[1]:


import astropy.units as un
import astropy.constants as cte
import func_for_boots
import glob
import numpy as np
import pandas as pd
import run_boots


# In[2]:


c_kms = cte.c.to(un.km / un.s).value

balmer_ab = ["Ha", "Hb"]
step = 1
rv_test   = np.arange(-55, 130, step)
teff_test = np.arange(5000, 26000, 250)
logg_test = np.arange(7, 8.75, 0.25)


info = pd.read_csv('info_stars_P116_centrales')

full_names = info['nombre_1'].tolist() #obsblock_nombre
obs_blocks = info['OB'].tolist()
targets = info['target'] #nombre
dates = info['date']
vel_helio = info['vhelio'].tolist() #solo 0s
central_Ha = info['central_1'].tolist()
central_Hb = info['central_2'].tolist()


espectros_norm_Ha = glob.glob('/Users/mariajesusfloresmoraga/Desktop/IFMR_3/P116/normalization/Ha_2006_ifmr/*_Ha.csv')
espectros_norm_Hb = glob.glob('/Users/mariajesusfloresmoraga/Desktop/IFMR_3/P116/normalization/Hb_2006_ifmr/*_Hb.csv')


model_dir = "/Volumes/discoSSD/WD_models/koester2_3"
all_models = func_for_boots.load_koester_models(model_dir)

modelos_aumentados = func_for_boots.aumentar_resolucion_models(all_models)


g00 = 4580 - 40
g01 = 4580 + 40
g10 = 5640 - 40
g11 = 5640 + 40

Ha = 6562.79
Hb = 4861.35


lineas = {
    "Ha": {"nplot":r"H$_{\alpha}$", "rest": Ha, "fixed_window": 135, "lim_inf": 6400, "lim_sup": 7800},
    "Hb": {"nplot":r"H$_{\beta}$" , "rest": Hb, "fixed_window": 150, "lim_inf": 4635, "lim_sup": 5050}}

model_grid_Ha = func_for_boots.convolution(modelos_aumentados, lineas, 'Ha', min_wave_search=Ha)
model_grid_Hb = func_for_boots.convolution(modelos_aumentados, lineas, 'Hb', min_wave_search=Hb)

model_grid_cache_Ha = model_grid_Ha
model_grid_cache_Hb = model_grid_Hb


# In[ ]:


# --- ESTO DEBE IR EN TU SCRIPT PRINCIPAL DE EJECUCIÓN ---

if __name__ == '__main__':
    # Todo lo que use multiprocesamiento DEBE estar dentro de este bloque


    for full_name in full_names:

        if 'cleaned_corr_spec4498579_20260315T085226383' in full_name:
        
             # 1. Ejecución para H-alpha
             # Desempaquetamos los 3 arrays que devuelve la nueva función boots
             rvs_Ha, teffs_Ha, loggs_Ha = run_boots.boots(
                 full_name,
                 "Ha",
                 N=5, plot_both = True,
                 plot_hist = True,
                 N_PROCS=8
             )
             # Si tu función guardar_resultados_rv solo procesa RV, la dejas así.
             # Si quieres, ahora también puedes pasarle teffs_Ha y loggs_Ha para                    guardar más info.
             #func_for_boots.guardar_resultados_rv(rvs_Ha, full_name, "Ha")
             
             
             # 2. Ejecución para H-beta
             rvs_Hb, teffs_Hb, loggs_Hb = run_boots.boots(
                 full_name,
                 "Hb",
                 N=5, plot_both = True,
                 plot_hist = True,
                 N_PROCS=8
             )
             #func_for_boots.guardar_resultados_rv(rvs_Hb, full_name, "Hb")
     
             print('-------------------------------------------------')


# In[ ]:




