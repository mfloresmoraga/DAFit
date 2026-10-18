#!/usr/bin/env python
# coding: utf-8

import os
import time  
import numpy as np
import pandas as pd  
import func_for_boots
from concurrent.futures import ProcessPoolExecutor
import astropy.units as un
import astropy.constants as cte
import matplotlib.pyplot as plt 

g10 = 5640 - 40
g11 = 5640 + 40

Ha = 6562.79
Hb = 4861.35


lineas = {
    "Ha": {"nplot":r"H$_{\alpha}$", "rest": Ha, "fixed_window": 135, "lim_inf": 6400, "lim_sup": 7000},
    "Hb": {"nplot":r"H$_{\beta}$" , "rest": Hb, "fixed_window": 150, "lim_inf": 4635, "lim_sup": 5050}}

model_dir = "/Volumes/discoSSD/WD_models/koester2_3"
all_models = func_for_boots.load_koester_models(model_dir)

modelos_aumentados = func_for_boots.aumentar_resolucion_models(all_models)

model_grid_Ha = func_for_boots.convolution(modelos_aumentados, lineas, 'Ha', min_wave_search=Ha)
model_grid_Hb = func_for_boots.convolution(modelos_aumentados, lineas, 'Hb', min_wave_search=Hb)

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

def give_model(linea, teff, logg, rv, xlims):

    if linea == 'Ha':
        model_grid = model_grid_Ha
        rest = Ha
    if linea == 'Hb':
        model_grid = model_grid_Hb
        rest = Hb

    wm_model, fm_model = func_for_boots.safe_get_model(
        {"fixed_window": lineas[linea]["fixed_window"],
         "line_center_obs": rest},
        linea, model_grid, teff, logg
    )

    
    wm_shifted = wm_model * (1 + rv/c_kms)
    

    if xlims != None:
        plt.xlim(xlims[0],xlims[1])

    #plt.plot(wm_model, fm_model,zorder=2)    
    plt.plot(wm_shifted, fm_model, c = 'red', lw=2, zorder=2)
    plt.axvline(rest)
    

    return

# 1. FUNCIÓN GLOBAL AUXILIAR (Modificada con Seed independiente)
def sample(
    nombre,
    outdir,
    teff_test,
    logg_test,
    rv_test,
    rv_guess,
    i,
    line_center_obs,
    fm_model,
    error,
    model_grid,
    linea,
    wm_model,
    v_helio,
    seed,
):
    t0 = time.time()

    # --- CLAVE: Generador aleatorio independiente por cada proceso ---
    rng = np.random.default_rng(seed)

    # Ruido gaussiano independiente usando el generador rng
    flux_bootstrap = fm_model + rng.normal(0, 1, len(fm_model)) * error
    wave_bootstrap_shift = wm_model * (1 + rv_guess / c_kms)
    eflux_norm_error = np.full_like(fm_model, error)

    chi2_bootstrap_tensor, _, _, _, _ = func_for_boots.calcular_CHI2_tensor(
        teff_test,
        logg_test,
        rv_test,
        linea,
        line_center_obs,
        model_grid,
        wave_bootstrap_shift,
        flux_bootstrap,
        eflux_norm_error,
        cut=False,
    )

    min_index_tuple_bs = np.unravel_index(
        np.argmin(chi2_bootstrap_tensor), chi2_bootstrap_tensor.shape
    )

    best_teff_linea = teff_test[min_index_tuple_bs[0]]
    best_logg_linea = logg_test[min_index_tuple_bs[1]]
    best_rv_linea = rv_test[min_index_tuple_bs[2]]

    t1 = time.time()

    print(f"Bootstrap {i} listo en {t1-t0:.2f} s", flush=True)

    return best_rv_linea, best_teff_linea, best_logg_linea


def _boots_worker(task_tuple):
    return sample(*task_tuple)


def boots(full_name, linea, N, plot_both=False, plot_hist=False, N_PROCS=4):
        
    target = full_name.split("_")[3]
    outdir = os.path.join(f"boots2206_p116/wds/{full_name}/", linea)
    os.makedirs(outdir, exist_ok=True)

    file_rv = os.path.join(outdir, f"bootstrap_rv_{N}.txt")
    file_teff = os.path.join(outdir, f"bootstrap_teff_{N}.txt")
    file_logg = os.path.join(outdir, f"bootstrap_logg_{N}.txt")

    read_rvs = []
    read_teffs = []
    read_loggs = []

    # Comprobación inicial si ya corrió previamente
    if (
        os.path.exists(file_rv)
        and os.path.exists(file_teff)
        and os.path.exists(file_logg)
    ):
        print(
            f"--> [{full_name} - {linea}] ya corrió previamente. Cargando datos del disco...",
            flush=True,
        )
        with open(file_rv, "r") as f:
            read_rvs = [float(line.strip()) for line in f if line.strip()]
        with open(file_teff, "r") as f:
            read_teffs = [float(line.strip()) for line in f if line.strip()]
        with open(file_logg, "r") as f:
            read_loggs = [float(line.strip()) for line in f if line.strip()]

        return np.array(read_rvs), np.array(read_teffs), np.array(read_loggs)

    print(
        f"--> [{full_name} - {linea}] no detectado. Iniciando procesamiento...",
        flush=True,
    )

    if linea == "Ha":
        index = 0
        model_grid = model_grid_Ha
    else:
        index = 1
        model_grid = model_grid_Hb

    info_ret = func_for_boots.informacion_star(target)
    if info_ret is None:
        raise SystemExit("No hay información de la estrella.")

    date, v_helio, centrales, *_ = info_ret
    wave, flux = func_for_boots.find_spec(full_name, linea)

    eflux_norm = np.full_like(flux, np.std(flux))
    line_center_obs = centrales[index]

    # Usamos explícitamente los arrays desde func_for_boots
    teff_test = func_for_boots.teff_test
    logg_test = func_for_boots.logg_test
    rv_test = func_for_boots.rv_test

    if full_name == 'cleaned_corr_spec4498551_20251025T001253572':
        w_change = 30
        
    else:

        w_change = 15

    # FIT GLOBAL
    chi2_tensor, flux_m, eflux_m, wave_m, model_grid = (
        func_for_boots.calcular_CHI2_tensor(
            teff_test,
            logg_test,
            rv_test,
            w_change,
            linea,
            line_center_obs,
            model_grid,
            wave,
            flux,
            eflux_norm,
        )
    )

    i0, j0, k0 = np.unravel_index(np.nanargmin(chi2_tensor), chi2_tensor.shape)
    best_teff = teff_test[i0]
    best_logg = logg_test[j0]
    best_rv = rv_test[k0]

    if plot_both:

        if full_name == 'cleaned_corr_spec4498551_20251025T001253572':
            give_model(
            linea,
            best_teff,
            best_logg,
            best_rv,
            xlims=[line_center_obs - 30, line_center_obs + 30],
            )

        else:
            
           give_model(
               linea,
               best_teff,
               best_logg,
               best_rv,
               xlims=[line_center_obs - 15, line_center_obs + 15],
               )
        plt.plot(wave, flux, c="k", lw=0.5, zorder=1)
        plt.savefig(f"{outdir}/plot_both.png", dpi=200)
        plt.close()

    print(
        f"Best Fit Global -> Teff: {best_teff}, Logg: {best_logg}, RV: {best_rv}",
        flush=True,
    )

    wm_model, fm_model = func_for_boots.safe_get_model(
        {
            "fixed_window": func_for_boots.lineas[linea]["fixed_window"],
            "line_center_obs": line_center_obs,
        },
        linea,
        model_grid,
        best_teff,
        best_logg,
    )

    mask = (wm_model >= wave_m.min()) & (wm_model <= wave_m.max())
    fm_interp = np.interp(wave_m, wm_model[mask], fm_model[mask])
    error = np.std(flux_m - fm_interp)

    # --- Generación de Semillas Estadísticamente Independientes ---
    ss = np.random.SeedSequence(12345)
    child_seeds = ss.spawn(N)

    # Preparar tareas para el Bootstrap paralelo
    tasks = []
    for i in range(N):
        tasks.append(
            (
                date,
                outdir,
                teff_test,
                logg_test,
                rv_test,
                best_rv,
                i,
                line_center_obs,
                fm_model,
                error,
                model_grid,
                linea,
                wm_model,
                v_helio,
                child_seeds[i],  # <--- Semilla única para cada iteración
            )
        )

    print(
        f"Iniciando {len(tasks)} cálculos en PARALELO usando {N_PROCS} procesadores...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
        resultados = list(executor.map(_boots_worker, tasks))

    for res in resultados:
        resultado_rv, resultado_teff, resultado_logg = res
        read_rvs.append(float(resultado_rv))
        read_teffs.append(float(resultado_teff))
        read_loggs.append(float(resultado_logg))

    # Guardar archivos en el disco
    with open(file_rv, "w") as f:
        for val in read_rvs:
            f.write(f"{val}\n")
    with open(file_teff, "w") as f:
        for val in read_teffs:
            f.write(f"{val}\n")
    with open(file_logg, "w") as f:
        for val in read_loggs:
            f.write(f"{val}\n")

    print("Resultados guardados exitosamente en paralelo.", flush=True)

    if plot_hist:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        ax1.hist(read_rvs, color="salmon", edgecolor="black")
        ax1.set_title("RV Distribution")
        ax1.set_xlabel("RV (km/s)")
        ax1.set_ylabel("Frequency")

        ax2.hist(read_teffs, color="lightgreen", edgecolor="black")
        ax2.set_title("Teff Distribution")
        ax2.set_xlabel("Teff (K)")

        ax3.hist(read_loggs, color="violet", edgecolor="black")
        ax3.set_title("logg Distribution")
        ax3.set_xlabel("log(g)")

        plt.tight_layout()
        plt.savefig(f"{outdir}/plot_hist.png", dpi=200)
        plt.close()

    return np.array(read_rvs), np.array(read_teffs), np.array(read_loggs)
    
#if __name__ == '__main__':
    
    # 1. MOVIDO AQUÍ: Primero leemos los datos antes de llamar al bucle for
#    info = pd.read_csv('info_stars_P116_centrales')
#    full_names = info['nombre_1'].tolist() 
#    obs_blocks = info['OB'].tolist()
#    targets = info['target'] 
#    dates = info['date']
#    vel_helio = info['vhelio'].tolist() 
#    central_Ha = info['central_1'].tolist()
#    central_Hb = info['central_2'].tolist()
#
#    # 2. Ahora sí corremos el bucle con las variables ya cargadas en memoria
#    for full_name in full_names:
#        
#        # Ejecución para H-alpha
#        rvs_Ha, teffs_Ha, loggs_Ha = boots(
#            full_name,
#            "Ha",
#            N=2,
#            N_PROCS=8
#        )
#        #func_for_boots.guardar_resultados_rv(rvs_Ha, full_name, "Ha")
#        
#        # Ejecución para H-beta
#        rvs_Hb, teffs_Hb, loggs_Hb = boots(
#            full_name,
#            "Hb",
#            N=2,
#            N_PROCS=8
#        )
#        #func_for_boots.guardar_resultados_rv(rvs_Hb, full_name, "Hb")
#
#        print('-------------------------------------------------')