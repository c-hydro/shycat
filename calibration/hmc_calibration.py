#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HMC tools - Calibration

__date__    = '20251216'
__version__ = '1.6.6'
__author__  = 'Andrea Libertino (andrea.libertino@cimafoundation.org)'
              'Lorenzo Campo (lorenzo.campo@cimafoundation.org)'
              'Lorenzo Alfieri (lorenzo.alfieri@cimafoundation.org)'
__library__ = 'HMC_calibration_tool'

General command line:
    python3 HMC_calibration -settings_file "FILE.json"

Changelog:
20201130 (0.0.1) -->    Beta release single domain
20210226 (1.0.0) -->    Separate multi-domain branch with discharge-only support
20220302 (1.1.0) -->    Changed calibration approach
20220513 (1.1.1) -->    Bug fixes
20230330 (1.2.0) -->    Add logarithmic scale sampling
                        Add lakes mask
20240605 (1.3.0) -->    Fix coercing of section data null values
20240920 (1.4.0) -->    Add check on the range of input maps
20250505 (1.5.0) -->    Fixed check on range of input maps with lakes
20250605 (1.5.1) -->    Skip stations with all-NaN observations, handle removed duplicate stations
20251006 (1.6.0) -->    Check that input maps respect the range (except lakes), if not, rescale them prior the start
                        Exclude lakes values from all calculations
                        Allow lakes values from json, or hard-coded defaults
                        Flag algorithm.flags.rescale_input_out_of_range to enable/disable pre-rescaling (default False)
20251216 (1.6.6) -->    General reorganization of the script
                        Adding diagnostic plots
"""
# -------------------------------------------------------------------------------------
# Libraries
import numpy as np
import os, pickle, math, time
import rasterio as rio
import pandas as pd
from pyDOE import lhs
from cdo import *
import logging
from datetime import date
from argparse import ArgumentParser
import json
import shutil
import warnings
from osgeo import gdal
import datetime as dt
import hydrostats as hs
import subprocess
from typing import Optional

# -------------------------------------------------------------------------------------
# Algorithm info
alg_name = 'HMC tools - Calibration'
alg_version = '1.6.6'
alg_release = '2025-12-16'
time_format = '%Y%m%d%H%M'


# -------------------------------------------------------------------------------------
# Script main

def main():
    """
    Main driver for multi-domain HMC calibration.
    1) Read settings.
    2) Prepare input rasters & masks.
    3) Read observed discharge.
    4) LHS sampling -> run HMC -> compute errors.
    5) Refine ranges & repeat until convergence or max iterations.
    6) Write out final best-fit parameter maps.
    """
    start_time = time.time()

    # ------------------- 1. Read settings, initialize -------------------
    alg_settings = get_args()
    data_settings = read_file_json(alg_settings)

    # Flags: presence check, default False if missing
    rescale_out_flag = data_settings.get("algorithm", {}).get("flags", {}).get(
        "rescale_input_out_of_range", False
    )
    delete_calib_data_flag = data_settings.get("algorithm", {}).get("flags", {}).get(
        "delete_calib_data", False
    )

    domain = data_settings['algorithm']['general']['domain_name']
    path_settings = data_settings["algorithm"]["path"]

    iMin = data_settings['algorithm']['general']['start_with_iteration_number']
    iMax = iMin + data_settings['algorithm']['general']['max_number_of_iterations'] - 1
    nExplor = data_settings['algorithm']['general']['number_of_points_first_iteration']

    calibrated_params = [
        var for var in data_settings['calibration']['parameters'].keys()
        if data_settings['calibration']['parameters'][var]["calibrate"]
    ]

    run_hydro_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["run_hydro_start"], "%Y-%m-%d %H:%M")
    run_hydro_end = dt.datetime.strptime(data_settings["algorithm"]["time"]["run_hydro_end"], "%Y-%m-%d %H:%M")
    calib_hydro_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["calib_hydro_start"], "%Y-%m-%d %H:%M")

    implemented_approaches = {
        "rescale": rescale_map,  # uses a base map and modifies it using the sampled parameter
        "mask": rescale_mask,  # uses the parameter-specific base mask map * par
        "uniform": rescale_value  # constant value over the whole domain mask
    }

    # ------------------- 2) Folders & logging -------------------
    os.makedirs(path_settings["work_path"], exist_ok=True)
    os.makedirs(path_settings["out_path"], exist_ok=True)
    os.makedirs(path_settings["log_path"], exist_ok=True)

    set_logging(logger_file=os.path.join(path_settings['log_path'], domain + "_calibration.log"))

    logging.info('============================================================================')
    logging.info(f'==> {alg_name} (Version: {alg_version} Release_Date: {alg_release})')
    logging.info(f'==> TIME : {date.today().strftime("%d-%B-%Y %H:%M")}')
    logging.info('==> START ...')
    logging.info('==> ALGORITHM SETTINGS <==')
    logging.info(f'--> Domain: {domain}')
    logging.info(' ')

    # Parameter limits
    param_limits = pd.DataFrame(index=['min', 'max', 'sigma', 'best'], columns=calibrated_params)

    # Log10 transform of min/max for log_scale params
    for par in calibrated_params:
        if not data_settings['calibration']['parameters'][par].get("log_scale", False):
            data_settings['calibration']['parameters'][par]["log_scale"] = False
        if data_settings['calibration']['parameters'][par]["log_scale"]:
            data_settings['calibration']["parameters"][par]['min'] = np.log10(
                data_settings['calibration']["parameters"][par]['min']
            )
            data_settings['calibration']["parameters"][par]['max'] = np.log10(
                data_settings['calibration']["parameters"][par]['max']
            )

    for par in calibrated_params:
        for lim in ['min', 'max']:
            param_limits.loc[lim][par] = data_settings['calibration']['parameters'][par][lim]
        param_limits.loc['sigma', par] = np.nan

    # ------------------- 3) Static rasters & base maps -------------------
    logging.info(' --> Import domain land data')
    logging.info(' ---> Load DEM...')
    maps_in = {}

    dem = rio.open(os.path.join(
        data_settings["calibration"]["input_gridded_data_folder"], domain + ".dem.txt"
    ))
    header = dem.profile
    header["driver"] = 'GTiff'

    # Domain mask (1 = valid, NaN = invalid)
    mask_path = os.path.join(data_settings["calibration"]["input_base_maps"], domain + ".mask.txt")
    if os.path.isfile(mask_path):
        logging.info(" ---> Mask file available, use it!")
        maps_in['mask'] = rio.open(mask_path).read(1)
        maps_in['mask'][maps_in['mask'] == 0] = np.nan
    else:
        logging.info(" ---> Mask file not available, compute domain from DEM!")
        dem_temp = rio.open(os.path.join(
            data_settings["calibration"]["input_gridded_data_folder"], domain + ".dem.txt"
        )).read(1)
        maps_in['mask'] = np.where(dem_temp >= -9000, 1, np.nan)

    maps_in['DEM'] = np.where(maps_in['mask'] == 1, dem.read(1), np.nan)

    logging.info(' ---> Load base maps for calibration...')
    available_approaches = set(
        data_settings['calibration']['parameters'][var]["approach"]
        for var in calibrated_params
    )

    calibrated_params_approach = {}
    for approach in available_approaches:
        if approach not in implemented_approaches:
            logging.warning(f" ---> WARNING! Approach '{approach}' not implemented!")

        calibrated_params_approach[approach] = [
            var for var in calibrated_params
            if data_settings['calibration']['parameters'][var]["approach"] == approach
        ]

        # NOTE:
        # - For 'rescale': load the parameter's base map, apply domain mask, ignore lakes (set to NaN),
        #   then check range and optionally pre-rescale if flag is True. Store into maps_in[par].
        # - For 'mask':   load the parameter's *mask base map* (area of parameter) into maps_in[par].
        # - For 'uniform': no base map is needed; it will use maps_in['mask'] at generation time.

        if approach in ("rescale", "mask"):
            for par in calibrated_params_approach[approach]:
                base_map_path = os.path.join(
                    data_settings["calibration"]["input_base_maps"], domain + "." + par + ".txt"
                )

                # Read base map in the same "space" as parameter bounds if approach=='rescale'
                if approach == "rescale" and data_settings['calibration']['parameters'][par]["log_scale"]:
                    lin_base = rio.open(base_map_path).read(1).astype(float)
                    # Robust log10: treat non-finite and non-positive values as NaN (they will be excluded by the mask)
                    lin_base[~np.isfinite(lin_base)] = np.nan
                    lin_base[lin_base <= 0] = np.nan
                    with np.errstate(divide='ignore', invalid='ignore'):
                        raw_arr = np.log10(lin_base)
                    if np.isnan(raw_arr).all():
                        raise ValueError(f"{par} base map becomes all-NaN after log10 (check nodata/zeros/negatives)")
                else:
                    raw_arr = rio.open(base_map_path).read(1)

                # Apply domain mask and optionally exclude lakes (keep lakes as NaN)
                if data_settings['calibration']['parameters'][par].get('lakes_mask', None):
                    lakes_arr = rio.open(data_settings['calibration']["parameters"][par]["lakes_mask"]).read(1)
                    valid_mask = (maps_in['mask'] == 1) & (lakes_arr != 1)
                    arr = np.where(valid_mask, raw_arr, np.nan)
                else:
                    arr = np.where(maps_in['mask'] == 1, raw_arr, np.nan)

                # Only for 'rescale': check input map range vs [min,max] and pre-rescale if requested
                if approach == "rescale":
                    par_min = data_settings['calibration']["parameters"][par]['min']
                    par_max = data_settings['calibration']["parameters"][par]['max']
                    vmin = np.nanmin(arr)
                    vmax = np.nanmax(arr)

                    if (vmax > par_max) or (vmin < par_min):
                        if rescale_out_flag:
                            logging.warning(
                                f" ---> '{par}' base map out of range ({vmin:.6g}..{vmax:.6g}) "
                                f"vs [{par_min},{par_max}]. Rescaling to fit."
                            )
                            valid = np.isfinite(arr)
                            if np.any(valid) and (vmax > vmin):
                                a = (par_max - par_min) / (vmax - vmin)
                                b = par_min - a * vmin
                                arr[valid] = a * arr[valid] + b
                            else:
                                logging.error(f" ---> '{par}': degenerate range, cannot rescale.")
                                raise ValueError(f"{par} map cannot be rescaled due to degenerate range.")
                        else:
                            logging.error(f" ---> ERROR! '{par}' map outside allowed range [{par_min}, {par_max}].")
                            raise ValueError(f"{par} map contains values outside the allowed range")

                # Store base for this parameter (used by rescale_map and rescale_mask)
                maps_in[par] = arr

        logging.info(
            f" ----> Param(s) calibrated with approach '{approach}': "
            + ", ".join(calibrated_params_approach[approach])
        )

    # Initial sigma
    for par in calibrated_params:
        if par in calibrated_params_approach.get("rescale", []):
            param_limits.loc['sigma'][par] = 0.5  # arctan neighborhood width
        else:
            param_limits.loc['sigma', par] = (
                    param_limits.loc['max', par] - param_limits.loc['min', par]
            )

    logging.info(' ---> Preparing domain land data.. OK!')

    # ------------------- 4) Observations -------------------
    logging.info(' ---> Read section data...')
    calibration_period = pd.date_range(
        calib_hydro_start,
        run_hydro_end,
        freq=data_settings["data"]["hydro"]["calib_hydro_resolution"]
    )

    section_data = {}
    section_file = os.path.join(
        data_settings["calibration"]["input_point_data_folder"],
        domain + ".info_section.txt"
    )
    area_file = os.path.join(
        data_settings["calibration"]["input_gridded_data_folder"],
        domain + ".area.txt"
    )

    sections = pd.read_csv(
        section_file,
        sep="\s",
        header=None,
        names=["row_HMC", "col_HMC", "basin", "name"],
        usecols=[0, 1, 2, 3]
    )
    sections["area_ncell"] = np.nan

    all_names = sections["name"].tolist()
    dup_names = [n for n in set(all_names) if all_names.count(n) > 1]
    if dup_names:
        logging.error(f" ---> ERROR: Duplicate station name(s) found in info_section.txt: {dup_names}")
        raise ValueError(f"Duplicate station names must be removed or renamed in {section_file}: {dup_names}")

    area = rio.open(area_file).read(1)

    logging.info('---> Search observed series')

    for section, basin in zip(sections["name"], sections["basin"]):
        logging.info(f'---> Section: {section}')

        row_idx = int(sections.loc[sections["name"] == section, "row_HMC"].iloc[0]) - 1
        col_idx = int(sections.loc[sections["name"] == section, "col_HMC"].iloc[0]) - 1
        sections.loc[sections["name"] == section, "area_ncell"] = area[row_idx, col_idx]

        file_name_sec = os.path.join(
            data_settings["data"]["hydro"]["folder"],
            data_settings["data"]["hydro"]["filename"]
        ).format(
            domain=domain,
            section_name=section,
            section_basin=basin
        )

        if not os.path.isfile(file_name_sec):
            logging.warning(f'---> WARNING! File not found for section: {section}')
            continue

        date_fmt = data_settings["data"]["hydro"]["date_fmt"]

        common_kwargs = dict(
            sep=data_settings["data"]["hydro"]["sep"],
            usecols=[data_settings["data"]["hydro"]["date_col"], data_settings["data"]["hydro"]["value_col"]],
            names=["date", "value"],
            index_col=["date"],
            parse_dates=["date"],
            na_values=data_settings["data"]["hydro"]["null_values"],
            header=None,  # ensure no header is inferred when 'names' is given
        )

        try:
            # pandas ≥ 2.0 fast-path (if available on your system)
            df_sec = pd.read_csv(file_name_sec, date_format=date_fmt, **common_kwargs)
        except TypeError:
            # pandas 1.x fallback
            df_sec = pd.read_csv(
                file_name_sec,
                date_parser=lambda x: dt.datetime.strptime(x, date_fmt),
                **common_kwargs
            )

        # subset and align to calibration period
        df_sec = df_sec[calib_hydro_start:run_hydro_end].reindex(
            calibration_period,
            method="nearest",
            tolerance="1" + data_settings["data"]["hydro"]["calib_hydro_resolution"]
        )

        # force numeric
        df_sec["value"] = pd.to_numeric(df_sec["value"], errors="coerce")

        if df_sec["value"].dropna().empty:
            logging.warning(
                f'---> WARNING! Section {section} has only NaN values in calibration period. Skipping.'
            )
            continue

        section_data[section] = df_sec
        logging.info(f'---> Section: {section} ... IMPORTED!')

    logging.info(
        f' ---> {len(section_data)} valid section(s) imported out of {len(sections)} total.'
    )
    logging.info(' ---> Read section data...DONE!')

    # ------------------- 5) Multi-iterative loop -------------------
    converges = False
    best_score_iter = {}

    for iIter in np.arange(iMin, iMax + 1):
        maps_iter = {}

        # Load previous iteration
        if iIter > 1:
            logging.info(f' ---> Loading results of iteration {iIter - 1:02d}')
            prev_path = os.path.join(
                data_settings["algorithm"]["path"]["out_path"],
                f'ITER{(iIter - 1):02d}_results.pickle'
            )
            with open(prev_path, "rb") as handle:
                previous_iter = pickle.load(handle)
            logging.info(f' ---> Loading results of iteration {iIter - 1:02d}...DONE')

            best_score_iter = previous_iter["best_score_iter"]

            if data_settings["algorithm"]["general"]["error_metrics"]["best_value"] == "max":
                idx_best = np.nanargmax(previous_iter["scores_iter"]["tot"].values) + 1
                best_score_iter[iIter - 1] = np.nanmax(previous_iter["scores_iter"]["tot"].values)
            else:
                idx_best = np.nanargmin(previous_iter["scores_iter"]["tot"].values) + 1
                best_score_iter[iIter - 1] = np.nanmin(previous_iter["scores_iter"]["tot"].values)

            logging.info(f" ---> Best combination for iteration {iIter - 1:02d} is combination: {idx_best:03d}")

            if iIter > 2:
                prev_best = best_score_iter[iIter - 1]
                prev_prev_best = best_score_iter[iIter - 2]
                improvement = abs((prev_best - prev_prev_best) / prev_best)
                thresh = data_settings["algorithm"]["general"]["percentage_min_improvement_quit_optimization"] / 100.0
                if improvement < thresh:
                    logging.info(f" --> Converged: improvement {improvement:.4f} < threshold {thresh:.4f}")
                    maps_out = previous_iter["maps_iter"][idx_best]
                    converges = True
                    break
                else:
                    logging.info(f" --> Improvement compared to previous iteration: {improvement:.4f}")

            nExplor = int(
                nExplor * data_settings["algorithm"]["general"]["percentage_samples_successive_iterations"] / 100
            )
            param_limits = previous_iter["param_limits"]
            param_bests = previous_iter["param"].loc[idx_best]

            for par in calibrated_params:
                if par in previous_iter["maps_iter"][idx_best]:
                    maps_in[par] = previous_iter["maps_iter"][idx_best][par]
                param_limits.loc['sigma', par] = (
                        param_limits.loc['sigma', par]
                        * (data_settings["algorithm"]["general"]["percentage_param_range_reduction"] / 100)
                )

                if par not in calibrated_params_approach.get("rescale", []):
                    new_min = np.max((
                        param_limits.loc['min', par],
                        param_bests[par] - param_limits.loc['sigma', par] / 2
                    ))
                    new_max = np.min((
                        param_limits.loc['max', par],
                        param_bests[par] + param_limits.loc['sigma', par] / 2
                    ))
                    param_limits.loc['min', par] = new_min
                    param_limits.loc['max', par] = new_max

        # 5.1 LHS sampling
        logging.info(f' --> Initialize iteration ITER{iIter:02d}')
        seedIter = pd.DataFrame(
            np.array(lhs(len(calibrated_params), nExplor)),
            index=np.arange(1, nExplor + 1),
            columns=calibrated_params
        )
        param = pd.DataFrame(index=np.arange(1, nExplor + 1), columns=calibrated_params).fillna(0.0)

        for par in calibrated_params:
            if par in calibrated_params_approach.get("rescale", []):
                # Asymmetric neighborhood around the base-map mean
                if "lakes_mask" in data_settings['calibration']["parameters"][par]:
                    lakes_mask = rio.open(data_settings['calibration']["parameters"][par]["lakes_mask"]).read(1)
                    map_clean = np.where(lakes_mask == 1, np.nan, maps_in[par])
                else:
                    map_clean = maps_in[par]

                diff_inf = abs(np.nanmean(map_clean) - param_limits[par]['min'])
                diff_sup = abs(np.nanmean(map_clean) - param_limits[par]['max'])
                min_scale = -(param_limits.loc['sigma'][par] * diff_inf / (diff_sup + diff_inf))
                max_scale = (param_limits.loc['sigma'][par] * diff_sup / (diff_sup + diff_inf))
            else:
                # mask/uniform: sample within [min,max]
                min_scale = param_limits[par]['min']
                max_scale = param_limits[par]['max']

            col = seedIter[par]
            param[par] = ((col - col.min()) / (col.max() - col.min())) * (max_scale - min_scale) + min_scale

        # 5.2 Prepare runs
        logging.info(' ---> Setup explorative runs...')
        translate_options = gdal.TranslateOptions(
            format="AAIGrid",
            outputType=gdal.GDT_Float32,
            noData=-9999,
            creationOptions=['FORCE_CELLSIZE=YES']
        )

        with open(data_settings["data"]["hmc"]["model_settings"]) as f:
            config_hmc_in = f.read()

        for iExplor in np.arange(1, nExplor + 1):
            logging.info(f' --->  ITER{iIter:02d}-{iExplor:03d}')
            iterPath = os.path.join(path_settings["work_path"], "simulations", f'ITER{iIter:02d}-{iExplor:03d}')
            os.makedirs(iterPath, exist_ok=True)

            # Copy static gridded inputs
            logging.info(' ----> Copy all static maps...')
            iter_gridded_path = os.path.join(iterPath, 'gridded')
            os.makedirs(iter_gridded_path, exist_ok=True)
            copy_all_files(data_settings["calibration"]["input_gridded_data_folder"], iter_gridded_path)

            # Generate parameter maps for this run
            logging.info(' ----> Generate parameters maps...')
            maps_out = {}
            for par in calibrated_params:
                maps_out[par] = implemented_approaches[
                    data_settings['calibration']['parameters'][par]["approach"]
                ](
                    par,
                    param[par][iExplor],
                    data_settings['calibration']["parameters"][par],
                    maps_in
                )

                with rio.open(os.path.join(iter_gridded_path, 'temp.tif'), 'w', **header) as dst:
                    out = maps_out[par].astype(float)
                    out[~np.isfinite(out)] = np.nan
                    if data_settings['calibration']['parameters'][par].get('log_scale', False):
                        with np.errstate(over='ignore', invalid='ignore'):
                            out = 10 ** out
                        out[~np.isfinite(out)] = np.nan
                    dst.write(np.nan_to_num(out, nan=-9999).astype(rio.float32), 1)

                gdal.Translate(
                    os.path.join(iter_gridded_path, f'{domain}.{par}.txt'),
                    os.path.join(iter_gridded_path, 'temp.tif'),
                    options=translate_options
                )
                os.remove(os.path.join(iter_gridded_path, 'temp.tif'))

            logging.info(' ---> Generate exploration static maps...DONE')

            # Copy point data
            logging.info(' ----> Copy point data...')
            iter_point_path = os.path.join(iterPath, 'point')
            os.makedirs(iter_point_path, exist_ok=True)
            copy_all_files(data_settings["calibration"]["input_point_data_folder"], iter_point_path)
            logging.info(' ----> Copy point data...DONE')

            # HMC executable and launcher
            logging.info(' ----> Copy and setup model executable...')
            iter_exe_path = os.path.join(iterPath, 'exe')
            iter_out_path = os.path.join(iterPath, 'outcome')
            os.makedirs(iter_exe_path, exist_ok=True)

            shutil.copy(
                data_settings["data"]["hmc"]["model_exe"],
                os.path.join(iter_exe_path, "HMC3_calib.x")
            )
            config_hmc_out = config_hmc_in.format(
                domain=f'"{domain}"',
                sim_length=str(int((run_hydro_end - run_hydro_start).total_seconds() / 3600)),
                run_hydro_start=run_hydro_start.strftime("%Y%m%d%H%M"),
                path_gridded=iter_gridded_path,
                path_point=iter_point_path,
                path_output=iter_out_path
            )
            with open(os.path.join(iter_exe_path, f"{domain}.info.txt"), "w") as f:
                f.write(config_hmc_out)

            make_launcher(iter_exe_path, domain, data_settings["data"]["hmc"]["system_env_libraries"])
            logging.info(' ----> Copy and setup model executable...DONE')

            maps_iter[iExplor] = maps_out

        # -------------------------------------------------------------------------------------
        # Diagnostics (pre-run): parameter boxplots
        # Generate boxplots right after creating the parameter maps, before launching HMC.
        flags = data_settings['algorithm'].get('flags', {})
        if flags.get('plot_param_boxplots', False):
            try:
                plots_dir = os.path.join(path_settings['out_path'], 'plots', 'boxplots', f'ITER{iIter:02d}')
                max_boxes = flags.get('boxplot_max_combinations', None)
                plot_param_boxplots_matplotlib(
                    domain=domain,
                    iiter=iIter,
                    maps_iter=maps_iter,
                    out_dir=plots_dir,
                    calibration_parameters=data_settings['calibration']['parameters'],
                    max_boxes=max_boxes
                )
                logging.info(' ---> Pre-run boxplots saved')
            except Exception as e:
                logging.warning(f' ---> Pre-run boxplots failed: {e}')

        # 5.3 Launch runs (disabled here; left as-is)
        bashCommand = (
                "for iIt in $(seq -f \"%03g\" 1 " + str(nExplor) + "); do "
                                                                   "cd " + os.path.join(path_settings["work_path"],
                                                                                        f"simulations/ITER{iIter:02d}") + "-$iIt/exe/; "
                                                                                                                          "chmod +x launcher.sh; ./launcher.sh & done\n wait"
        )
        subprocess.run(bashCommand, shell=True, executable="/bin/bash")
        logging.info(' --> Simulation runs... OK!')

        # 5.4 Read HMC output
        logging.info(' --> Read model output...')
        hmc_results = {}
        for iExplor in np.arange(1, nExplor + 1):
            logging.info(f'---> Read results of ITER{iIter:02d}-{iExplor:03d}')
            iterPath = os.path.join(path_settings["work_path"], "simulations", f'ITER{iIter:02d}-{iExplor:03d}')
            iter_settings_file = os.path.join(iterPath, 'exe', f"{domain}.info.txt")

            with open(iter_settings_file, 'r') as input_f:
                iter_out_path = None
                for line in input_f:
                    if 'sPathData_Output_TimeSeries' in line:
                        iter_out_path = line.split("=")[1].strip().replace('"', '').replace("'", "")
                        break

            if iter_out_path is None:
                logging.error(f" ---> Cannot find 'sPathData_Output_TimeSeries' in {iter_settings_file}")
                hmc_results[iExplor] = None
                continue

            try:
                df_out = read_discharge_hmc(
                    output_path=iter_out_path,
                    col_names=sections["name"].values,
                    start_time=calib_hydro_start
                )
                df_out = df_out.reindex(
                    calibration_period,
                    method="nearest",
                    tolerance="1" + data_settings["data"]["hydro"]["calib_hydro_resolution"]
                )

                if df_out.dropna(how="all", axis=1).empty:
                    logging.warning(f" ---> HMC output for ITER{iIter:02d}-{iExplor:03d} is all-NaN. Skipping.")
                    hmc_results[iExplor] = None
                else:
                    hmc_results[iExplor] = df_out

            except FileNotFoundError:
                logging.error(f" ---> WARNING! HMC output time-series file not found at path {iter_out_path}")
                hmc_results[iExplor] = None
            except IOError as e:
                logging.error(f" ---> ERROR reading HMC output for iteration {iIter:02d}-{iExplor:03d}: {e}")
                hmc_results[iExplor] = None

        logging.info(' ---> Read model output data...DONE')

        # 5.5 Scores
        logging.info(" --> Calculate iteration scores..")
        scores = pd.DataFrame(index=np.arange(1, nExplor + 1), columns=section_data.keys())
        scores_iter = pd.DataFrame(index=np.arange(1, nExplor + 1), columns=["tot"])

        eval_score = getattr(hs, data_settings["algorithm"]["general"]["error_metrics"]["function"])

        for iExplor in np.arange(1, nExplor + 1):
            logging.info(f" ---> Parameters set {iExplor:03d}...")
            if hmc_results[iExplor] is None:
                for section in section_data.keys():
                    scores.loc[iExplor, section] = np.nan
                scores_iter.loc[iExplor, "tot"] = np.nan
                continue

            for section in section_data.keys():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    sim_vals = hmc_results[iExplor][section].values
                    obs_vals = section_data[section]["value"].values

                    if sim_vals.ndim > 1:
                        sim_vals = sim_vals.squeeze()
                    if obs_vals.ndim > 1:
                        obs_vals = obs_vals.squeeze()

                    if sim_vals.shape != obs_vals.shape:
                        logging.error(
                            f"    [ERROR] Size mismatch for section '{section}': "
                            f"sim_vals={sim_vals.shape}, obs_vals={obs_vals.shape}."
                        )
                        raise ValueError("Simulated vs observed length mismatch.")

                    scores.loc[iExplor, section] = eval_score(sim_vals, obs_vals)

            if data_settings["algorithm"]["general"]["error_metrics"]["minimum_ins_inf"]:
                row_scores = scores.loc[iExplor]
                row_scores[row_scores < -1] = -1
                scores.loc[iExplor] = row_scores
                data_settings["algorithm"]["general"]["error_metrics"]["shift_for_positive"] = 1

            areas = sections.set_index("name").loc[list(section_data.keys()), "area_ncell"]
            section_scores = scores.loc[iExplor, list(section_data.keys())].astype(float)

            numerator = np.nansum(
                np.log(areas.values) * (
                        data_settings["algorithm"]["general"]["error_metrics"]["shift_for_positive"]
                        + section_scores.values
                )
            )
            denominator = np.nansum(np.log(areas.values))

            scores_iter.loc[iExplor, "tot"] = numerator / denominator
            scores.loc[iExplor, "tot"] = scores_iter.loc[iExplor, "tot"]

        logging.info(" --> Calculate iteration scores..DONE")

        # 5.6 Diagnostics plots (optional)
        flags = data_settings['algorithm'].get('flags', {})

        if flags.get('plot_timeseries_plotly', False):
            try:
                plots_dir = os.path.join(path_settings['out_path'], 'plots', 'timeseries_plotly', f'ITER{iIter:02d}')
                best_value = data_settings['algorithm']['general']['error_metrics']['best_value']
                plot_iter_timeseries_plotly(domain, iIter, plots_dir, section_data, hmc_results, scores_iter, best_value)
                logging.info(' ---> Plotly time series saved')
            except Exception as e:
                logging.warning(f' ---> Plotly time series failed: {e}')


        # 5.6 Persist results
        logging.info(" ---> Save outputs...")
        csv_path = os.path.join(
            data_settings["algorithm"]["path"]["out_path"],
            f'ITER{iIter:02d}_sections_scores.csv'
        )
        scores.to_csv(csv_path)

        pickle_path = os.path.join(
            data_settings["algorithm"]["path"]["out_path"],
            f'ITER{iIter:02d}_results.pickle'
        )
        with open(pickle_path, "wb") as handle:
            pickle.dump({
                "param_limits": param_limits,
                "param": param,
                "scores_iter": scores_iter,
                "maps_iter": maps_iter,
                "best_score_iter": best_score_iter
            }, handle)

        logging.info(" ---> Save outputs...DONE")

    # ------------------- 6) Fallback best from last iteration -------------------
    if iIter == iMax and not converges:
        logging.warning(' ---> Max iterations reached without convergence!')

        final_pickle = os.path.join(
            data_settings["algorithm"]["path"]["out_path"],
            f'ITER{iIter:02d}_results.pickle'
        )
        with open(final_pickle, "rb") as handle:
            current_iter = pickle.load(handle)
        logging.info(f' ---> Loading results of iteration {iIter:02d}...DONE')

        best_score_iter = current_iter["best_score_iter"]
        if data_settings["algorithm"]["general"]["error_metrics"]["best_value"] == "max":
            idx_best = np.nanargmax(current_iter["scores_iter"]["tot"].values) + 1
            best_score_iter[iIter] = np.nanmax(current_iter["scores_iter"]["tot"].values)
        else:
            idx_best = np.nanargmin(current_iter["scores_iter"]["tot"].values) + 1
            best_score_iter[iIter] = np.nanmin(current_iter["scores_iter"]["tot"].values)

        logging.info(f" ---> Best combination for iteration {iIter:02d} is {idx_best:03d}")
        improvement = abs((best_score_iter[iIter] - best_score_iter[iIter - 1]) / best_score_iter[iIter])
        logging.info(f" --> Improvement: {improvement:.4f}")
        maps_out = current_iter["maps_iter"][idx_best]

    # ------------------- 7) Write best maps -------------------
    logging.info(" --> Write resulting best maps...")
    os.makedirs(os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded"), exist_ok=True)

    for par in calibrated_params:
        tmp_path = os.path.join(data_settings["algorithm"]["path"]["out_path"], "gridded", 'temp.tif')
        with rio.open(tmp_path, 'w', **header) as dst:
            out = maps_out[par].astype(float)
            out[~np.isfinite(out)] = np.nan
            if data_settings['calibration']['parameters'][par].get('log_scale', False):
                with np.errstate(over='ignore', invalid='ignore'):
                    out = 10 ** out
                out[~np.isfinite(out)] = np.nan
            dst.write(np.nan_to_num(out, nan=-9999).astype(rio.float32), 1)

        ascii_out = os.path.join(
            data_settings["algorithm"]["path"]["out_path"], "gridded", f'{domain}.{par}.txt'
        )
        gdal.Translate(ascii_out, tmp_path, options=translate_options)
        os.remove(tmp_path)

    logging.info(" --> Write resulting best maps...DONE")

    # ------------------- 8) Final log -------------------
    time_elapsed = round(time.time() - start_time, 1)
    logging.info(' ')
    logging.info(f'==> {alg_name} (Version: {alg_version} Release_Date: {alg_release})')
    logging.info(f'==> TIME ELAPSED: {time_elapsed} seconds')
    logging.info('==> ... END')
    logging.info('==> Bye, Bye')
    logging.info('============================================================================')


# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_values = parser_handle.parse_args()
    if parser_values.alg_settings:
        return parser_values.alg_settings
    else:
        return 'configuration.json'


# -------------------------------------------------------------------------------------
# Method to read JSON settings (with environment variable expansion)
def read_file_json(file_name):
    """
    Read a JSON file that may contain $ENV_VAR references.
    Expands any occurrences of '$ENV_VAR' in the JSON text to the actual environment value.
    Returns the parsed JSON dictionary.
    """
    env_ws = {env_item: env_value for env_item, env_value in os.environ.items()}

    with open(file_name, "r") as file_handle:
        json_block = []
        for file_row in file_handle:
            for env_key, env_value in env_ws.items():
                env_tag = f'${env_key}'
                if env_tag in file_row:
                    val = env_value.strip("'\\'")
                    file_row = file_row.replace(env_tag, val)
                    file_row = file_row.replace('//', '/')
            json_block.append(file_row)
            if file_row.startswith('}'):
                json_dict = json.loads(''.join(json_block))
                json_block = []
                return json_dict

    return json.loads(''.join(json_block))


# -------------------------------------------------------------------------------------
# Method to set logging
def set_logging(logger_file='log.txt', logger_format=None):
    """
    Configure Python logging to write INFO+ messages to both:
    1) The specified log file (overwrite if exists)
    2) Standard output (console)
    """
    if logger_format is None:
        logger_format = (
            '%(asctime)s %(name)-12s %(levelname)-8s '
            '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'
        )

    if os.path.exists(logger_file):
        os.remove(logger_file)

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    handler_file = logging.FileHandler(logger_file, 'w')
    handler_console = logging.StreamHandler()
    handler_file.setLevel(logging.INFO)
    handler_console.setLevel(logging.INFO)

    formatter = logging.Formatter(logger_format)
    handler_file.setFormatter(formatter)
    handler_console.setFormatter(formatter)

    logging.getLogger('').addHandler(handler_file)
    logging.getLogger('').addHandler(handler_console)


# -------------------------------------------------------------------------------------
# Generate a Latin Hypercube sample: n points in p dimensions
def lhssample(n, p):
    """
    Create an n×p Latin Hypercube sample in [0,1].
    Each column is a random permutation of the n strata midpoints.
    Returns: NumPy array of shape (n, p).
    """
    x = np.random.uniform(size=[n, p])
    for i in range(p):
        x[:, i] = (np.argsort(x[:, i]) + 0.5) / n
    return x


# -------------------------------------------------------------------------------------
# Function for treating lakes

def assign_lakes(arr, par_name, par_settings, in_log_space):
    """
    If 'lakes_mask' is provided, fill lake cells with the configured lake value.
    - in_log_space=True: write log10(lake_value)
    - in_log_space=False: write lake_value (linear)
    Raises ValueError if lakes_mask is present but no lake_value can be resolved.
    """
    if "lakes_mask" not in par_settings:
        return arr

    lakes_mask = rio.open(par_settings["lakes_mask"]).read(1)
    lake_val = lake_value_for(par_name=par_name, par_settings=par_settings)
    if lake_val is None:
        raise ValueError(
            f"Parameter '{par_name}' uses lakes_mask but no 'lake_value' is provided "
            f"and it is not a standard parameter. Please add 'lake_value' in JSON."
        )
    v = np.log10(float(lake_val)) if in_log_space else float(lake_val)
    return np.where(lakes_mask == 1, v, arr)


# -------------------------------------------------------------------------------------
# Approach: rescale (base map + arctan neighborhood)
def rescale_map(map_name, par, par_settings, maps_in):
    """
    Rescale a base map via arctan neighborhood.
    If log_scale=True, maps_in[map_name], min, max, and 'par' are all in log10 space.
    """
    map_max = par_settings["max"] * maps_in["mask"]
    map_min = par_settings["min"] * maps_in["mask"]

    scalaATan = (
        (1 - np.double((2 - (1 - np.sign(par))) > 0)) * (maps_in[map_name] - map_min)
        + np.double((2 - (1 - np.sign(par))) > 0) * (map_max - maps_in[map_name])
    )
    new_map = maps_in[map_name] + (
        (scalaATan / (math.pi / 2))
        * np.arctan(2 * (map_max - map_min) * ((math.pi / 2) / scalaATan) * par)
    )

    # lakes in the same space of new_map (log if log_scale, linear otherwise)
    in_log = bool(par_settings.get("log_scale", False))
    new_map = assign_lakes(new_map, map_name, par_settings, in_log_space=in_log)
    return new_map


# -------------------------------------------------------------------------------------
# Approach: mask (area map * parameter)
def rescale_mask(map_name, par, par_settings, maps_in):
    """
    Area-mask approach.
    - log_scale=False: new_map = base_mask * par (linear)
    - log_scale=True:  new_map_log = par on active cells, -inf elsewhere (so 10**(-inf)=0 in output)
    Lakes are filled with lake_value in the appropriate space.
    """
    base = maps_in[map_name].astype(float)  # expected >0 for active area
    active = np.isfinite(base) & (base > 0)

    if par_settings.get("log_scale", False):
        new_map = np.full_like(base, -np.inf, dtype=float)
        new_map[active] = float(par)  # par is log10
        return assign_lakes(new_map, map_name, par_settings, in_log_space=True)
    else:
        new_map = base * float(par)
        new_map[new_map < 0] = -9999
        return assign_lakes(new_map, map_name, par_settings, in_log_space=False)


# -------------------------------------------------------------------------------------
# Approach: uniform
def rescale_value(map_name, par, par_settings, maps_in):
    """
    Uniform approach.
    - log_scale=False: new_map = domain_mask * par (linear)
    - log_scale=True:  new_map_log = par where domain_mask==1, NaN elsewhere (will be 10**par at write)
    Lakes are filled with lake_value in the appropriate space.
    """
    dom = maps_in['mask']

    if par_settings.get("log_scale", False):
        new_map = np.where(dom == 1, float(par), np.nan)  # par is log10
        return assign_lakes(new_map, map_name, par_settings, in_log_space=True)
    else:
        new_map = dom * float(par)
        return assign_lakes(new_map, map_name, par_settings, in_log_space=False)

# -------------------------------------------------------------------------------------
# Copy all files from source_folder → destination_folder (used for point & gridded data)
def copy_all_files(source_folder, destination_folder):
    """
    Iterate over each file in source_folder; if it's a regular file, copy to destination_folder.
    """
    for file_name in os.listdir(source_folder):
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        if os.path.isfile(source):
            shutil.copy(source, destination)


# -------------------------------------------------------------------------------------
# Create a simple HMC launcher script in each run’s 'exe' folder
def make_launcher(iter_exe_path, domain_name, env_path):
    """
    Writes 'launcher.sh' in iter_exe_path that:
      1) sources the provided env_path (e.g. library exports)
      2) cds into iter_exe_path
      3) runs the HMC binary with '<domain>.info.txt' as input.
    """
    with open(os.path.join(iter_exe_path, "launcher.sh"), "w") as launcher:
        launcher.write("#!/bin/bash\n")
        launcher.write(f"source {env_path}\n")
        launcher.write(f"cd {iter_exe_path}\n")
        launcher.write("chmod 777 HMC3_calib.x\n")
        launcher.write("ulimit -s unlimited\n")
        launcher.write(f"./HMC3_calib.x {domain_name}.info.txt\n")


# -------------------------------------------------------------------------------------
# Read HMC’s ASCII output time‐series file into a pandas DataFrame
def read_discharge_hmc(output_path='', col_names=None, output_name="hmc.hydrograph.txt",
                       format='txt', start_time=None, end_time=None):
    """
    Reads the HMC output time‐series (ASCII) file:
      - index_col=0 is parsed as a datetime (format '%Y%m%d%H%M')
      - subsequent columns are unnamed; we assign col_names if provided.
    If col_names length ≠ number of columns, raises IOError.
    Subsets the DF to [start_time : end_time], if given; otherwise uses the full range.
    """
    if format == 'txt':
        custom_date_parser = lambda x: dt.datetime.strptime(x, "%Y%m%d%H%M")
        if col_names is None:
            print(' ---> ERROR! Columns names parameter not provided!')
            raise IOError("Section list should be provided as col_names parameter!")

        hmc_discharge_df = pd.read_csv(
            os.path.join(output_path, output_name),
            header=None,
            delimiter=r"\s+",
            parse_dates=[0],
            index_col=[0],
            date_parser=custom_date_parser
        )

        if len(col_names) == len(hmc_discharge_df.columns):
            hmc_discharge_df.columns = col_names
        else:
            print(' ---> ERROR! Number of hmc output columns is not consistent with the number of stations!')
            raise IOError("Verify your section file, your run setup or provide a personal column setup!")

        if start_time is None:
            start_time = min(hmc_discharge_df.index)
        if end_time is None:
            end_time = max(hmc_discharge_df.index)

        return hmc_discharge_df[start_time:end_time]
    else:
        raise NotImplementedError("Only 'txt' format is supported for HMC output.")


# -------------------------------------------------------------------------------------
# (Optional) cost function using Kling‐Gupta Efficiency (not invoked by default)
def costiIdro(matSim, matObs, stations):
    """
    Example of a hydro‐cost function using KGE transformed into a cost via arctan:
      - matSim, matObs: pandas DataFrames (each column = station timeseries)
      - stations: object with .area attribute to supply station areas (for weighting).
    Returns an array J with one value per station, plus a weighted aggregate.
    """
    J = np.empty((1, len(matSim.columns.values) + 1))
    ind = np.empty((1, len(matSim.columns.values) + 1))
    ii = 0

    for staz in matSim.columns.values:
        xSim = matSim[staz]
        xObs = matObs[staz]
        KGE = 1 - np.sqrt(
            (np.corrcoef(xSim, xObs)[0, 1] - 1) ** 2
            + (((np.std(xSim) / np.mean(xSim)) / (np.std(xObs) / np.mean(xObs)) - 1)) ** 2
            + ((np.mean(xSim) / np.mean(xObs) - 1)) ** 2
        )
        J[0, ii] = (2 / np.pi) * np.arctan(1 - KGE)
        ind[0, ii] = np.log(stations.area[staz])
        ii += 1

    J[0, -1] = np.nansum(J[0, 0:-1] * ind[0, 0:-1]) / np.nansum(ind[0, 0:-1] * (J[0, 0:-1] / J[0, 0:-1]))
    return J


# ----------------------------------------------------------------------------
def lake_value_for(par_name: str, par_settings):
    """Resolve the value to assign to lake cells for a given parameter.

    Priority:
      1) If the parameter JSON includes a "lake_value", return that (float).
      2) Otherwise, return a hard-coded default for standard parameters:
         - soil_ksat_drain -> 0.03
         - soil_ksat_infilt -> 100.0
         - ct -> 0.9
         - soil_vmax -> 4500.0
         - cn -> 5.0
      3) If neither is available, return None (caller will raise if lakes_mask is used).
    """
    try:
        if isinstance(par_settings, dict):
            if "lake_value" in par_settings and par_settings["lake_value"] is not None:
                return float(par_settings["lake_value"])
    except Exception:
        pass

    LAKE_DEFAULTS = {
        "soil_ksat_drain": 0.03,
        "soil_ksat_infilt": 100.0,
        "ct": 0.9,
        "soil_vmax": 4500.0,
        "cn": 5.0,
    }
    return LAKE_DEFAULTS.get(str(par_name), None)


# ----------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Plotting utilities

def plot_param_boxplots_matplotlib(domain, iiter, maps_iter, out_dir, calibration_parameters, max_boxes=None):
    """Create per-parameter boxplots of map values across parameter combinations.

    The boxplots are meant as a *sampling diagnostic* and should be generated right after
    maps are created (before launching HMC).

    - X axis: combinations (001..N)
    - Y axis: distribution of valid (finite) grid values over the *calibratable* domain
      (domain mask == 1) excluding lakes pixels when a `lakes_mask` is provided for that
      parameter.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    comb_ids = sorted([k for k in maps_iter.keys()])
    if not comb_ids:
        logging.warning(" ---> Boxplots: no combinations found in maps_iter")
        return

    # Optional cap on number of boxes (useful when there are many combinations)
    if max_boxes is not None:
        try:
            max_boxes = int(max_boxes)
            if max_boxes > 0 and len(comb_ids) > max_boxes:
                comb_ids = comb_ids[:max_boxes]
        except Exception:
            pass

    first_maps = maps_iter[comb_ids[0]]

    # Domain mask: only pixels where mask == 1 (if present)
    dom_mask = None
    if isinstance(first_maps, dict) and "mask" in first_maps:
        m = first_maps["mask"]
        try:
            dom_mask = np.isfinite(m) & (m == 1)
        except Exception:
            dom_mask = None

    # Plot only keys that look like parameters (exclude the generic 'mask')
    par_names = [k for k in first_maps.keys() if k != "mask"]

    for par in par_names:

        # Lakes mask is defined per parameter in the JSON
        lakes_mask = None
        par_cfg = calibration_parameters.get(par, {})
        lakes_path = par_cfg.get("lakes_mask", None)
        if lakes_path:
            try:
                lakes_arr = rio.open(lakes_path).read(1)
                lakes_mask = (lakes_arr == 1)
            except Exception as e:
                logging.warning(f" ---> Boxplots: could not read lakes_mask for '{par}': {e}")
                lakes_mask = None

        data = []
        labels = []

        for comb_id in comb_ids:
            arr = maps_iter[comb_id].get(par, None)
            if arr is None:
                continue

            a = arr.astype(float)

            valid = np.isfinite(a)
            if dom_mask is not None:
                valid &= dom_mask
            if lakes_mask is not None:
                valid &= ~lakes_mask

            vals = a[valid]
            if vals.size == 0:
                continue

            data.append(vals)
            labels.append(f"{comb_id:03d}")

        if not data:
            logging.info(f" ---> Boxplots: skip '{par}' (no valid cells after masking)")
            continue

        fig_w = max(10.0, len(labels) * 0.25)
        plt.figure(figsize=(fig_w, 6))
        plt.boxplot(data)  # showfliers=True by default
        plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
        plt.ylabel(par)
        plt.title(f"{domain} - ITER {iiter:02d} - {par}")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{domain}_ITER{iiter:02d}_{par}_boxplot.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
def plot_iter_timeseries_plotly(domain, iiter, out_dir, section_data, hmc_results, scores_iter, best_value):
    """Plot OBS vs SIM for one iteration (per section) and save as HTML (Plotly).

    - OBS: dashed line
    - BEST simulation: thick line
    - All other simulations: thin + transparent lines
    - Simulations are aggregated to the OBS time scale using mean (no thinning/subsampling).

    Output
    ------
    One HTML per section:
        <out_dir>/<domain>_ITERXX_<section>.html
    """
    import plotly.graph_objects as go

    os.makedirs(out_dir, exist_ok=True)

    # best index (1-based)
    if best_value == "max":
        idx_best = int(np.nanargmax(scores_iter["tot"].values) + 1)
    else:
        idx_best = int(np.nanargmin(scores_iter["tot"].values) + 1)

    for section in section_data.keys():
        fig = go.Figure()

        # OBS
        obs = section_data[section][["value"]].copy()
        obs = obs[obs["value"].notna()]
        obs_index = obs.index

        fig.add_trace(go.Scatter(
            x=obs_index, y=obs["value"].values,
            mode="lines",
            name="obs",
            line=dict(dash="dash", width=2)
        ))

        # SIM all (excluding best)
        for iexplor in sorted(hmc_results.keys()):
            df_sim = hmc_results.get(iexplor, None)
            if df_sim is None or section not in df_sim.columns:
                continue
            if iexplor == idx_best:
                continue

            sim = df_sim[[section]].copy()
            sim = sim.dropna()
            if sim.empty:
                continue

            # aggregate to obs time scale (mean) then align on obs index
            target_freq = pd.infer_freq(obs_index)
            if target_freq is not None:
                sim_s = sim[section].resample(target_freq).mean()
                sim_s = sim_s.reindex(obs_index)
            else:
                sim_s = sim[section].reindex(obs_index, method="nearest")

            fig.add_trace(go.Scatter(
                x=obs_index,
                y=sim_s.values,
                mode="lines",
                name=f"sim {iexplor:03d}",
                line=dict(width=1),
                opacity=0.15,
                showlegend=False
            ))

        # BEST
        df_best = hmc_results.get(idx_best, None)
        if df_best is not None and section in df_best.columns:
            best = df_best[[section]].copy().dropna()
            if not best.empty:
                target_freq = pd.infer_freq(obs_index)
                if target_freq is not None:
                    best_s = best[section].resample(target_freq).mean()
                    best_s = best_s.reindex(obs_index)
                else:
                    best_s = best[section].reindex(obs_index, method="nearest")

                fig.add_trace(go.Scatter(
                    x=obs_index, y=best_s.values,
                    mode="lines",
                    name=f"best {idx_best:03d}",
                    line=dict(width=4)
                ))

        fig.update_layout(
            title=f"{domain} - ITER {iiter:02d} - {section}",
            xaxis_title="time",
            yaxis_title="discharge"
        )

        out_path = os.path.join(out_dir, f"{domain}_ITER{iiter:02d}_{section}.html")
        fig.write_html(out_path, include_plotlyjs="cdn")

if __name__ == "__main__":
    main()
