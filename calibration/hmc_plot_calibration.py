#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hmc_plot_calibration.py

(Re)plot OBS vs SIM time series for a given calibration iteration, using the same JSON settings
as the calibration workflow.

Usage:
  python3 hmc_plot_calibration_fixed.py -settings_file configuration.json -iter 2 -mode running
  python3 hmc_plot_calibration_fixed.py -settings_file configuration.json -iter 2 -mode final

Modes:
- running: does NOT select a best run (works even while the iteration is still running)
- final:   selects the best run using scores_iter (pickle preferred, CSV fallback)

Important:
- There is ONLY ONE plotting function used: tools_plot.plot_iter_timeseries(...)
- The 'running' behavior is controlled by the flag running=True/False passed to that function.
"""

import os
import json
import logging
import warnings
import datetime as dt
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from tools.tools_hmc import read_discharge_hmc
from tools.tools_plot import plot_iter_timeseries


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-settings_file", dest="settings_file", default="configuration.json",
                        help="Path to the JSON configuration file")
    parser.add_argument("-iter", dest="iter_num", required=True, type=int,
                        help="Iteration number to plot (e.g., 1, 2, 3...)")
    parser.add_argument("-mode", dest="mode", default="running", choices=["running", "final"],
                        help="Plot mode: running (no best) or final (select best). Default: running")
    parser.add_argument("-out_dir", dest="out_dir", default=None,
                        help="Optional output directory for plots (default: <out_path>/plots/timeseries/ITERxx)")
    return parser.parse_args()


def read_file_json(file_name):
    env_ws = {k: v for k, v in os.environ.items()}
    with open(file_name, "r") as fh:
        json_block = []
        for row in fh:
            for env_key, env_value in env_ws.items():
                env_tag = f"${env_key}"
                if env_tag in row:
                    val = str(env_value).strip("'\\'")
                    row = row.replace(env_tag, val)
                    row = row.replace("//", "/")
            json_block.append(row)
            if row.startswith("}"):
                return json.loads("".join(json_block))
    return json.loads("".join(json_block))


def set_logging(logger_file):
    logger_format = (
        "%(asctime)s %(name)-12s %(levelname)-8s "
        "%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s"
    )
    os.makedirs(os.path.dirname(logger_file), exist_ok=True)
    if os.path.exists(logger_file):
        os.remove(logger_file)

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode="w")
    logging.getLogger("").addHandler(logging.StreamHandler())


def read_sections_and_obs(data_settings):
    domain = data_settings["algorithm"]["general"]["domain_name"]
    calib_hydro_start = dt.datetime.strptime(data_settings["algorithm"]["time"]["calib_hydro_start"], "%Y-%m-%d %H:%M")
    run_hydro_end = dt.datetime.strptime(data_settings["algorithm"]["time"]["run_hydro_end"], "%Y-%m-%d %H:%M")

    calibration_period = pd.date_range(
        calib_hydro_start,
        run_hydro_end,
        freq=data_settings["data"]["hydro"]["calib_hydro_resolution"],
    )

    section_file = os.path.join(data_settings["calibration"]["input_point_data_folder"], domain + ".info_section.txt")
    sections = pd.read_csv(section_file, sep=r"\s+", header=None,
                           names=["row_HMC", "col_HMC", "basin", "name"], usecols=[0, 1, 2, 3])

    # Duplicate station names check
    names = sections["name"].tolist()
    dup = [n for n in set(names) if names.count(n) > 1]
    if dup:
        raise ValueError(f"Duplicate station name(s) in info_section file: {dup}")

    # Observations
    section_data = {}
    for section, basin in zip(sections["name"], sections["basin"]):
        file_name_sec = os.path.join(
            data_settings["data"]["hydro"]["folder"],
            data_settings["data"]["hydro"]["filename"],
        ).format(domain=domain, section_name=section, section_basin=basin)

        if not os.path.isfile(file_name_sec):
            logging.warning(f"[OBS] File not found for section '{section}': {file_name_sec}")
            continue

        date_fmt = data_settings["data"]["hydro"]["date_fmt"]
        common_kwargs = dict(
            sep=data_settings["data"]["hydro"]["sep"],
            usecols=[data_settings["data"]["hydro"]["date_col"], data_settings["data"]["hydro"]["value_col"]],
            names=["date", "value"],
            index_col=["date"],
            parse_dates=["date"],
            na_values=data_settings["data"]["hydro"]["null_values"],
            header=None,
        )

        try:
            df_sec = pd.read_csv(file_name_sec, date_format=date_fmt, **common_kwargs)
        except TypeError:
            df_sec = pd.read_csv(file_name_sec, date_parser=lambda x: dt.datetime.strptime(x, date_fmt), **common_kwargs)

        df_sec = df_sec[calib_hydro_start:run_hydro_end].reindex(
            calibration_period,
            method="nearest",
            tolerance="1" + data_settings["data"]["hydro"]["calib_hydro_resolution"],
        )
        df_sec["value"] = pd.to_numeric(df_sec["value"], errors="coerce")
        if df_sec["value"].dropna().empty:
            continue

        section_data[section] = df_sec

    return sections, section_data, calibration_period, calib_hydro_start


def find_output_path_from_info(info_txt_path):
    if not os.path.isfile(info_txt_path):
        return None
    with open(info_txt_path, "r") as fh:
        for line in fh:
            if "sPathData_Output_TimeSeries" in line:
                try:
                    return line.split("=")[1].strip().replace('"', "").replace("'", "")
                except Exception:
                    return None
    return None


def read_hmc_results_for_iteration(data_settings, iter_num, sections, calibration_period, calib_hydro_start):
    domain = data_settings["algorithm"]["general"]["domain_name"]
    work_path = data_settings["algorithm"]["path"]["work_path"]
    sim_base = os.path.join(work_path, "simulations")

    # Estimate number of combinations, then (if present) refine using CSV length
    n0 = int(data_settings["algorithm"]["general"]["number_of_points_first_iteration"])
    pct = float(data_settings["algorithm"]["general"]["percentage_samples_successive_iterations"]) / 100.0
    n_explor = max(1, int(round(n0 * (pct ** max(iter_num - 1, 0)))))

    out_path = data_settings["algorithm"]["path"]["out_path"]
    csv_scores = os.path.join(out_path, f"ITER{iter_num:02d}_sections_scores.csv")
    if os.path.isfile(csv_scores):
        try:
            df_scores = pd.read_csv(csv_scores, index_col=0)
            n_explor = int(df_scores.index.astype(int).max())
        except Exception:
            pass

    hmc_results = {}
    for i_explor in range(1, n_explor + 1):
        sim_dir = os.path.join(sim_base, f"ITER{iter_num:02d}-{i_explor:03d}")
        info_txt = os.path.join(sim_dir, "exe", f"{domain}.info.txt")
        out_ts_path = find_output_path_from_info(info_txt)

        if out_ts_path is None:
            hmc_results[i_explor] = None
            continue

        try:
            df_out = read_discharge_hmc(output_path=out_ts_path,
                                        col_names=sections["name"].values,
                                        start_time=calib_hydro_start)
            df_out = df_out.reindex(calibration_period,
                                    method="nearest",
                                    tolerance="1" + data_settings["data"]["hydro"]["calib_hydro_resolution"])
            hmc_results[i_explor] = None if df_out.dropna(how="all", axis=1).empty else df_out
        except Exception:
            hmc_results[i_explor] = None

    return hmc_results


def read_scores_iter(data_settings, iter_num):
    out_path = data_settings["algorithm"]["path"]["out_path"]

    pkl = os.path.join(out_path, f"ITER{iter_num:02d}_results.pickle")
    if os.path.isfile(pkl):
        import pickle
        with open(pkl, "rb") as fh:
            obj = pickle.load(fh)
        scores_iter = obj.get("scores_iter", None)
        return scores_iter

    csv_scores = os.path.join(out_path, f"ITER{iter_num:02d}_sections_scores.csv")
    if os.path.isfile(csv_scores):
        df = pd.read_csv(csv_scores, index_col=0)
        if "tot" in df.columns:
            return df[["tot"]]
    return None


def main():
    args = get_args()
    data_settings = read_file_json(args.settings_file)

    domain = data_settings["algorithm"]["general"]["domain_name"]
    log_dir = data_settings["algorithm"]["path"]["log_path"]
    set_logging(os.path.join(log_dir, f"{domain}_plot_iter{args.iter_num:02d}_{args.mode}.log"))

    if args.out_dir is None:
        out_dir = os.path.join(data_settings["algorithm"]["path"]["out_path"], "plots", "timeseries", f"ITER{args.iter_num:02d}")
    else:
        out_dir = args.out_dir

    sections, section_data, calibration_period, calib_hydro_start = read_sections_and_obs(data_settings)
    if not section_data:
        raise RuntimeError("No observed time series loaded (section_data is empty).")

    hmc_results = read_hmc_results_for_iteration(data_settings, args.iter_num, sections, calibration_period, calib_hydro_start)

    running = (args.mode == "running")
    best_value = data_settings["algorithm"]["general"]["error_metrics"]["best_value"]

    scores_iter = None
    if not running:
        scores_iter = read_scores_iter(data_settings, args.iter_num)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_iter_timeseries(
            domain=domain,
            iiter=args.iter_num,
            out_dir=out_dir,
            section_data=section_data,
            hmc_results=hmc_results,
            scores_iter=scores_iter,
            best_value=best_value,
            running=running
        )

    logging.info(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
