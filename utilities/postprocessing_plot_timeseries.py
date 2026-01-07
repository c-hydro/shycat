#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocessing_plot_timeseries.py

Author: Andrea Libertino (CIMA Foundation)
Created: 2026-01-07

Description
-----------
Post-processing utility to:
- read HMC hydrograph outputs for multiple experiments
- optionally overlay observed discharge
- compute metrics (user-configurable mapping to hydrostats.metrics)
- write interactive HTML plots per section
- export metrics CSV per experiment

Configuration
-------------
This script reads a JSON configuration file using the standard CLI convention:
  --settings_file <path_to_json>

Changelog
---------
- 2026-01-07: refactor to JSON-driven configuration + logging/args aligned to hmc_exec_calibration.
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Libraries
from typing import List, Tuple, Optional, Dict, Callable
import os
import json
import argparse
import logging
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hydrostats.metrics as hm
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Script main
def main():
    settings_file = get_args()
    cfg = read_file_json(settings_file)

    # Paths
    alg_path = cfg.get("algorithm", {}).get("path", {})
    log_path = alg_path.get("log_path", "./log")
    out_path = alg_path.get("out_path", "./out")

    plot_path = os.path.join(out_path, "plot")
    metrics_path = os.path.join(out_path, "metrics")

    safe_makedirs(log_path)
    safe_makedirs(plot_path)
    safe_makedirs(metrics_path)

# Logging (file in log_path)
    log_file = os.path.join(log_path, "log_postprocessing.txt")
    set_logging(logger_file=log_file)

    logging.info(' ==> START ... ')
    logging.info(' ==> Settings file: %s', settings_file)

    # 1) Read configuration
    section_file = cfg["data"]["input"]["section_file"]
    hydrographs = cfg["data"]["input"]["hydrographs"]
    discharge_obs_cfg = cfg["data"]["input"].get("discharge_obs", {})

    plot_filename = cfg["data"]["output"]["plot_filename"]

    date_start = cfg["algorithm"]["time"].get("date_start", None)
    date_end = cfg["algorithm"]["time"].get("date_end", None)

    plot_rain_cfg = cfg.get("algorithm", {}).get("general", {}).get("plot_rain", {})
    if plot_rain_cfg.get("exec", False):
        raise NotImplementedError("Plotting of rain with section is not implemented yet.")

    metrics_map = resolve_metrics_map(cfg.get("algorithm", {}).get("metrics", {}))

    # Parse dates
    try:
        p_start = pd.to_datetime(date_start) if date_start else None
        p_end = pd.to_datetime(date_end) if date_end else None
    except ValueError as e:
        logging.error("Date parsing failed. Check 'date_start'/'date_end'. Details: %s", str(e))
        return

    labels = list(hydrographs.keys())
    hydrograph_outputs = [hydrographs[k] for k in labels]

    # Sections
    section_df = read_section(section_file)
    section_names = section_df["section"].tolist()

    # Read model outputs
    logging.info("Reading model outputs (%d experiments)...", len(hydrograph_outputs))
    model_dfs: List[pd.DataFrame] = []
    for p in hydrograph_outputs:
        df = read_discharge_hmc(output_path=p, col_names=section_names, start_time=p_start, end_time=p_end)
        df[df < 0] = np.nan
        model_dfs.append(df)

    if not model_dfs:
        raise ValueError("No model data loaded.")

    global_start = model_dfs[0].index.min()
    global_end = model_dfs[0].index.max()
    logging.info("Analysis period: %s to %s", global_start.date(), global_end.date())

    metric_names = list(metrics_map.keys()) + ["Perc_Valid"]
    metrics_storage = {
        label: pd.DataFrame(index=section_names, columns=metric_names)
        for label in labels
    }

    logging.info("Processing %d sections...", len(section_names))

    for section_name in section_names:
        obs_series = None
        if discharge_obs_cfg:
            obs_series = read_discharge_obs(discharge_obs_cfg, section_name)

        x_range = (global_start, global_end)

        fig = build_fig_for_section(
            discharge_list=model_dfs,
            labels=labels,
            section_name=section_name,
            x_range=x_range,
            obs_series=obs_series,
        )

        out_file = plot_filename.format(section=section_name)
        out_path_section = os.path.join(plot_path, out_file)
        fig.write_html(out_path_section, include_plotlyjs="cdn")

        if obs_series is not None and not obs_series.empty:
            full_date_range = pd.date_range(start=global_start, end=global_end, freq="D")
            total_days = len(full_date_range)

            obs_in_period = obs_series.reindex(full_date_range)
            valid_days = obs_in_period.count()
            perc_valid = (valid_days / total_days) * 100 if total_days > 0 else 0.0

            for i, model_df in enumerate(model_dfs):
                label = labels[i]
                if section_name not in model_df.columns:
                    continue
                sim_series = model_df[section_name]
                metrics_result = calculate_metrics_direct(sim_series, obs_series, metrics_map=metrics_map)
                metrics_result["Perc_Valid"] = perc_valid
                metrics_storage[label].loc[section_name] = metrics_result
        else:
            for label in labels:
                metrics_storage[label].loc[section_name, "Perc_Valid"] = 0.0

    # Save metrics (in metrics_path)
    logging.info("Saving metrics CSVs...")
    for label in labels:
        df_metrics = metrics_storage[label]
        safe_label = label.replace(" ", "_").replace("/", "-")
        csv_name = f"{safe_label}_metrics.csv"
        csv_path = os.path.join(metrics_path, csv_name)
        df_metrics.index.name = "Section"
        df_metrics.to_csv(csv_path)
        logging.info("Saved metrics for %s: %s", label, csv_path)

    logging.info(' ==> ... END ')
    logging.info("Done.")
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def get_args() -> str:
    """
    Read CLI args, aligned to hmc_exec_calibration.py.

    Returns
    -------
    settings_file : str
        Path to settings JSON. If not provided, defaults to 'configuration.json'
        in the current working directory.
    """
    parser_handle = argparse.ArgumentParser()
    parser_handle.add_argument('--settings_file', '-settings_file', action="store", dest="alg_settings")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        return parser_values.alg_settings
    return 'configuration.json'
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def set_logging(logger_file: str = 'log.txt', logger_format: Optional[str] = None) -> None:
    """
    Configure Python logging to write INFO+ messages to both:
    1) The specified log file (overwrite if exists)
    2) Standard output (console)

    Note: this mirrors the behavior/style used in hmc_exec_calibration.py.
    """
    if logger_format is None:
        logger_format = (
            '%(asctime)s %(name)-12s %(levelname)-8s '
            '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'
        )

    log_dir = os.path.dirname(logger_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

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

# -------------------------------------------------------------------------------------
def read_file_json(file_name: str) -> Dict:
    """
    Read a JSON file that may contain $ENV_VAR references.
    Expands any occurrences of '$ENV_VAR' in the JSON text to the actual environment value.
    Returns the parsed JSON dictionary.
    """
    env_ws = {env_item: env_value for env_item, env_value in os.environ.items()}

    json_block = []
    with open(file_name, "r", encoding="utf-8") as file_handle:
        for file_row in file_handle:
            for env_key, env_value in env_ws.items():
                env_tag = f'${env_key}'
                if env_tag in file_row:
                    val = env_value.strip("'\\'")
                    file_row = file_row.replace(env_tag, val)
            json_block.append(file_row)

    return json.loads(''.join(json_block))
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def safe_makedirs(folder_path: str) -> None:
    """Create folder if it doesn't exist."""
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def resolve_metrics_map(metrics_cfg: Dict) -> Dict[str, Callable]:
    """
    Resolve a metrics mapping from config.

    Expected config structure:
        metrics_cfg = {
            "functions": {
                "NSE": "nse",
                "KGE": "kge_2012",
                ...
            }
        }

    Values can be:
    - function name in hydrostats.metrics (e.g. "nse", "kge_2012")
    - or "hm.<name>" (will be stripped)
    """
    if not isinstance(metrics_cfg, dict) or "functions" not in metrics_cfg:
        raise ValueError("Missing 'algorithm.metrics.functions' configuration.")

    func_map = metrics_cfg.get("functions", {})
    if not isinstance(func_map, dict) or not func_map:
        raise ValueError("'algorithm.metrics.functions' must be a non-empty dict.")

    out: Dict[str, Callable] = {}
    for metric_name, func_name in func_map.items():
        if not isinstance(func_name, str):
            raise ValueError(f"Metric function for '{metric_name}' must be a string.")

        fn = func_name.strip()
        if fn.startswith("hm."):
            fn = fn.replace("hm.", "", 1)

        if not hasattr(hm, fn):
            raise ValueError(f"Unknown hydrostats.metrics function '{func_name}' for metric '{metric_name}'.")

        out[metric_name] = getattr(hm, fn)

    return out
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def read_section(section_file: str,
                 column_names: List[str] = None,
                 sep: str = r"\s+",
                 fmt: str = "tabular") -> pd.DataFrame:
    """Read HMC section info table."""
    if column_names is None:
        column_names = ["r_HMC", "c_HMC", "basin", "section"]

    if fmt != "tabular":
        raise NotImplementedError("Only 'tabular' format is supported.")

    section_df = pd.read_csv(section_file, sep=sep, header=None)

    if section_df.shape[1] > len(column_names):
        section_df = section_df.iloc[:, :len(column_names)].copy()
    if section_df.shape[1] < len(column_names):
        raise IOError("Verify your section file or provide a personal column setup!")

    section_df.columns = column_names
    return section_df
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def read_discharge_hmc(output_path: str,
                       col_names: List[str],
                       output_name: str = "hmc.hydrograph.txt",
                       fmt: str = "txt",
                       start_time: Optional[pd.Timestamp] = None,
                       end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Read one HMC discharge file."""
    if fmt != "txt":
        raise NotImplementedError("Only 'txt' format is supported.")

    file_path = output_path
    if os.path.isdir(output_path):
        file_path = os.path.join(output_path, output_name)

    if col_names is None:
        raise IOError("Section list should be provided as col_names parameter!")

    custom_date_parser = lambda x: dt.datetime.strptime(x, "%Y%m%d%H%M")

    hmc_discharge_df = pd.read_csv(
        file_path,
        header=None,
        delimiter=r"\s+",
        parse_dates=[0],
        index_col=[0],
        date_parser=custom_date_parser
    ).resample("D").mean()

    if len(col_names) == len(hmc_discharge_df.columns):
        hmc_discharge_df.columns = col_names
    else:
        raise IOError(f"HMC columns: {len(hmc_discharge_df.columns)}, Sections provided: {len(col_names)}")

    if start_time is None:
        start_time = hmc_discharge_df.index.min()
    if end_time is None:
        end_time = hmc_discharge_df.index.max()

    hmc_discharge_df = hmc_discharge_df.loc[
        (hmc_discharge_df.index >= start_time) & (hmc_discharge_df.index <= end_time)
    ]

    return hmc_discharge_df
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def read_discharge_obs(discharge_obs_cfg: Dict,
                       section_name: str) -> Optional[pd.Series]:
    """
    Read observed discharge for a section, following calibration-like config keys.

    discharge_obs_cfg keys (suggested):
      - folder
      - filename (supports {section_name})
      - date_col (int)
      - value_col (int)
      - sep
      - date_fmt (optional)
      - null_values (optional list)
      - resolution (e.g. "D")
    """
    folder = discharge_obs_cfg.get("folder", None)
    if folder is None:
        return None

    filename_tpl = discharge_obs_cfg.get("filename", "{section_name}.csv")
    file_name = filename_tpl.format(section_name=section_name)
    csv_path = os.path.join(folder, file_name)

    if not os.path.exists(csv_path):
        return None

    date_col = int(discharge_obs_cfg.get("date_col", 0))
    value_col = int(discharge_obs_cfg.get("value_col", 1))
    sep = discharge_obs_cfg.get("sep", ",")
    date_fmt = discharge_obs_cfg.get("date_fmt", None)
    null_values = discharge_obs_cfg.get("null_values", [])
    resolution = discharge_obs_cfg.get("resolution", "D")

    df = pd.read_csv(csv_path, sep=sep, header=None)

    max_col = max(date_col, value_col)
    if df.shape[1] <= max_col:
        return None

    df = df.iloc[:, [date_col, value_col]].copy()
    df.columns = ["date", "obs"]

    if null_values:
        df.replace(null_values, np.nan, inplace=True)

    if date_fmt:
        df["date"] = pd.to_datetime(df["date"], format=date_fmt, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["obs"] = pd.to_numeric(df["obs"], errors="coerce")
    df.loc[df["obs"] < 0, "obs"] = np.nan

    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date").resample(resolution).mean()

    return df["obs"]
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def calculate_metrics_direct(sim_series: pd.Series,
                             obs_series: pd.Series,
                             metrics_map: Dict[str, Callable]) -> pd.Series:
    """Calculate hydrological metrics."""
    df = pd.concat([sim_series, obs_series], axis=1)
    df.dropna(inplace=True)

    if len(df) < 2:
        return pd.Series({k: np.nan for k in metrics_map.keys()})

    sim = df.iloc[:, 0].values
    obs = df.iloc[:, 1].values

    results = {}
    for metric_name, metric_func in metrics_map.items():
        try:
            results[metric_name] = metric_func(sim, obs)
        except Exception:
            results[metric_name] = np.nan

    return pd.Series(results)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def build_fig_for_section(discharge_list: List[pd.DataFrame],
                          labels: List[str],
                          section_name: str,
                          x_range: Tuple[pd.Timestamp, pd.Timestamp],
                          obs_series: Optional[pd.Series] = None) -> go.Figure:
    """Build a Plotly figure for one section."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    for df, label in zip(discharge_list, labels):
        if section_name not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[section_name].values,
                mode="lines",
                name=f"{label}",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f} m³/s<extra>" + label + "</extra>",
            )
        )

    if obs_series is not None and not obs_series.empty:
        obs_plot = obs_series.loc[(obs_series.index >= x_range[0]) & (obs_series.index <= x_range[1])]
        if not obs_plot.empty:
            fig.add_trace(
                go.Scatter(
                    x=obs_plot.index,
                    y=obs_plot.values,
                    mode="markers",
                    name="Observed",
                    marker=dict(color="black", size=5, symbol="circle"),
                    hovertemplate="Obs %{x|%Y-%m-%d}<br>%{y:.3f} m³/s<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"Discharge – {section_name}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=70, b=50),
    )
    fig.update_xaxes(title_text="Date", range=[x_range[0], x_range[1]])
    fig.update_yaxes(title_text="Discharge (m³/s)", range=[0, None])
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------return fig
    
if __name__ == "__main__":
    main()
