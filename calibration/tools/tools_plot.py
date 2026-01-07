import os
import pandas as pd
import datetime as dt
import shutil
import numpy as np
import logging
import rasterio as rio

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

def plot_iter_timeseries(
    domain,
    iiter,
    out_dir,
    section_data,
    hmc_results,
    scores_iter=None,
    best_value="min",
    running=False,
):
    """Plot OBS vs SIM for one iteration (per section) and save as HTML (Plotly).

    Requisiti:
    - OBS a puntini (markers), con buchi dove manca l'osservato
    - Simulazioni spengibili singolarmente (toggle individuale in legenda)
    - BEST più spessa quando running=False e scores_iter disponibile
    - Tutte le simulazioni visibili all'apertura

    Output:
      <out_dir>/<domain>_ITERXX_<section>.html
    """
    import plotly.graph_objects as go

    os.makedirs(out_dir, exist_ok=True)

    # Decide best (1-based) if requested
    idx_best = None
    if not running:
        if scores_iter is None or "tot" not in scores_iter.columns or scores_iter["tot"].dropna().empty:
            # if scores not available, behave like running
            running = True
        else:
            if best_value == "max":
                idx_best = int(np.nanargmax(scores_iter["tot"].values) + 1)
            else:
                idx_best = int(np.nanargmin(scores_iter["tot"].values) + 1)

    for section in section_data.keys():
        fig = go.Figure()

        # SIM (each one toggleable individually)
        for iexplor in sorted(hmc_results.keys()):
            df_sim = hmc_results.get(iexplor,None)
            if df_sim is None or section not in df_sim.columns:
                continue

            sim = df_sim[[section]].copy().dropna()
            if sim.empty:
                continue

            # Keep simulated series on its own time axis (do NOT clip to obs timestamps)
            sim_s = sim[section]
            is_best = (idx_best is not None and iexplor == idx_best)

            line_width = 4 if (is_best and not running) else 1
            opacity = 1.0 if (is_best and not running) else 0.25

            fig.add_trace(
                go.Scatter(
                    x=sim_s.index,
                    y=sim_s.values,
                    mode="lines",
                    name=(f"best {iexplor:03d}" if (is_best and not running) else f"sim {iexplor:03d}"),
                    line=dict(width=line_width),
                    opacity=opacity,
                    showlegend=True,
                    visible=True,  # tutte visibili all'apertura
                    # IMPORTANT: no legendgroup -> toggles are individual
                )
            )

        # OBS (markers; keep gaps where obs is missing)
        obs = section_data[section]["value"].astype(float).resample("D").mean()

        fig.add_trace(
            go.Scatter(
                x=obs.index,
                y=obs.values,
                mode="markers",
                name="obs",
                marker=dict(size=5, color="black"),
            )
        )
        
        title_suffix = " (running)" if running else ""
        fig.update_layout(
            title=f"{domain} - ITER {iiter:02d} - {section}{title_suffix}",
            xaxis_title="time",
            yaxis_title="discharge",
        )

        out_path = os.path.join(out_dir, f"{domain}_ITER{iiter:02d}_{section}.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
