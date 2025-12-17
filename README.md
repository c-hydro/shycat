# HMC Calibration Tool

This tool performs **automatic calibration of Continuum model parameters** over one or more domains.  
It explores the parameter space, runs HMC simulations, compares simulated and observed discharge,
and iteratively refines parameter ranges until convergence or a maximum number of iterations is reached.

---

## Overview

The calibration tool performs **iterative automatic calibration** of HMC model parameters.
It explores the parameter space using Latin Hypercube Sampling (LHS), runs ensembles of HMC simulations,
evaluates performance against observed discharge, and progressively narrows parameter ranges.

The final output consists of **best-fit parameter maps** and full diagnostics of the calibration process.

---

## How the calibration works

At each iteration, the tool:

1. Loads static gridded and point data (DEM, masks, base parameter maps, sections)
2. Reads observed discharge time series for each calibration section
3. Generates an ensemble of parameter sets using **Latin Hypercube Sampling (LHS)**
4. For each parameter set:
   - builds new parameter maps (according to the chosen approach)
   - creates a dedicated run folder
   - generates an HMC `.info` configuration file from a template
   - runs the HMC executable
5. Computes performance metrics (e.g. NRMSE) for each section and an aggregated score
6. Selects the best-performing parameter set
7. Reduces parameter ranges and proceeds to the next iteration

At the end of the process, the tool writes the **best calibrated parameter maps** and saves all intermediate results.

---

## Running the script

```bash
python3 HMC_calibration.py -settings_file configuration.json
```

If `-settings_file` is not provided, the script looks for `configuration.json` in the working directory.

---

## Settings file (`configuration.json`)

The settings file is a JSON document composed of three main sections:

- `algorithm`
- `data`
- `calibration`

Below is the **complete configuration structure**, with **all available keys** and inline comments.

---

## Complete configuration example (commented)

## Complete configuration template (commented)

```jsonc
{
  "algorithm": {

    "flags": {
      "delete_calib_data": false,               // If true, delete temporary calibration run folders
      "rescale_input_out_of_range": true,        // If true, auto-rescale base maps when outside [min,max]
      "plot_param_boxplots": true,               // If true, save sampled-parameter boxplots
      "boxplot_max_combinations": 200,           // Max combinations shown in boxplots (to avoid huge plots)
      "plot_timeseries_plotly": false            // If true, save interactive Plotly discharge time series
    },

    "general": {
      "domain_name": "DOMAIN_NAME",              // Domain identifier (consistent with Continuum one)

      "start_with_iteration_number": 1,          // Restart calibration from this iteration (1 if new start)
      "max_number_of_iterations": 10,            // Maximum number of calibration iterations

      "number_of_points_first_iteration": 50,   // LHS samples in first iteration
      "percentage_samples_successive_iterations": 80, // % of samples used in later iterations
      "percentage_param_range_reduction": 50,    // % shrink of parameter ranges after each iteration

      "percentage_min_improvement_quit_optimization": 1, // Stop if improvement (%) is below this value

      "error_metrics": {
        "__comment__": "best_value: choose min or max; set shift_for_positive if negative values are possible",
        "function": "nrmse_mean",                // Metric name (from hydrostats)
        "best_value": "min",                     // 'min' or 'max'
        "shift_for_positive": 0,                 // Add a shift so the metric can work with non-positive values
        "minimum_ins_inf": false                 // If true, enforce minimum/Inf handling
      }
    },

    "time": {
      "run_hydro_start": "2000-01-01 00:00",     // Simulation start (HMC run window)
      "run_hydro_end": "2005-12-31 23:00",       // Simulation end
      "calib_hydro_start": "2001-01-01 00:00"    // Start of the evaluation window for scoring
    },

    "path": {
      "work_path": "/path/to/work_dir",          // Working directory for simulations
      "out_path": "/path/to/output_dir",         // Output directory (results, plots, final maps)
      "log_path": "/path/to/log_dir"             // Log directory
    }
  },

  "data": {

    "hydro": {
      "folder": "/path/to/observations",         // Observed discharge folder
      "filename": "{section_name}.csv",          // Filename template for station/section series

      "date_col": 0,                             // Date column (index or name). Example uses index=0
      "value_col": 1,                            // Discharge column (index or name). Example uses index=1
      "sep": ",",                                // CSV separator
      "date_fmt": "%d-%b-%Y",                    // Datetime format
      "calib_hydro_resolution": "D",             // Observation time step (e.g., D daily, H hourly)
      "null_values": [-9999, -999, "NA"]         // Missing value flags
    },

    "hmc": {
      "model_exe": "/path/to/hmc_executable",                              // HMC binary/executable
      "model_settings": "/path/to/info_template/<domain>.info_calib.txt",  // Path to the INFO TEMPLATE (version-dependent)
      "system_env_libraries": "/path/to/env/setup.sh"                      // Script to load libs/env before running HMC
    }
  },

  "calibration": {

    "input_point_data_folder": "/path/to/static/point/",    // Sections and point static data
    "input_gridded_data_folder": "/path/to/static/gridded/",// DEM, area, masks, etc.
    "input_base_maps": "/path/to/static/gridded/",          // Base parameter maps (often same as gridded)

    "parameters": {
      // "PARAMETER_NAME is taken from Continuum static maps. All the parameters to be calibrated are listed here"
      "PARAMETER_NAME": {
        "calibrate": true,                      // Enable calibration for this parameter
        "approach": "rescale",                  // rescale | mask | uniform
        "min": 0.1,                             // Minimum allowed value
        "max": 10.0,                            // Maximum allowed value
        "log_scale": false,                     // If true, sample in log10 space (optional)

        "lakes_mask": "/path/to/waterbodies.tif" // Optional: exclude lakes using a mask raster
      }

      // Add as many parameters as needed...
    }
  }
}
```

---

## IMPORTANT: HMC model settings (`info_template`)

- **All HMC settings must be edited in the template file**
- Templates are stored in the `info_template` directory
- Keep **one template per HMC version**
- Point `data.hmc.model_settings` to the correct template

Do **not** manually edit the `.info` files generated during calibration.

### ⚠️ Forcing files paths

Always verify and update forcing paths inside the template, especially:

```
sPathData_Forcing_Gridded=/path/to/forcing/gridded
```

This is the most common source of errors when changing domain or machine.

---
