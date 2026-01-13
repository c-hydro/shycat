# tools_hmc.py
import os
import pandas as pd
import datetime as dt
import shutil

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
# Create a simple HMC launcher script in each run’s 'exe' folder
def make_launcher(iter_exe_path, domain_name, env_path=None, use_conda_env=False, conda=None):
    """
    Writes 'launcher.sh' in iter_exe_path.

    Modes:
      - Classic (default): source `env_path` (e.g. exports LD_LIBRARY_PATH/PATH)
      - Conda: if `use_conda_env=True`, activate conda using `conda` dict:
            conda = {"virtualenv_folder": "...", "virtualenv_name": "..."}

    Notes:
      - Backward compatible with the old signature (iter_exe_path, domain_name, env_path).
      - We intentionally keep this script very small and explicit.
    """
    launcher_path = os.path.join(iter_exe_path, "launcher.sh")

    if use_conda_env:
        if not isinstance(conda, dict):
            raise ValueError("conda configuration must be a dict when use_conda_env=True")

        virtualenv_folder = str(conda.get("virtualenv_folder", "")).strip()
        virtualenv_name = str(conda.get("virtualenv_name", "")).strip()

        if not virtualenv_folder or not virtualenv_name:
            raise ValueError(
                "Missing conda settings. Provide data.hmc.conda.virtualenv_folder and data.hmc.conda.virtualenv_name"
            )

        script_folder = iter_exe_path

        env_lines = [
            f'virtualenv_folder="{virtualenv_folder}"',
            f'virtualenv_name="{virtualenv_name}"',
            f'script_folder="{script_folder}"',
            'export PATH="$virtualenv_folder/bin:$PATH"',
            'if [ -f "$virtualenv_folder/etc/profile.d/conda.sh" ]; then',
            '  source "$virtualenv_folder/etc/profile.d/conda.sh"',
            '  conda activate "$virtualenv_name"',
            'else',
            '  source activate "$virtualenv_name"',
            'fi',
            'export PYTHONPATH="${PYTHONPATH}:$script_folder"',
        ]
    else:
        env_path = "" if env_path is None else str(env_path).strip()
        if not env_path:
            raise ValueError("env_path is required when use_conda_env=False")
        env_lines = [f"source {env_path}"]

    with open(launcher_path, "w") as launcher:
        launcher.write("#!/bin/bash\n")
        for ln in env_lines:
            launcher.write(ln + "\n")
        launcher.write(f"cd {iter_exe_path}\n")
        launcher.write("chmod 777 HMC3_calib.x\n")
        launcher.write("ulimit -s unlimited\n")
        launcher.write(f"./HMC3_calib.x {domain_name}.info.txt\n")