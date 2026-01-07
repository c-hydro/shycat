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