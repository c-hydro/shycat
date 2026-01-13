#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_cloud.py

AWS EC2 helpers for HMC calibration (EC2 + EFS).
LOGIC:
1. Setup: Mount EFS (/share).
2. Stage: Copy input FROM /share TO /scratch (Local NVMe).
3. Run: Model writes inputs/outputs strictly ON /scratch.
4. Sync: Rsync results FROM /scratch TO /share only at the end.
5. Safety: Uses bash 'trap' to ensure SHUTDOWN happens even on script crash.
"""

# Libraries
import os
import time
import json
import logging
import subprocess

# Helpers ---------
default_aws_user_data_template = r"""#!/bin/bash
# --------- Placeholders (filled by calibrator) ---------
domain="DOMAIN"
path_file_lock_def_start="LOCK_START"
path_file_lock_def_end="LOCK_END"
time_now="TIME_NOW"
exe_path_share="EXE_PATH"
efs_dns="EFS_DNS"
work_path_cloud="WORK_PATH_CLOUD"
# ------------------------------------------------------

# --- Logging function ---
log_debug() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --------- SAFETY NET: TRAP SHUTDOWN ---------
cleanup_and_shutdown() {
    log_debug "[TRAP] Script exit detected. Initiating Shutdown Sequence..."

    if [[ -d "${dest_share}" ]]; then
        sudo -u floodproofs cp "${log_cloud}" "${dest_share}/" 2>/dev/null || true
    fi

    log_debug "[SHUTDOWN] Powering off in 5s..."
    sleep 5
    shutdown -h now
}

trap cleanup_and_shutdown EXIT

# ---------------------------------------------

log_debug "[BOOT] Starting User Data Script"

# --------- Basic packages ---------
dnf update -y
dnf install -y gcc-gfortran m4 libcurl-devel git rsync
dnf install -y netcdf-fortran netcdf-devel hdf5-devel openmpi-devel

# --------- Create floodproofs user + SSH keys ---------
if ! id -u floodproofs >/dev/null 2>&1; then
  useradd -m floodproofs
fi
mkdir -p /home/floodproofs/.ssh
chmod 700 /home/floodproofs/.ssh

SSH_KEYS_BLOCK

# --------- Mount shared storage (EFS) ---------
mkdir -p /share
if ! mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport "${efs_dns}:/" /share; then
    log_debug "[EFS] Mount failed, retrying..."
    mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport "${efs_dns}:/" /share
fi

# --------- Locks folder (On EFS) ---------
mkdir -p "$(dirname "$path_file_lock_def_start")" "$(dirname "$path_file_lock_def_end")"
chown -R floodproofs:floodproofs "$(dirname "$path_file_lock_def_start")" || true

# --------- Lock START ---------
time_step="$(date +"%Y-%m-%d %H:%M")"
sudo -u floodproofs rm -f "$path_file_lock_def_end" || true
sudo -u floodproofs touch "$path_file_lock_def_start"
{
  echo " ==== EXECUTION START REPORT ==== "
  echo " ==== Execution Mode: LOCAL SCRATCH (High Performance)"
  echo " ==== Domain: $domain"
  echo " ==== EXE_PATH_SHARE (Source): $exe_path_share"
  echo " ==== WORK_PATH_CLOUD (Target): $work_path_cloud"
} >> "$path_file_lock_def_start"

# --------- Derive Paths ---------
# 1. SHARED PATHS (EFS)
iter_exe_rel="${exe_path_share#/share/}"       
iter_root_rel="$(dirname "$iter_exe_rel")"     
iter_root_share="/share/${iter_root_rel}"       
dest_share="${iter_root_share}/outcome"

# 2. LOCAL PATHS (SCRATCH)
iter_root_cloud="${work_path_cloud}/${iter_root_rel}" 
exe_path_cloud="${iter_root_cloud}/exe"

# --------- STAGE: Copy EFS -> Scratch ---------
log_debug "[SETUP] Copying inputs from EFS to Local Scratch..."
mkdir -p "${iter_root_cloud}"

rsync -a "${iter_root_share}/gridded" "${iter_root_cloud}/" || true
rsync -a "${iter_root_share}/point"   "${iter_root_cloud}/" || true
rsync -a "${iter_root_share}/exe"     "${iter_root_cloud}/" || true

# --------- CONFIG: Patch .info.txt AND launcher.sh ---------
info_file="${exe_path_cloud}/${domain}.info.txt"
launcher_file="${exe_path_cloud}/launcher.sh"

# Defined LOCAL output paths (Roots only)
path_static_gridded="${iter_root_cloud}/gridded"
path_static_point="${iter_root_cloud}/point"
path_output="${iter_root_cloud}/outcome"

# --- PATCH INFO FILE ---
if [[ -f "${info_file}" ]]; then
  log_debug "[CONFIG] Patching ${info_file} with LOCAL paths"
  # Replace global share path with global cloud path
  sed -i "s|${iter_root_share}|${iter_root_cloud}|g" "${info_file}" || true
  # Fallback for old placeholders
  sed -i "s|{path_output}|${path_output}|g" "${info_file}" || true
else
  log_debug "[CONFIG] WARNING: Info file not found at ${info_file}"
fi

# --- PATCH LAUNCHER SCRIPT (NEW!) ---
if [[ -f "${launcher_file}" ]]; then
  log_debug "[CONFIG] Patching ${launcher_file} to force LOCAL execution"
  # If the launcher has absolute paths like /share/simulations/ITER..., point them to /scratch/...
  sed -i "s|${iter_root_share}|${iter_root_cloud}|g" "${launcher_file}" || true
else
  log_debug "[CONFIG] WARNING: Launcher file not found at ${launcher_file}"
fi

# --------- PERMISSIONS FIX ---------
log_debug "[PERMS] Chown local workspace to floodproofs..."
chown -R floodproofs:floodproofs "${work_path_cloud}" || true

# --------- RUN MODEL (Local) ---------
mkdir -p "${path_output}"
chown -R floodproofs:floodproofs "${path_output}"

log_cloud="${iter_root_cloud}/outcome/run.log"
touch "${log_cloud}"
chown floodproofs:floodproofs "${log_cloud}"

echo "[RUN] Launching simulation on local disk..." >> "$path_file_lock_def_start"

# Execute inside local folder as user floodproofs
# We export ADDR2LINE to fix Conda crash
if cd "${exe_path_cloud}" && chmod +x ./launcher.sh && sudo -u floodproofs -H bash -lc "export ADDR2LINE=; ./launcher.sh" > "${log_cloud}" 2>&1
then
  log_debug "[RUN] Simulation SUCCESS"
  status="SUCCESS"
else
  log_debug "[RUN] Simulation FAILED"
  status="FAILED"
fi

# --------- SYNC: Scratch -> EFS ---------
log_debug "[SYNC] Offloading results from Scratch to EFS..."

if [[ ! -d "${path_output}" ]]; then
    log_debug "[SYNC] WARNING: Output folder ${path_output} missing! Recreating..."
    mkdir -p "${path_output}"
fi

mkdir -p "${dest_share}"
chown floodproofs:floodproofs "${dest_share}" || true

touch "${dest_share}/rsync_debug.log"
chown floodproofs:floodproofs "${dest_share}/rsync_debug.log"

if sudo -u floodproofs rsync -av "${path_output}/" "${dest_share}/" >> "${dest_share}/rsync_debug.log" 2>&1; then
    log_debug "[SYNC] Rsync OK"
else
    log_debug "[SYNC] Rsync ERRORS"
fi

sudo -u floodproofs cp "${log_cloud}" "${dest_share}/" || true

# --------- FINISH ---------
sudo -u floodproofs touch "$path_file_lock_def_end"
{
  echo " ==== EXECUTION END REPORT ==== "
  echo " ==== Status: $status"
  echo " ==== Log Synced to: $dest_share"
} >> "$path_file_lock_def_end"
"""


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text(file_path: str, text: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def build_user_data_from_template(
        template_text: str,
        domain: str,
        lock_start: str,
        lock_end: str,
        time_now: str,
        exe_path_share: str,
        efs_dns: str,
        work_path_cloud: str,
        ssh_keys: list[str],
) -> str:
    # Replace simple placeholders
    user_data = template_text
    user_data = user_data.replace("DOMAIN", str(domain))
    user_data = user_data.replace("LOCK_START", str(lock_start))
    user_data = user_data.replace("LOCK_END", str(lock_end))
    user_data = user_data.replace("TIME_NOW", str(time_now))
    user_data = user_data.replace("EXE_PATH", str(exe_path_share))
    user_data = user_data.replace("EFS_DNS", str(efs_dns))
    user_data = user_data.replace("WORK_PATH_CLOUD", str(work_path_cloud))

    # SSH keys setup
    keys = [str(k).strip() for k in (ssh_keys or []) if str(k).strip()]
    if not keys:
        raise RuntimeError("cloud.ssh_keys is empty: provide at least one SSH public key.")

    lines = []
    lines.append('authorized_keys_path="/home/floodproofs/.ssh/authorized_keys"')
    for i, k in enumerate(keys):
        redir = ">" if i == 0 else ">>"
        lines.append(f"echo '{k}' {redir} \"$authorized_keys_path\"")
    lines.append("chown -R floodproofs:floodproofs /home/floodproofs")
    lines.append("chmod 700 /home/floodproofs/.ssh")
    lines.append("chmod 600 /home/floodproofs/.ssh/authorized_keys")

    keys_block = "\n".join(lines) + "\n"

    if "SSH_KEYS_BLOCK" not in user_data:
        raise RuntimeError("SSH_KEYS_BLOCK placeholder missing in user-data template")

    user_data = user_data.replace("SSH_KEYS_BLOCK", keys_block)
    return user_data


def aws_launch_instances(region: str, launch_template_id: str, user_data_path: str, tags: dict) -> list[dict]:
    tags_str = ",".join([f"{{Key={k},Value={v}}}" for k, v in tags.items()])
    cmd = (
        f'aws ec2 run-instances '
        f'--region {region} '
        f'--launch-template "LaunchTemplateId={launch_template_id}" '
        f'--tag-specifications "ResourceType=instance,Tags=[{tags_str}]" '
        f'--metadata-options "InstanceMetadataTags=enabled" '
        f'--instance-initiated-shutdown-behavior terminate '
        f'--user-data "file://{user_data_path}" '
        f'--output json'
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if res.returncode != 0:
        raise RuntimeError(f"AWS run-instances failed: {res.stderr.strip()}")

    try:
        payload = json.loads(res.stdout)
        instances_info = []
        for x in payload.get("Instances", []):
            iid = x.get("InstanceId")
            pip = x.get("PrivateIpAddress", "N/A")
            if iid:
                instances_info.append({"id": iid, "ip": pip})

    except Exception as e:
        raise RuntimeError(f"AWS run-instances returned non-JSON output: {e}. Raw stdout: {res.stdout[:500]}")

    if not instances_info:
        raise RuntimeError(f"AWS run-instances did not return InstanceId(s). Raw stdout: {res.stdout[:500]}")

    return instances_info


def aws_terminate_instances_best_effort(region: str, instance_ids: list[str]) -> None:
    if not instance_ids:
        return
    ids = " ".join(instance_ids)
    cmd = f"aws ec2 terminate-instances --region {region} --instance-ids {ids}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if res.returncode != 0:
        logging.warning(f"AWS terminate-instances failed (best effort): {res.stderr.strip()}")


def wait_for_lock_end(lock_end_paths: list[str], poll_seconds: int, max_wait_seconds: int) -> None:
    t0 = time.time()
    while True:
        missing = [p for p in lock_end_paths if not os.path.exists(p)]
        if not missing:
            return
        if (time.time() - t0) > max_wait_seconds:
            raise TimeoutError(f"Timeout waiting for LOCK_END files. Missing: {missing[:10]}")
        time.sleep(poll_seconds)


def save_iteration_instance_ids(out_path: str, i_iter: int, iter_to_instance_info: dict) -> str:
    """
    Saves CSV: IterTag,InstanceID,PrivateIP
    """
    instance_id_file = os.path.join(out_path, f"ITER{i_iter:02d}.instance_ids")
    lines: list[str] = []

    for itag in sorted(iter_to_instance_info.keys()):
        info_list = iter_to_instance_info[itag]
        for item in info_list:
            iid = item["id"]
            pip = item["ip"]
            lines.append(f"{itag},{iid},{pip}")

    write_text(instance_id_file, "\n".join(lines) + ("\n" if lines else ""))
    return instance_id_file


def launch_runs_aws_ec2(
        domain: str,
        i_iter: int,
        n_explor: int,
        path_settings: dict,
        cloud_settings: dict,
        time_now: str,
) -> None:
    """
    Main entry point. Launches runs, waits for locks, saves IDs.
    """
    region = cloud_settings["region"]
    launch_template_id = cloud_settings["launch_template_id"]
    efs_dns = cloud_settings["efs_dns"]
    # This comes from JSON (e.g. "/scratch/hmc_calib").
    # It ensures we use LOCAL DISK for computation.
    work_path_cloud = cloud_settings["work_path_cloud"]

    poll_seconds = int(cloud_settings.get("poll_seconds", 30))
    max_wait_seconds = int(cloud_settings.get("max_wait_seconds", 86400))

    ssh_keys = cloud_settings.get("ssh_keys", None)
    if ssh_keys is None:
        raise RuntimeError("Missing cloud.ssh_keys in JSON.")

    out_path = path_settings.get("out_path", path_settings["work_path"])
    locks_folder = os.path.join(out_path, "locks", "calibration")

    tmp_user_data_dir = os.path.join(out_path, "tmp_user_data")
    os.makedirs(tmp_user_data_dir, exist_ok=True)

    lock_end_paths: list[str] = []
    launched_instance_ids: list[str] = []
    iter_to_instance_info: dict = {}

    for i_explor in range(1, int(n_explor) + 1):
        iter_tag = f"ITER{i_iter:02d}-{i_explor:03d}"

        # Lock files are checked on the SHARED path (out_path) so Python can see them
        lock_start = os.path.join(locks_folder, domain, f"{iter_tag}_START.txt")
        lock_end = os.path.join(locks_folder, domain, f"{iter_tag}_END.txt")
        os.makedirs(os.path.dirname(lock_start), exist_ok=True)

        # Source executable path (on Shared/EFS)
        exe_path_share = os.path.join(path_settings["work_path"], "simulations", iter_tag, "exe")

        user_data = build_user_data_from_template(
            template_text=default_aws_user_data_template,
            domain=domain,
            lock_start=lock_start,
            lock_end=lock_end,
            time_now=time_now,
            exe_path_share=exe_path_share,
            efs_dns=efs_dns,
            work_path_cloud=work_path_cloud,
            ssh_keys=ssh_keys,
        )

        user_data_path = os.path.join(tmp_user_data_dir, f"ist_script_{domain}_{iter_tag}.sh")
        write_text(user_data_path, user_data)

        tags = {
            "Name": f"hmc-calib-{domain}-{iter_tag}",
            "APPLICATION": cloud_settings.get("tag_application", "fp_calibrate"),
            "Domain": domain,
            "Iter": f"{i_iter:02d}",
            "Explor": f"{i_explor:03d}",
        }

        instances_info = aws_launch_instances(region, launch_template_id, user_data_path, tags)

        current_ids = [x["id"] for x in instances_info]
        launched_instance_ids.extend(current_ids)
        iter_to_instance_info[iter_tag] = instances_info
        lock_end_paths.append(lock_end)

    instance_id_file = save_iteration_instance_ids(out_path, i_iter, iter_to_instance_info)
    logging.info(f"[AWS] Saved instance ids and IPs to: {instance_id_file}")

    try:
        wait_for_lock_end(lock_end_paths, poll_seconds=poll_seconds, max_wait_seconds=max_wait_seconds)
    except TimeoutError:
        logging.error("[AWS] Timeout waiting for LOCK_END. Terminating instances.")
        aws_terminate_instances_best_effort(region, launched_instance_ids)
        raise