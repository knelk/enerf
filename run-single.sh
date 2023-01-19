#!/usr/bin/env bash

# given single config file, run enerf

config=$1
BASE_DIR="./Runs"  # << Insert your OUTDIR here
DT="$(date +%m-%d-%Y_%H%M%S)"
PID=$$
JOB_DIR="$BASE_DIR/${DT}_${PID}"
CODEDIR="."
SLURMSTART="slurm_start_train.sbatch"

mkdir -p "$JOB_DIR"
echo "$JOB_DIR"
# Copy Code to Job-Dir
cp "$CODEDIR/activation.py" "$JOB_DIR"
cp "$CODEDIR/encoding.py" "$JOB_DIR"
cp "$CODEDIR/environment.yml" "$JOB_DIR"
cp "$CODEDIR/loss.py" "$JOB_DIR"
cp "$CODEDIR/main_nerf.py" "$JOB_DIR"
cp "$CODEDIR/readme.md" "$JOB_DIR"
cp "$CODEDIR/requirements.txt" "$JOB_DIR"
cp "run-single.sh" "$JOB_DIR"
cp "$SLURMSTART" "$JOB_DIR"

cp -r "$CODEDIR/configs" "$JOB_DIR"
cp -sR "$PWD/$CODEDIR/ffmlp/" "$JOB_DIR/" # source must be abs-path, for SYMLINK to work  use -sR for symlink.
cp -r "$CODEDIR/gridencoder" "$JOB_DIR"
cp -r "$CODEDIR/hashencoder" "$JOB_DIR"
cp -r "$CODEDIR/nerf" "$JOB_DIR"
cp -r "$CODEDIR/raymarching" "$JOB_DIR"
cp -r "$CODEDIR/scripts" "$JOB_DIR"
cp -r "$CODEDIR/shencoder" "$JOB_DIR"
cp -r "$CODEDIR/testing" "$JOB_DIR"
cp -r "$CODEDIR/utils" "$JOB_DIR"

cd "$JOB_DIR"

CONFFILE="${1}"
sbatch "$SLURMSTART" "$CONFFILE"
