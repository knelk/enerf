#!/bin/bash
#SBATCH --job-name="sim-train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --mail-type=NONE	
#SBATCH --exclude=node7
#SBATCH --output=OUTDIR/slurm-%j.out
#SBATCH --error=OUTDIR/error-%j.out

echo "$PWD/$SLURM_JOBID"
mkdir -p "$PWD/slurm_id_$SLURM_JOBID"
CONF="${1}"
echo "$CONF"

srun python main_nerf.py --config "$CONF"
