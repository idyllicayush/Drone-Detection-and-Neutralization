#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name DIO
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --error=error
#SBATCH --output=output
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_NTASKS="$SLURM_NTASKS

ulimit -s unlimited
ulimit -c unlimited

#source /home/apps/DL/DL-CondaPy3.7/bin/activate
#source activate /scratch/ritali.ai.iitmandi/.conda/envs/pyan
##python -m bitsandbytes
python3 /scratch/ritali.ai.iitmandi/d.py > output_today_optimized.txt