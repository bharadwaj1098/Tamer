#!/bin/bash 

#SBATCH --jobname=auto_encoder_type1
#SBATCH --partition=HERCULES
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu = 16GB
#SBATCH --gres=gpu:TitanV:1
#SBATCH --time=24:00:00

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Node List   : $SLURM_JOB_NODELIST"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo "" 

module load python
module load torch
module load torchvision

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"

