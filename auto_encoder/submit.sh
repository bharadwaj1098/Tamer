#!/bin/bash 

#SBATCH --job-name=auto_encoder_type1
#SBATCH --partition=Hercules
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:TitanV:1
#SBATCH --time=24:00:00


module load pytorch/1.6.0-anaconda3-cuda10.2

pip install --no-index --upgrade pip

conda install -c akode atari-py  

python Type_1.py

