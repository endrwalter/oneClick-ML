#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH --nodelist=hilbert
#SBATCH --partition=cpu
#SBATCH --job-name=neuro_aa
#SBATCH --output=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML/batchResults/%j_stdOut.txt
#SBATCH --error=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML/batchResults/%j_stdErr.txt
#SBATCH --container-image=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML/enroot_img/cpu_cont2.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML
#SBATCH --container-remap-root

printf "Starting dir $(pwd)\n"
printf "Moving to working dir.\n"
cd /storage/DSH/projects/neuroart/pd/motor_symptoms_ML/code_ensemble
printf "current dir $(pwd)\n\n"
source /storage/DSH/projects/neuroart/pd/motor_symptoms_ML/enroot_img/venv/bin/activate

printf "\nLaunching PD script.\n"
python3 main.py --config config_freezing.ini &
python3 main.py --config config_motor_f.ini &
python3 main.py --config config.ini &
wait

