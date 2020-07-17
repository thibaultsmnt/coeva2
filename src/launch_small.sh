#!/bin/bash -l

#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p batch
#SBATCH --time=0-0:3:00
#SBATCH --qos=qos-besteffort
#SBATCH -J PREPARE
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=thibault.simonetto.001@student.uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
module restore python3
source ../adv/bin/activate

python "$@"
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh
