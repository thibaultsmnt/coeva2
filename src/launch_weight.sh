#!/bin/bash -l

#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p batch
#SBATCH --time=0-4:00:00
#SBATCH --qos=qos-besteffort
#SBATCH -J WEIGHT-SEARCH
#SBATCH --mail-type=all
#SBATCH --mail-user=thibault.simonetto.001@student.uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
module load swenv/default-env/v1.1-20180716-production lang/Python/3.6.4-foss-2018a
source ../adv/bin/activate

python "$@"
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh
