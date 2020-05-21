#!/bin/bash -l

#SBATCH -o %x_%j.out
#SBATCH -n 1
#SBATCH -p batch
#SBATCH --time=0-0:02:00
#SBATCH --qos=qos-batch
#SBATCH -J FITNESS-RANDOM-SEARCH
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=thibault.simonetto.001@student.uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
module restore python3
source ../adv/bin/activate

python fitness_random_search.py "$@"
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh
