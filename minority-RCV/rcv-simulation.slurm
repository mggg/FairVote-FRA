#!/bin/bash
#SBATCH --job-name="rcv-sim"
#SBATCH --time=4-00:00:00 # days-hh:mm:ss
#SBATCH --nodes=1 # how many computers do we need?
#SBATCH --ntasks-per-node=1 # how many cores per node do we need?
#SBATCH --mem=16000 # how many MB of memory do we need (16GB here)
#SBATCH --partition=largemem # which queue to use
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=apizzi02@tufts.edu

# Environment setup
source ~/.bashrc
conda activate rcv

# Navigate to the correct directory.
cd /cluster/tufts/mggg/apizzi02/FairVote/minority-RCV

# Do some analysis!
python rcv-simulation.py ${1} ${2} ${3}
