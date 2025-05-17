#!/bin/bash
#SBATCH --time=3-10:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mem=1GB

source .venv/bin/activate

python src/main.py --agent_type all --episodes 60 --num_runs 5

