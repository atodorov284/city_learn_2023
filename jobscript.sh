#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --mem=16GB

source .venv/bin/activate

python src/main.py --agent_type all --episodes 1

