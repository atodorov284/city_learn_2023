#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mem=16GB

source .venv/bin/activate

python src/main.py --agent_type all --episodes 200

