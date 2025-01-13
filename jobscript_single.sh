#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB

source .venv/bin/activate

python src/main.py --agent_type centralized --episodes 100

