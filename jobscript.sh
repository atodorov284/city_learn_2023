#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --mem=8GB

source .venv/bin/activate

python src/main.py --agent_type all --episodes 200

