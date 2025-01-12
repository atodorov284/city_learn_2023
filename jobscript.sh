#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB

source .venv/bin/activate

python src/main.py --agent_type maml --episodes 150

