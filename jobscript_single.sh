#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB

source .venv/bin/activate

python src/main.py --agent_type maml --episodes 2


