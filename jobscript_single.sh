#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB

source .venv/bin/activate

python src/main.py --agent_type maml --episodes 100 --k_shots 50 --learning_rate 0.001

