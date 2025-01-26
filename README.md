# 🌍🌳 CityLearn 2023 Grid Managment Meta Learning Project ⚡⚡
 
<a target="_blank" href="https://citylearn.net">
    <img src="https://img.shields.io/badge/CityLearn-Challenge-004080?logo=data:image/png;base64,..." />
</a>

Buildings play a significant role in the total energy use and carbon emissions in the United States. The US Department of Energy estimates that residential buildings consume around 21% of total energy use and emit 15% of total greenhouse gas emissions as of 2024. As climate change concerns intensify, modeling energy consumption in the residential building sector becomes increasingly vital. By understanding how homes use energy, we can develop targeted strategies for energy efficiency improvements and electrification, potentially leading to substantial reductions in both energy use and carbon emissions.

This study focuses on introducing the meta-RL paradigm to the energy-grid management problem, applied to the CityLearn 2023 environment, which models urban energy systems in
the US. The research investigates the performance of the usefulness and performance of meta-RL, compared to the currently existing top performers in the environment. Hence, we investigate 1) how a centralized learning, decentralized execution Model-Agnostic Meta-Learning agent, in combination with a Soft-Actor Critic, performs in terms of expected reward when compared to the decentralized learning, decentralized execution Soft-Actor Critic. Moreover, we 2) further compare them in terms of total time to train and convergence speed, as well as to the centralized learning, centralized execution Soft-Actor Critic, which is a state-of-the-art in the energy-management problem, but faces scalability issues.

---

## 🏃‍♂️ Running Source Code
### 🛠️ Set-Up

**Clone the Repository**: 
Start by cloning the repository to your local machine.
   ```bash
   git clone https://github.com/atodorov284/city_learn_2023.git
   cd city_learn_2023
   ```
**Set Up Package Environment**:
    Download uv package manager by running the following command:
    
   ```bash
   pip install uv
   ```
    
   Make sure all dependencies are installed by running the following command:
   ```bash
   uv sync
   ```
### 🏋️‍♂️ Training the Agents

To train the MARL agents, use the parser through a following command:

```bash
python ./src/main.py arg1 value1 ... argN valueN
```

**⚙️ Parser Arguments**

| Argument         | Type    | Default  | Description                                                                 |
|------------------|---------|----------|-----------------------------------------------------------------------------|
| `--seed`         | `int`   | `0`      | Seed for random number generation to ensure reproducibility.               |
| `--agent_type`   | `str`   | `"all"`  | Type of agent to use. Choices: `centralized`, `decentralized`, `maml`, `all`. |
| `--episodes`     | `int`   | `150`    | Number of episodes to run the simulation or training for.                  |
| `--hidden_size`  | `int`   | `256`    | Size of hidden layers in the neural network.                              |
| `--buffer_size`  | `int`   | `100000` | Size of the replay buffer for experience replay.                           |
| `--batch_size`   | `int`   | `256`    | Batch size used during training.                                           |
| `--learning_rate`| `float` | `3e-4`   | Learning rate for the optimizer.                                           |
| `--gamma`        | `float` | `0.99`   | Discount factor for future rewards in reinforcement learning.              |
| `--tau`          | `float` | `0.01`   | Soft update parameter for the target networks.                                 |
| `--alpha`        | `float` | `0.05`   | Coefficient for entropy regularization to guide exploration.           |
| `--k_shots`      | `int`   | `3`      | Number of inner adapation steps.                      |

Example use:
```bash
python ./src/main.py --seed 0 --agent_type maml --tau  0.02 
```

Each execution of code creates a unique experiment ID, which allows you to track the progress of training without overwriting plots.

---
### 👁️ Visualization 

Running the previous script will produce two plots in the **\plots** folder, following a format:

***step_rewards_agent-type_experiment-id_train***: a training curve of the agent with daily rewards on 1 month data with 3 buildings, and
***step_rewards_agent-type_experiment-id_eval***: an evaluation curve of the agent with daily rewards on 3 month data with unseen 3 buildings.

---

## Project Structure

```plaintext
.
├── data                   <- Simulated data
├── plots                  <- Folder with plots from last experiments
├── src                    <- Source code for the project
│   ├── experiment_runners <- Methods to train the agents, called in the parser
│   └── utils              <- Helper functions (replay buffer, custom reward)
│   └── agents             <- Implementation of MARL agents
│       ├── base_models    <- Base agents used in this study (SAC, random agent)
│       └── wrappers       <- Wrappers for the CityLearn environment to provide unified functionality
├── jobscript_all.sh       <- Script to run parallelized experiments on Habrok
├── jobscript_all.sh       <- Script to run a single experiment on Habrok
├── README.md              <- Project documentation
└── uv.lock                <- Dependency list
```

---

## Acknowledgements

This project is developed under the CityLearn 2023 framework developed for an AI crowd challenge. Special thanks to the organizers and contributors of the CityLearn platform for providing a comprehensive environment for advancing energy optimization research.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

