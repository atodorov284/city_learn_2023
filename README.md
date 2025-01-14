# CityLearn 2023 Challenge

<a target="_blank" href="https://citylearn.net">
    <img src="https://img.shields.io/badge/CityLearn-Challenge-004080?logo=data:image/png;base64,..." />
</a>

The CityLearn 2023 Challenge is an international competition focused on designing and optimizing multi-agent reinforcement learning (MARL) algorithms to manage energy systems in urban environments. Participants develop solutions that aim to minimize energy consumption, enhance grid stability, and optimize renewable energy usage. This repository contains our solution, submitted as part of the challenge.

---

## Project Overview

Urban energy systems face challenges related to efficiency, sustainability, and scalability. The CityLearn Challenge provides a simulation environment for researchers and practitioners to test their algorithms in a realistic urban setting. The environment models interactions between multiple buildings, energy storage systems, and the power grid.

Our solution incorporates state-of-the-art MARL techniques, leveraging recent advancements in reinforcement learning to address these challenges effectively.

---

## Features

- **Environment:** Integration with the CityLearn simulation environment.
- **Agents:** Custom MARL agents designed for energy optimization.
- **Training Pipeline:** A robust training pipeline to evaluate and fine-tune agent performance.
- **Visualization:** Tools to analyze agent behaviors and energy usage patterns.

---

## Installation

### Prerequisites

- Python 3.9 or later
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/atodorov284/city_learn_2023.git
cd city_learn_2023
```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Training the Agents

To train the MARL agents, use the following command:

```bash
python main.py --mode train
```

Optional arguments:
- `--episodes`: Number of training episodes (default: 1000)
- `--config`: Path to the configuration file (default: `configs/default.json`)

### Evaluating the Agents

To evaluate the performance of trained agents:

```bash
python main.py --mode evaluate
```

### Visualization

To visualize agent performance and energy usage patterns:

```bash
python visualize.py
```

---

## Project Structure

```plaintext
.
├── LICENSE                <- Open-source MIT license
├── README.md              <- Project documentation
├── configs                <- Configuration files for training and evaluation
├── data                   <- Simulated data and results
├── logs                   <- Training and evaluation logs
├── models                 <- Saved models and checkpoints
├── notebooks              <- Jupyter notebooks for analysis
├── requirements.txt       <- Dependency list
├── scripts                <- Utility scripts
├── src                    <- Source code for the project
│   ├── agents             <- Implementation of MARL agents
│   ├── environments       <- Wrappers for the CityLearn environment
│   ├── training           <- Training pipeline
│   ├── evaluation         <- Evaluation scripts
│   └── utils              <- Helper functions
└── visualize.py           <- Visualization tool for energy usage
```

---

## Acknowledgements

This project is developed for the CityLearn 2023 Challenge. Special thanks to the organizers and contributors of the CityLearn platform for providing a comprehensive environment for advancing energy optimization research.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

