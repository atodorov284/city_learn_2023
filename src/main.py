import argparse
import uuid
from experiment_runners import experiment_runner


def main() -> None:
    """Setup a parser to parse command line arguments and run the CityLearn benchmark with the specified agent type."""
    parser = argparse.ArgumentParser(
        description="Run the CityLearn benchmark with the specified agent type"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["centralized", "decentralized", "maml", "all"],
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--k_shots", type=int, default=3)

    args = parser.parse_args()
    
    experiment_id = str(uuid.uuid4().hex)

    hyperparameters_dict = {
        "hidden_dim": args.hidden_size,
        "buffer_size": args.buffer_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "tau": args.tau,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "k_shots": args.k_shots,
    }

    if args.agent_type == "all":
        experiment_runner.setup_all_agents(
            seed=args.seed,
            episodes=args.episodes,
            hyperparameters_dict=hyperparameters_dict,
            experiment_id=experiment_id
        )
    else:
        experiment_runner.setup_single_agent(
            agent_type=args.agent_type,
            seed=args.seed,
            episodes=args.episodes,
            hyperparameters_dict=hyperparameters_dict,
            experiment_id=experiment_id
        )


if __name__ == "__main__":
    main()
