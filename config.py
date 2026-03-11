import torch

# Random seed for reproducibility
SEED = None  # 42

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"

# Simulation parameters
SIMULATION_CONFIG = {
    "num_honest": 3,  # Number of honest clients
    "num_byzantine": 1,  # Number of Byzantine clients
    "rounds": 200,  # Number of communication rounds
    "batch_size": 64,
    "device": DEVICE,
    "dataset_name": "MNIST",  # Options: MNIST, CIFAR10, FashionMNIST
}

# Model parameters
MODEL_CONFIG = {
    "model_name": "cnn_mnist",  # Options: cnn_mnist, mlp_mnist, resnet18_cifar
    "loss_name": "CrossEntropyLoss",  # Options: NLLLoss, CrossEntropyLoss
    "optimizer_name": "SGD",  # Options: SGD, Adam
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "learning_rate_decay": 0.25,
}

# Data distribution parameters
DATA_DISTRIBUTION_CONFIG = {
    "distribution_name": "iid",  # Options: iid, dirichlet_niid, pathological_niid
    "distribution_parameter": 0.5,  # For Dirichlet: concentration parameter
    "store_per_client_metrics": True,  # Whether to store metrics per client
}

# Server parameters
SERVER_CONFIG = {
    "optimizer_name": MODEL_CONFIG["optimizer_name"],
    "learning_rate": MODEL_CONFIG["learning_rate"],
    "weight_decay": MODEL_CONFIG["weight_decay"],
    "milestones": [1000],  # Learning rate decay milestones
    "learning_rate_decay": 0.25,
    "use_pre_aggregation": False,  # Whether to use pre-aggregation defenses
}

# Client parameters
CLIENT_CONFIG = {
    "momentum": 0.9,
    "nb_labels": 10,  # Number of classes in dataset
    "store_per_client_metrics": True,
    "label_flipping": False,  # Whether to use label flipping attack
}

# Attack parameters
ATTACK_CONFIG = {
    "attack_name": "InnerProductManipulation",
    # Options: SignFlipping, Noise, InnerProductManipulation, Optimal_InnerProductManipulation
    "attack_parameters": {
        "tau": 3.0,  # For InnerProductManipulation
        # "scale": 10.0,  # For SignFlip
        # "noise_std": 0.1,  # For Noise attack
    },
}

# Aggregator parameters
AGGREGATOR_CONFIG = {
    "aggregators_to_compare": [
        "Average",
        "MultiKrum",
        "TrMean",
        "Median",
        "Krum",
        "SMEA",
        "CAF",
        "MDA",
        "PCAEigenvalueAggregator",
        "PCAEigenvalueAggregatorV2",
        "RobustPCAEigenvalueAggregator",
    ],
    "single_aggregator": "TrMean",
    "pre_aggregation_defenses": [
        {"name": "Clipping", "parameters": {"c": 2.0}},
        {"name": "NNM", "parameters": {"f": SIMULATION_CONFIG["num_byzantine"]}},
    ],
}

# Output/Result parameters
results_suffix = f"_ALL_CUSTOM_{ATTACK_CONFIG['attack_name']}_{MODEL_CONFIG["optimizer_name"]}"
OUTPUT_CONFIG = {
    "plot_save_path": f"results/defenses/fl_comparison{results_suffix}_{SIMULATION_CONFIG['num_byzantine']}_{SIMULATION_CONFIG['num_honest']}.png",
    "results_save_path": f"results/defenses/simulation_results{results_suffix}_{SIMULATION_CONFIG['num_byzantine']}_{SIMULATION_CONFIG['num_honest']}.json",
    "verbose": True,
    "save_models": False,
}


def print_config():
    """Print configuration"""
    print("=" * 50)
    print("ByzFL Federated Learning Simulation")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Model: {MODEL_CONFIG['model_name']}")
    print(
        f"Clients: {SIMULATION_CONFIG['num_honest']} honest, "
        f"{SIMULATION_CONFIG['num_byzantine']} Byzantine"
    )
    print(f"Rounds: {SIMULATION_CONFIG['rounds']}")
    print(f"Data Distribution: {DATA_DISTRIBUTION_CONFIG['distribution_name']}")
    print(f"Attack: {ATTACK_CONFIG['attack_name']}")
    print(f"Aggregator: {AGGREGATOR_CONFIG['single_aggregator']}")
    print("=" * 50)
