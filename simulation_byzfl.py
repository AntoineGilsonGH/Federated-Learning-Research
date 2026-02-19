import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Any
import copy
import numpy as np

import torch
from torchvision import datasets, transforms
from byzfl import Client, Server, ByzantineClient, DataDistributor
import byzfl.aggregators as byzfl_agg  # Import the specific module

from utils.train import train
from utils.aggregators import CUSTOM_AGGREGATORS


class ByzFLSimulation:
    """
    Federated Learning Simulation using ByzFL framework.
    """

    def __init__(
        self,
        dataset_name: str = "MNIST",
        num_honest_clients: int = 8,
        num_byzantine_clients: int = 2,
        num_rounds: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        model_config: Dict[str, Any] = None,
        aggregator_config: Dict[str, Any] = None,
        attack_config: Dict[str, Any] = None,
        data_distribution_config: Dict[str, Any] = None,
        server_config: Dict[str, Any] = None,
        client_config: Dict[str, Any] = None,
    ):
        """
        Initialize the simulation.
        """
        self.dataset_name = dataset_name
        self.num_honest_clients = num_honest_clients
        self.num_byzantine_clients = num_byzantine_clients
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.device = device
        self.total_clients = num_honest_clients + num_byzantine_clients

        # Configurations
        self.model_config = model_config or {}
        self.aggregator_config = aggregator_config or {}
        self.attack_config = attack_config or {}
        self.data_distribution_config = data_distribution_config or {}
        self.server_config = server_config or {}
        self.client_config = client_config or {}

        # Results storage
        self.results = {}

        self.custom_aggregators = CUSTOM_AGGREGATORS
        self.builtin_aggregators = ["TrMean", "Krum", "Median", "Average", "MultiKrum", "MDA", "SMEA", "CAF"]

        self._patch_aggregators()

        # Setup
        self._prepare_data()
        self._setup_clients()

    def _patch_aggregators(self):
        """
        Patch custom aggregators into ByzFL's aggregators module.
        This makes them available to the Server class via getattr.
        """
        print("Patching custom aggregators into byzfl.aggregators.aggregators...")
        for agg_name, agg_class in self.custom_aggregators.items():
            setattr(byzfl_agg, agg_name, agg_class)
        print(f"Patched {len(self.custom_aggregators)} custom aggregators: {list(self.custom_aggregators.keys())}")

    def _get_dataset_transform(self):
        """Get appropriate transform for the dataset."""
        if self.dataset_name == "MNIST":
            return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        elif self.dataset_name == "CIFAR10":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        elif self.dataset_name == "FashionMNIST":
            return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        else:
            return transforms.ToTensor()

    def _prepare_data(self):
        """Prepare train and test datasets."""
        transform = self._get_dataset_transform()

        # Load dataset
        if self.dataset_name == "MNIST":
            train_dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            )
        elif self.dataset_name == "CIFAR10":
            train_dataset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        elif self.dataset_name == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                root="./data", train=False, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Distribute data among clients
        self.data_distributor = DataDistributor(
            {
                "data_distribution_name": self.data_distribution_config.get(
                    "distribution_name", "iid"
                ),
                "distribution_parameter": self.data_distribution_config.get(
                    "distribution_parameter", 0.5
                ),
                "nb_honest": self.num_honest_clients,
                "data_loader": train_loader,
                "batch_size": self.batch_size,
            }
        )

        self.client_dataloaders = self.data_distributor.split_data()

    def _setup_clients(self):
        """Initialize honest clients."""
        # Common client configuration
        base_client_config = {
            "model_name": self.model_config.get("model_name", "cnn_mnist"),
            "device": self.device,
            "loss_name": self.model_config.get("loss_name", "NLLLoss"),
            "LabelFlipping": self.client_config.get("label_flipping", False),
            "momentum": self.client_config.get("momentum", 0.9),
            "nb_labels": self.client_config.get("nb_labels", 10),
            "store_per_client_metrics": self.client_config.get(
                "store_per_client_metrics", True
            ),
        }

        # Create honest clients
        self.honest_clients = []
        for i in range(self.num_honest_clients):
            client_config = copy.deepcopy(base_client_config)
            client_config["training_dataloader"] = self.client_dataloaders[i]
            self.honest_clients.append(Client(client_config))

    def _setup_server(self, aggregator_name: str):
        """Initialize server with specified aggregator."""
        # Determine aggregator parameters
        if aggregator_name in self.builtin_aggregators:
            aggregator_info = self._setup_builtin_aggregator(aggregator_name)
        elif aggregator_name in self.custom_aggregators:
            aggregator_info = self._setup_custom_aggregator(aggregator_name)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator_name}"
                             f"Built-in: {self.builtin_aggregators}, "
                             f"Custom: {list(self.custom_aggregators.keys())}")

        # Configure pre-aggregation defenses
        pre_agg_list = []
        if self.aggregator_config.get("use_pre_aggregation", False):
            pre_agg_list = self.aggregator_config.get("pre_aggregation_defenses", [])

        # Server configuration
        server_config = {
            "device": self.device,
            "model_name": self.model_config.get("model_name", "cnn_mnist"),
            "test_loader": self.test_loader,
            "optimizer_name": self.server_config.get("optimizer_name", "SGD"),
            "learning_rate": self.server_config.get("learning_rate", 0.1),
            "weight_decay": self.server_config.get("weight_decay", 0.0001),
            "milestones": self.server_config.get("milestones", [1000]),
            "learning_rate_decay": self.server_config.get("learning_rate_decay", 0.25),
            "aggregator_info": aggregator_info,
            "pre_agg_list": pre_agg_list,
        }

        return Server(server_config)

    def _setup_builtin_aggregator(self, aggregator_name):
        aggregator_info = {}
        if aggregator_name == "TrMean":
            aggregator_info = {
                "name": "TrMean",
                "parameters": {"f": self.num_byzantine_clients},
            }
        elif aggregator_name == "Krum":
            aggregator_info = {
                "name": "Krum",
                "parameters": {"f": self.num_byzantine_clients},
            }
        elif aggregator_name == "Median":
            aggregator_info = {"name": "Median", "parameters": {}}
        elif aggregator_name == "Average":
            aggregator_info = {"name": "Average", "parameters": {}}
        elif aggregator_name == "MultiKrum":
            aggregator_info = {
                "name": "MultiKrum",
                "parameters": {"f": self.num_byzantine_clients, "m": 1},
            }
        elif aggregator_name == "MDA":
            aggregator_info = {
                "name": "MDA",
                "parameters": {"f": self.num_byzantine_clients},
            }
        elif aggregator_name == "SMEA":
            aggregator_info = {
                "name": "SMEA",
                "parameters": {"f": self.num_byzantine_clients},
            }
        elif aggregator_name == "CAF":
            aggregator_info = {
                "name": "CAF",
                "parameters": {"f": self.num_byzantine_clients},
            }
        else:
            raise ValueError(f"Unknown aggregator: {aggregator_name}")
        
        return aggregator_info

    def _setup_custom_aggregator(self, aggregator_name: str) -> Dict[str, Any]:
        """
        Set up custom aggregator.
        
        Args:
            aggregator_name: Name of the custom aggregator
            
        Returns:
            Dictionary with aggregator info for server configuration
        """
        # Get the aggregator class
        aggregator_class = self.custom_aggregators[aggregator_name]
        
        # Get custom parameters from config
        custom_params = self.aggregator_config.get("custom_parameters", {})
        agg_params = custom_params.get(aggregator_name, {})
        
        # Ensure f is included
        agg_params["f"] = self.num_byzantine_clients
        
        # Create aggregator instance
        aggregator_instance = aggregator_class(**agg_params)
        
        # Return in format expected by ByzFL Server
        # Note: This assumes ByzFL can handle a 'custom_aggregator' field
        return {
            "name": aggregator_name,
            "parameters": agg_params,
            "custom_aggregator": aggregator_instance,  # Pass the instance directly
            "is_custom": True,
        }

    def _setup_byzantine_client(self):
        """Initialize Byzantine client with specified attack."""
        attack_config = {
            "name": self.attack_config.get("attack_name", "SignFlipping"),
            "f": self.num_byzantine_clients,
            "parameters": self.attack_config.get("attack_parameters", {}),
        }

        return ByzantineClient(attack_config)

    def run_single_aggregator_(
        self, aggregator_name: str = None, save_results: bool = True
    ) -> List[float]:
        """
        Run simulation with a single aggregator.

        Args:
            aggregator_name: Name of aggregator to use
            save_results: Whether to save results internally

        Returns:
            List of accuracy values per round
        """
        if aggregator_name is None:
            aggregator_name = self.aggregator_config.get("single_aggregator", "TrMean")

        print(f"\n--- Running simulation with {aggregator_name} aggregator ---")
        print(f"Honest clients: {self.num_honest_clients}")
        print(f"Byzantine clients: {self.num_byzantine_clients}")

        # Setup server and Byzantine client
        self._setup_clients()
        self._prepare_data()
        server = self._setup_server(aggregator_name)
        byz_client = self._setup_byzantine_client()

        # Run training
        print("Starting training...")
        accuracy_history = train(
            server, self.num_rounds, self.honest_clients, byz_client
        )

        if save_results:
            self.results[aggregator_name] = accuracy_history

        # Print final results
        print(f"\nFinal Accuracy: {accuracy_history[-1]:.2%}")
        print(f"Best Accuracy: {max(accuracy_history):.2%}")

        return accuracy_history

    def run_single_aggregator(
            self, aggregator_name: str = None, save_results: bool = True
        ) -> List[float]:
        """
        Run simulation with a single aggregator (built-in or custom).

        Args:
            aggregator_name: Name of aggregator to use
            save_results: Whether to save results internally

        Returns:
            List of accuracy values per round
        """
        if aggregator_name is None:
            aggregator_name = self.aggregator_config.get("single_aggregator", "TrMean")

        # Determine aggregator type for printing
        if aggregator_name in self.builtin_aggregators:
            agg_type = "built-in"
        elif aggregator_name in self.custom_aggregators:
            agg_type = "custom"
        else:
            raise ValueError(f"Unknown aggregator: {aggregator_name}")

        print(f"\n--- Running simulation with {agg_type} aggregator: {aggregator_name} ---")
        print(f"Honest clients: {self.num_honest_clients}")
        print(f"Byzantine clients: {self.num_byzantine_clients}")

        # Setup server and Byzantine client
        self._setup_clients()
        self._prepare_data()
        server = self._setup_server(aggregator_name)
        byz_client = self._setup_byzantine_client()

        # Run training
        print("Starting training...")
        accuracy_history = train(
            server, self.num_rounds, self.honest_clients, byz_client
        )

        if save_results:
            self.results[aggregator_name] = accuracy_history

        # Print final results
        print(f"\nFinal Accuracy: {accuracy_history[-1]:.2%}")
        print(f"Best Accuracy: {max(accuracy_history):.2%}")

        return accuracy_history

    def compare_aggregators(
        self, aggregator_names: List[str] = None, save_plots: bool = True
    ):
        """
        Compare multiple aggregators.

        Args:
            aggregator_names: List of aggregator names to compare
            save_plots: Whether to save comparison plots
        """
        if aggregator_names is None:
            aggregator_names = self.aggregator_config.get(
                "aggregators_to_compare", ["Average", "TrMean", "Median", "Krum"]
            )

        print(f"\n--- Comparing {len(aggregator_names)} aggregators ---")

        for agg_name in aggregator_names:
            print(f"\nRunning {agg_name}...")
            self.run_single_aggregator(agg_name, save_results=True)

        if save_plots:
            self.plot_results()

    def plot_results(
        self,
        aggregator_names: List[str] = None,
        save_path: str = None,
        title: str = None,
    ):
        """Plot simulation results."""
        if not self.results:
            raise ValueError("No results to plot. Run simulation first.")

        # Default: plot all available aggregators
        if aggregator_names is None:
            aggregator_names = list(self.results.keys())

        plt.figure(figsize=(10, 6))

        colors = ["red", "green", "blue", "orange", "purple", "brown"]
        colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 colors
        linestyles = ["-", "--", "-.", ":", "-", "--"]

        x_step = 10  # distance between evaluation points

        for idx, name in enumerate(aggregator_names):
            if name not in self.results:
                continue

            y_values = self.results[name]
            x_values = [i * x_step for i in range(len(y_values))]

            plt.plot(
                x_values,
                y_values,
                label=name,
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2,
            )

        if title is None:
            title = f"ByzFL: {self.num_byzantine_clients}/{self.total_clients} Byzantine Clients"

        # ----- Axis formatting -----
        plt.title(title)
        plt.xlabel("Training Steps")
        plt.ylabel("Test Accuracy")

        # Force plot to start at 0 and end at last point
        if aggregator_names:
            first_name = aggregator_names[0]
            n_points = len(self.results[first_name])
            plt.xlim(0, (n_points - 1) * x_step)

        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_results(self, save_path: str = None):
        """Save simulation results to JSON file."""
        if not self.results:
            print("Warning: No results to save.")
            return

        if save_path is None:
            save_path = "results/simulation_results.json"

        # Prepare results dictionary
        results_dict = {
            "simulation_config": {
                "dataset": self.dataset_name,
                "num_honest_clients": self.num_honest_clients,
                "num_byzantine_clients": self.num_byzantine_clients,
                "num_rounds": self.num_rounds,
                "batch_size": self.batch_size,
                "device": self.device,
            },
            "model_config": self.model_config,
            "attack_config": self.attack_config,
            "aggregator_config": self.aggregator_config,
            "results": self.results,
            "final_accuracies": {name: acc[-1] for name, acc in self.results.items()},
            "best_accuracies": {name: max(acc) for name, acc in self.results.items()},
        }

        # Save to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"Results saved to {save_path}")

    def get_results_summary(self):
        """Get a summary of simulation results."""
        summary = {}
        for agg_name, acc_history in self.results.items():
            summary[agg_name] = {
                "final_accuracy": acc_history[-1],
                "best_accuracy": max(acc_history),
                "average_accuracy": sum(acc_history) / len(acc_history),
            }
        return summary
