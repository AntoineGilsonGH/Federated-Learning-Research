"""
Bound-Challenging Experiment for PCAEA Defense.

For each coeff in a sweep, we:
  1. Compute honest gradients normally each round
  2. Compute the base Byzantine vector via the chosen attack
  3. Estimate mu_B and sigma^2 from the current honest gradients
  4. Rescale the Byzantine vector so that:
         ||mu_M - mu_B||^2 = coeff * (6 * sigma^2 / epsilon)
  5. Run a full independent training with that fixed scaling strategy

A special "baseline" run (coeff=None) is also included, using the
unmodified Byzantine vectors straight from ByzFL with no rescaling.

Per-round metrics tracked:
  - Test accuracy          (every 10 steps)
  - True Positive Rate     (TPR): malicious correctly flagged for removal
  - False Negative Rate    (FNR): malicious wrongly kept as clean  (= 1 - TPR)
  - Actual bound ratio     ||mu_M - mu_B||^2 / (6*sigma^2/epsilon)

Reads all simulation parameters from config_attacks.py — no duplication.
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from byzfl import ByzantineClient, Client, DataDistributor, Server
import byzfl.aggregators as byzfl_agg
from torchvision import datasets, transforms

import config_attacks as config
from utils.aggregators_bound import CUSTOM_AGGREGATORS


# ──────────────────────────────────────────────────────────────────────────────
# Experiment knobs  (only things not already in config_attacks.py)
# ──────────────────────────────────────────────────────────────────────────────

# None = baseline: unmodified Byzantine vectors, no rescaling
COEFFS = [None, 2.0, 1.0, 0.5]

# Single attack & aggregator to study
# ATTACK = {"attack_name": "ALittleIsEnough", "attack_parameters": {"tau": 2.5}}
# ATTACK = {"attack_name": "SignFlipping", "attack_parameters": {}}
ATTACK = {"attack_name": "Optimal_InnerProductManipulation", "attack_parameters": {}}

AGGREGATOR_NAME = "PCAEigenvalueAggregator"

OUTPUT_DIR = "results/bound_challenge_test_OptiSignFlip"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Setup helpers  (mirror your existing simulation classes, using config)
# ──────────────────────────────────────────────────────────────────────────────

def patch_aggregators():
    for name, cls in CUSTOM_AGGREGATORS.items():
        setattr(byzfl_agg, name, cls)


def get_transform(dataset_name):
    if dataset_name == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010))
        ])
    elif dataset_name == "FashionMNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return transforms.ToTensor()


def load_data():
    """Returns (client_dataloaders, test_loader) — reads config."""
    name      = config.SIMULATION_CONFIG["dataset_name"]
    bs        = config.SIMULATION_CONFIG["batch_size"]
    n_honest  = config.SIMULATION_CONFIG["num_honest"]
    transform = get_transform(name)

    DS = {"MNIST": datasets.MNIST,
          "CIFAR10": datasets.CIFAR10,
          "FashionMNIST": datasets.FashionMNIST}[name]

    train_ds = DS(root="./data", train=True,  download=True, transform=transform)
    test_ds  = DS(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=bs, shuffle=False)

    distributor = DataDistributor({
        "data_distribution_name":  config.DATA_DISTRIBUTION_CONFIG["distribution_name"],
        "distribution_parameter":  config.DATA_DISTRIBUTION_CONFIG["distribution_parameter"],
        "nb_honest":               n_honest,
        "data_loader":             train_loader,
        "batch_size":              bs,
    })
    return distributor.split_data(), test_loader


def make_clients(client_dataloaders):
    base = {
        "model_name":               config.MODEL_CONFIG["model_name"],
        "device":                   config.SIMULATION_CONFIG["device"],
        "loss_name":                config.MODEL_CONFIG["loss_name"],
        "LabelFlipping":            config.CLIENT_CONFIG["label_flipping"],
        "momentum":                 config.CLIENT_CONFIG["momentum"],
        "nb_labels":                config.CLIENT_CONFIG["nb_labels"],
        "store_per_client_metrics": config.CLIENT_CONFIG["store_per_client_metrics"],
    }
    clients = []
    for dl in client_dataloaders:
        cfg = copy.deepcopy(base)
        cfg["training_dataloader"] = dl
        clients.append(Client(cfg))
    return clients


def make_server(test_loader):
    f = config.SIMULATION_CONFIG["num_byzantine"]

    agg_info = {
        "name":       AGGREGATOR_NAME,
        "parameters": {"f": f},
    }
    server_cfg = {
        "device":              config.SIMULATION_CONFIG["device"],
        "model_name":          config.MODEL_CONFIG["model_name"],
        "test_loader":         test_loader,
        "optimizer_name":      config.SERVER_CONFIG["optimizer_name"],
        "learning_rate":       config.SERVER_CONFIG["learning_rate"],
        "weight_decay":        config.SERVER_CONFIG["weight_decay"],
        "milestones":          config.SERVER_CONFIG["milestones"],
        "learning_rate_decay": config.SERVER_CONFIG["learning_rate_decay"],
        "aggregator_info":     agg_info,
        "pre_agg_list":        [],
    }
    return Server(server_cfg)


def make_byz_client():
    return ByzantineClient({
        "name":       ATTACK["attack_name"],
        "f":          config.SIMULATION_CONFIG["num_byzantine"],
        "parameters": ATTACK["attack_parameters"],
    })


# ──────────────────────────────────────────────────────────────────────────────
# Rescaling
# ──────────────────────────────────────────────────────────────────────────────

def scale_byz_vector(byz_vec, mu_b, coeff, sigma2, epsilon):
    """
    Rescale byz_vec so that ||mu_M - mu_B||^2 = coeff * 6*sigma^2/epsilon.

    coeff=0  → Byzantine vector sits exactly on the honest mean (no attack).
    """
    target_sq_norm = coeff * 6.0 * sigma2 / epsilon

    if target_sq_norm <= 0.0:
        return mu_b.clone()

    direction   = byz_vec - mu_b
    dir_norm_sq = torch.dot(direction, direction).item()

    if dir_norm_sq < 1e-12:
        # Degenerate: attack == honest mean, perturb randomly
        direction   = torch.randn_like(byz_vec)
        dir_norm_sq = torch.dot(direction, direction).item()

    scale = (target_sq_norm / dir_norm_sq) ** 0.5
    return mu_b + scale * direction


# ──────────────────────────────────────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_with_coeff(coeff, client_dataloaders, test_loader):
    """
    Full independent FL training for one coeff value (or None = baseline).

    Returns
    -------
    accuracy_history     : list[float]  — one entry per eval point (every 10 steps)
    tpr_history          : list[float]  — True Positive Rate per round
    fnr_history          : list[float]  — False Negative Rate per round
    ratio_history        : list[float]  — actual ||mu_M-mu_B||^2 / (6s2/eps) per round
    """
    label   = "baseline" if coeff is None else f"coeff={coeff:.2f}"
    n_byz   = config.SIMULATION_CONFIG["num_byzantine"]
    n_honest= config.SIMULATION_CONFIG["num_honest"]
    n_total = n_honest + n_byz
    epsilon = n_byz / n_total
    rounds  = config.SIMULATION_CONFIG["rounds"]

    # Fresh clients, server, byz client for each run
    honest_clients = make_clients(client_dataloaders)
    server     = make_server(test_loader)
    byz_client = make_byz_client()

    # ByzFL instantiates its OWN aggregator instance internally — our agg object
    # is never called. The real instance lives at server.robust_aggregator.aggregator.
    agg = server.robust_aggregator.aggregator

    accuracy_history = []
    tpr_history      = []
    fnr_history      = []
    ratio_history    = []

    for step in range(rounds):

        # ── Evaluate every 10 steps ──────────────────────────
        if step % 10 == 0:
            acc = server.compute_test_accuracy()
            accuracy_history.append(acc)
            print(f"  [{label}] step {step:>4}/{rounds}  acc={acc:.4f}")

        # ── Honest gradients ──────────────────────────────────
        for client in honest_clients:
            client.compute_gradients()
        honest_grads = [c.get_flat_gradients_with_momentum() for c in honest_clients]
        honest_stack = torch.stack(honest_grads)   # (n_honest, d) — on model device

        mu_b   = honest_stack.mean(dim=0)
        # sum over all d dims — consistent with unnormalized ||mu_M - mu_B||^2
        sigma2 = honest_stack.var(dim=0).sum().item()

        # ── Byzantine vector ──────────────────────────────────
        raw_byz = byz_client.apply_attack(honest_grads)
        # ByzFL returns a list of f identical vectors; take the first
        if isinstance(raw_byz, list):
            raw_byz = raw_byz[0]
        if not torch.is_tensor(raw_byz):
            raw_byz = torch.tensor(raw_byz)
        raw_byz = raw_byz.to(device=mu_b.device, dtype=mu_b.dtype)

        # ── Rescale (or keep as-is for baseline) ─────────────
        if coeff is None:
            scaled_byz = raw_byz
        else:
            scaled_byz = scale_byz_vector(raw_byz, mu_b, coeff, sigma2, epsilon)

        # ── Diagnostic ratio ──────────────────────────────────
        actual_sq = torch.dot(scaled_byz - mu_b, scaled_byz - mu_b).item()
        bound_val = 6.0 * sigma2 / epsilon if sigma2 > 0 else 1.0
        ratio_history.append(actual_sq / bound_val)

        # ── Build gradient list: honest first, byz last ───────
        # Byzantine indices are known: [n_honest, n_honest + n_byz)
        byz_list  = [scaled_byz.clone() for _ in range(n_byz)]
        gradients = honest_grads + byz_list

        # ── Server update (calls instrumented aggregator) ─────
        server.update_model_with_gradients(gradients)

        # ── TPR / FNR directly from aggregator ───────────────
        removed     = set(getattr(agg, 'last_removed_indices', []))
        byz_indices = set(range(n_honest, n_honest + n_byz))

        tp  = len(removed & byz_indices)
        fn  = n_byz - tp
        tpr = tp / n_byz
        fnr = fn / n_byz

        tpr_history.append(tpr)
        fnr_history.append(fnr)

        # ── Broadcast updated model ───────────────────────────
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)

    # Final evaluation
    acc = server.compute_test_accuracy()
    accuracy_history.append(acc)
    print(f"  [{label}] FINAL acc={acc:.4f}")

    return accuracy_history, tpr_history, fnr_history, ratio_history


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def coeff_label(coeff):
    return "baseline (no rescaling)" if coeff is None else f"coeff={coeff:.2f}"


def make_colors(n):
    """n-1 colors from RdYlGn for numeric coeffs + 1 distinct color for baseline."""
    numeric_colors = plt.cm.RdYlGn(np.linspace(0.05, 0.95, max(n - 1, 1)))
    baseline_color = np.array([0.2, 0.2, 0.8, 1.0])   # blue for baseline
    return [baseline_color] + list(numeric_colors)


def smooth(arr, window=10):
    if len(arr) < window:
        return arr
    return list(np.convolve(arr, np.ones(window) / window, mode="valid"))


def plot_results(all_results):
    coeffs  = [r["coeff"]   for r in all_results]
    labels  = [coeff_label(c) for c in coeffs]
    colors  = make_colors(len(all_results))
    x_step  = 10

    # ── 1. Accuracy curves ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for r, color, label in zip(all_results, colors, labels):
        acc = r["accuracy"]
        xs  = [i * x_step for i in range(len(acc))]
        ls  = "--" if r["coeff"] is None else "-"
        ax.plot(xs, acc, label=label, color=color, linewidth=2, linestyle=ls)
    ax.axhline(0.1, color="black", linestyle=":", linewidth=1, label="random baseline")
    ax.set_title(
        f"PCAEA Defense — Bound Challenge\n"
        f"Attack: {ATTACK['attack_name']} | "
        f"n={config.SIMULATION_CONFIG['num_honest'] + config.SIMULATION_CONFIG['num_byzantine']} "
        f"f={config.SIMULATION_CONFIG['num_byzantine']}"
    )
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, "accuracy_curves.png")

    # ── 2. Final accuracy vs coeff (phase transition) ────────
    numeric = [r for r in all_results if r["coeff"] is not None]
    if numeric:
        num_coeffs  = [r["coeff"]      for r in numeric]
        final_accs  = [r["accuracy"][-1] for r in numeric]
        best_accs   = [max(r["accuracy"]) for r in numeric]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(num_coeffs, final_accs, "o-",  color="steelblue",  linewidth=2, label="Final accuracy")
        ax.plot(num_coeffs, best_accs,  "s--", color="darkorange", linewidth=2, label="Best accuracy")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Theoretical bound (coeff=1)")
        ax.set_title("Final / Best Accuracy vs Bound Coefficient\n(phase transition expected near coeff=1)")
        ax.set_xlabel("coeff  (‖μ_M − μ_B‖² = coeff × 6σ²/ε)")
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        _save(fig, "final_acc_vs_coeff.png")

    # ── 3. TPR curves ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for r, color, label in zip(all_results, colors, labels):
        tpr_s = smooth(r["tpr"])
        xs    = list(range(len(tpr_s)))
        ls    = "--" if r["coeff"] is None else "-"
        ax.plot(xs, tpr_s, label=label, color=color, linewidth=2, linestyle=ls)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="perfect detection")
    ax.set_title("True Positive Rate — malicious correctly flagged (smoothed)")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("TPR")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, "tpr_curves.png")

    # ── 4. FNR curves ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for r, color, label in zip(all_results, colors, labels):
        fnr_s = smooth(r["fnr"])
        xs    = list(range(len(fnr_s)))
        ls    = "--" if r["coeff"] is None else "-"
        ax.plot(xs, fnr_s, label=label, color=color, linewidth=2, linestyle=ls)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1, label="perfect detection")
    ax.set_title("False Negative Rate — malicious wrongly kept as clean (smoothed)")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("FNR")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fnr_curves.png")

    # ── 5. Bound ratio diagnostics ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for r, color, label in zip(all_results, colors, labels):
        xs = list(range(len(r["ratio"])))
        ls = "--" if r["coeff"] is None else "-"
        ax.plot(xs, r["ratio"], label=label, color=color, linewidth=1.5, linestyle=ls, alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Separability threshold")
    ax.set_title("Actual ‖μ_M − μ_B‖² / (6σ²/ε) per Round  (sanity check — should track target coeff)")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, "ratio_diagnostics.png")

    # ── 6. Combined: accuracy + TPR side by side ─────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for r, color, label in zip(all_results, colors, labels):
        acc   = r["accuracy"]
        tpr_s = smooth(r["tpr"])
        xs_a  = [i * x_step for i in range(len(acc))]
        xs_t  = list(range(len(tpr_s)))
        ls    = "--" if r["coeff"] is None else "-"
        ax1.plot(xs_a, acc,   label=label, color=color, linewidth=2, linestyle=ls)
        ax2.plot(xs_t, tpr_s, label=label, color=color, linewidth=2, linestyle=ls)

    ax1.set_title("Test Accuracy"); ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Training Step"); ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=7, loc="lower right")

    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax2.set_title("TPR (smoothed)"); ax2.set_ylim(-0.05, 1.1); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Training Round"); ax2.set_ylabel("TPR")
    ax2.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"{ATTACK['attack_name']} vs {AGGREGATOR_NAME} — bound coefficient sweep",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, "combined.png")


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    patch_aggregators()
    config.print_config()

    print("=" * 60)
    print("Bound-Challenging Experiment")
    print(f"  Attack      : {ATTACK['attack_name']} {ATTACK['attack_parameters']}")
    print(f"  Defense     : {AGGREGATOR_NAME}")
    print(f"  Coefficients: {COEFFS}")
    print("=" * 60)

    # Load data ONCE — all runs share the same split for fair comparison
    client_dataloaders, test_loader = load_data()

    all_results = []
    for coeff in COEFFS:
        print(f"\n{'─' * 55}")
        print(f"  coeff = {coeff_label(coeff)}")
        if coeff is not None:
            print(f"  target: ‖μ_M − μ_B‖² = {coeff:.2f} × 6σ²/ε")
        print(f"{'─' * 55}")

        acc, tpr, fnr, ratio = train_with_coeff(coeff, client_dataloaders, test_loader)
        all_results.append({
            "coeff":    coeff,
            "accuracy": acc,
            "tpr":      tpr,
            "fnr":      fnr,
            "ratio":    ratio,
        })

    plot_results(all_results)

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'run':>26}  {'final_acc':>10}  {'best_acc':>9}  {'mean_tpr':>9}  {'mean_ratio':>10}")
    print("-" * 70)
    for r in all_results:
        label      = coeff_label(r["coeff"])
        mean_tpr   = float(np.mean(r["tpr"]))   if r["tpr"]   else float("nan")
        mean_ratio = float(np.mean(r["ratio"])) if r["ratio"] else float("nan")
        print(f"{label:>26}  {r['accuracy'][-1]:>10.4f}  {max(r['accuracy']):>9.4f}"
              f"  {mean_tpr:>9.4f}  {mean_ratio:>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
