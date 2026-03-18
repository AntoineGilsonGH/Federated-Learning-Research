"""
Runs the bound-challenge experiment for multiple attacks sequentially.
Each attack gets its own output directory.
Imports and calls main() from main_bound_challenge_v5 after patching
the ATTACK and OUTPUT_DIR module-level variables.
"""

import importlib
import sys
import os

# ── Attack configurations ─────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "attack":      {"attack_name": "Optimal_ALittleIsEnough",        "attack_parameters": {}},
        "output_dir":  "results/bound_challenge_test_OptiALIE",
    },
    {
        "attack":      {"attack_name": "Optimal_InnerProductManipulation", "attack_parameters": {}},
        "output_dir":  "results/bound_challenge_test_OptiIPM",
    },
    {
        "attack":      {"attack_name": "SignFlipping",                    "attack_parameters": {"scale": 5}},
        "output_dir":  "results/bound_challenge_test_SignFlip",
    },
    {
        "attack":      {"attack_name": "Inf",                             "attack_parameters": {}},
        "output_dir":  "results/bound_challenge_test_Inf",
    },
]

# ── Import the experiment module once ────────────────────────────────────────
# We add the repo root to sys.path so the import works regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main_bound_challenge as exp   # imports the module without running main()

# ── Run each experiment ───────────────────────────────────────────────────────
for i, cfg in enumerate(EXPERIMENTS):
    attack_name = cfg["attack"]["attack_name"]
    print("\n" + "=" * 70)
    print(f"EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {attack_name}")
    print("=" * 70)

    # Patch module-level variables before calling main()
    exp.ATTACK      = cfg["attack"]
    exp.OUTPUT_DIR  = cfg["output_dir"]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    try:
        exp.main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Experiment {attack_name} failed: {e}")
        traceback.print_exc()
        print("Continuing to next experiment...\n")

print("\n" + "=" * 70)
print("All experiments finished.")
print("=" * 70)
