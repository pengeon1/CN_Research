"""
Master Script: Generates all Tables and Figures from experimental results.
Adheres strictly to the Hard Rule: ZERO fabricated numbers.
Every data point originates from results/*.json produced by experimental runs.
"""

from typing import Dict, Any, List
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def setup_matplotlib():
    """Sets publication-quality styling for figures."""
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    matplotlib.rcParams["axes.edgecolor"] = "#333333"
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["grid.color"] = "#E0E0E0"
    matplotlib.rcParams["grid.linestyle"] = "--"
    matplotlib.rcParams["grid.linewidth"] = 0.5


def generate_baseline_plots(results_dir: str = "results", figures_dir: str = "figures"):
    """Generates Figure 7 (RMSE comparisons) and Figure 13 (SLA violations)."""
    os.makedirs(figures_dir, exist_ok=True)
    bfile = os.path.join(results_dir, "baselines.json")
    if not os.path.exists(bfile):
        print(f"Skipping baseline plots: {bfile} not found.")
        return

    with open(bfile, "r") as f:
        data = json.load(f)

    # Aggregate across seeds
    slices = ["Facebook", "YouTube", "Google", "Instagram"]
    dlstm_means = []
    dlstm_stds = []
    cent_means = []

    for s in slices:
        s_rmses = [seed_data["DLSTM"][s]["mean_rmse"] for seed_data in data.values()]
        dlstm_means.append(np.mean(s_rmses))
        dlstm_stds.append(np.std(s_rmses))
        c_rmses = [seed_data["Centralized"][s]["rmse"] for seed_data in data.values()]
        cent_means.append(np.mean(c_rmses))

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    x = np.arange(len(slices))
    width = 0.35

    ax.bar(x - width / 2, dlstm_means, width, yerr=dlstm_stds, label="DLSTM (Decentralized)", color="#5C6BC0", capsize=3)
    ax.bar(x + width / 2, cent_means, width, label="Centralized LSTM", color="#26A69A")

    ax.set_ylabel("Average Test RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(slices)
    ax.legend(frameon=True)
    ax.grid(True, axis="y")
    ax.set_title("Multi-Slice Forecasting Baseline Comparison (Mean $\pm$ Std)")

    out_fig = os.path.join(figures_dir, "fig_7_rmse_baselines.pdf")
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"Saved: {out_fig}")


def generate_fl_plots(results_dir: str = "results", figures_dir: str = "figures"):
    """Generates Figure 10 (Omega participation sweep) and Figure 12 (mu proximal sweep)."""
    flfile = os.path.join(results_dir, "fl_experiments.json")
    if not os.path.exists(flfile):
        print(f"Skipping FL plots: {flfile} not found.")
        return

    with open(flfile, "r") as f:
        data = json.load(f)

    setup_matplotlib()

    # 1. Mu sweep plot (Figure 12)
    mu_keys = ["mu_0.001", "mu_0.01", "mu_0.1", "mu_0.5", "mu_1.0"]
    mu_vals = [0.001, 0.01, 0.1, 0.5, 1.0]
    mu_means = []
    mu_stds = []

    for k in mu_keys:
        vals = [seed_data["FPLSTM_mu_sweep"][k]["test_rmse"] for seed_data in data.values()]
        mu_means.append(np.mean(vals))
        mu_stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    ax.errorbar(range(len(mu_vals)), mu_means, yerr=mu_stds, marker="o", color="#E53935", linewidth=1.5, capsize=4)
    ax.set_xticks(range(len(mu_vals)))
    ax.set_xticklabels([str(m) for m in mu_vals])
    ax.set_xlabel("Proximal Hyperparameter $\mu$")
    ax.set_ylabel("Test RMSE")
    ax.set_title("FPLSTM Sensitivity to Proximal Parameter $\mu$ (Figure 12 Re-creation)")
    ax.grid(True)
    out_mu = os.path.join(figures_dir, "fig_12_mu_sweep.pdf")
    plt.tight_layout()
    plt.savefig(out_mu)
    plt.close()
    print(f"Saved: {out_mu}")

    # 2. Omega sweep plot (Figure 10)
    omega_keys = ["omega_0.1", "omega_0.3", "omega_0.5", "omega_0.7", "omega_1.0"]
    omega_vals = [0.1, 0.3, 0.5, 0.7, 1.0]
    omega_means = []
    omega_stds = []
    comm_costs = []

    for k in omega_keys:
        vals = [seed_data["FPLSTM_omega_sweep"][k]["test_rmse"] for seed_data in data.values()]
        omega_means.append(np.mean(vals))
        omega_stds.append(np.std(vals))
        # Comm cost is invariant across seeds
        first_seed = next(iter(data.values()))
        comm_costs.append(first_seed["FPLSTM_omega_sweep"][k]["communication_cost_mb"])

    fig, ax1 = plt.subplots(figsize=(6, 3.5), dpi=300)
    color = "#1E88E5"
    ax1.set_xlabel("Local Agent Participation Rate $\Omega$")
    ax1.set_ylabel("Test RMSE", color=color)
    ax1.errorbar(omega_vals, omega_means, yerr=omega_stds, marker="s", color=color, capsize=4, label="RMSE")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color2 = "#FB8C00"
    ax2.set_ylabel("Communication Cost (MB)", color=color2)
    ax2.plot(omega_vals, comm_costs, marker="^", color=color2, linestyle="--", label="Comm Cost")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.grid(True)
    plt.title("Effect of Client Participation Rate $\Omega$ (Figure 10)")
    out_om = os.path.join(figures_dir, "fig_10_omega_sweep.pdf")
    plt.tight_layout()
    plt.savefig(out_om)
    plt.close()
    print(f"Saved: {out_om}")


def generate_all():
    """Generates all figures and tables."""
    generate_baseline_plots()
    generate_fl_plots()


if __name__ == "__main__":
    generate_all()

