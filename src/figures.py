"""
Figure Generator
----------------
All figures for the manuscript companion repo.
Run after experiment.py: python src/figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# --- Design tokens ---
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#21262d"
TEXT     = "#c9d1d9"
MUTED    = "#6e7681"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f78166"
YELLOW   = "#e3b341"

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


# ---------------------------------------------------------------------------
# Fig 1: Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix():
    ms = json.load(open(RESULTS_DIR / "baseline.json"))["manuscript"]

    cm = np.array([
        [ms["true_negatives"],  ms["false_positives"]],
        [ms["false_negatives"], ms["true_positives"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(DARK_BG)

    im = ax.imshow(cm, cmap="Blues")
    ax.set_facecolor(PANEL_BG)

    labels = [["TN\n(Normal → Normal)", "FP\n(Normal → Attack)"],
              ["FN\n(Attack → Normal)", "TP\n(Attack → Attack)"]]
    colors = [["white", "white"], ["white", "white"]]

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}\n{labels[i][j]}",
                    ha="center", va="center", fontsize=10,
                    color="black" if val > cm.max() * 0.5 else TEXT,
                    fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Normal", "Predicted: Attack"],
                       fontsize=9, color=TEXT)
    ax.set_yticklabels(["Actual: Normal", "Actual: Attack"],
                       fontsize=9, color=TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    ax.set_title(f"Confusion Matrix — NSL-KDD Baseline\nAccuracy: 92.51%",
                 fontsize=11, fontweight="bold", pad=10, color=TEXT)
    fig.patch.set_facecolor(DARK_BG)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] confusion_matrix.png")


# ---------------------------------------------------------------------------
# Fig 2: Classification report bar chart
# ---------------------------------------------------------------------------
def plot_classification_report():
    ms = json.load(open(RESULTS_DIR / "baseline.json"))["manuscript"]

    metrics = ["Precision", "Recall", "F1-Score"]
    normal  = [ms["precision_normal"],  ms["recall_normal"],  ms["f1_normal"]]
    attack  = [ms["precision_attack"],  ms["recall_attack"],  ms["f1_attack"]]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)

    b1 = ax.bar(x - width/2, normal, width, label="Class 0 (Normal)",
                color=GREEN, alpha=0.85, edgecolor=DARK_BG)
    b2 = ax.bar(x + width/2, attack, width, label="Class 1 (Attack)",
                color=RED,   alpha=0.85, edgecolor=DARK_BG)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=10, color=TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.12)
    style(ax, "Classification Report — Baseline Isolation Forest (NSL-KDD)", "", "Score")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    # Annotate macro avg
    ax.axhline(ms["macro_avg_f1"], color=MUTED, linewidth=1, linestyle="--",
               label=f"Macro Avg F1 = {ms['macro_avg_f1']:.2f}")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "classification_report.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] classification_report.png")


# ---------------------------------------------------------------------------
# Fig 3: Feature sensitivity — top 10 horizontal bar
# ---------------------------------------------------------------------------
def plot_feature_sensitivity():
    ms_data = json.load(open(RESULTS_DIR / "sensitivity_manuscript.json"))
    computed = json.load(open(RESULTS_DIR / "feature_sensitivity.json"))

    # Manuscript top 10
    features = list(ms_data.keys())
    ms_vals  = list(ms_data.values())

    # Computed top 10 (align by feature name where possible)
    comp_dict = {d["feature"]: d["abs_correlation"] for d in computed}
    comp_vals = [comp_dict.get(f, 0.0) for f in features]

    y = np.arange(len(features))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)

    b1 = ax.barh(y + height/2, ms_vals,   height, label="Manuscript (real NSL-KDD)",
                 color=ACCENT, alpha=0.85, edgecolor=DARK_BG)
    b2 = ax.barh(y - height/2, comp_vals, height, label="Synthetic replication",
                 color=MUTED,  alpha=0.70, edgecolor=DARK_BG)

    for bar in b1:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=8, color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_", " ") for f in features], fontsize=9)
    ax.set_xlim(0, 1.05)
    style(ax, "Feature Sensitivity Matrix — Top 10 Leverage Points\n"
          "|Pearson r| between feature and Isolation Forest anomaly score",
          "|Pearson r|", "")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    # Goldilocks annotation
    ax.axvline(0.80, color=YELLOW, linewidth=1.2, linestyle="--", alpha=0.7,
               label="High-leverage threshold (r=0.80)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_sensitivity.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] feature_sensitivity.png")


# ---------------------------------------------------------------------------
# Fig 4: Evasion findings — three experiments + comparison bar
# ---------------------------------------------------------------------------
def plot_evasion_comparison():
    data = json.load(open(RESULTS_DIR / "evasion_findings.json"))
    exps = data["experiments"]

    labels = ["A: Single-Feature\n(dst_host_srv_count ×0.2)",
              "B: Multi-Feature\n(3 features ×0.5)",
              "C: Extreme Stealth\n(5 features ×0.05)"]
    rates  = [exps["experiment_a"]["evasion_rate"] * 100,
              exps["experiment_b"]["evasion_rate"] * 100,
              exps["experiment_c"]["evasion_rate"] * 100]
    evaded = [exps["experiment_a"]["evaded"],
              exps["experiment_b"]["evaded"],
              exps["experiment_c"]["evaded"]]
    colors = [ACCENT, YELLOW, RED]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Left: evasion rate
    ax = axes[0]
    bars = ax.bar(labels, rates, color=colors, edgecolor=DARK_BG, alpha=0.88)
    for bar, rate, ev in zip(bars, rates, evaded):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.2f}%\n({ev:,} samples)", ha="center", va="bottom",
                fontsize=9, color=TEXT, fontweight="bold")
    style(ax, "Evasion Success Rate by Experiment\n(% of 49,870 True Positives)",
          "", "Evasion Rate (%)")
    ax.set_ylim(0, max(rates) * 1.4)

    # Right: absolute counts
    ax = axes[1]
    bars2 = ax.bar(labels, evaded, color=colors, edgecolor=DARK_BG, alpha=0.88)
    for bar, ev in zip(bars2, evaded):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{ev:,}", ha="center", va="bottom",
                fontsize=10, color=TEXT, fontweight="bold")
    style(ax, "Absolute Evasion Count\n(samples evading detection)",
          "", "Evaded Samples")
    ax.set_ylim(0, max(evaded) * 1.35)

    # Arrow annotation showing paradox direction
    axes[1].annotate("", xy=(2, evaded[2] + 15), xytext=(0, evaded[0] - 15),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    axes[1].text(1.05, (evaded[0] + evaded[2]) / 2,
                 "Extremity\nParadox", ha="center", va="center",
                 fontsize=8, color=MUTED, style="italic")

    plt.suptitle("Adversarial Evasion Findings — Three Perturbation Scenarios",
                 fontsize=12, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "evasion_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] evasion_comparison.png")


# ---------------------------------------------------------------------------
# Fig 5: Security Decay curves — all three evasion experiments
# ---------------------------------------------------------------------------
def plot_security_decay():
    data = json.load(open(RESULTS_DIR / "evasion_findings.json"))

    curve_a = data["curve_a"]
    curve_b = data["curve_b"]
    curve_c = data["curve_c"]

    ra = [c[0] for c in curve_a]; va = [c[1]*100 for c in curve_a]
    rb = [c[0] for c in curve_b]; vb = [c[1]*100 for c in curve_b]
    rc = [c[0] for c in curve_c]; vc = [c[1]*100 for c in curve_c]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(ra, va, color=ACCENT,  marker="o", linewidth=2.5, markersize=5,
            label="Exp A: Single-feature (linear growth)")
    ax.plot(rb, vb, color=YELLOW,  marker="s", linewidth=2.5, markersize=5,
            label="Exp B: Multi-feature coordinated (non-linear)")
    ax.plot(rc, vc, color=RED,     marker="^", linewidth=2.5, markersize=5,
            label="Exp C: Extreme stealth — Extremity Paradox (inverted)")

    ax.fill_between(ra, va, alpha=0.07, color=ACCENT)
    ax.fill_between(rb, vb, alpha=0.07, color=YELLOW)
    ax.fill_between(rc, vc, alpha=0.07, color=RED)

    # Mark the Extremity Trap
    ax.annotate("Extremity Trap:\nnear-zero values\nbecome MORE detectable",
                xy=(95, vc[-1]), xytext=(70, 0.55),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    # Mark Exp B peak
    peak_b_x = rb[vb.index(max(vb))]
    peak_b_y = max(vb)
    ax.annotate(f"Conflicting feature\ncombinations detected\nas novel anomaly",
                xy=(peak_b_x, peak_b_y), xytext=(peak_b_x - 25, peak_b_y + 0.12),
                fontsize=8, color=YELLOW,
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2))

    style(ax,
          "Security Decay Curves — Evasion Rate vs. Perturbation Intensity\n"
          "Isolation Forest UEBA on NSL-KDD (49,870 True Positive samples)",
          "Feature Reduction (%)", "Evasion Rate (%)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT,
              loc="upper left")
    ax.set_xlim(-2, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "security_decay.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] security_decay.png")


# ---------------------------------------------------------------------------
# Fig 6: Goldilocks Zone
# ---------------------------------------------------------------------------
def plot_goldilocks():
    data   = json.load(open(RESULTS_DIR / "evasion_findings.json"))
    curve  = data["goldilocks"]

    reductions = [c[0] for c in curve]
    rates      = [c[1] * 100 for c in curve]
    peak_r     = reductions[rates.index(max(rates))]
    peak_v     = max(rates)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(reductions, rates, color=YELLOW, marker="o", linewidth=2.5,
            markersize=5, label="Evasion rate (5 features)")
    ax.fill_between(reductions, rates, alpha=0.10, color=YELLOW)

    # Shade the Goldilocks window
    ax.axvspan(16, 28, alpha=0.12, color=GREEN,
               label="Goldilocks Zone (≈16%–28% reduction)")
    ax.axvline(peak_r, color=GREEN, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(peak_r + 0.5, peak_v * 0.95,
            f"Peak evasion\n@{peak_r}% reduction\n({peak_v:.2f}%)",
            fontsize=8, color=GREEN)

    # Shade the danger zones
    ax.axvspan(0, 8, alpha=0.08, color=MUTED, label="Under-perturbation zone")
    ax.axvspan(32, 40, alpha=0.08, color=RED, label="Over-perturbation zone")

    style(ax,
          "Goldilocks Zone Analysis — Optimal Evasion Window\n"
          "Fine-grained sweep: 5 high-leverage features, 0–40% reduction",
          "Feature Reduction (%)", "Evasion Rate (%)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "goldilocks_zone.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] goldilocks_zone.png")


# ---------------------------------------------------------------------------
# Fig 7: Contamination sweep
# ---------------------------------------------------------------------------
def plot_contamination_sweep():
    data = json.load(open(RESULTS_DIR / "contamination_sweep.json"))

    contam  = [d["contamination"] for d in data]
    acc     = [d["accuracy"]        for d in data]
    f1_atk  = [d["f1_attack"]       for d in data]
    recall  = [d["recall_attack"]   for d in data]
    fp      = [d["false_positives"] for d in data]
    fn      = [d["false_negatives"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    ax.plot(contam, acc,    color=ACCENT,  marker="o", linewidth=2, label="Accuracy")
    ax.plot(contam, f1_atk, color=GREEN,   marker="s", linewidth=2, label="F1 (Attack)")
    ax.plot(contam, recall, color=YELLOW,  marker="^", linewidth=2, label="Recall (Attack)")
    ax.axvline(0.01, color=MUTED, linewidth=1.2, linestyle="--",
               label="Manuscript config (0.01)")
    style(ax, "Detection Quality vs. Contamination", "Contamination", "Score")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    ax = axes[1]
    ax.plot(contam, fp, color=YELLOW, marker="o", linewidth=2, label="False Positives")
    ax.plot(contam, fn, color=RED,    marker="s", linewidth=2, label="False Negatives")
    ax.axvline(0.01, color=MUTED, linewidth=1.2, linestyle="--",
               label="Manuscript config (0.01)")
    style(ax, "FP / FN Trade-off vs. Contamination", "Contamination", "Count")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    plt.suptitle("Contamination Hyperparameter Sensitivity",
                 fontsize=12, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "contamination_sweep.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] contamination_sweep.png")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_classification_report()
    plot_feature_sensitivity()
    plot_evasion_comparison()
    plot_security_decay()
    plot_goldilocks()
    plot_contamination_sweep()
    print("\n[Done] All figures saved to /figures")
