"""Regenerate the compact, large-type comparison panels used in Figure 7."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT = Path(__file__).resolve().parent / "figures"

NAVY = "#1f4e79"
ORANGE = "#d95f02"
TEAL = "#2a9d8f"
RED = "#b22222"
GRID = "#d9dde3"
TEXT = "#1f2933"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.edgecolor": "#aab4be",
            "axes.linewidth": 1.1,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def gate1() -> None:
    names = ["DistilBERT", "BERT", "RoBERTa"]
    accuracy = [0.9280, 0.9253, 0.9653]
    f1 = [0.9284, 0.9249, 0.9655]
    runtime = [21.126, 36.443, 37.460]
    x = range(3)
    width = 0.30

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    accuracy_bars = ax.bar([i - width / 2 for i in x], accuracy, width, color=NAVY)
    f1_bars = ax.bar([i + width / 2 for i in x], f1, width, color=ORANGE)
    ax.set_ylim(0.88, 0.985)
    ax.set_ylabel("Score")
    ax.set_xticks(list(x), names)

    for index, bar in enumerate(f1_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            f1[index] + 0.002,
            f"{f1[index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold" if index == 2 else "normal",
            color=RED if index == 2 else TEXT,
        )

    runtime_axis = ax.twinx()
    runtime_axis.plot(list(x), runtime, color=TEAL, marker="o", markersize=7, linewidth=2.5)
    runtime_axis.set_ylabel("Runtime (s)")
    runtime_axis.set_ylim(0, 48)

    legend = [
        accuracy_bars[0],
        f1_bars[0],
        Line2D([0], [0], color=TEAL, marker="o", linewidth=2.5),
    ]
    ax.legend(legend, ["Accuracy", "Macro-F1", "Runtime"], loc="upper left", ncol=1)
    save(fig, "checker1_comparison_publication.png")


def gate2() -> None:
    names = ["Cosine", "Linear", "MLP"]
    accuracy = [0.9628, 0.8756, 0.9517]
    f1 = [0.9634, 0.8836, 0.9520]
    auroc = [0.9922, 0.9555, 0.9885]
    x = range(3)
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    accuracy_bars = ax.bar([i - width for i in x], accuracy, width, color=NAVY)
    f1_bars = ax.bar(list(x), f1, width, color=ORANGE)
    auroc_bars = ax.bar([i + width for i in x], auroc, width, color=TEAL)
    ax.set_ylim(0.84, 1.01)
    ax.set_ylabel("Score")
    ax.set_xticks(list(x), names)

    for index, bar in enumerate(f1_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            f1[index] + 0.003,
            f"{f1[index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold" if index == 0 else "normal",
            color=RED if index == 0 else TEXT,
        )

    ax.legend(
        [accuracy_bars[0], f1_bars[0], auroc_bars[0]],
        ["Accuracy", "F1", "AUROC"],
        loc="upper left",
        ncol=1,
    )
    save(fig, "checker2_comparison_publication.png")


if __name__ == "__main__":
    configure()
    gate1()
    gate2()
