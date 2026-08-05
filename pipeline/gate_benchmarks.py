"""Distilled checker experiment metrics for the production dashboard.

These are offline evaluation results, not live request latency. Keeping this small
summary in production avoids loading training datasets or heavyweight ML stacks.
"""

from __future__ import annotations

from typing import Any


GATE_BENCHMARKS: list[dict[str, Any]] = [
    {"gate": "Gate 1", "method": "DistilBERT error classifier", "trained": True, "accuracy": 0.9280, "precision": 0.9444, "recall": 0.9280, "f1": 0.9284, "auroc": None, "experiment_seconds": 21.126},
    {"gate": "Gate 1", "method": "BERT error classifier", "trained": True, "accuracy": 0.9253, "precision": 0.9314, "recall": 0.9253, "f1": 0.9249, "auroc": None, "experiment_seconds": 36.443},
    {"gate": "Gate 1", "method": "RoBERTa error classifier", "trained": True, "accuracy": 0.9653, "precision": 0.9667, "recall": 0.9653, "f1": 0.9655, "auroc": None, "experiment_seconds": 37.460},
    {"gate": "Gate 2", "method": "CLIP cosine threshold", "trained": False, "accuracy": 0.9628, "precision": 0.9474, "recall": 0.9800, "f1": 0.9634, "auroc": 0.9922, "experiment_seconds": 0.015},
    {"gate": "Gate 2", "method": "CLIP linear probe", "trained": True, "accuracy": 0.8756, "precision": 0.8301, "recall": 0.9444, "f1": 0.8836, "auroc": 0.9555, "experiment_seconds": 1.085},
    {"gate": "Gate 2", "method": "CLIP MLP probe", "trained": True, "accuracy": 0.9517, "precision": 0.9462, "recall": 0.9578, "f1": 0.9520, "auroc": 0.9885, "experiment_seconds": 1.311},
]


def benchmark_rows(gate: str | None = None) -> list[dict[str, Any]]:
    rows = GATE_BENCHMARKS if gate is None else [row for row in GATE_BENCHMARKS if row["gate"] == gate]
    return [
        {
            "Gate": row["gate"],
            "Method": row["method"],
            "Trained": "Yes" if row["trained"] else "No",
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "AUROC": row["auroc"],
            "Experiment time (s)": row["experiment_seconds"],
        }
        for row in rows
    ]
