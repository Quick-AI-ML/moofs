"""Pareto-front visualization helpers (matplotlib).

Every function accepts a ``Result``, a list of ``Solution`` or a 2-D
objective array, returns the matplotlib ``Axes`` for further styling, and
never calls ``plt.show()`` — the caller stays in control.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .metrics import _as_F, merge_reference_front

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_COLORS = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5",
           "#993556", "#854F0B", "#3B6D11", "#5F5E5A"]


def plot_pareto_front(result, label=None, ax=None, color=None, marker="o",
                      annotate=False):
    """Scatter plot of one Pareto front (error % vs. subset size).

    Parameters
    ----------
    result : Result, list of Solution, or ndarray
    label : str, optional
        Legend label (defaults to the algorithm name if available).
    ax : matplotlib.axes.Axes, optional
    annotate : bool, default=False
        Write the subset size next to each point.
    """
    F = _as_F(result)
    if label is None:
        label = getattr(result, "algorithm", None)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(F[:, 1])
    F = F[order]
    ax.scatter(F[:, 1], F[:, 0], label=label, color=color or _COLORS[0],
               marker=marker, s=45, zorder=3)
    ax.plot(F[:, 1], F[:, 0], color=color or _COLORS[0], alpha=0.35,
            linewidth=1, zorder=2)
    if annotate:
        for f1, f2 in F[:, [0, 1]]:
            ax.annotate(f"{int(f2)}", (f2, f1), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
    _style(ax)
    if label:
        ax.legend()
    return ax


def plot_fronts(results, ax=None, reference=False):
    """Overlay several Pareto fronts for comparison.

    Parameters
    ----------
    results : dict
        Mapping ``name -> Result`` (or front).
    reference : bool, default=False
        Also draw the merged reference front as a dashed line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))
    for i, (name, res) in enumerate(results.items()):
        F = _as_F(res)
        order = np.argsort(F[:, 1])
        F = F[order]
        c = _COLORS[i % len(_COLORS)]
        m = _MARKERS[i % len(_MARKERS)]
        ax.scatter(F[:, 1], F[:, 0], label=name, color=c, marker=m, s=45,
                   alpha=0.9, zorder=3)
        ax.plot(F[:, 1], F[:, 0], color=c, alpha=0.3, linewidth=1, zorder=2)
    if reference:
        R = merge_reference_front(*results.values())
        R = R[np.argsort(R[:, 1])]
        ax.plot(R[:, 1], R[:, 0], "k--", linewidth=1.2, alpha=0.7,
                label="Reference front", zorder=1)
    _style(ax)
    ax.legend()
    return ax


def plot_selector(selector, ax=None):
    """Plot a fitted ``MOFSSelector`` front and highlight the chosen subset."""
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(selector, "support_")
    ax = plot_pareto_front(selector.pareto_front_,
                           label=selector.algorithm, ax=ax)
    err, size = selector.selected_objectives_
    ax.scatter([size], [err], s=180, facecolors="none",
               edgecolors="#D85A30", linewidths=2, zorder=4,
               label=f"selected ({selector.strategy})")
    ax.legend()
    return ax


def _style(ax):
    ax.set_xlabel("F2: number of selected features")
    ax.set_ylabel("F1: classification error (%)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
