"""
Performance curve visualizations (ROC, PR, Learning curves).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# NumPy 2.x compatibility: trapz was renamed to trapezoid
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
if _trapz is None:
    def _trapz(y, x):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ._style import get_colors, setup_figure, style_axis, SEMANTIC_COLORS
from tuiml.evaluation.metrics.classification import _binary_roc_curve, auc

def _roc_curve_binary(y_true_bin: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute (fpr, tpr, auc) for a single binary-labelled vector y_true_bin in {0,1}."""
    fpr, tpr, _ = _binary_roc_curve(y_true_bin, y_score, pos_label=1)
    return fpr, tpr, auc(fpr, tpr)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_auc: bool = True,
    label: str = None,
    show_grid: bool = False,
    classes: Optional[List] = None,
):
    """Plot ROC curve(s) — binary or multiclass (one-vs-rest).

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True class labels (binary or multiclass).
    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        For binary: probabilities of the positive class (1-D).
        For multiclass: per-class probabilities (2-D); one OvR curve is
        drawn for each class and a macro-average curve is overlaid.
    classes : list, optional
        Class labels in the column order of `y_score`. Used to label the
        per-class curves. Defaults to ``np.unique(y_true)``.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    unique_labels = np.unique(y_true)

    if y_score.ndim == 1 and len(unique_labels) > 2:
        raise ValueError(
            "Multiclass ROC curves require class probabilities/scores with "
            "shape (n_samples, n_classes). A 1-D vector, such as predicted "
            "class labels, cannot produce meaningful ROC curves."
        )

    # ── Multiclass path (one-vs-rest) ────────────────────────────────
    if y_score.ndim == 2 and y_score.shape[1] > 2:
        if classes is None:
            classes = list(unique_labels)
        n_classes = y_score.shape[1]
        if len(classes) != n_classes:
            raise ValueError(
                "Length of classes must match the number of y_score columns"
            )
        colors = get_colors(n_classes)

        fig, ax = setup_figure(figsize=figsize)
        # Common FPR grid for macro-averaging
        all_fpr = np.linspace(0.0, 1.0, 200)
        mean_tpr = np.zeros_like(all_fpr)
        per_class_auc = []

        for k in range(n_classes):
            y_true_bin = (y_true == classes[k]).astype(int)
            fpr_k, tpr_k, auc_k = _roc_curve_binary(y_true_bin, y_score[:, k])
            per_class_auc.append(auc_k)
            ax.plot(fpr_k, tpr_k, lw=2.0, color=colors[k],
                    label=f'{classes[k]} (AUC = {auc_k:.3f})')
            mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)

        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, lw=3.0, linestyle=':',
                color=SEMANTIC_COLORS.get('primary', 'k'),
                label=f'macro-avg (AUC = {macro_auc:.3f})')
        ax.plot([0, 1], [0, 1], '--', lw=1.5, color=SEMANTIC_COLORS['neutral'], label='Random')

        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
        style_axis(ax, title=f'{title} (one-vs-rest)',
                   xlabel='False Positive Rate', ylabel='True Positive Rate',
                   legend=True, legend_loc='lower right', grid=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
        plt.show()
        return all_fpr, mean_tpr, macro_auc

    # ── Binary path ──────────────────────────────────────────────────
    # If a 2-column proba matrix was passed, use the positive-class column.
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]

    # Normalise true labels to {0,1}
    uniq = np.unique(y_true)
    if not (set(uniq.tolist()) <= {0, 1}):
        pos_label = uniq[-1]
        y_true_bin = (y_true == pos_label).astype(int)
    else:
        y_true_bin = y_true.astype(int)

    fpr, tpr, auc_score = _roc_curve_binary(y_true_bin, y_score)

    colors = get_colors(2)
    fig, ax = setup_figure(figsize=figsize)
    if label is None:
        label = f'ROC (AUC = {auc_score:.3f})' if show_auc else 'ROC'
    elif show_auc:
        label = f'{label} (AUC = {auc_score:.3f})'

    ax.plot(fpr, tpr, lw=3.0, label=label, color=colors[0])
    ax.plot([0, 1], [0, 1], '--', lw=2.0, label='Random', color=SEMANTIC_COLORS['neutral'])
    ax.fill_between(fpr, tpr, alpha=0.2, color=colors[0])
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    style_axis(ax, title=title, xlabel='False Positive Rate', ylabel='True Positive Rate',
               legend=True, legend_loc='lower right', grid=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()
    return fpr, tpr, auc_score

def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_ap: bool = True,
    label: str = None,
    show_grid: bool = False,
):
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_score : ndarray
        Predicted probabilities for positive class.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show_ap : bool, default=True
        Show Average Precision in legend.
    label : str, optional
        Label for the curve.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Calculate PR curve
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]

    precision_list = [1.0]
    recall_list = [0.0]

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.array(precision_list)
    recall = np.array(recall_list)

    # Calculate Average Precision
    sorted_idx = np.argsort(recall)
    recall_sorted = recall[sorted_idx]
    precision_sorted = precision[sorted_idx]
    ap = _trapz(precision_sorted, recall_sorted)

    # Plot
    fig, ax = setup_figure(figsize=figsize)

    if label is None:
        label = f'PR (AP = {ap:.3f})' if show_ap else 'PR'
    elif show_ap:
        label = f'{label} (AP = {ap:.3f})'

    ax.plot(recall_sorted, precision_sorted, lw=3.0, label=label, color=colors[0])

    # Fill area under curve
    ax.fill_between(recall_sorted, precision_sorted, alpha=0.2, color=colors[0])

    # Baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color=SEMANTIC_COLORS['neutral'], linestyle='--', lw=2.0,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    style_axis(
        ax,
        title=title,
        xlabel='Recall',
        ylabel='Precision',
        legend=True,
        legend_loc='lower left',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

    return recall_sorted, precision_sorted, ap

def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = 'Learning Curve',
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
    metric_name: str = 'Score',
    show_std: bool = True,
    show_grid: bool = False,
):
    """
    Plot learning curve showing performance vs training set size.

    Parameters
    ----------
    train_sizes : ndarray
        Training set sizes.
    train_scores : ndarray of shape (n_sizes,) or (n_sizes, n_splits)
        Training scores.
    test_scores : ndarray of shape (n_sizes,) or (n_sizes, n_splits)
        Test/validation scores.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    metric_name : str
        Name of the metric.
    show_std : bool, default=True
        Show standard deviation bands.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    train_sizes = np.asarray(train_sizes)
    train_scores = np.asarray(train_scores)
    test_scores = np.asarray(test_scores)

    # Handle both 1D and 2D arrays
    if train_scores.ndim == 1:
        train_mean = train_scores
        train_std = np.zeros_like(train_mean)
        test_mean = test_scores
        test_std = np.zeros_like(test_mean)
    else:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

    fig, ax = setup_figure(figsize=figsize)

    # Training curve
    ax.plot(train_sizes, train_mean, 'o-', color=colors[0], lw=3.0,
            markersize=10, label='Training score')
    if show_std and train_scores.ndim > 1:
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=colors[0])

    # Validation curve
    ax.plot(train_sizes, test_mean, 's-', color=colors[1], lw=3.0,
            markersize=10, label='Cross-validation score')
    if show_std and test_scores.ndim > 1:
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                        alpha=0.2, color=colors[1])

    style_axis(
        ax,
        title=title,
        xlabel='Training Set Size',
        ylabel=metric_name,
        legend=True,
        legend_loc='best',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()
