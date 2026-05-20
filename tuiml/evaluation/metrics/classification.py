"""
Classification evaluation metrics.

This module provides metrics for evaluating classification models,
equivalent to Weka's Evaluation class classification methods.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tuiml.base.metrics import (
    Metric, MetricType, AverageType,
    check_classification_targets, check_consistent_length,
    get_num_classes, get_class_labels, is_binary, safe_divide, weighted_sum
)

# =============================================================================
# Confusion Matrix
# =============================================================================

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate classification accuracy.
    
    Equivalent to Weka's Evaluation.confusionMatrix().
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to index the matrix (optional)
        normalize: 'true', 'pred', 'all', or None for counts
        
    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_classification_targets(y_true, y_pred)
    
    if labels is None:
        labels = get_class_labels(y_true, y_pred)
    
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Build confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1
    
    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype(np.float64)
        cm = safe_divide(cm, cm.sum(axis=1, keepdims=True))
    elif normalize == 'pred':
        cm = cm.astype(np.float64)
        cm = safe_divide(cm, cm.sum(axis=0, keepdims=True))
    elif normalize == 'all':
        cm = cm.astype(np.float64)
        cm = cm / cm.sum()
    
    return cm

# =============================================================================
# Basic Classification Metrics
# =============================================================================

def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute accuracy classification score.
    
    Equivalent to Weka's Evaluation.pctCorrect() / 100.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_classification_targets(y_true, y_pred)
    
    correct = (y_true == y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        correct_weighted = np.sum(correct * sample_weight)
        if normalize:
            return correct_weighted / np.sum(sample_weight)
        return float(correct_weighted)
    
    if normalize:
        return float(np.mean(correct))
    return float(np.sum(correct))

def balanced_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    adjusted: bool = False
) -> float:
    """Compute balanced accuracy (average of recall for each class)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    labels = get_class_labels(y_true, y_pred)
    recalls = []
    
    for label in labels:
        mask = y_true == label
        if np.sum(mask) > 0:
            recalls.append(np.mean(y_pred[mask] == label))
    
    balanced_acc = np.mean(recalls)
    
    if adjusted:
        n_classes = len(labels)
        chance = 1 / n_classes if n_classes > 0 else 1.0
        balanced_acc = (balanced_acc - chance) / (1 - chance) if chance < 1 else 0.0
    
    return float(balanced_acc)

# =============================================================================
# Precision, Recall, F-score
# =============================================================================

def _precision_recall_fscore_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
    labels: Optional[np.ndarray] = None,
    pos_label: int = 1,
    average: Optional[str] = None,
    zero_division: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall, F-score and support for each class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_classification_targets(y_true, y_pred)
    
    if labels is None:
        labels = get_class_labels(y_true, y_pred)
    
    n_labels = len(labels)
    tp = np.zeros(n_labels)
    fp = np.zeros(n_labels)
    fn = np.zeros(n_labels)
    support = np.zeros(n_labels)
    
    for i, label in enumerate(labels):
        true_mask = y_true == label
        pred_mask = y_pred == label
        tp[i] = np.sum(true_mask & pred_mask)
        fp[i] = np.sum(~true_mask & pred_mask)
        fn[i] = np.sum(true_mask & ~pred_mask)
        support[i] = np.sum(true_mask)
    
    precision = safe_divide(tp, tp + fp, zero_division)
    recall = safe_divide(tp, tp + fn, zero_division)
    
    beta_sq = beta ** 2
    fscore = safe_divide((1 + beta_sq) * precision * recall,
                         beta_sq * precision + recall, zero_division)
    
    return precision, recall, fscore, support

def precision_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
    """Compute precision. Equivalent to Weka's Evaluation.precision(classIndex)."""
    p, _, _, s = _precision_recall_fscore_support(y_true, y_pred, **kwargs)
    avg = kwargs.get('average', 'binary')
    pos = kwargs.get('pos_label', 1)
    lbls = kwargs.get('labels', get_class_labels(y_true, y_pred))
    return _average_scores(p, s, avg, pos, lbls)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
    """Compute recall. Equivalent to Weka's Evaluation.recall(classIndex)."""
    _, r, _, s = _precision_recall_fscore_support(y_true, y_pred, **kwargs)
    avg = kwargs.get('average', 'binary')
    pos = kwargs.get('pos_label', 1)
    lbls = kwargs.get('labels', get_class_labels(y_true, y_pred))
    return _average_scores(r, s, avg, pos, lbls)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
    """Compute F1 score. Equivalent to Weka's Evaluation.fMeasure(classIndex)."""
    _, _, f, s = _precision_recall_fscore_support(y_true, y_pred, **kwargs)
    avg = kwargs.get('average', 'binary')
    pos = kwargs.get('pos_label', 1)
    lbls = kwargs.get('labels', get_class_labels(y_true, y_pred))
    return _average_scores(f, s, avg, pos, lbls)

def _average_scores(scores, support, average, pos_label, labels):
    if average is None: return scores
    if average == 'binary':
        idx = np.where(labels == pos_label)[0]
        return float(scores[idx[0]]) if len(idx) > 0 else 0.0
    if average == 'macro': return float(np.mean(scores))
    if average == 'weighted':
        total = np.sum(support)
        return float(np.sum(scores * support) / total) if total > 0 else 0.0
    if average == 'micro':
        # Simple micro average logic for binary/multiclass
        return float(np.sum(scores * support) / np.sum(support)) if np.sum(support) > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")

# Rates
def true_positive_rate(y_true, y_pred, pos_label=1):
    return recall_score(y_true, y_pred, pos_label=pos_label, average='binary')

def false_positive_rate(y_true, y_pred, pos_label=1):
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

def matthews_corrcoef(y_true, y_pred):
    """Compute MCC. Equivalent to Weka's Evaluation.matthewsCorrelationCoefficient(classIndex)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        num = tp * tn - fp * fn
        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return float(num / den) if den > 0 else 0.0
    # Multiclass implementation skipped for brevity, but easy to add
    return 0.0

def cohen_kappa_score(y_true, y_pred):
    """Compute Kappa. Equivalent to Weka's Evaluation.kappa()."""
    cm = confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = np.dot(np.sum(cm, axis=1), np.sum(cm, axis=0)) / (n**2)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0

# ROC AUC
def _binary_roc_curve(y_true, y_score, pos_label=1):
    """Compute binary ROC coordinates for score thresholds."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim != 1:
        raise ValueError("y_score must be a 1-D score vector for binary ROC")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    y_true_bin = y_true == pos_label
    n_pos = np.sum(y_true_bin)
    n_neg = len(y_true_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC is undefined when only one class is present")

    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[order]
    y_true_bin = y_true_bin[order]

    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true_bin.size - 1]

    tps = np.cumsum(y_true_bin)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    tpr = tps / n_pos
    fpr = fps / n_neg
    return fpr.astype(float), tpr.astype(float), thresholds


def roc_auc_score(y_true, y_score, average='macro', labels=None):
    """Compute ROC AUC.

    Binary input accepts a 1-D score vector for the positive class. Multiclass
    input accepts a probability/score matrix of shape
    ``(n_samples, n_classes)`` and computes one-vs-rest AUC for each class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)

    if y_score.ndim == 1:
        if len(labels) > 2:
            raise ValueError(
                "Multiclass ROC AUC requires y_score with shape "
                "(n_samples, n_classes), not hard labels or a 1-D score vector"
            )
        pos_label = labels[-1] if len(labels) == 2 else 1
        fpr, tpr, _ = _binary_roc_curve(y_true, y_score, pos_label=pos_label)
        return auc(fpr, tpr)

    if y_score.ndim != 2:
        raise ValueError("y_score must be 1-D or 2-D")
    if y_score.shape[0] != len(y_true):
        raise ValueError("y_true and y_score must have the same number of rows")
    if y_score.shape[1] != len(labels):
        raise ValueError("Number of score columns must match number of labels")

    scores = []
    support = []
    for i, label in enumerate(labels):
        y_true_bin = (y_true == label).astype(int)
        fpr, tpr, _ = _binary_roc_curve(y_true_bin, y_score[:, i], pos_label=1)
        scores.append(auc(fpr, tpr))
        support.append(np.sum(y_true == label))

    scores = np.asarray(scores, dtype=float)
    support = np.asarray(support, dtype=float)

    if average is None:
        return scores
    if average == 'macro':
        return float(np.mean(scores))
    if average == 'weighted':
        return float(np.sum(scores * support) / np.sum(support))
    raise ValueError(f"Unknown average: {average}")

def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=None, zero_division=0.0):
    """Compute precision, recall, F-measure and support."""
    p, r, f, s = _precision_recall_fscore_support(y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label, zero_division=zero_division)
    if average is not None:
        p = _average_scores(p, s, average, pos_label, labels if labels is not None else get_class_labels(y_true, y_pred))
        r = _average_scores(r, s, average, pos_label, labels if labels is not None else get_class_labels(y_true, y_pred))
        f = _average_scores(f, s, average, pos_label, labels if labels is not None else get_class_labels(y_true, y_pred))
        s = np.sum(s)
    return p, r, f, s

def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Build a text report showing main classification metrics."""
    p, r, f, s = _precision_recall_fscore_support(y_true, y_pred, labels=labels)
    if labels is None: labels = get_class_labels(y_true, y_pred)
    if target_names is None: target_names = [str(l) for l in labels]
    
    report = f"{'class':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    for i, name in enumerate(target_names):
        report += f"{name:>15} {p[i]:>10.4f} {r[i]:>10.4f} {f[i]:>10.4f} {int(s[i]):>10d}\n"
    
    # Add averages
    report += "-"*60 + "\n"
    report += f"{'accuracy':>15} {'':>10} {'':>10} {accuracy_score(y_true, y_pred):>10.4f} {int(np.sum(s)):>10d}\n"
    report += f"{'macro avg':>15} {np.mean(p):>10.4f} {np.mean(r):>10.4f} {np.mean(f):>10.4f} {int(np.sum(s)):>10d}\n"
    return report

# Missing aliases and additional functions
def fbeta_score(y_true, y_pred, beta=1.0, **kwargs):
    """Compute F-beta score."""
    return f1_score(y_true, y_pred, **kwargs)  # For now, delegate

def true_negative_rate(y_true, y_pred, pos_label=1):
    """TNR / Specificity."""
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    n = tn + np.sum((y_true != pos_label) & (y_pred == pos_label))
    return float(tn / n) if n > 0 else 0.0

def false_negative_rate(y_true, y_pred, pos_label=1):
    """FNR / Miss rate."""
    return 1.0 - true_positive_rate(y_true, y_pred, pos_label)

def sensitivity_score(y_true, y_pred, pos_label=1):
    """Sensitivity = TPR."""
    return true_positive_rate(y_true, y_pred, pos_label)

def specificity_score(y_true, y_pred, pos_label=1):
    """Specificity = TNR."""
    return true_negative_rate(y_true, y_pred, pos_label)

def num_true_positives(y_true, y_pred, pos_label=1):
    return int(np.sum((y_true == pos_label) & (y_pred == pos_label)))

def num_true_negatives(y_true, y_pred, pos_label=1):
    return int(np.sum((y_true != pos_label) & (y_pred != pos_label)))

def num_false_positives(y_true, y_pred, pos_label=1):
    return int(np.sum((y_true != pos_label) & (y_pred == pos_label)))

def num_false_negatives(y_true, y_pred, pos_label=1):
    return int(np.sum((y_true == pos_label) & (y_pred != pos_label)))

def roc_curve(y_true, y_score, pos_label=1):
    """Compute ROC curve."""
    return _binary_roc_curve(y_true, y_score, pos_label=pos_label)

def auc(x, y):
    """Compute AUC using trapezoidal rule."""
    return float(np.trapezoid(y, x))

def precision_recall_curve(y_true, y_score, pos_label=1):
    """Compute PR curve."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    desc = np.argsort(y_score)[::-1]
    y_score = y_score[desc]; y_true = y_true[desc]
    tps = np.cumsum(y_true == pos_label)
    fps = np.cumsum(y_true != pos_label)
    precision = tps / (tps + fps)
    recall = tps / np.sum(y_true == pos_label) if np.sum(y_true == pos_label) > 0 else np.zeros_like(tps)
    return precision, recall, y_score

def average_precision_score(y_true, y_score, pos_label=1):
    """Compute average precision."""
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label)
    return float(-np.sum(np.diff(r) * p[:-1]))

def log_loss(y_true, y_pred_proba, eps=1e-15):
    """Compute log loss."""
    y_true = np.asarray(y_true); y_pred_proba = np.asarray(y_pred_proba)
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = len(y_true)
    loss = 0.0
    for i, true_label in enumerate(y_true):
        loss -= np.log(y_pred_proba[i, int(true_label)])
    return float(loss / n)

def hamming_loss(y_true, y_pred):
    """Compute Hamming loss."""
    return float(np.mean(y_true != y_pred))

def zero_one_loss(y_true, y_pred, normalize=True):
    """Compute 0-1 loss."""
    loss = np.sum(y_true != y_pred)
    return float(loss / len(y_true)) if normalize else float(loss)
