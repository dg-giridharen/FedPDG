"""
utils/metrics.py — All Evaluation Metrics
Closed-set, open-set, and Byzantine detection metrics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report,
    average_precision_score
)


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics."""
    metrics = {}

    # Multi-class metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_macro'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0)
    
    # Store per-class F1 to track minority collapse
    metrics['f1_per_class'] = f1_score(
        y_true, y_pred, average=None, zero_division=0)

    # Binary metrics (benign=0 vs attack=1+)
    y_bin_true = (y_true > 0).astype(int)
    y_bin_pred = (y_pred > 0).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(
            y_bin_true, y_bin_pred, labels=[0, 1]).ravel()
        metrics['detection_rate'] = tp / (tp + fn + 1e-8)
        metrics['false_alarm_rate'] = fp / (fp + tn + 1e-8)
        metrics['precision_binary'] = tp / (tp + fp + 1e-8)
        metrics['f1_binary'] = f1_score(y_bin_true, y_bin_pred, zero_division=0)
    except ValueError:
        metrics['detection_rate'] = 0.0
        metrics['false_alarm_rate'] = 0.0
        metrics['precision_binary'] = 0.0
        metrics['f1_binary'] = 0.0

    # AUC-ROC (if probability scores provided)
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(
                y_bin_true, y_prob, multi_class='ovr')
        except Exception:
            metrics['auc_roc'] = 0.0

    return metrics


def compute_zeroday_metrics(distances, is_zeroday, threshold):
    """
    Metrics for zero-day detection.

    Args:
        distances:   array of distances to nearest prototype
        is_zeroday:  boolean array (True if sample is truly unseen)
        threshold:   decision threshold for flagging unknown

    Returns:
        dict of zero-day metrics
    """
    predicted_zd = (distances > threshold)

    tp = np.sum(predicted_zd & is_zeroday)
    fp = np.sum(predicted_zd & ~is_zeroday)
    fn = np.sum(~predicted_zd & is_zeroday)
    tn = np.sum(~predicted_zd & ~is_zeroday)

    zd_precision = tp / (tp + fp + 1e-8)
    zd_recall = tp / (tp + fn + 1e-8)
    zd_f1 = 2 * zd_precision * zd_recall / (zd_precision + zd_recall + 1e-8)
    fpr_at_95 = fp / (fp + tn + 1e-8)  # simplified

    try:
        auc_roc = roc_auc_score(is_zeroday, distances)
        pr_auc = average_precision_score(is_zeroday, distances)
    except Exception:
        auc_roc = 0.0
        pr_auc = 0.0

    return {
        'zd_precision': zd_precision,
        'zd_recall': zd_recall,
        'zd_f1': zd_f1,
        'zd_detection_rate': zd_recall,
        'fpr_at_95tpr': fpr_at_95,
        'zd_auc_roc': auc_roc,
        'zd_pr_auc': pr_auc,
    }


def compute_byzantine_metrics(true_byz_ids, detected_byz_ids, num_clients):
    """
    Byzantine detection performance.

    Args:
        true_byz_ids:     list of actual Byzantine client IDs
        detected_byz_ids: list of detected Byzantine client IDs
        num_clients:      total number of clients

    Returns:
        dict of Byzantine detection metrics
    """
    true_set = set(true_byz_ids)
    det_set = set(detected_byz_ids)

    tp = len(true_set & det_set)
    fp = len(det_set - true_set)
    fn = len(true_set - det_set)
    tn = num_clients - len(true_set | det_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'byz_precision': precision,
        'byz_recall': recall,
        'byz_f1': f1,
        'byz_accuracy': (tp + tn) / num_clients,
    }


def get_classification_report(y_true, y_pred, label_names=None):
    """Get full per-class classification report."""
    return classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0)
