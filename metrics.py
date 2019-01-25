import numpy as np


def precision(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the precision score from the given predicted and true documents.
    """
    true_positive = 0
    for doc_pred in pred:
        if doc_pred in true:
            true_positive += 1
    return true_positive / len(pred)


def recall(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Computes the recall score from the given predicted and true documents.
    """
    true_positive = 0
    for doc_pred in pred:
        if doc_pred in true:
            true_positive += 1
    return true_positive / len(true)


def e_measure(pred: np.ndarray, true: np.ndarray, beta=None) -> float:
    """
    Computes the E measure from the given predicted and true documents.
    """
    precision_score = precision(pred, true)
    recall_score = recall(pred, true)
    if beta is None:
        beta = precision_score / recall_score
    return 1 - precision_score * recall_score * (1 + beta ** 2) / (recall_score + precision_score * beta ** 2)


def f_measure(pred: np.ndarray, true: np.ndarray, beta=1) -> float:
    """
    Computes the F measure from the given predicted and true documents.
    """
    return 1 - e_measure(pred, true, beta)
