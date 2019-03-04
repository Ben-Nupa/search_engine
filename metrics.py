import numpy as np
from matplotlib import pyplot as plt


def compute_precision(pred: np.ndarray, true) -> float:
    """
    Computes the precision score from the given predicted and true documents.

    Attributes
    ----------
    pred: ndarray
        The predicted documents.
    true: list, ndarray or set
        The true documents for the searched query.

    Returns
    ----------
    out: float
        Precision score.
    """
    if type(true) != set:
        true = set(true)
    true_positives = set(pred).intersection(true)
    return len(true_positives) / len(pred)


def compute_recall(pred: np.ndarray, true) -> float:
    """
    Computes the recall score from the given predicted and true documents.
    """
    if type(true) != set:
        true = set(true)
    true_positives = set(pred).intersection(true)
    return len(true_positives) / len(true)


def compute_e_measure(pred: np.ndarray, true, beta=None) -> float:
    """
    Computes the E measure from the given predicted and true documents.
    """
    precision_score = compute_precision(pred, true)
    recall_score = compute_recall(pred, true)
    if beta is None:
        beta = precision_score / recall_score
    return 1 - precision_score * recall_score * (1 + beta ** 2) / (recall_score + precision_score * beta ** 2)


def compute_f_measure(pred: np.ndarray, true, beta=1) -> float:
    """
    Computes the F measure from the given predicted and true documents.
    """
    return 1 - compute_e_measure(pred, true, beta)


def compute_r_measure(pred: np.ndarray, true, max_rank=None) -> float:
    """
    Computes the R-measure (precision at rank R) from the given predicted and true documents.

    Attributes
    ----------
    pred: ndarray
        The predicted documents, ranked in order of relevance.
    true: list, ndarray or set
        The true documents for the searched query.
    max_rank: int
        Maximum rank to consider when computing the precision. If None, max_rank is set to the number of relevant
        documents (i.e. len(true)).

    Returns
    ----------
    out: float
        R-measure.
    """
    if max_rank is None:
        max_rank = len(true)
    elif max_rank > len(pred):
        max_rank = len(pred)
    return compute_precision(pred[:max_rank], true)


def compute_interpolated_precisions(pred: np.ndarray, true) -> list:
    """
    Computes the interpolated precisions score from the given predicted and true documents.

    Attributes
    ----------
    pred: ndarray
        The predicted documents, ranked in order of relevance.
    true: list, ndarray or set
        The true documents for the searched query.

    Returns
    ----------
    out: list
        Interpolated precisions.
    """
    precision = []
    recall = []
    true = set(true)  # Faster computation to do it now.
    # Compute precision and recall scores by adding documents one by one in order of return.
    for k in range(1, len(pred) + 1):
        precision.append(compute_precision(pred[:k], true))
        recall.append(compute_recall(pred[:k], true))

    interpolated_precisions = []
    idx_min_recall = 0
    # Computes interpolated precisions
    for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        while idx_min_recall < len(recall) and recall[idx_min_recall] < r:
            idx_min_recall += 1

        if idx_min_recall == len(recall):
            interpolated_precisions.append(0)
        else:
            interpolated_precisions.append(np.max(precision[idx_min_recall:]))

    return interpolated_precisions


def compute_average_precision(pred: np.ndarray, true) -> float:
    """
    Computes the average precision score from the given predicted and true documents.

    Attributes
    ----------
    pred: ndarray
        The predicted documents, ranked in order of relevance.
    true: list, ndarray or set
        The true documents for the searched query.

    Returns
    ----------
    out: float
        Average precision score.
    """
    return np.sum(compute_interpolated_precisions(pred, true)) / 11


def compute_mean_average_precision(queries_pred: list, queries_true: list) -> float:
    """
    Computes the mean average precision score from the given queries.

    Attributes
    ----------
    queries_pred: list[ndarray]
        The predicted documents, ranked in order of relevance, of each query.
    queries_true: list[list], list[ndarray] or list[set]
        The true documents of each query.

    Returns
    ----------
    out: float
        Mean average precision score.
    """
    average_precisions = []
    for idx_query in range(len(queries_pred)):
        average_precisions.append(compute_average_precision(queries_pred[idx_query], queries_true[idx_query]))
    return np.mean(average_precisions)


def plot_precision_recall_curve(pred: np.ndarray, true):
    """
    Plots the recall-precision curve. Must call plt.show() to show.
    """
    recalls = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    interpolated_precisions = compute_interpolated_precisions(pred, true)

    # Rearrange lists for a better plot
    x = []
    y = []
    for k in range(11):
        if k == 0:
            x.append(recalls[k])
            y.append(interpolated_precisions[k])
            y.append(interpolated_precisions[k])
        elif k == 10:
            x.append(recalls[k])
            x.append(recalls[k])
            y.append(interpolated_precisions[k])
        else:
            x.append(recalls[k])
            x.append(recalls[k])
            y.append(interpolated_precisions[k])
            y.append(interpolated_precisions[k])

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-recall curve')


# if __name__ == '__main__':
#     pred = np.arange(14)
#     true = [0, 1, 3, 5, 12, 50]
#
#     plot_precision_recall_curve(pred, true)
#     plt.show()
