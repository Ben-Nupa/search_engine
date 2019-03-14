import numpy as np
from matplotlib import pyplot as plt
from time import time
import re

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
    if len(pred) > 0:
        return len(true_positives) / len(pred)
    return None


def compute_recall(pred: np.ndarray, true) -> float:
    """
    Computes the recall score from the given predicted and true documents.
    """
    if type(true) != set:
        true = set(true)
    true_positives = set(pred).intersection(true)
    if len(true) > 0:
        return len(true_positives) / len(true)
    return None


def compute_e_measure(pred: np.ndarray, true, beta=None) -> float:
    """
    Computes the E measure from the given predicted and true documents.
    """
    precision_score = compute_precision(pred, true)
    recall_score = compute_recall(pred, true)
    if beta is None:
        try:
            beta = precision_score / recall_score
        except:
            beta=np.inf
    try:
        return 1 - precision_score * recall_score * (1 + beta ** 2) / (recall_score + precision_score * beta ** 2)
    except:
        return 1


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

    
def read_truth(qrel_path: str, query_path:str=None):
    truth = np.loadtxt(qrel_path, dtype=str, delimiter="\n")
    query_docs = [[] for i in range(66)]
    for row in truth:
        row = re.split("\s+", row)
        id_query = int(row[0])
        id_doc = int(row[1])
        query_docs[id_query] +=[id_doc]
    
    if query_path is None:
        return query_docs

    
    query_texts = [""]  # querys start at 1
    raw_file = np.loadtxt(query_path, dtype=str, delimiter="\n")
    keep_element = False

    for line in raw_file:
        if line.startswith(".W"):
            keep_element = True
            query_texts += [""]
        elif line.startswith("."):
            keep_element = False
        else:
            if keep_element:
                query_texts[-1] += line + " "
      
    return query_docs, query_texts


def full_query_report(index, query, true_results):
    
    start = time()
    docs = index.treat_query(query, show_tree=True)
    print("Query treated in {:.1f}s".format(time()-start))


    print("Docs found for query : {}".format(query))
    print(docs)
    print("Docs we should have found :")
    print(true_results)
    precision = compute_precision(docs, true_results)
    recall = compute_recall(docs, true_results)
    e_measure = compute_e_measure(docs, true_results)
    f_measure = compute_f_measure(docs, true_results)
    r_measure = compute_r_measure(docs, true_results)
    interpolated_precisions = compute_interpolated_precisions(docs, true_results)
    average_precision = compute_average_precision(docs, true_results)
    
    print("precision :\t\t\t{:.2f}".format(precision))
    print("recall :\t\t\t{:.2f}".format(recall))
    print("e_measure :\t\t\t{:.2f}".format(e_measure))
    print("f_measure :\t\t\t{:.2f}".format(f_measure))
    print("r_measure :\t\t\t{:.2f}".format(r_measure))
    print("interpolated_precisions :\t{}".format(interpolated_precisions))
    print("average_precision :\t\t{:.2f}".format(average_precision))

    
    plot_precision_recall_curve(docs, true_results)
    plt.show()


def short_query_report(index, id_query, query, true_results):
    t = time()
    result = index.treat_query(query)
    t = time()-t
    precision = compute_precision(result, true_results)
    recall = compute_recall(result, true_results)
    f_measure = compute_f_measure(result, true_results)
    report_str = "query {} :\t".format(id_query)
    if recall is not None:
        report_str += "r : {:.2f}\t".format(recall)
    else : 
        report_str += "r : NaN\t\t"
    if precision is not None:
        report_str += "p : {:.2f}\t".format(precision)
    else : 
        report_str += "p : NaN\t\t"        
    if f_measure is not None:
        report_str += "f1 : {:.2f}\t".format(f_measure)
    else : 
        report_str += "f1 : NaN\t\t"      
    report_str += "({:.2f}s)".format(t)
    print(report_str)
    return result
    

if __name__ == '__main__':
    pred = np.arange(14)
    true = [0, 1, 3, 5, 12, 50]
    # true = np.array([131,4334,53442,434])
    print(compute_mean_average_precision(pred,true))
    plot_precision_recall_curve(pred, true)
    plt.show()
