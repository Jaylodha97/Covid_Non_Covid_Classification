from inference import *


def get_accuracy(tp, tn, total):
    """
    Function to determine Model accuracy
    Formula: (tp+tn)/total

    Args:
    tp: True positives, tn: True negatives, total: Total no. of images used

    Returns:
    a: Model accuracy
    """

    a = round((((tp + tn) / (total)) * 100), 2)

    return a


def get_precision(tp, fp):
    """
    Function to determine Model precision
    Formula: (tp)/(tp+fp)

    Args:
    tp: True positives, fp: false positives

    Returns:
    p: Model precision
    """
    p = round(((tp / (tp + fp)) * 100), 2)

    return p


def get_recall(tn, fn):
    """
    Function to determine Model recall
    Formula: (tn)/(tn+fn)

    Args:
    tn: True negatives, fn: false negatives

    Returns:
    r: Model recall
    """

    r = round(((tn / (tn + fn)) * 100), 2)

    return r


def get_f1_score(precision, recall):
    """
    Function to determine Model f1 score (Harmonic mean)
    Formula: 2*(precision*recall)/(precision+recall)

    Args:
    precision, recall

    Returns:
    f: Model's F1 score
    """

    f = (2 * precision * recall) / (precision + recall)

    return f
