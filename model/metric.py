# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras


def calculate_auroc(predictions, labels):
    """
    Calculate auroc.
    :param predictions: predictions
    :param labels: labels
    :return: fpr_list, tpr_list, auroc
    """
    if np.max(labels) ==1 and np.min(labels)==0:
        fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
        auroc = metrics.roc_auc_score(labels, predictions)
    else:
        fpr_list, tpr_list = [], []
        auroc = np.nan
    return fpr_list, tpr_list, auroc


def calculate_aupr(predictions, labels):
    """
    Calculate aupr.
    :param predictions: predictions
    :param labels: labels
    :return: precision_list, recall_list, aupr
    """
    if np.max(labels) == 1 and np.min(labels) == 0:
        precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
        aupr = metrics.average_precision_score(labels, predictions)
    else:
        precision_list, recall_list = [], []
        aupr = np.nan
    return precision_list, recall_list, aupr