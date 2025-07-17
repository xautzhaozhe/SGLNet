# -*- coding: utf-8 -*-
from sklearn import metrics
import numpy as np
import torch


def Residual(contr_data, org_data):

    row, col, band = org_data.shape
    residual = np.square(org_data - contr_data)
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            R = np.mean(residual[i, j, :])
            result[i, j] = R

    return result


def ROC_AUC(target2d, groundtruth):
    """
    :param target2d: the 2D anomaly component
    :param groundtruth: the groundtruth
    :return: auc: the AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    target2d = target2d.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    print('The AUC Value: ', auc)

    return 0


def Mahalanobis(data):

    row, col, band = data.shape
    data = data.reshape(row * col, band)
    mean_vector = np.mean(data, axis=0)
    mean_matrix = np.tile(mean_vector, (row * col, 1))
    re_matrix = data - mean_matrix
    matrix = np.dot(re_matrix.T, re_matrix) / (row * col - 1)
    variance_covariance = np.linalg.pinv(matrix)

    distances = np.zeros([row * col, 1])
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = np.dot(re_array, variance_covariance)
        distances[i] = np.dot(re_var, np.transpose(re_array))
    distances = distances.reshape(row, col)

    return distances


# define dataset
class MyTrainData(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt

    def __len__(self):
        return 1
