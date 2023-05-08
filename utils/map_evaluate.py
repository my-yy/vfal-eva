import numpy as np
import scipy
import scipy.spatial


def cos_dist(query_matrix, result_matrix):
    return scipy.spatial.distance.cdist(query_matrix, result_matrix, 'cosine')


def fx_calc_map_label(query_matrix, result_matrix, labels, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(query_matrix, result_matrix, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(query_matrix, result_matrix, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []

    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if labels[i] == labels[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


def fx_calc_map_label_v2(dist, label, k=0):
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []

    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


def fx_calc_map_label_v3(ord, label, k=0):
    numcases = ord.shape[0]
    if k == 0:
        k = numcases
    res = []

    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


if __name__ == "__main__":
    pass
