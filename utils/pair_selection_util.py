import ipdb
import numpy as np
import torch


def contrastive_loss(f_emb, v_emb, margin, tau):
    # 1. positive pair distances
    loss_pos = parallel_l2_distance(f_emb, v_emb).mean()

    # 2.negative pairs
    negative_tuple_list = negative_pair_selection(v_emb, f_emb, tau)
    f_emb2 = f_emb[[tup[0] for tup in negative_tuple_list]]
    v_emb2 = v_emb[[tup[1] for tup in negative_tuple_list]]
    dist = parallel_l2_distance(f_emb2, v_emb2)
    loss_neg = torch.nn.functional.relu(margin - dist).mean()
    loss = loss_pos + loss_neg
    return loss


def negative_pair_selection(f_emb, v_emb, tau):
    # calculate pari-wise similarity:
    f_emb_npy = f_emb.detach().cpu().numpy()
    v_emb_npy = v_emb.detach().cpu().numpy()
    f2v_distance = pairwise_l2_distance(f_emb_npy, v_emb_npy)

    # create (anchor,neg) pairs:
    batch_size = len(v_emb)
    pair_list = []

    for i in range(batch_size):
        # anchor-negative distance list:
        distance_list = f2v_distance[i]
        distance_sorted_list = np.argsort(distance_list)[::-1]
        # [big_distance_index,.....,small_distance_index]
        for j in range(int(tau * batch_size)):
            neg_idx = distance_sorted_list[j]
            if neg_idx == i:  # find positive sample，early exit
                break
            pair_list.append([i, j])
    return pair_list


def parallel_l2_distance(matrix_a, matrix_b):
    # matrix_a: (batch_a,dim)
    # matrix_b: (batch_b,dim)
    # output： (batch)

    c = matrix_a - matrix_b
    return torch.sqrt(torch.sum(c * c, dim=1))


def pairwise_l2_distance(matrix_a, matrix_b):
    # matrix_a: (batch_a,dim)
    # matrix_b: (batch_b,dim)

    matrix_dot = np.dot(matrix_a, np.transpose(matrix_b))
    # (batch_a,batch_b)

    a_square = np.sum(matrix_a * matrix_a, axis=1)
    # (batch_a)

    b_square = np.sum(matrix_b * matrix_b, axis=1)
    # (batch_b)

    a_square_2d = np.expand_dims(a_square, axis=1)
    # (1,batch_a)

    b_square_2d = np.expand_dims(b_square, axis=0)
    # (batch_b,1)

    distance_matrix_squired = a_square_2d - 2.0 * matrix_dot + b_square_2d

    distance_matrix = np.maximum(distance_matrix_squired, 0.0)
    distance_matrix = np.sqrt(distance_matrix)
    return distance_matrix
