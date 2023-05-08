import mkl
import collections

mkl.get_max_threads()
import faiss
from utils import wb_util, distance_util
import numpy as np
import torch


def do_k_means(matrix, ncentroids):
    niter = 20
    verbose = True
    d = matrix.shape[1]

    kmeans = faiss.Kmeans(d,
                          ncentroids,
                          niter=niter,
                          verbose=verbose,
                          spherical=False,
                          min_points_per_centroid=3,
                          max_points_per_centroid=100000,
                          gpu=False,
                          )

    kmeans.train(matrix)

    D, I = kmeans.index.search(matrix, 1)

    cluster_label = I.squeeze()
    similarity_array = []
    for i in range(len(matrix)):
        sample_vec = matrix[i]
        sample_label = I[i][0]
        center_vec = kmeans.centroids[sample_label]
        similarity = distance_util.cosine_similarity(sample_vec, center_vec)
        similarity_array.append(similarity)
    similarity_array = np.array(similarity_array)

    sorted_similarity_array = similarity_array.copy()
    sorted_similarity_array.sort()

    return cluster_label, similarity_array


def get_center_matrix(v_emb, f_emb, cluster_label, ncentroids):
    tmp_dict = collections.defaultdict(list)

    for v, f, label in zip(v_emb, f_emb, cluster_label):
        tmp_dict[label].append(v)
        tmp_dict[label].append(f)

    tmp_arr = []
    for i in range(ncentroids):
        vec = np.mean(tmp_dict[i], axis=0)
        tmp_arr.append(vec)

    center_matrix = np.array(tmp_arr)
    return center_matrix


def extract_embeddings(ordered_iter, model):
    model.eval()
    all_emb = []
    all_emb_v = []
    all_emb_f = []
    all_keys = []
    for data in ordered_iter:
        with torch.no_grad():
            data = [i.cuda() for i in data]
            voice_data, face_data, label = data
            v_emb, f_emb = model(voice_data, face_data)
            # [v-f]
            the_emb = torch.cat([v_emb, f_emb], dim=1).detach().cpu().numpy()
            label_npy = label.squeeze().detach().cpu().numpy().tolist()

        for emb, label_int in zip(the_emb, label_npy):
            all_emb.append(emb)
            all_emb_v.append(emb[0:128])
            all_emb_f.append(emb[128:])
            all_keys.append(ordered_iter.dataset.train_movie_list[label_int])
    model.train()
    return all_keys, all_emb, all_emb_v, all_emb_f


def do_cluster(ordered_iter, ncentroids, model, input_emb_type="all"):
    all_keys, all_emb, all_emb_v, all_emb_f = extract_embeddings(ordered_iter, model)

    if input_emb_type == "v":
        input_emb = np.array(all_emb_v)
    elif input_emb_type == "f":
        input_emb = np.array(all_emb_f)
    elif input_emb_type == "all":
        input_emb = np.array(all_emb)
    else:
        raise Exception("wrong type")

    cluster_label, similarity_array = do_k_means(input_emb, ncentroids)

    movie2label = {}
    for label, key, sim in zip(cluster_label, all_keys, similarity_array):
        movie2label[key] = label

    center_vector = get_center_matrix(all_emb_v, all_emb_f, cluster_label, ncentroids)
    return movie2label, center_vector


def do_cluster_v2(all_keys, all_emb, all_emb_v, all_emb_f, ncentroids, input_emb_type="all"):
    if input_emb_type == "v":
        input_emb = np.array(all_emb_v)
    elif input_emb_type == "f":
        input_emb = np.array(all_emb_f)
    elif input_emb_type == "all":
        input_emb = np.array(all_emb)
    else:
        raise Exception("wrong type")

    cluster_label, similarity_array = do_k_means(input_emb, ncentroids)

    movie2label = {}
    for label, key, sim in zip(cluster_label, all_keys, similarity_array):
        movie2label[key] = label

    center_vector = get_center_matrix(all_emb_v, all_emb_f, cluster_label, ncentroids)
    return movie2label, center_vector
