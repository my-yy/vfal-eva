import numpy
import numpy as np
import scipy.spatial


def calc_inter_distance(embedding):
    matrix_dot = numpy.dot(embedding, numpy.transpose(embedding))
    # (batch,batch)

    l2_norm_squired = numpy.diagonal(matrix_dot)

    distance_matrix_squired = numpy.expand_dims(l2_norm_squired, axis=0) + numpy.expand_dims(l2_norm_squired,
                                                                                             axis=1) - 2.0 * matrix_dot

    distance_matrix = numpy.maximum(distance_matrix_squired, 0.0)
    distance_matrix = numpy.sqrt(distance_matrix)
    return distance_matrix


def calc_matrix_distance(matrix_a, matrix_b):
    # matrix_a: (batch_a,dim)
    # matrix_b: (batch_b,dim)

    matrix_dot = numpy.dot(matrix_a, numpy.transpose(matrix_b))
    # (batch_a,batch_b)

    a_square = numpy.sum(matrix_a * matrix_a, axis=1)
    # (batch_a)

    b_square = numpy.sum(matrix_b * matrix_b, axis=1)
    # (batch_b)

    a_square_2d = numpy.expand_dims(a_square, axis=1)
    # (1,batch_a)

    b_square_2d = numpy.expand_dims(b_square, axis=0)
    # (batch_b,1)

    distance_matrix_squired = a_square_2d - 2.0 * matrix_dot + b_square_2d

    distance_matrix = numpy.maximum(distance_matrix_squired, 0.0)
    distance_matrix = numpy.sqrt(distance_matrix)
    return distance_matrix


def parallel_distance(a, b):
    a = numpy.array(a)
    b = numpy.array(b)

    assert len(a) == len(b)

    c = a - b
    return numpy.sqrt(numpy.sum(c * c, axis=1))


def parallel_distance_cosine_based_distance(a, b):
    assert len(a.shape) == 2
    assert a.shape == b.shape
    ab = np.sum(a * b, axis=1)
    # (batch_size,)

    a_norm = np.sqrt(np.sum(a * a, axis=1))
    b_norm = np.sqrt(np.sum(b * b, axis=1))
    cosine = ab / (a_norm * b_norm)

    dist = 1 - cosine
    # 0~2
    return dist


def distance_of_2point(a, b):
    return parallel_distance([a], [b])[0]


def cosine_similarity(v1, v2):
    return (1 - scipy.spatial.distance.cosine(v1, v2) + 1) / 2.0
