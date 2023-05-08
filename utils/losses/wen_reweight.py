import collections
import copy

import ipdb
import numpy as np


def not_zero_count(the_dict):
    return len([v for v in the_dict.values() if not np.isclose(v, 0.0)])


# old_weights:
# 1) empty: init dict; the bottom 30% to 1.0, others to 0.0
# 2）not empty: set 22 identities weights to 1.0 where "old_weight==0 " ; others weight=weight*0.99
def update_weight(old_weights, hardness, init_ratio=0.3, k=22, alpha=0.99):
    # 1.sorted by hardness
    # [hard,....,easy]
    tup_list = [(k, v) for k, v in hardness.items()]
    tup_list.sort(key=lambda x: x[-1], reverse=True)
    sorted_identities_list = [tup[0] for tup in tup_list]

    if len(old_weights) == 0:  # init
        print("init weight dict")
        new_weights = {}
        rare_point = int(len(sorted_identities_list) * (1 - init_ratio))
        for i in range(len(sorted_identities_list)):
            if i > rare_point:
                new_weights[i] = 1
            else:
                new_weights[i] = 0
        return new_weights

    # 2.find identities with zero weights:
    # [hard,....,easy]
    zero_weight_identities = [i for i in sorted_identities_list if np.isclose(old_weights[i], 0.0)]

    # 3.find the easiest 22 identities:
    if len(zero_weight_identities) > k:
        zero_weight_identities_bottom = zero_weight_identities[int(len(zero_weight_identities) - k):]
        assert len(zero_weight_identities_bottom) == k
    else:
        zero_weight_identities_bottom = zero_weight_identities
    zero_weight_identities_bottom = set(zero_weight_identities_bottom)

    # 4.assign new weights:
    new_weights = copy.deepcopy(old_weights)
    for k, v in new_weights.items():
        if k in zero_weight_identities_bottom:
            new_v = 1.0
        else:
            new_v = v * alpha
        new_weights[k] = new_v
    return new_weights


def update_hardness(old_hardness, label_list, loss_list, beta=0.9):
    new_hardness = copy.deepcopy(old_hardness)
    for label, loss in zip(label_list, loss_list):
        old_h = old_hardness.get(label, loss)
        new_h = beta * old_h + (1 - beta) * loss
        new_hardness[label] = new_h
    return new_hardness
