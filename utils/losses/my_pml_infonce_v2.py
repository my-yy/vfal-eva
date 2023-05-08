import torch
import ipdb
import numpy


# 目标：只是扩充ap、an set，不要增加过多的a

def my_info_nce(anchor, pos_neg, l_anchor, l_pos_neg, temperature=0.07, reduction="mean"):
    # 1.获得可行的组合
    a1, p, a2, n = get_all_pairs_indices(l_anchor, l_pos_neg)
    # (a1,p)  AP样本对在sim_matrix矩阵中对应的（行下标，列下标)
    # (a2,n)  AN样本对在sim_matrix矩阵中对应的（行下标，列下标)

    # 2.计算相似度矩阵
    sim_matrix = cosine_similarity(anchor, pos_neg)

    # 3.获得ap、an的相似度值
    pos_pairs = sim_matrix[a1, p]
    # [total_AP] 此数量作为最终的batch数
    neg_pairs = sim_matrix[a2, n]
    # [total_AN]

    # 4.相似度值分别除以温度系数
    pos_pairs = pos_pairs.unsqueeze(1) / temperature
    neg_pairs = neg_pairs / temperature

    # 5.为每个(a,p)构建对应的负样本集合

    # 5.1 先假定假设所有的负样本都是可行的，因此获得 [total_AP,total_AN] 大小的矩阵
    neg_pairs_2D = neg_pairs.repeat(len(pos_pairs), 1)

    # 5.2 创建mask,将隶属于同一个anchor的(a,p)(a,n)处为1，其余为0
    mask = a1.unsqueeze(1) == a2.unsqueeze(0)
    # ipdb.set_trace()
    # [total_AP,total_AN]

    # 5.3 将0处的score值设置为负无穷，即不考虑这些位置
    neg_pairs_2D[mask == 0] = torch.finfo(torch.float32).min

    # 6.相当于分子、分母都有了，然后计算-log(exp())即可
    loss = neg_log_exp(pos_pairs, neg_pairs_2D)
    # ipdb.set_trace()
    if reduction == "mean":
        return loss.mean()
    return loss, a1


def neg_log_exp(pos_pairs, neg_pairs):
    # ipdb.set_trace()
    max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]).detach()

    numerator = torch.exp(pos_pairs - max_val).squeeze(1)

    denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator

    p = numerator / denominator

    loss = - torch.log(p + torch.finfo(torch.float32).tiny)
    return loss


def get_all_pairs_indices(labels_anchor, labels_ref):
    # 根据label，生成所有的ap、an搭配
    labels1 = labels_anchor.unsqueeze(1)
    labels2 = labels_ref.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    # [4,10]

    diffs = matches ^ 1
    # matches.fill_diagonal_(0)  # 这里我注释掉了
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)

    return a1_idx, p_idx, a2_idx, n_idx


def cosine_similarity(a, pn):
    a = torch.nn.functional.normalize(a)
    pn = torch.nn.functional.normalize(pn)
    similarity_matrix = torch.matmul(a, pn.T)
    return similarity_matrix


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature, reduction):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, emb_anchor, emb_pn, label_anchor, label_pn):
        return my_info_nce(emb_anchor, emb_pn, label_anchor, label_pn, self.temperature, self.reduction)


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # https://pytorch.org/docs/stable/notes/randomness.html


if __name__ == "__main__":
    import random
    import numpy
    import torch

    numpy.set_printoptions(linewidth=180, precision=5, suppress=True)
    torch.set_printoptions(linewidth=180, sci_mode=False)
    set_seed(10086)
    emb_dim = 2
    embedding_anchor = torch.rand([10, emb_dim])
    embedding_pn = torch.rand([20, emb_dim])
    l1 = torch.LongTensor([1, 2, 1, 3, 0, 1, 2, 1, 2, 0])
    l2 = torch.LongTensor([1, 2, 1, 3, 0, 1, 2, 1, 2, 0, 1, 2, 1, 3, 0, 1, 2, 1, 2, 0])

    print(my_info_nce(embedding_anchor, embedding_pn, l1, l2))
