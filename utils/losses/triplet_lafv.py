import torch
import numpy as np
import ipdb


def triplet_loss(v, f_p, f_n):
    d_ap = l2_distance(v, f_p)
    d_an = l2_distance(v, f_n)
    cated = torch.cat([d_ap, d_an], dim=1)
    probs = torch.nn.functional.softmax(cated, dim=1)
    batch_size = len(v)
    target = torch.FloatTensor([[0, 1]] * batch_size).cuda()
    loss = l2_distance(probs, target)
    loss_mean = loss.mean()
    return loss_mean


def l2_distance(v, f):
    # [batch,emb] -> [batch,1]
    return torch.sqrt(torch.sum((v - f) ** 2, dim=1, keepdim=True) + 1e-12)


if __name__ == "__main__":
    from utils import seed_util

    seed_util.set_seed(1)
    import ipdb

    v = torch.rand(4, 10).cuda()
    fp = torch.rand(4, 10).cuda()
    fn = torch.rand(4, 10).cuda()
    triplet_loss(v, fp, fn)
