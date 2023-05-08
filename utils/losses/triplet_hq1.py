import torch
import ipdb

def l2_distance(v, f):
    # [batch,emb] -> [batch,1]
    return torch.sqrt((v - f).pow(2).sum(dim=1, keepdim=True) + 1e-12)


def triplet_loss(v_origin, f_origin, label, alpha=0.6, beta=0.2, lamd=0.1):
    # 变为单位向量
    v = torch.nn.functional.normalize(v_origin, p=2, dim=1)
    f = torch.nn.functional.normalize(f_origin, p=2, dim=1)

    label = label.squeeze()
    # [batch]

    # 计算欧几里得距离
    l2_dist_matrix = torch.cdist(v, f)
    dis_ap = torch.diagonal(l2_dist_matrix)

    # 寻找Hardest样本：
    mask = (label.unsqueeze(1) == label.unsqueeze(0)).byte()
    # 相同标签的部分值为1
    MAX_VAL = l2_dist_matrix.max()
    tmp_l2_dist_matrix = mask * MAX_VAL + l2_dist_matrix

    # 然后找这一行中的最小值：
    dis_hardest_an, indices = tmp_l2_dist_matrix.min(dim=1)

    # ipdb.set_trace()
    # 第一部分的损失
    loss1 = torch.nn.functional.relu(alpha + dis_ap - dis_hardest_an)

    # 第二部分的损失
    # emb_hardest = f[indices]
    # dis_pn = l2_distance(f, emb_hardest)
    # loss2 = torch.nn.functional.relu(beta - dis_pn)

    # loss = loss1 + lamd * loss2
    loss = loss1
    return loss.mean()


if __name__ == "__main__":
    from utils import seed_util

    seed_util.set_seed(1)
    import ipdb

    batch_size = 5
    v = torch.rand(batch_size, 10).cuda()
    f = torch.rand(batch_size, 10).cuda()
    label = torch.LongTensor([0, 0, 1, 2, 3]).cuda()
    triplet_loss(v, f, label)
