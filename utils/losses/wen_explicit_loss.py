import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss function of equation (10) of paper "Seeking the Shape of Sound: An Adaptive Framework for Learning Voice-Face Association"

# From https://github.com/KID-7391/seeking-the-shape-of-sound   models/backbones.py

def cross_logit(x, v):
    # x、v就是两组embedding向量，比如声音、人脸向量，同一行对应的是同一个人的
    dist = l2dist(F.normalize(x).unsqueeze(0), v.unsqueeze(1))
    # [batch,batch]
    # 默认情况下，x、v都不是单位向量， F.normalize将x变为单位向量

    one_hot = torch.zeros(dist.size()).to(x.device)
    # 全0矩阵，[batch,batch]

    one_hot.scatter_(1, torch.arange(len(x)).view(-1, 1).long().to(x.device), 1)
    # 将对角线变为1  [\]

    pos = (one_hot * dist).sum(-1, keepdim=True)
    # 将那个对角线上的值取出来，这个是同一个人的声音与人脸向量

    logit = (1.0 - one_hot) * (dist - pos)
    # "不同人音、脸的距离"，比"同一人" 音脸之间的距离大多少

    loss = torch.log(1 + torch.exp(logit).sum(-1) + 3.4)

    return loss


def l2dist(a, b):
    # L2 distance
    dist = (a * b).sum(-1)
    return dist
