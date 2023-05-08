import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature, reduction="mean"):
        super(InfoNCE, self).__init__()
        self.cel = nn.CrossEntropyLoss(reduction=reduction)
        self.temperature = temperature

    # 两个batch是平行对应的
    def forward(self, anchor, positive, need_logits=False):
        batch_size = anchor.size(0)
        # 1.变成单位向量
        anchor = F.normalize(anchor)
        positive = F.normalize(positive)

        # 2.计算相似度矩阵，也即logits
        similarity_matrix = torch.matmul(anchor, positive.T)  # (bs, bs)

        # 3.对角线元素为"分类目标"
        logits = similarity_matrix / self.temperature
        labels = torch.LongTensor([i for i in range(batch_size)]).cuda()
        loss = self.cel(logits, labels)
        if need_logits:
            return loss, logits
        return loss
