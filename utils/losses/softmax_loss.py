# 带参数的cel损失
import torch
import torch.nn as nn


class SoftmaxLoss(nn.Module):
    def __init__(self, feature_dim, num_class, reduction="mean"):
        super(SoftmaxLoss, self).__init__()
        self.in_feats = feature_dim
        self.W = torch.nn.Parameter(torch.randn(feature_dim, num_class))
        self.cel = nn.CrossEntropyLoss(reduction=reduction)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, embedding, labels):
        assert embedding.size()[0] == labels.size()[0]
        assert embedding.size()[1] == self.in_feats
        logits = torch.mm(embedding, self.W)
        loss = self.cel(logits, labels)
        return loss


if __name__ == "__main__":
    feature_dim = 128
    num_class = 10
    batch_size = 3

    from utils import seed_util

    seed_util.set_seed(10086)
    embedding = torch.randn(batch_size, feature_dim)
    label = torch.randint(0, num_class, (batch_size,), dtype=torch.long)

    criteria = MySoftmaxLoss(feature_dim, num_class).cuda()
    loss = criteria(embedding.cuda(), label.cuda())
    print(loss)
