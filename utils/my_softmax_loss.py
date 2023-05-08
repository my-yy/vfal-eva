import torch
import torch.nn as nn


class MySoftmaxLoss(nn.Module):
    def __init__(self, feature_dim, num_class):
        super(MySoftmaxLoss, self).__init__()
        self.in_feats = feature_dim
        self.W = torch.nn.Parameter(torch.randn(feature_dim, num_class))
        self.cel = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, embedding, labels):
        assert embedding.size()[0] == labels.size()[0]
        assert embedding.size()[1] == self.in_feats
        logits = torch.mm(embedding, self.W)
        loss = self.cel(logits, labels)
        return loss
