import ipdb
import torch
import torch.nn as nn


# 单纯的center loss，其中"类中心矩阵"是可以训练的参数
class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, use_gpu):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        # 1 计算出与类中心之间的 欧几里得距离
        # 1）先计算a^2+b^2
        A = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        B = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = A + B

        # 2)进一步-2ab
        distmat.addmm_(1, -2, x, self.centers.t())

        # 2.将位于真实label位置处的距离抠出来：
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


if __name__ == "__main__":
    from utils import seed_util

    seed_util.set_seed(10086)
    batch_size = 3
    feature_dim = 128
    num_class = 4

    embedding = torch.rand([batch_size, feature_dim])
    label = torch.LongTensor([0, 1, 3])

    fun = CenterLoss(num_class, feature_dim, use_gpu=False)
    loss = fun(embedding, label)
    print(loss)
