import torch
import torch.nn as nn
import ipdb


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        # loss
        c_diff = (c - torch.eye(D, device="cuda")).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param

        loss = c_diff.sum()

        return loss


class MyBarlowTwinsLoss(nn.Module):
    def __init__(self, feature_dim, num_out_dim):
        super(MyBarlowTwinsLoss, self).__init__()
        self.in_feats = feature_dim
        self.W = torch.nn.Parameter(torch.randn(feature_dim, num_out_dim))
        nn.init.xavier_normal_(self.W, gain=1)
        self.fun_barlow = BarlowTwinsLoss()

    def forward(self, v_emb, f_emb):
        v_out = torch.mm(v_emb, self.W)
        f_out = torch.mm(f_emb, self.W)
        loss = self.fun_barlow(v_out, f_out)
        return loss
