import torch


def CORALV2(source, target):
    d = source.data.shape[1]
    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)

    # frobenius norm between source and target
    tmp = torch.sum(torch.mul((xc - xct), (xc - xct)))
    loss = tmp / (4 * d * d)

    return loss
