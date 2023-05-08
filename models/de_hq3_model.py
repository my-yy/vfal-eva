import ipdb
import torch
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F


class MyL2(nn.Module):
    def __init__(self, scale=1.0):
        super(MyL2, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=1, p=2) * self.scale


def create_encoder(in_dim=512):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
    )


def create_decoder(out_dim=512):
    return torch.nn.Sequential(
        MyL2(),
        torch.nn.Linear(128, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, out_dim),
        torch.nn.Tanh(),
    )


class Model(nn.Module):
    def __init__(self, num_user, args):
        super(Model, self).__init__()
        self.args = args
        # 编码器
        self.face_encoder_common = create_encoder()
        self.face_encoder_private = create_encoder()
        self.voice_encoder_common = create_encoder(in_dim=192)
        self.voice_encoder_private = create_encoder(in_dim=192)

        # 解码器
        self.face_decoder = create_decoder()
        self.voice_decoder = create_decoder(out_dim=192)

        # 损失
        self.id_classifier = torch.nn.Linear(128, num_user)
        self.fun_mse = torch.nn.MSELoss()
        self.fun_cross_entropy = torch.nn.CrossEntropyLoss()

    def face_encoder(self, f):
        return self.face_encoder_common(f)

    def voice_encoder(self, v):
        return self.voice_encoder_common(v)

    def forward(self, v, f, label, step):
        # 1.变成embeding
        f_emb_common = self.face_encoder_common(f)
        f_emb_private = self.face_encoder_private(f)
        v_emb_common = self.voice_encoder_common(v)
        v_emb_private = self.voice_encoder_private(v)

        # ============= id分类器
        id_logits_f = self.id_classifier(f_emb_common)
        id_logits_v = self.id_classifier(v_emb_common)
        loss_id = self.fun_cross_entropy(id_logits_f, label) + self.fun_cross_entropy(id_logits_v, label)

        # ============= 解纠缠
        loss_instance_level = horizontal_cosine_similarity(v_emb_common, v_emb_private) \
                              + horizontal_cosine_similarity(f_emb_common, f_emb_private)
        loss_mutual_level = frobenius_norm(f_emb_common @ f_emb_private.T) \
                            + frobenius_norm(v_emb_common @ v_emb_private.T)

        emb_and = f_emb_common * v_emb_common
        emb_or = f_emb_private + v_emb_private
        loss_and_or = frobenius_norm(emb_and @ emb_or.T)

        loss_orth = loss_instance_level + loss_mutual_level + loss_and_or

        # ============= 重构损失
        # 重构时必须要存在自己的私有的部分
        f_enc1 = self.face_decoder(f_emb_common + f_emb_private)
        f_enc2 = self.face_decoder(v_emb_common + f_emb_private)
        v_enc1 = self.voice_decoder(v_emb_common + v_emb_private)
        v_enc2 = self.voice_decoder(f_emb_common + v_emb_private)
        mse = self.fun_mse
        loss_rec = mse(f, f_enc1) + mse(f, f_enc2) + mse(v, v_enc1) + mse(v, v_enc2)

        # 最后的损失聚合
        loss = loss_id + self.args.ratio_rec * loss_rec + self.args.ratio_orth * loss_orth

        info = {
            "loss_ratio_rec": (self.args.ratio_rec * loss_rec).item() / loss.item(),
            "loss_ratio_orth": (self.args.ratio_orth * loss_orth).item() / loss.item(),
            # "f_emb_common": f_emb_common,
            # "f_emb_private": f_emb_private,
            # "v_emb_common": v_emb_common,
            # "v_emb_private": v_emb_private,
        }

        return loss, info


def horizontal_cosine_similarity(emb1, emb2):
    unit_emb1 = F.normalize(emb1)
    unit_emb2 = F.normalize(emb2)
    ans = torch.sum(unit_emb1 * unit_emb2, dim=1)
    return ans.sum()


def frobenius_norm(A):
    # if torch.sum(A ** 2).item() < 0:
    #     ipdb.set_trace()
    return torch.norm(A, p='fro')
    # return torch.sqrt(torch.sum(A ** 2))
    # return torch.sqrt(torch.abs(torch.sum(A ** 2)))
