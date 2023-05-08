import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class IR_CMPC(nn.Module):
    def __init__(self, temperature, delta, ka, R):
        super(IR_CMPC, self).__init__()
        self.inst_inst_criterion = InstInstCLR(temperature=temperature).cuda()
        self.inst_proto_criterion = InstProtoCLR(temperature=temperature).cuda()
        self.delta = delta
        self.ka = ka
        self.R = R

    def forward(self, audio_emb, frame_emb, audio_cluster_result, frame_cluster_result, video_index):
        bs = audio_emb.size(0)

        loss_v2f_ii = self.inst_inst_criterion(audio_emb, frame_emb)
        loss_f2v_ii = self.inst_inst_criterion(frame_emb, audio_emb)

        loss_v2f_ip = self.inst_proto_criterion(audio_emb, frame_cluster_result, video_index)
        loss_f2v_ip = self.inst_proto_criterion(frame_emb, audio_cluster_result, video_index)

        # print(loss_v2f_ii.shape, loss_f2v_ii.shape, loss_v2f_ip.shape, loss_f2v_ip.shape)
        w = torch.ones(bs).cuda()
        if frame_cluster_result is not None:
            features_audio = F.normalize(audio_emb, dim=1)
            features_frame = F.normalize(frame_emb, dim=1)
            audio_frame_matrix = torch.mm(features_audio, features_frame.transpose(0, 1)).detach().cpu().numpy()

            inst2cluster_matrix = np.zeros((bs, bs))

            for i in range(self.R):
                inst2cluster_voice = audio_cluster_result['inst2cluster'][i][video_index]
                inst2cluster_face = frame_cluster_result['inst2cluster'][i][video_index]
                inst2cluster_matrix += (
                    torch.mm(
                        audio_cluster_result['centroids'][i][inst2cluster_voice],
                        frame_cluster_result['centroids'][i][inst2cluster_face].transpose(0, 1),
                    )
                        .cpu()
                        .numpy()
                )

            rho = audio_frame_matrix - inst2cluster_matrix
            rho = rho.diagonal()

            sorted_rho = np.sort(rho)
            argsort_rho = np.argsort(rho)

            mu = rho.mean() + self.delta * rho.std()
            sigma = rho.std() * self.ka ** (1 / 2)

            y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (sorted_rho - mu)) ** 2)
            y = y.cumsum()
            y /= y[-1]

            w = y[np.argsort(argsort_rho)]
            w = torch.tensor(w).cuda()

        loss_v2f = torch.sum(w * (loss_v2f_ii + loss_v2f_ip)) / torch.sum(w)
        loss_f2v = torch.sum(w * (loss_f2v_ii + loss_f2v_ip)) / torch.sum(w)

        loss = loss_v2f + loss_f2v

        return loss


class InstInstCLR(nn.Module):
    def __init__(self, temperature):
        super(InstInstCLR, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, anchor, pos):
        batch_size = anchor.size(0)

        anchor = F.normalize(anchor)  # (bs, out_dim)

        pos = F.normalize(pos)  # (bs, out_dim)

        similarity_matrix = torch.matmul(anchor, pos.T)  # (bs, bs)
        # mask the main diagonal for positives
        mask = torch.eye(batch_size, dtype=torch.bool)  # (bs, bs)

        assert similarity_matrix.shape == mask.shape

        # select and combine multiple positives
        positives = similarity_matrix[mask].view(batch_size, -1)  # (bs, 1)

        # select only the negatives the negatives

        negatives = similarity_matrix[~mask].view(batch_size, -1)  # (bs, bs-1)
        # combine pos and neg
        logits = torch.cat([positives, negatives], dim=1)  # (bs, bs)

        labels = torch.zeros(batch_size, dtype=torch.long).cuda()  # (bs)

        logits = logits / self.temperature

        loss = self.ce(logits, labels)
        return loss


class InstProtoCLR(nn.Module):
    def __init__(self, temperature):
        super(InstProtoCLR, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, anchor, cluster_result=None, index=None):
        batch_size = anchor.size(0)

        loss_proto = torch.zeros(batch_size).cuda()
        if cluster_result is None:
            return loss_proto

        anchor = F.normalize(anchor)  # (bs, out_dim)

        for n, (inst2cluster, prototypes, density) in enumerate(zip(cluster_result['inst2cluster'], cluster_result['centroids'], cluster_result['density'])):
            prototypes = F.normalize(prototypes)

            # get positive prototypes

            pos_proto_id = inst2cluster[index]
            pos_prototypes = prototypes[pos_proto_id]

            # embed()
            proto_similarity_matrix = torch.matmul(anchor, pos_prototypes.T)  # [bs, dim]x[dim, bs]->[bs, bs]
            # mask the main diagonal for positives
            mask = torch.eye(batch_size, dtype=torch.bool)  # (bs, bs)
            assert proto_similarity_matrix.shape == mask.shape

            # select and combine multiple positives
            proto_positives = proto_similarity_matrix[mask].view(batch_size, -1)  # (bs, 1)
            # select only the negatives the negatives
            proto_negatives = proto_similarity_matrix[~mask].view(batch_size, -1)  # (bs, bs-1)
            # combine pos and neg
            proto_logits = torch.cat([proto_positives, proto_negatives], dim=1)  # (bs, bs)
            # targets for prototype assignment
            proto_labels = torch.zeros(batch_size, dtype=torch.long).cuda()  # (bs)

            # scaling temperatures for the selected prototypes
            temp_proto = density[pos_proto_id]
            proto_logits /= temp_proto
            loss_proto += self.ce(proto_logits, proto_labels)

            # average loss across all sets of prototypes
            loss_proto /= len(cluster_result)

        return loss_proto
