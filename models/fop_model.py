# Fusion and Orthogonal Projection for Improved Face-Voice Association,ICASSP,2022
# FROM https://github.com/msaadsaeed/FOP
import torch
import torch.nn as nn

class FopModel(nn.Module):
    def __init__(self):
        super(FopModel, self).__init__()
        self.encoder_v = EmbedBranch(192, 128)
        self.encoder_f = EmbedBranch(512, 128)
        self.gated_fusion = GatedFusion(embed_dim_in=128, mid_att_dim=128, emb_dim_out=128)
        self.tanh_mode = False

    def forward(self, voice_input, face_input):
        tmp_v_emb = self.encoder_v(voice_input)
        tmp_f_emb = self.encoder_f(face_input)
        return self.gated_fusion(tmp_v_emb, tmp_f_emb)

    def face_encoder(self, data):
        return torch.tanh(self.encoder_f(data))

    def voice_encoder(self, data):
        return torch.tanh(self.encoder_v(data))


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x)
        return x


class GatedFusion(nn.Module):
    def __init__(self, embed_dim_in, mid_att_dim, emb_dim_out):
        super(GatedFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim_in * 2, mid_att_dim),
            nn.BatchNorm1d(mid_att_dim),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(mid_att_dim, emb_dim_out)
        )

    def forward(self, voice_input, face_input):
        concat = torch.cat((face_input, voice_input), dim=1)
        attention_out = torch.sigmoid(self.attention(concat))
        face_emb = torch.tanh(face_input)
        voice_emb = torch.tanh(voice_input)
        fused_emb = face_emb * attention_out + (1.0 - attention_out) * voice_emb
        return voice_emb, face_emb, fused_emb




# class FopModel(nn.Module):
#     def __init__(self, encoder):
#         super(FopModel, self).__init__()
#         self.encoder = encoder
#         self.gated_fusion = GatedFusion(embed_dim_in=128, mid_att_dim=128, emb_dim_out=128)
#
#     def forward(self, voice_input, face_input):
#         v_emb0, f_emb0 = self.encoder(voice_input, face_input)
#         return self.gated_fusion(v_emb0, f_emb0)
#
#     def face_encoder(self, data):
#         return torch.tanh(self.encoder.face_encoder(data))
#
#     def voice_encoder(self, data):
#         return torch.tanh(self.encoder.voice_encoder(data))

