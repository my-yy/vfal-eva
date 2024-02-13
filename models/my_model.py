import torch
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, voice_size=192, face_size=512, mid_dim=256, embedding_size=128, shared=True):
        super(Encoder, self).__init__()

        if shared:
            face_rare = self.create_rare(mid_dim, embedding_size)
            voice_rare = face_rare
        else:
            face_rare = self.create_rare(mid_dim, embedding_size)
            voice_rare = self.create_rare(mid_dim, embedding_size)

        self.face_encoder = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(face_size, mid_dim),
            torch.nn.ReLU(),
            face_rare
        )

        self.voice_encoder = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(voice_size, mid_dim),
            torch.nn.ReLU(),
            voice_rare
        )

    def create_rare(self, mid_dim, embedding_size):
        return torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mid_dim, embedding_size),
        )

    def forward(self, voice_data, face_data):
        v_emb = self.voice_encoder(voice_data)
        f_emb = self.face_encoder(face_data)
        return v_emb, f_emb


class EncoderWithProjector(torch.nn.Module):
    def __init__(self, voice_size=192, face_size=512, mid_dim=256, embedding_size=128, shared=True):
        super(EncoderWithProjector, self).__init__()

        if shared:
            face_rare = self.create_rare(mid_dim, embedding_size)
            voice_rare = face_rare
        else:
            face_rare = self.create_rare(mid_dim, embedding_size)
            voice_rare = self.create_rare(mid_dim, embedding_size)

        self.face_encoder = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(face_size, mid_dim),
            torch.nn.ReLU(),
            face_rare
        )

        self.voice_encoder = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(voice_size, mid_dim),
            torch.nn.ReLU(),
            voice_rare
        )

        self.face_projector = torch.nn.Linear(embedding_size, embedding_size)
        self.voice_projector = torch.nn.Linear(embedding_size, embedding_size)

    def create_rare(self, mid_dim, embedding_size):
        return torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mid_dim, embedding_size),
        )

    def forward(self, voice_data, face_data, need_projector=False):
        v_emb = self.voice_encoder(voice_data)
        f_emb = self.face_encoder(face_data)

        if need_projector:
            pf = self.face_projector(f_emb)
            pv = self.voice_projector(v_emb)
            return v_emb, f_emb, pv, pf
        return v_emb, f_emb
