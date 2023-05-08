import torch
from models.my_model import Encoder


class Decoder(torch.nn.Module):
    def __init__(self, voice_size=192, face_size=512, embedding_size=128, shared=True):
        super(Decoder, self).__init__()
        # 128->Drop-fc256-Relu-fc256-Relu-xxx

        # 共享层
        mid_dim = 256

        def create_rare():
            return torch.nn.Sequential(
                torch.nn.Dropout(),
                torch.nn.Linear(embedding_size, mid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mid_dim, mid_dim),
                torch.nn.ReLU(),
            )

        face_rare = create_rare()
        if shared:
            voice_rare = face_rare
        else:
            voice_rare = create_rare()

        self.dec_face = torch.nn.Sequential(
            face_rare,
            torch.nn.Linear(mid_dim, voice_size),
        )
        self.dec_voice = torch.nn.Sequential(
            voice_rare,
            torch.nn.Linear(mid_dim, face_size)
        )

    def forward(self, v_emb, f_emb):
        f_out = self.dec_voice(v_emb)
        v_out = self.dec_face(f_emb)
        return v_out, f_out


class CAE(torch.nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder(shared=True)
        self.face_encoder = self.encoder.face_encoder
        self.voice_encoder = self.encoder.voice_encoder

        self.decoder = Decoder(shared=False)
        self.fun_loss_mse = torch.nn.MSELoss()

    def forward(self, voice_data, face_data, only_emb=False):
        v_emb = self.voice_encoder(voice_data)
        f_emb = self.face_encoder(face_data)
        if only_emb:
            return v_emb, f_emb
        v_out, f_out = self.decoder(v_emb, f_emb)

        fun_loss_mse = self.fun_loss_mse
        loss_dec = fun_loss_mse(voice_data, v_out) + fun_loss_mse(face_data, f_out)
        loss_emb = fun_loss_mse(v_emb, f_emb)
        return loss_emb, loss_dec
