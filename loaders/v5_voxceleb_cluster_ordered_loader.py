import ipdb
import numpy as np
import torch
from utils import pickle_util, worker_util
from utils.path_util import look_up

from torch.utils.data import DataLoader
import collections


def extract_embeddings(name2face_emb, name2voice_emb, model):
    face_iter = get_ordered_iter(512, name2face_emb, name2voice_emb, is_face=True)
    movies, emb_face = extract_embeddings_core(face_iter, model.face_encoder)

    voice_iter = get_ordered_iter(512, name2face_emb, name2voice_emb, is_face=False)
    movies2, emb_voice = extract_embeddings_core(voice_iter, model.voice_encoder)

    assert len(movies2) == len(movies)
    final_emb = np.hstack([emb_voice, emb_face])
    return movies, final_emb, emb_voice, emb_face


def extract_embeddings_core(ordered_iter, encoder):
    # 1.extract embedding
    encoder.eval()
    the_dict = collections.defaultdict(list)
    for data in ordered_iter:
        with torch.no_grad():
            batch_movie, tensor = data
            # ipdb.set_trace()
            batch_emb = encoder(tensor.cuda()).detach().cpu().numpy()
            for emb, movie in zip(batch_emb, batch_movie):
                the_dict[movie].append(emb)
    encoder.train()

    # 2. merge embedding by video
    final_dict = {}
    for key, arr in the_dict.items():
        # arr:[batch,emb]
        final_dict[key] = np.mean(arr, axis=0)

    # 3.sort
    videos = list(final_dict.keys())
    videos.sort()
    emb_array = np.array([final_dict[key] for key in videos])

    return videos, emb_array


def get_ordered_iter(batch_size, name2face_emb, name2voice_emb, is_face):
    train_iter = DataLoader(OrderedDataSet(is_face, name2face_emb, name2voice_emb),
                            batch_size=batch_size, shuffle=False,
                            pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class OrderedDataSet(torch.utils.data.Dataset):

    def __init__(self, is_face, name2face_emb, name2voice_emb):
        name2movies = pickle_util.read_pickle(look_up("./dataset/info/name2movies.pkl"))
        train_names = pickle_util.read_pickle(look_up("./dataset/info/train_valid_test_names.pkl"))["train"]

        # 3.数据
        all_jpgs = []
        all_wavs = []

        for name in train_names:
            movies = name2movies[name]
            for movie in movies:
                # movie.keys: ['jpgs', 'wavs', 'movie', 'person']
                wavs = movie["wavs"]
                jpgs = movie["jpgs"]
                # wav: 'A.J._Buckley/J9lHsKG98U8/00025.wav'
                # jpg: 'A.J._Buckley/J9lHsKG98U8/0010225.jpg'
                for short_path in jpgs:
                    movie_name = "/".join(short_path.split("/")[0:2])
                    # A.J._Buckley/J9lHsKG98U8
                    all_jpgs.append([movie_name, short_path])
                for short_path in wavs:
                    movie_name = "/".join(short_path.split("/")[0:2])
                    all_wavs.append([movie_name, short_path])

        if is_face:
            self.data = all_jpgs
            self.name2emb = name2face_emb
        else:
            self.data = all_wavs
            self.name2emb = name2voice_emb

        self.is_face = is_face

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        movie, short_path = self.data[index]
        tensor = torch.FloatTensor(self.name2emb[short_path])
        if self.is_face:
            assert len(tensor) == 512
        # ipdb.set_trace()
        return movie, tensor
