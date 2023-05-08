import ipdb
import numpy as np
import torch
from utils import pickle_util, sample_util, worker_util, vec_util
from torch.utils.data import DataLoader


def get_iter(batch_size, full_length, name2face_emb, name2voice_emb, movie2label):
    train_iter = DataLoader(DataSet(name2face_emb, name2voice_emb, full_length, movie2label),
                            batch_size=batch_size, shuffle=False, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class DataSet(torch.utils.data.Dataset):

    def __init__(self, name2face_emb, name2voice_emb, full_length, movie2label):
        self.train_movie_list = list(movie2label.keys())
        self.full_length = full_length
        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb
        self.movie2label = movie2label

        # create movie2jpg, movie2wav dict
        self.movie2jpg_path = {}
        self.movie2wav_path = {}
        name2movies = pickle_util.read_pickle("./dataset/info/name2movies.pkl")
        for name, movie_list in name2movies.items():
            for movie_obj in movie_list:
                movie_name = movie_obj['person'] + "/" + movie_obj["movie"]
                # A.J._Buckley/J9lHsKG98U8
                self.movie2wav_path[movie_name] = movie_obj["wavs"]
                self.movie2jpg_path[movie_name] = movie_obj["jpgs"]

    def __len__(self):
        return self.full_length

    def __getitem__(self, index):
        movie = sample_util.random_element(self.train_movie_list)
        label = self.movie2label[movie]

        img = sample_util.random_element(self.movie2jpg_path[movie])
        wav = sample_util.random_element(self.movie2wav_path[movie])
        wav, img = self.to_tensor([wav, img])

        return wav, img, torch.LongTensor([label])

    def to_tensor(self, path_arr):
        ans = []
        for path in path_arr:
            if ".wav" in path:
                emb = self.name2voice_emb[path]
            else:
                emb = self.name2face_emb[path]
            emb = torch.FloatTensor(emb)
            ans.append(emb)
        return ans
