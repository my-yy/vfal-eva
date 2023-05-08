import torch
from utils import pickle_util, sample_util, worker_util, vec_util
from torch.utils.data import DataLoader
from utils.config import face_emb_dict, voice_emb_dict
import numpy as np


# 无监督采样器，先选择一个movie，然后选择一个clip

def get_iter(batch_size, dataset_length):
    train_iter = DataLoader(DataSet(dataset_length), batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class DataSet(torch.utils.data.Dataset):

    def __init__(self, dataset_length):
        train_names = pickle_util.read_pickle("./dataset/info/train_valid_test_names.pkl")["train"]
        name2movies = pickle_util.read_pickle("./dataset/info/name2movies.pkl")
        train_movies = []
        for name in train_names:
            train_movies += name2movies[name]
        self.train_movies = train_movies
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        # find a movie
        movie_id = index % len(self.train_movies)
        movie_obj = self.train_movies[movie_id]

        # sample an image and a voice clip
        jpg_path = sample_util.random_element(movie_obj["jpgs"])
        wav_path = sample_util.random_element(movie_obj["wavs"])

        voice_tensor = voice_emb_dict[wav_path]
        face_tensor = face_emb_dict[jpg_path]
        return voice_tensor, face_tensor, torch.tensor(movie_id).long()
