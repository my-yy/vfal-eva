import ipdb
import torch
from utils import pickle_util, sample_util, worker_util
from torch.utils.data import DataLoader
from utils.config import face_emb_dict, voice_emb_dict


# 用于LAFV，最后适用于triplet loss
def get_iter(batch_size, full_length):
    train_iter = DataLoader(DataSet(full_length), batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class DataSet(torch.utils.data.Dataset):

    def __init__(self, dataset_length):
        train_names = pickle_util.read_pickle("./dataset/info/train_valid_test_names.pkl")["train"]
        train_names.sort()
        name2id = {train_names[i]: i for i in range(len(train_names))}
        self.train_names = train_names
        self.name2id = name2id

        self.name2movies = pickle_util.read_pickle("./dataset/info/name2movies.pkl")
        self.name2gender = pickle_util.read_pickle("./dataset/info/name2gender.pkl")
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        name1, name2 = sample_util.random_elements(self.train_names, 2)
        v1, f1 = self.load_one_person(name1)
        v2, f2 = self.load_one_person(name2)
        return v1, f1, v2, f2

    def load_one_person(self, name1):
        movie_obj = sample_util.random_element(self.name2movies[name1])
        wav_name = sample_util.random_element(movie_obj["wavs"])
        jpg_name = sample_util.random_element(movie_obj["jpgs"])
        voice_tensor = torch.FloatTensor(voice_emb_dict[wav_name])
        face_tensor = torch.FloatTensor(face_emb_dict[jpg_name])
        return voice_tensor, face_tensor
