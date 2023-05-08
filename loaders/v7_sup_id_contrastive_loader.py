import ipdb
import torch
from utils import pickle_util, sample_util, worker_util
from torch.utils.data import DataLoader
from utils.config import face_emb_dict, voice_emb_dict


# 用于配合pytorch的默认contrastive loss

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
        is_same_person = index % 2
        if is_same_person == 1:
            name1 = sample_util.random_element(self.train_names)
            name2 = name1
        else:
            name1, name2 = sample_util.random_elements(self.train_names, 2)

        jpg_path = sample_util.random_element(sample_util.random_element(self.name2movies[name1])["jpgs"])
        wav_path = sample_util.random_element(sample_util.random_element(self.name2movies[name2])["wavs"])

        voice_tensor = voice_emb_dict[wav_path]
        face_tensor = face_emb_dict[jpg_path]

        is_same_person = torch.as_tensor(is_same_person).long()
        return voice_tensor, face_tensor, is_same_person
