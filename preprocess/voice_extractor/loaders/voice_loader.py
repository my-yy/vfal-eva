import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, all_wavs):
        self.all_wavs = all_wavs

    def __len__(self):
        return len(self.all_wavs)

    def __getitem__(self, index):
        return {
            "key": self.all_wavs[index],
            "data": torchaudio.load(self.all_wavs[index])[0]
        }


def collate_fn(item_list):
    data_list = [i['data'] for i in item_list]
    the_lengths = np.array([i.shape[-1] for i in data_list])
    max_len = np.max(the_lengths)
    len_ratio = the_lengths / max_len

    batch_size = len(item_list)
    output = torch.zeros([batch_size, max_len])
    for i in range(batch_size):
        cur = data_list[i]
        cur_len = data_list[i].shape[-1]
        output[i, :cur_len] = cur.squeeze()

    len_ratio = torch.FloatTensor(len_ratio)
    keys = [i['key'] for i in item_list]
    return output, len_ratio, keys


def get_loader(num_workers, batch_size, all_wavs):
    loader = DataLoader(Dataset(all_wavs),
                        num_workers=num_workers, batch_size=batch_size,
                        shuffle=False, pin_memory=True, collate_fn=collate_fn)
    return loader


if __name__ == "__main__":
    pass
