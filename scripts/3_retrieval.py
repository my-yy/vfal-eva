import sys
from utils import pickle_util
from utils import seed_util, sample_util
import random


def one_gropu(names, name2wav, name2jpg, jpg_set, wav_set, max_person):
    if max_person is not None:
        names = sample_util.random_elements(names, max_person)
        print("重新采样人数:", max_person)

    result = []

    # 每个人10个声音，10个人脸
    wav_size = 10
    jpg_size = 10

    for name in names:
        wavs = name2wav[name]
        random.shuffle(wavs)
        wavs = wavs[0:wav_size]
        # 选10个声音

        jpgs = name2jpg[name]
        random.shuffle(jpgs)
        jpgs = jpgs[0:jpg_size]
        # 选10个人脸

        for wav, jpg in zip(wavs, jpgs):
            tup = (wav, jpg, name)
            result.append(tup)
            jpg_set.add(jpg)
            wav_set.add(wav)
    return result


def fun(names, name2wav, name2jpg, group=3, max_person=None):
    seed_util.set_seed(100)

    jpg_set = set()
    wav_set = set()

    # 设置4组
    result = []
    for i in range(group):
        arr = one_gropu(names, name2wav, name2jpg, jpg_set, wav_set, max_person)
        result.append(arr)

    obj = {
        "wav_set": wav_set,
        "jpg_set": jpg_set,
        "retrieval_lists": result
    }

    return obj


if __name__ == "__main__":
    name_list_dict = pickle_util.read_pickle("./dataset/info/train_valid_test_names.pkl")
    name2jpgs_wavs = pickle_util.read_pickle("./dataset/info/name2jpgs_wavs.pkl")
    name2gender = pickle_util.read_pickle("./dataset/info/name2gender.pkl")
    name2wav = name2jpgs_wavs["name2wavs"]
    name2jpg = name2jpgs_wavs["name2jpgs"]
    pickle_util.save_pickle("./dataset/evals/test_retrieval.pkl", fun(name_list_dict["test"], name2wav, name2jpg))
