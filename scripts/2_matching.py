import ipdb
import numpy as np
import ipdb
import os
from utils import pickle_util, sample_util, seed_util


def fun(names, iter_num, name2wav, name2pic):
    seed_util.set_seed(100)
    data = []
    wav_set = set()
    jpg_set = set()
    for i in range(iter_num):
        # 1.选两个人
        name1, name2 = np.random.choice(names, 2, replace=False)

        voice1 = sample_util.random_element(name2wav[name1])
        voice2 = sample_util.random_element(name2wav[name2])

        face1 = sample_util.random_element(name2pic[name1])
        face2 = sample_util.random_element(name2pic[name2])

        # 统计出现的内容
        assert name1 != name2
        assert voice1 != voice2
        assert face1 != face2
        obj = (name1, voice1, face1, name2, voice2, face2)
        data.append(obj)

        wav_set.add(voice1)
        wav_set.add(voice2)
        jpg_set.add(face1)
        jpg_set.add(face2)

    obj = {
        "wav_set": wav_set,
        "jpg_set": jpg_set,
        "match_list": data
    }
    return obj


def fun_g(names, iter_num, name2wav, name2pic):
    seed_util.set_seed(100)

    male_names = [name for name in names if is_male(name2gender[name])]
    female_names = [name for name in names if not is_male(name2gender[name])]

    data = []
    wav_set = set()
    jpg_set = set()
    for i in range(iter_num):
        # 1.选两个人
        if i % 2 == 0:
            name1, name2 = np.random.choice(female_names, 2, replace=False)
        else:
            name1, name2 = np.random.choice(male_names, 2, replace=False)

        voice1 = sample_util.random_element(name2wav[name1])
        voice2 = sample_util.random_element(name2wav[name2])

        face1 = sample_util.random_element(name2pic[name1])
        face2 = sample_util.random_element(name2pic[name2])

        # 统计出现的内容
        assert name1 != name2
        assert voice1 != voice2
        assert face1 != face2
        obj = (name1, voice1, face1, name2, voice2, face2)
        data.append(obj)

        wav_set.add(voice1)
        wav_set.add(voice2)
        jpg_set.add(face1)
        jpg_set.add(face2)
        # ipdb.set_trace()

    obj = {
        "wav_set": wav_set,
        "jpg_set": jpg_set,
        "match_list": data
    }
    return obj


def is_male(gender):
    if gender in ['f', 0]:
        return False

    assert gender in ['m', 1]
    return True


def fun_1n(names, iter_num, name2wav, name2pic, N):
    data = []
    wav_set = set()
    jpg_set = set()
    for i in range(iter_num):
        # 1.选择N个人
        name_list = sample_util.random_elements(names, N)
        # 2.选择样本（对应位置上，人员是一样的）
        voices = [sample_util.random_element(name2wav[name]) for name in name_list]
        faces = [sample_util.random_element(name2pic[name]) for name in name_list]
        data.append([voices, faces])
        for v in voices:
            wav_set.add(v)

        for f in faces:
            jpg_set.add(f)

    obj = {
        "wav_set": wav_set,
        "jpg_set": jpg_set,
        "match_list": data
    }
    return obj


if __name__ == "__main__":
    name_list_dict = pickle_util.read_pickle("./dataset/info/train_valid_test_names.pkl")
    name2jpgs_wavs = pickle_util.read_pickle("./dataset/info/name2jpgs_wavs.pkl")
    name2gender = pickle_util.read_pickle("./dataset/info/name2gender.pkl")
    name2wav = name2jpgs_wavs["name2wavs"]
    name2jpg = name2jpgs_wavs["name2jpgs"]

    # 基本测试
    count = 10000
    pickle_util.save_pickle("./dataset/evals/test_matching.pkl", fun(name_list_dict["test"], count, name2wav, name2jpg))
    pickle_util.save_pickle("./dataset/evals/test_matching_g.pkl", fun_g(name_list_dict["test"], count, name2wav, name2jpg))
    pickle_util.save_pickle("./dataset/evals/test_matching_10.pkl", fun_1n(name_list_dict["test"], count, name2wav, name2jpg, 10))
