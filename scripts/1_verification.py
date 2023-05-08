import numpy as np
from utils import pickle_util, sample_util, seed_util


def fun(names, count, name2wav, name2jpg):
    seed_util.set_seed(100)

    result = []
    jpg_set = set()
    wav_set = set()

    for i in range(count):
        label = np.random.randint(0, 2)
        if label == 1:
            # 同一个人
            name1 = sample_util.random_element(names)
            name2 = name1
        else:
            name1, name2 = sample_util.random_elements(names, 2)

        wav = sample_util.random_element(name2wav[name1])
        face = sample_util.random_element(name2jpg[name2])
        obj = [wav, face, label]
        wav_set.add(wav)
        jpg_set.add(face)
        result.append(obj)

    obj = {
        "wav_set": wav_set,
        "jpg_set": jpg_set,
        "list": result,
        "script": open(__file__).read(),
    }
    return obj


def is_male(gender):
    if gender in ['f', 0]:
        return False

    assert gender in ['m', 1]
    return True


def fun_g(names, count, name2wav, name2jpg, name2gender):
    seed_util.set_seed(100)
    result = []
    jpg_set = set()
    wav_set = set()

    male_names = []
    female_names = []

    for name in names:
        gender = name2gender[name]
        if is_male(gender):
            male_names.append(name)
        else:
            female_names.append(name)

    for i in range(count):
        label = np.random.randint(0, 2)
        if i % 2 == 0:
            the_names = female_names
        else:
            the_names = male_names

        if label == 1:
            # 同一个人
            name1 = sample_util.random_element(the_names)
            name2 = name1
        else:
            name1, name2 = sample_util.random_elements(the_names, 2)

        wav = sample_util.random_element(name2wav[name1])
        face = sample_util.random_element(name2jpg[name2])
        obj = [wav, face, label]
        wav_set.add(wav)
        jpg_set.add(face)
        result.append(obj)

    obj = {
        "wav_set": wav_set,
        "jpg_set": jpg_set,
        "list": result,
        "script": open(__file__).read(),
    }
    return obj


if __name__ == '__main__':
    name_list_dict = pickle_util.read_pickle("./dataset/info/train_valid_test_names.pkl")
    name2jpgs_wavs = pickle_util.read_pickle("./dataset/info/name2jpgs_wavs.pkl")
    name2gender = pickle_util.read_pickle("./dataset/info/name2gender.pkl")

    count = 10000
    pickle_util.save_pickle("./dataset/evals/valid_verification.pkl", fun(name_list_dict["valid"], count, name2jpgs_wavs["name2wavs"], name2jpgs_wavs["name2jpgs"]))
    pickle_util.save_pickle("./dataset/evals/test_verification.pkl", fun(name_list_dict["test"], count, name2jpgs_wavs["name2wavs"], name2jpgs_wavs["name2jpgs"]))
    pickle_util.save_pickle("./dataset/evals/test_verification_g.pkl", fun_g(name_list_dict["test"], count, name2jpgs_wavs["name2wavs"], name2jpgs_wavs["name2jpgs"], name2gender))
