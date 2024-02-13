import glob
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import time
from utils import pickle_util
import numpy as np


def get_emb(the_path):
    fpath = Path(the_path)
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    return embed.tolist()


def core(all_wavs):
    start_tiem = time.time()
    result_dict = {}
    counter = 0
    for wav_path in all_wavs:
        counter += 1
        try:
            emb = get_emb(wav_path)
        except Exception as e:
            print("error:", e, wav_path)
            continue

        result_dict[wav_path] = np.array(emb)

        if counter % 100 == 0:
            progress = counter / len(all_wavs)
            time_cost = (time.time() - start_tiem) / 3600.0
            total_time = time_cost / progress
            print("total_time:%.1fh;progress:%.3f" % (total_time, progress))
    return result_dict


if __name__ == '__main__':
    encoder = VoiceEncoder("cuda")
    all_wavs = glob.glob("wav/*.wav")
    result_dict = core(all_wavs)
    pickle_util.save_pickle("resemblyzer.pkl", result_dict)
