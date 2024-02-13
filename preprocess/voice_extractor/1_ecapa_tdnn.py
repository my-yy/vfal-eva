import torch
from utils import model_util, pickle_util
from .loaders import voice_loader
import time
import glob


def generate_emb_dict(wav_list, batch_size=16):
    loader = voice_loader.get_loader(4, batch_size, wav_list)
    the_dict = {}
    counter = 0
    start_time = time.time()
    for data, lens, keys in loader:
        try:
            core_step(data, lens, model, keys, the_dict)
        except Exception as e:
            print("error:", e)
            continue

        counter += 1
        if counter % 10 == 0:
            processed = len(the_dict)
            progress = processed / len(loader.dataset)
            time_cost = time.time() - start_time
            total_time = time_cost / progress / 3600.0
            print("progress:", progress, "total_time:", total_time)
    return the_dict


def core_step(wavs, lens, model, keys, the_dict):
    with torch.no_grad():
        feats = fun_compute_features(wavs.cuda())
        feats = fun_mean_var_norm(feats, lens)
        embedding = model(feats, lens)
        embedding_npy = embedding.detach().cpu().numpy().squeeze()
        # (batch,192)
    for key, emb in zip(keys, embedding_npy):
        the_dict[key] = emb


def get_ecapa_model():
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    n_mels = 80
    channels = [1024, 1024, 1024, 1024, 3072]
    kernel_sizes = [5, 3, 3, 3, 1]
    dilations = [1, 2, 3, 4, 1]
    attention_channels = 128
    lin_neurons = 192
    model = ECAPA_TDNN(input_size=n_mels, channels=channels,
                       kernel_sizes=kernel_sizes, dilations=dilations,
                       attention_channels=attention_channels,
                       lin_neurons=lin_neurons
                       )
    # print(model)
    return model


def get_fun_compute_features():
    from speechbrain.lobes.features import Fbank

    n_mels = 80
    left_frames = 0
    right_frames = 0
    deltas = False
    compute_features = Fbank(n_mels=n_mels, left_frames=left_frames, right_frames=right_frames, deltas=deltas)
    return compute_features


def get_fun_norm():
    from speechbrain.processing.features import InputNormalization
    return InputNormalization(norm_type="sentence", std_norm=False)


if __name__ == "__main__":
    # 1.get model
    model = get_ecapa_model().cuda()
    pkl_path = "ecapa_acc0.9854.pkl"
    model_util.load_model(pkl_path, model)
    model.eval()

    fun_compute_features = get_fun_compute_features().cuda()
    fun_mean_var_norm = get_fun_norm().cuda()

    # 2.get all wav files
    wav_list = glob.glob("/your_path/*.wav")

    the_dict = generate_emb_dict(wav_list)
    pickle_util.save_pickle("voice_emb.pkl", the_dict)
