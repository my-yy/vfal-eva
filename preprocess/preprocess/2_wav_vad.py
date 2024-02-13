# https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb#scrollTo=nd2zX-kJ84bb
import glob
import torch
import tqdm

SAMPLING_RATE = 16000

torch.set_num_threads(1)

USE_ONNX = False  # change this to True if you want to test onnx model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)


def fun(input_wav, output_wav):
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    wav = read_audio(input_wav, sampling_rate=SAMPLING_RATE)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    save_audio(output_wav, collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)


if __name__ == '__main__':
    all_wavs = glob.glob("./data/test/*.wav")
    for wav_path in tqdm.tqdm(all_wavs):
        fun(wav_path, wav_path + "_vad.wav")
