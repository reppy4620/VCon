import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa

from librosa.display import specshow
from resemblyzer import preprocess_wav, VoiceEncoder
from nn import VCModel
from utils import get_config, get_wav_mel


def model_test():
    params = get_config('./config.json')
    inp = torch.randn(8, 128, 80)
    model = VCModel(params)


def load_wav_file_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    wav, mel = get_wav_mel(fn, pt=False)

    plt.plot(wav)
    plt.show()

    specshow(np.log1p(mel.T))
    plt.show()

    wav = librosa.feature.inverse.mel_to_audio(mel.T)
    plt.plot(wav)
    plt.show()


def embed_test():
    fn = 'D:/dataset/VCTK-Corpus/wav48/p225/p225_001.wav'
    wav = preprocess_wav(fn)
    ve = VoiceEncoder()
    embed = ve.embed_utterance(wav)
    print(embed)


if __name__ == '__main__':
    load_wav_file_test()
