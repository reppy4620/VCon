import librosa
import torch
import webrtcvad
import struct
import numpy as np

from scipy.io.wavfile import write
from scipy.ndimage.morphology import binary_dilation


# load wav data from file
def load_wav(fn):
    wav, sr = librosa.load(fn)
    wav = trim_long_silences(wav)
    wav = 0.95 * librosa.util.normalize(wav)
    return torch.from_numpy(wav).float(), sr


# load raw wav and mel-spectrogram from fn by using official melgan's converter
def get_wav_mel(fn, to_mel=None):
    if to_mel is None:
        to_mel = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    wav, sr = librosa.load(fn)
    wav = trim_long_silences(wav)
    wav = 0.95 * librosa.util.normalize(wav)
    mel = to_mel(torch.tensor(wav, dtype=torch.float).unsqueeze(0)).squeeze(0)
    return wav, mel.cpu()


# save sample
def save_sample(file_path, audio, sr=22050):
    audio = (audio * 32768).astype("int16")
    write(file_path, sr, audio)


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width = 8
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length = 6
    sampling_rate = 16000
    int16_max = (2 ** 15) - 1
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]
