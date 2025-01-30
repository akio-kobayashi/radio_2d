import librosa
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')

mel_basis = {}
hann_window = {}

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mag_spectrogram(y, n_fft, hop_size, win_size, center=False):
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.abs(spec) + 1.e-9 

    return spec

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    spec = torch.abs(spec) + 1.e-9
    #spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec[:, 1:, :] # omit 0-dim

def get_mel_spectrogram(path, resample_rate=22050, num_mels=80, n_fft=1024, 
                        hop_size=256, win_size=1024, fmin=0, fmax=8000):
    # HiFi GAN configuration
    #resample_rate = 22050
    #num_mels=80
    #n_fft=1024
    #hop_size=256
    #win_size=1024
    #fmin=0
    #fmax=8000
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    waveform, sample_rate = torchaudio.load(path)
    resampled_waveform = F.resample(waveform.to(device), sample_rate, resample_rate, lowpass_filter_width=6)
    melspec = mel_spectrogram(resampled_waveform, n_fft, num_mels, resample_rate, hop_size, win_size, fmin, fmax)

    return melspec

def get_mag_spectrogram(path, resample_rate=22050, n_fft=1024, 
                        hop_size=256, win_size=1024):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    waveform, sample_rate = torchaudio.load(path)
    resampled_waveform = F.resample(waveform.to(device), sample_rate, resample_rate, lowpass_filter_width=6)
    spec = mag_spectrogram(resampled_waveform, n_fft, resample_rate, hop_size, win_size)

    return spec