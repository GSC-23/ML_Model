## Dependencies
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks

## Constants 
FRAME_SIZE = 1024
HOP_SIZE = 512
red = "r"
blue = "b"
green = "g"


def amplitude_envolpe(signal,frame_size, hop_len):
    amplitude_envolpe = []
    plt.figure(figsize=(25,10))
    for i in range (0, len(signal), hop_len):
        current_frame_amplitude_envolpe = max(signal[i:i+ frame_size])
        amplitude_envolpe.append(current_frame_amplitude_envolpe)

    amp_signal = np.array(amplitude_envolpe)
    frames = range(0, amp_signal.size)
    t = librosa.frames_to_time(frames=frames)
    # plt.subplot(5,1,1)
    plt.title("Amplitude Plot")
    plt.plot(t, amp_signal, color= "r")



def Frequency_Magnitude(signal, title, sr, f_ratio):
    plt.figure(figsize=(25,10))

    ft = np.fft.fft(signal)
    mag_spec = np.abs(ft)
    freq_bin = np.linspace(0, sr, len(mag_spec))
    num_freq_bins = int(len(freq_bin)* f_ratio)
    # plt.subplot(5,1,2)
    plt.plot(freq_bin[:num_freq_bins], mag_spec[:num_freq_bins])
    plt.xlabel("frequency (Hz)")
    plt.title(title)
    

def Spectrogram(signal,sr,hop_length, yaxis="linera"):
    plt.figure(figsize=(25,15))
    signal = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=hop_length)
    signal= np.abs(signal)
    signal = librosa.power_to_db(signal)
    # plt.subplot(5,1,3)
    plt.title("Spectogram")
    librosa.display.specshow(signal, sr=sr, hop_length=hop_length, x_axis="time", y_axis=yaxis)
    plt.colorbar(format="%+2.f")


def pitch_plot(signal,sr,plot = True):
    T = 1/sr
    N = len(signal)
    t = N / sr  
    spec = np.fft.fft(signal)[0:int(N/2)]/N
    spec[1:] = 2 * spec[1:]
    p_spec = np.abs(spec)
    f = sr * np.arange((N/2)) / N
    auto = sm.tsa.acf(signal, nlags = 2000)
    peaks = find_peaks(auto)[0]
    lag = peaks[0]
    pitch = sr / lag
    number_peaks = len(peaks)
    if plot ==True:
        fig, ax = plt.subplots()
        plt.plot(f[0:5000], p_spec[0:5000], linewidth =2)
        plt.xlabel("freq")
        plt.ylabel("amp")
    
    return pitch,number_peaks



def mel_Spectogram(signal,sr):
    filter_banks = librosa.filters.mel(n_fft=2048, sr = 22050, n_mels=10)
    mel_spectogram = librosa.feature.melspectrogram(y=signal,n_mels=90,sr=sr,n_fft=2048,hop_length=512)
    log_mel_specto = librosa.power_to_db(mel_spectogram)
    plt.figure(figsize=(25,10))
    plt.title("Mel Spectogram")
    librosa.display.specshow(log_mel_specto, x_axis="time", y_axis="mel", sr=sr)
    plt.colorbar(format="%+2.f")


def MFCC(signal,sr):
    mfcc = librosa.feature.mfcc(y=signal,n_mfcc=13, sr=sr)
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    comp_mfcc = np.concatenate((mfcc,delta_mfcc,delta2_mfcc))
    plt.figure(figsize=(25,10))
    plt.title("MFCC Spectogram")
    librosa.display.specshow(comp_mfcc, x_axis="time", sr=sr)
    plt.colorbar(format="%+2.f")


def spectral_centroid(signal,sr):
    signal = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)[0]
    frame = range(len(signal))
    t1 = librosa.frames_to_time(frames=frame)
    plt.figure(figsize=(25,10))
    plt.title(" Spectral Centroid")
    plt.plot(t1, signal, color = green)



def band_energy_plot(signal,sr):
    signal = librosa.stft(y=signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

    def spliting_freq(spectogram,split_freq, sr):
        frequency_range = sr /2
        freq_delta_per_bin = frequency_range/ spectogram.shape[0]
        slip_freq_bin = np.floor(split_freq/ freq_delta_per_bin)
        
        return int(slip_freq_bin)
    
    def cal_bandRation(spectrogram, split_freq, sr):
        split_freq_bin= spliting_freq(spectrogram, split_freq, sr)
        power_spec = np.abs(spectrogram) ** 2
        power_spec = power_spec.T

        band_energy_ratio = []
    
        for frequencies in power_spec:
            sum_power_low_frequencies = np.sum(frequencies[:split_freq_bin])
            sum_power_high_frequencies = np.sum(frequencies[split_freq_bin:])
            band_crrent_frame = sum_power_low_frequencies /sum_power_high_frequencies
            band_energy_ratio.append(band_crrent_frame)

        return np.array(band_energy_ratio)
    
    signal = cal_bandRation(signal,2000,sr)

    plt.figure(figsize=(25,10))
    frames1 = range(len(signal))
    t2 = librosa.frames_to_time(frames=frames1, hop_length=HOP_SIZE)
    plt.title("Band Energy ratio Plot")
    plt.plot(t2, signal, color ="b")
    plt.plot(t2, signal, color ="r")





def main():
    ## loading audio filess
    path = input("Enter the Path of Audio File:- ")
    print("Plotting Graphs.........")
    signal, sr = librosa.load(path) # sr is sampling rate here
    sample_duration = 1 /sr 
    duration = sample_duration * len(signal)

    
    
    amplitude_envolpe(signal, frame_size=FRAME_SIZE,hop_len=HOP_SIZE)
    Frequency_Magnitude(signal=signal,title="Frequency plot", sr=sr,f_ratio=2000)
    Spectrogram(signal=signal, sr=sr, hop_length=HOP_SIZE,yaxis="log")
    pitch, peaks = pitch_plot(signal=signal,sr=sr,plot=False)
    mel_Spectogram(signal=signal,sr=sr)
    MFCC(signal=signal, sr=sr)
    spectral_centroid(signal=signal, sr=sr)
    band_energy_plot(signal=signal, sr=sr)

    
    print(pitch)
    print(peaks)
    
    
    plt.show()



if __name__ == "__main__":
    main()
