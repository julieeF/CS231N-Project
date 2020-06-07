#!/usr/bin/python

def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)





def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    
    #Masked Signal
    #masked_signal = _istft(sig_stft_db_masked, hop_length, win_length)
    #masked_spec = _amp_to_db(
    #    np.abs(_stft(masked_signal, n_fft, hop_length, win_length))
    #)
    
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    #masked_signal = _istft(sig_stft_amp, hop_length, win_length)
    return recovered_signal


def denoise_and_generate_spectogram(orig_wav_vector, start_time, end_time, noise_removal=0):
    sr = 44100
    audio_clip_duration = 3
    leading_and_trailing_zeros = 10000
    #orig_wav_vector, _sr = librosa.load(wav_file, sr = sr)
    #print(_sr)
    #sr=_sr
    #start_time = 5.53
    #end_time = 7.7
    a_clip_time = (end_time - start_time)
    print("Audio Clip Time ", (end_time - start_time))
    
    start_frame = math.floor(start_time * sr)
    end_frame = math.floor(end_time * sr)
    
    
    
    print("Start frame", start_frame)
    print("End frame", end_frame)
    total_size = audio_clip_duration * sr
    truncated_wav_vector = np.zeros(total_size)
    #pdb.set_trace()
    ts = end_frame - start_frame
    
    if noise_removal==0:
        noise_vec = pickle.load(open("/home/mandeep_stanford/cs231n_project/code/Mandeep/noise_vector.pkl", 'rb'))
        truncated_wav_vector = noise_vec[0:total_size]
    
    t_w_v_shape = truncated_wav_vector[0:ts].shape[0]
    o_w_v_shape = orig_wav_vector[start_frame:end_frame + 1].shape[0]
    
    if t_w_v_shape < o_w_v_shape:
        o_w_v_shape = t_w_v_shape
        truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:start_frame+o_w_v_shape]
    if t_w_v_shape > o_w_v_shape:
        o_w_v_shape = t_w_v_shape
        truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:start_frame+o_w_v_shape]
    if t_w_v_shape == o_w_v_shape:
        o_w_v_shape = t_w_v_shape
        truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:start_frame+o_w_v_shape]
    
    if noise_removal == 0:
        return truncated_wav_vector
    
    '''
    if (end_time - start_time) == 3.0 :
        #pdb.set_trace()
        if truncated_wav_vector[0:ts].shape[0] > orig_wav_vector[start_frame:end_frame + 1].shape[0]:
            truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame -1]
        if truncated_wav_vector[0:ts].shape[0] < orig_wav_vector[start_frame:end_frame + 1].shape[0]:
            if truncated_wav_vector[0:ts].shape[0] == orig_wav_vector[start_frame:end_frame].shape[0]:
                truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame]
            else:
                truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame-1]
                
        else:
            truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame]
    else:
        #if end_frame - start_frame + 1 == ts:
        if (end_time - start_time) > 3.0 and (end_time - start_time) < 3.001 :
            if truncated_wav_vector[0:ts].shape[0] < orig_wav_vector[start_frame:end_frame + 1].shape[0]:
                truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame]
            else:
                truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame + 1]
        if end_frame - start_frame == ts:
            truncated_wav_vector[0:ts] = orig_wav_vector[start_frame:end_frame]
        #truncated_wav_vector[0:ts] = orig_wav_vector
    '''    
    
    pre = np.zeros((leading_and_trailing_zeros))
    post = np.zeros((leading_and_trailing_zeros))
    truncated_wav_vector = np.concatenate([pre,truncated_wav_vector, post])
    
    
    #ADD NOISE
    noise_len = 4 # seconds
    noise = band_limited_noise(min_freq=1, max_freq = 30000, samples=len(truncated_wav_vector), samplerate=sr)*10
    print(noise.shape)
    noise_clip = noise[:sr*noise_len]
    audio_clip_band_limited = truncated_wav_vector+(1*noise)
    
    if noise_removal == 0:
        truncated_wav_vector = audio_clip_band_limited[leading_and_trailing_zeros+1:total_size+leading_and_trailing_zeros+1]
        return truncated_wav_vector
    
    if noise_removal == 1:
        output = removeNoise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip,    n_std_thresh=2,prop_decrease=0.9,verbose=False,visual=False)
        output = removeNoise(audio_clip=output, noise_clip=noise_clip,    n_std_thresh=2,prop_decrease=0.9,verbose=False,visual=False)
        output = removeNoise(audio_clip=output, noise_clip=noise_clip,    n_std_thresh=2,prop_decrease=0.9,verbose=False,visual=False)
        out = output[leading_and_trailing_zeros+1:total_size+leading_and_trailing_zeros+1]
    
    return out

def create_spectogram(input_vec, file_index, sess, noise_removal=0):
                truncated_wav_vector = input_vec
                X = librosa.stft(truncated_wav_vector)
                Xdb = librosa.amplitude_to_db(abs(X))
                plt.figure(figsize=(14, 5))
                #plt.figure()
                librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                plt.colorbar()
                
                #pdb.set_trace()
                #/home/mandeep_stanford/cs231n_project/code/pre-processed_data/images_denoised
                if noise_removal == 1:
                    imf = "/with_noise_cleanup/session{}/".format(sess)
                if noise_removal == 0:
                    imf = "/without_noise_cleanup/session{}/".format(sess)
                        
                plt.savefig(iemocap_pre_processed_data_path + imf + str(emotion) +"_"+ str(val) +"_"+ str(act) +"_"+ str(dom) + str(file_index) + '.png')
                plt.close('all')
                plt.close()
                print("FIG NUMBERS",plt.get_fignums())
                #librosa.cache.clear()
                del Xdb
                del X
                plot_spectrogram1(truncated_wav_vector, "--", noise_removal, file_index, sess)
                #time.sleep(1)
             
def plot_spectrogram1(signal, title, noise_removal, file_index, sess):
    fig, ax = plt.subplots(figsize=(4, 3))
    
    n_fft=2048
    win_length=2048
    hop_length=512
    #pdb.set_trace()
    #recovered_signal = _istft(signal, hop_length, win_length)
    signal1 = _amp_to_db(np.abs(_stft(signal, n_fft, hop_length, win_length)))#
    #librosa.display.specshow(signal1, sr=sr, x_axis='time', y_axis='hz')
    #recovered_spec = _amp_to_db(np.abs(_stft(recovered_signal, n_fft, hop_length, win_length)))
    x_max = 100*signal1.shape[0]/sr
    #cax = ax.matshow(
    cax = ax.imshow(    
        signal1,
        extent=[0.0, x_max, 0, 20000],
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-60,
        vmax=60,
    )
    #    vmin=-1 * np.max(np.abs(signal1)),
    #    vmax=np.max(np.abs(signal1)),
    #)
    #fig.colorbar(cax)
    #fig.colorbar().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    #ax.axes.get_colorbar().set_visible(False)
    #ax.set_title(title)
    fig.colorbar(cax).remove()
    plt.tight_layout()
    ax.set_xlim([0.0, x_max])
    ax.set_ylim([0, 20000])
    plt.show()
    if noise_removal == 1:
                    imf = "/cb_removed/with_noise_cleanup/session{}/".format(sess)
    if noise_removal == 0:
                    imf = "/cb_removed/without_noise_cleanup/session{}/".format(sess)
                        
    plt.savefig(iemocap_pre_processed_data_path + imf + str(emotion) +"_"+ str(val) +"_"+ str(act) +"_"+ str(dom) + str(file_index) + 'Same_ColorBAR.png') 
    plt.close('all')
    plt.close()
    print("FIG NUMBERS",plt.get_fignums())
    del signal
    del signal1
    #del recovered_spec

import cv2 
import re
from glob import glob
from moviepy.editor import *

import math

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite("v_frames/frame%d.png" % count, image) 
  
        count += 1
  
  
    # Calling the function 
    #FrameCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4") 

def clip_video(path, start_time,end_time, wav_file_name):
                    #import pdb
                    #pdb.set_trace()
                    video = VideoFileClip(path)
                    if end_time>video.duration:
                        end_time=video.duration
                    print("wav_file_name {},start time {},end time {}".format(wav_file_name, start_time, end_time))

                    video=video.subclip(
                        start_time, end_time)
                    video.write_videofile(
                        wav_file_name,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True)
                    del video
                    return wav_file_name
    
def extract_video_images(path, st, et, wav_file_name,iemocap_pre_processed_data_path,imf):
                #if et - st < 3.01:
                #    path_to_clip = path
                #else:
                import pdb
                #pdb.set_trace()
                base_name = os.path.basename(wav_file_name)
                base_name = base_name.split(".mp4")[0]
                
                num_images = 19
                path_to_clip = clip_video(path, st,et, wav_file_name)
                    
                cap = cv2.VideoCapture(path_to_clip)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(length)
                interval=length//num_images
                frameRate = cap.get(5)  # frame rate
                print(frameRate)
                x = 1
                while (cap.isOpened()):
                    frameId = cap.get(1)  # current frame number
                    ret, frame = cap.read()
                    if (ret != True):
                        break
                    if length%num_images==0:
                        length-=1
                    if (frameId<=(length-length%num_images)) and (frameId % math.floor(interval) == 0):
                        filename = iemocap_pre_processed_data_path + imf + str(base_name) + "_" + str(int(x)) + ".jpg";
                        x += 1
                        #pdb.set_trace()
                        print("Frame shape Before resize", frame.shape)
                        m_f_i = base_name.split("_")
                        m_f_l = m_f_i[4][-1]
                        m_f_r = m_f_i[6][0]
                        y1 = frame.shape[0]
                        w1 = frame.shape[1]
                        new_x = np.int(w1/2)
                        yy = np.int(y1/4)
                        if m_f_r == m_f_l:
                            #Get left part of image
                            frame = frame[yy:3*yy,0:new_x,:]
                        else:
                            frame = frame[yy:3*yy,new_x:w1,:]
                            #Get right part of image
                            print("")
                        print("After", frame.shape)
                        cv2.imwrite(filename, frame)
                        #######
                        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        #_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
                        #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        #cnt = contours[0]
                        #x1,y,w,h = cv2.boundingRect(cnt)
                        
                        #frame = frame[y:y+h,x1:x1+w]
                        ######
                        
                        #frame = cv2.resize(frame, (256,256))
                        #cv2.imwrite(filename, frame)
                        #import pdb
                        #pdb.set_trace()

                cap.release()
                print("Done!")
    
import sys
#print 'Number of arguments:', len(sys.argv), 'arguments.'
if sys.argv[1] == None:
    print("No argument given. Exiting...")
    exit()
    
wav_file = str(sys.argv[1])
print("File Name", wav_file)
import os
os.environ['LIBROSA_CACHE_DIR'] = '/home/mandeep_stanford/librosa_temp'
import pickle
import re
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import librosa
import math
import random
import pandas as pd
import IPython.display
import librosa.display
import pdb
from scipy.io import wavfile
import scipy.signal
ms.use('seaborn-muted')
#############%matplotlib inline
import time
from datetime import timedelta as td

iemocap_full_release_path = "/home/mandeep_stanford/cs231n_project/IEMOCAP_full_release/"

iemocap_pre_processed_data_path = "/home/mandeep_stanford/cs231n_project/code/pre-processed_data/4_EMOTIONS/"

data_dir = '/home/mandeep_stanford/cs231n_project/code/pre-processed_data/'
iemocap_dir = iemocap_full_release_path
labels_df_path = '{}df_iemocap.csv'.format(data_dir)
audio_vectors_path = '{}audio_vectors_'.format(data_dir)

labels_df = pd.read_csv(labels_df_path)

#print("hello")
extract_emotion = ['hap','sad','ang','neu','fea', 'exc', 'fru']
base_name = os.path.basename(wav_file)
base_name = [base_name]

sr = 44100
iterationn = str(sys.argv[2])
for sess in [iterationn]:  # using one session due to memory constraint, can replace [5] with range(1, 6)
    #audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
    
    #wav_file_path = '{}/Session{}/dialog/wav/'.format(iemocap_dir, sess)
    #pdb.set_trace()
    #orig_wav_files = os.listdir(wav_file_path)
    #for orig_wav_file in tqdm(orig_wav_files):
    for orig_wav_file in tqdm(base_name):
        print("Wav file is ", orig_wav_file)
        x = re.search("^Ses.*", orig_wav_file)
        if x == None:
            #print("Skiping file", orig_wav_file)
            continue
        try:
            #orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            #pdb.set_trace()
            orig_wav_file, file_format = orig_wav_file.split('.')
            #if orig_wav_file in audio_vectors.keys():
            #    orig_wav_vector = audio_vectors[orig_wav_file]
            #else:
            #    print("Key", orig_wav_file, "Not in pkl", '{}{}.pkl'.format(audio_vectors_path, sess))
            #    continue
            print("Working on file - ", orig_wav_file)
            #pdb.set_trace()
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                
                if emotion in extract_emotion:
                    print("Extracting images for ", emotion )
                else:
                    continue
                    
                #if truncated_wav_file_name in audio_vectors.keys():
                    #orig_wav_vector = audio_vectors[truncated_wav_file_name]
                #else:
                    #print("Key", truncated_wav_file_name, "Not in pkl", '{}{}.pkl'.format(audio_vectors_path, sess))
                #    continue
                
                #pdb.set_trace()
                
                ##Just create spectogram fro entie time
                #start_frame = math.floor(start_time * sr)
                #end_frame = math.floor(end_time * sr)
                #truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                #create_spectogram(truncated_wav_vector, "_Whole_Length", sess)
                
                total_time = end_time - start_time 
                num_slots = np.int(total_time/3.0)
                
                '''
                if total_time % 3.0 == 0:
                    num_slots = num_slots
                else:
                    num_slots = num_slots + 1
                if total_time < 3.01:
                    num_slots = 1
                '''    
                #if num_slots == 0:
                #    num_slots = 1
                print("Total 3 sec slots", num_slots)
                print(start_time, end_time, total_time)
                #pdb.set_trace()
                if total_time > 3.01:
                    print("+++")
                    for i in range(num_slots):
                        if i == 0:
                            s_t = start_time
                            e_t = s_t + 3.0
                        else:
                            s_t = s_t + 3.0
                            e_t = s_t + 3.0
                        if i == num_slots -1 :
                            e_t = end_time
                        print(i, s_t, e_t)
                        if (e_t - s_t) > 2.99 :
                            #truncated_wav_vector = denoise_and_generate_spectogram(orig_wav_vector, s_t, e_t, 0)
                            #create_spectogram(truncated_wav_vector, i, sess, 0)
                            #truncated_wav_vector = denoise_and_generate_spectogram(orig_wav_vector, s_t, e_t, 1)
                            #create_spectogram(truncated_wav_vector, i, sess, 1)
                            #create_video_frames()
                            imf = "/cb_removed/without_noise_cleanup/session{}/".format(sess)
                            file_path_name = iemocap_pre_processed_data_path + imf + str(emotion) +"_"+ str(val) +"_"+ str(act) +"_"+ str(dom) + "_" + str(i) + "_" + str(truncated_wav_file_name) + '_video_clip.mp4'
                            extract_video_images(wav_file, s_t, e_t, file_path_name,iemocap_pre_processed_data_path,imf)
                        else:
                            print("Discarding the last part which is less than 3 sec")
                else:
                    print("---")
                    #extract_video_images(path, s_t, e_t)
                    imf = "/cb_removed/without_noise_cleanup/session{}/".format(sess)
                    aa = 0
                    file_path_name = iemocap_pre_processed_data_path + imf + str(emotion) +"_"+ str(val) +"_"+ str(act) +"_"+ str(dom) + "_" + str(aa) + "_" + str(truncated_wav_file_name) + '_video_clip.mp4'
                    extract_video_images(wav_file, start_time, end_time, file_path_name,iemocap_pre_processed_data_path,imf)
                    #truncated_wav_vector = denoise_and_generate_spectogram(orig_wav_vector, start_time, end_time, 0)
                    #create_spectogram(truncated_wav_vector, 0, sess, 0)
                    #truncated_wav_vector = denoise_and_generate_spectogram(orig_wav_vector, start_time, end_time, 1)
                    #create_spectogram(truncated_wav_vector, 0, sess, 1)
                
                
                
                #start_frame = math.floor(start_time * sr)
                #end_frame = math.floor(end_time * sr)
                #print("In for loop",start_time, end_time)
                #pdb.set_trace()
                #librosa.cache.clear()
                #del truncated_wav_vector
                #del orig_wav_vector
        except:
            print('An exception occured for {}'.format(orig_wav_file))
            raise
        #librosa.cache.clear()
print("Done")






