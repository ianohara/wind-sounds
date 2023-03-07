#!/usr/bin/env python
from dataclasses import dataclass
import numpy as np
import ffmpeg
import tempfile
import scipy.io
import scipy.fft as fft

# TODO(imo): Figure out why the heck logging's setLevel is not working as expected
# and replace with proper logging
VERBOSE=False

def info(*args, **kwargs):
    print(*args, **kwargs)

def verbose(*args, **kwargs):
    if not VERBOSE:
        return
    print(*args, **kwargs)

@dataclass
class AudioData:
    sample_frequency: int
    left: np.array
    right: np.array

def get_audio_data(filename, time_window, verbose_ffmpeg):
    if not filename:
        raise ValueError("Must supply filename to load aduio from")
    
    tmp_audio_filename= tempfile.mktemp(suffix=".wav")

    ffmpeg_dag = ffmpeg.input(filename).audio

    if time_window and time_window[0]:
        verbose(f"atrim with start={time_window[0]}")
        ffmpeg_dag = ffmpeg_dag.filter("atrim", start=time_window[0])
    if time_window and time_window[1]:
        verbose(f"atrim with end={time_window[1]}")
        ffmpeg_dag = ffmpeg_dag.filter("atrim", end=time_window[1])

    ffmpeg_dag = ffmpeg_dag.filter("highpass", frequency=40)
    ffmpeg_dag = ffmpeg_dag.filter("lowpass", frequency=5000)

    ffmpeg_dag = ffmpeg_dag.output(tmp_audio_filename, f="wav").overwrite_output().run(quiet=not verbose_ffmpeg)
    audio_data = scipy.io.wavfile.read(tmp_audio_filename)

    return AudioData(sample_frequency=audio_data[0], left=audio_data[1][:,0], right=audio_data[1][:,1])

def get_freq_hz(data, sample_frequency):
    """
    Given sampled data (arbitrary intensity units) sampled at sample_frequency, return
    the fft frequency bins and the corresponding complex fft value at the center of
    each bin.

    For instance, if you provide 20000 data points sampled at 44000 hz, fft_freqs
    will contain the frequencies in the 0-44000 hz range that each fft_data point
    corresponds to.
    """
    fft_data = fft.fftshift(fft.fft(data))
    fft_freqs = fft.fftshift(fft.fftfreq(len(data), 1/float(sample_frequency)))

    return (fft_freqs, fft_data)

def get_sample_times(audio: AudioData):
    audio_duration_s = len(audio.left)/float(audio.sample_frequency)
    
    return np.linspace(0,audio_duration_s, len(audio.left))

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Takes video files and isolates interesting audio from them.")

    ap.add_argument("in_file", type=str, help="The input video file.")
    ap.add_argument("--start", type=str, help="Time in video to start analysis in mm:ss format.")
    ap.add_argument("--end", type=str, help="Time in video to end analysis in mm:ss format")
    ap.add_argument("--verbose", action="store_true", help="Turn on verbose log output")
    ap.add_argument("--verbose_ffmpeg", action="store_true", help="Turn on full ffmpeg output")
    ap.add_argument("--plot", action="store_true", help="Plot useful intermediate and final data")
    args = ap.parse_args()

    if args.verbose:
        VERBOSE = True

    audio_data = get_audio_data(args.in_file, time_window=(args.start, args.end), verbose_ffmpeg=args.verbose_ffmpeg)

    (fft_sample_freqs, fft_values) = get_freq_hz(audio_data.left, audio_data.sample_frequency)

    time_plot = plt.subplot(2,1,1)
    freq_plot = plt.subplot(2,1,2)
    time_plot.plot(get_sample_times(audio_data), audio_data.left)
    freq_plot.plot(fft_sample_freqs, np.abs(fft_values), "")
    plt.show()