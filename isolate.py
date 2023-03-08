#!/usr/bin/env python
from dataclasses import dataclass
import numpy as np
import ffmpeg
import tempfile
import scipy.io
import scipy.fft as fft
import math

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

@dataclass
class AnalysisSegment:
    sample_frequency: int
    audio: np.array
    time_points: np.array
    frequencies: np.array
    fourier: np.array

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
    max_audio_value = np.iinfo(audio_data[1].dtype).max

    return AudioData(sample_frequency=audio_data[0], left=audio_data[1][:,0]/float(max_audio_value), right=audio_data[1][:,1]/float(max_audio_value))

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


    return (fft_freqs[int(len(fft_freqs)/2):], fft_data[int(len(fft_data)/2):])

def get_sample_times(audio: AudioData):
    audio_duration_s = len(audio.left)/float(audio.sample_frequency)
    
    return np.linspace(0,audio_duration_s, len(audio.left))

def segment(sample_count, number_of_segments):
    segment_length = math.ceil(sample_count / number_of_segments)
    segment_ranges = []
    for seg_idx in range(number_of_segments):
        seg_start = seg_idx*segment_length
        seg_end = min(sample_count, (seg_idx+1)*segment_length)
        segment_ranges.append(range(seg_start, seg_end))
    return segment_ranges

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
    ap.add_argument("--segment_time", type=float, default=0.5, help="Chop up the input into durations this long in [s], and plot each fourier separately")
    args = ap.parse_args()

    if args.verbose:
        VERBOSE = True

    if args.segment_time <= 0:
        raise ValueError("Segment time must be > 0s")

    audio_data = get_audio_data(args.in_file, time_window=(args.start, args.end), verbose_ffmpeg=args.verbose_ffmpeg)

    sample_times = get_sample_times(audio_data)
    sample_count = len(sample_times)
    full_time_span = max(sample_times) - min(sample_times)
    number_of_segments = math.ceil(full_time_span / args.segment_time)
    segment_ranges = segment(sample_count, number_of_segments)

    segment_data = []
    for segment_idxs in segment_ranges:
        (fft_sample_freqs, fft_values) = get_freq_hz(audio_data.left.take(segment_idxs), audio_data.sample_frequency)
        segment_sample_times = sample_times.take(segment_idxs)
        segment_audio_data = audio_data.left.take(segment_idxs)
        segment_data.append(
            AnalysisSegment(
            sample_frequency=audio_data.sample_frequency,
            audio=segment_audio_data,
            time_points=segment_sample_times,
            frequencies=fft_sample_freqs,
            fourier=fft_values
            )
        )

    plot_row_count = len(segment_data) + 1 # + 1 for the time domain plot
    time_plot = plt.subplot(plot_row_count,1,1)
    time_plot.plot(sample_times, audio_data.left)
    freq_x_axis = None
    for (idx, seg) in enumerate(segment_data):
        row = idx + 2 # +1 for time domain plot, and +1 because plots are 1 indexed
        freq_plot = plt.subplot(plot_row_count, 1, row, sharex=freq_x_axis)
        freq_plot.plot(seg.frequencies, np.abs(seg.fourier))
        if not freq_x_axis:
            freq_x_axis = freq_plot 
    plt.show()