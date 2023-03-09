#!/usr/bin/env python
from dataclasses import dataclass
import numpy as np
import ffmpeg
import tempfile
import scipy.io
import scipy.fft as fft
import math
import logging

logging.basicConfig()
logger = logging.getLogger("isolate")

def info(*args, **kwargs):
    logger.info(*args, **kwargs)

def verbose(*args, **kwargs):
    logger.debug(*args, **kwargs)

def setVerbose(enable):
    if enable:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

@dataclass
class AudioData:
    """
    Raw stereo audio data.  first and second are the raw soundwave data (2 channels)
    and shouldn't be thought of as having any sane units (other than comparing to maximum
    sound intensity allowed by the size of each element in their respective np.array)
    """
    sample_frequency: int
    first: np.array
    second: np.array

    def first_max_possible(self):
        return AudioData._max_possible(self.first)

    def second_max_possible(self):
        return AudioData._max_possible(self.second)

    def first_sample_times(self):
        return AudioData._get_sample_times(self.first)

    def second_sample_times(self):
        return AudioData._get_sample_times(self.second)

    @staticmethod
    def _max_possible(audio_data: np.array):
        if not audio_data:
            raise ValueError("Cannot get bit depth of None or empty array")

        return np.iinfo(audio_data.dtype).max

    @staticmethod
    def _get_sample_times(audio_data, sample_frequency):
        """
        Return the times, starting with 0 as the start of the audio signal, of each
        audio signal data point in [s]
        """
        audio_duration_s = len(audio_data)/float(sample_frequency)

        return np.linspace(0,audio_duration_s, len(audio_data))

@dataclass
class AnalysisSegment:
    """
    This contains all the information about the fourier analysis of a single segment of audio.

    sample_frequency: the sampling frequency of the original audio segment
    audio: the original audio segment
    time_points: the points in time, starting at 0, corresponding to each of the points in the audio segment
    frequencies: the centers of all of the frequency bins used in the discrete fourier.  This is ordered
        such that the frequencies are ascending and >= 0 (the negative frequencies are dropped and not
        included)
    fourier: The complex fourier value for each frequency in frequencies.
    """
    sample_frequency: int
    audio: np.array
    time_points: np.array
    frequencies: np.array
    fourier: np.array

def get_audio_data(filename, time_window, verbose_ffmpeg=False):
    """
    Use ffmpeg to load the video file specified by filename, and return the
    audio data from the time_window in the video.

    time_window is a tuple of strings that can be used with ffmpeg's atrim filter.
    time_window[0] is the start time, time_window[1] the end time (not duration).
    """
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

    ffmpeg_dag = ffmpeg_dag.output(tmp_audio_filename, f="wav").overwrite_output().run(quiet=not verbose_ffmpeg)
    audio_data = scipy.io.wavfile.read(tmp_audio_filename)
    max_audio_value = np.iinfo(audio_data[1].dtype).max

    return AudioData(sample_frequency=audio_data[0], first=audio_data[1][:,0]/float(max_audio_value), second=audio_data[1][:,1]/float(max_audio_value))

def get_freq_hz(data, sample_frequency):
    """
    Given sampled data (arbitrary intensity units) sampled at sample_frequency, return
    the fft frequency bins and the corresponding complex fft value at the center of
    each bin.

    For instance, if you provide 20000 data points sampled at 44000 hz, fft_freqs
    will contain the frequencies in the 0-44000/2 hz range that each fft_data point
    corresponds to.

    This drops the negative frequencies and only returns the >=0 frequency bins and
    corresponding fourier values.
    """
    fft_data = fft.fftshift(fft.fft(data))
    fft_freqs = fft.fftshift(fft.fftfreq(len(data), 1/float(sample_frequency)))


    return (fft_freqs[int(len(fft_freqs)/2):], fft_data[int(len(fft_data)/2):])

def segment(sample_count, number_of_segments):
    """
    Return an array of number_of_segment ranges that, together, split the range 0-sample_count
    into equal length segments (except the last segment, which might be shorter).

    These ranges are useful in combination with np.array.take.
    """
    segment_length = math.ceil(sample_count / number_of_segments)
    segment_ranges = []
    for seg_idx in range(number_of_segments):
        seg_start = seg_idx*segment_length
        seg_end = min(sample_count, (seg_idx+1)*segment_length)
        segment_ranges.append(range(seg_start, seg_end))
    return segment_ranges

def get_segments(audio_data: AudioData, segment_time: float):
    sample_times = audio_data.first_sample_times()
    sample_count = len(sample_times)
    full_time_span = max(sample_times) - min(sample_times)
    number_of_segments = math.ceil(full_time_span / args.segment_time)
    segment_ranges = segment(sample_count, number_of_segments)

    segment_data = []
    for segment_idxs in segment_ranges:
        (fft_sample_freqs, fft_values) = get_freq_hz(audio_data.first.take(segment_idxs), audio_data.sample_frequency)
        segment_sample_times = sample_times.take(segment_idxs)
        segment_audio_data = audio_data.first.take(segment_idxs)
        segment_data.append(
            AnalysisSegment(
            sample_frequency=audio_data.sample_frequency,
            audio=segment_audio_data,
            time_points=segment_sample_times,
            frequencies=fft_sample_freqs,
            fourier=fft_values
            )
        )

    return segment_data

def add_frequency_subplot(segment, row_count, row, sharex):
    freq_plot = plt.subplot(row_count, 1, row, sharex=sharex)
    freq_plot.plot(segment.frequencies, np.abs(segment.fourier))

    return freq_plot

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Takes video files and isolates interesting audio from them.")

    ap.add_argument("in_file", type=str, help="The input video file.")
    ap.add_argument("out_file", type=str, help="The output file to write the resulting plot to")
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
    segment_data = get_segments(audio_data, args.segment_time)

    plot_row_count = len(segment_data) + 1 # + 1 for the time domain plot
    time_plot = plt.subplot(plot_row_count,1,1)
    time_plot.plot(audio_data.first_sample_times(), audio_data.first)

    freq_x_axis = None
    for (idx, seg) in enumerate(segment_data):
        row = idx + 2 # +1 for time domain plot, and +1 because plots are 1 indexed
        freq_ax = add_frequency_subplot(seg, plot_row_count, row, sharex=freq_x_axis)
        if sharex is None:
            sharex = freq_ax

    plt.savefig(args.out_file, format="svg", transparent=True)

    if args.plot:
        plt.show()