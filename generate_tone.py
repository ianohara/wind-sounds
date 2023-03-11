#!/usr/bin/env python
import scipy.io
import numpy as np

def get_tone(frequency, duration, sample_rate):
    """
    Get numpy tone data.  This is cribbed from the scipy.io.wavfile.write docs.
    """
    time_points = np.linspace(0, duration, int(duration * sample_rate))
    amplitude = np.iinfo(np.int16).max
    tone_data = amplitude * np.sin(2 * np.pi * frequency * time_points)

    return tone_data

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a wav file with a single sine wave tone.")

    ap.add_argument("frequency", type=float, help="Frequency of tone in [Hz]")
    ap.add_argument("out_file", type=str, help="File to output tone to.  .wav will be appended if not present")
    ap.add_argument("--duration", type=float, default=5.0, help="Duration of tone in wav file")
    ap.add_argument("--sample_rate", type=int, default=44100, help="Sample rate to use in wav file")

    args = ap.parse_args()

    if args.frequency <= 0.0:
        raise ValueError("Frequency must be > 0")

    if args.duration <= 0.0:
        raise ValueError("Tone duration must be > 0")

    tone_data = get_tone(args.frequency, args.duration, args.sample_rate)

    filename = args.out_file
    if not filename.endswith(".wav"):
        filename += ".wav"

    print(f"Writing tone to '{filename}'")
    scipy.io.wavfile.write(filename, args.sample_rate, tone_data.astype(np.int16))