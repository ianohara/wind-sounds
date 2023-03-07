#!/usr/bin/env python

import numpy as np

def tone(sample_rate, freq, duration_s, mag=10):
    """
    Generates a single frequency tone sine wave with frequency freq (in hz) sampled at sample_rate
    (samples per second) for duration_s (seconds) with magnitude mag (arbitrary units)

    Returns (np.array[...sample times...], np.array[...magnitude at each sample time...])
    """
    total_samples = sample_rate*duration_s
    sample_points = np.linspace(0, duration_s, total_samples)
    return (sample_points, mag*np.sin(sample_points * (freq * 360 * np.pi / 180)))