"""Transient detection using short peaks relative to their immediate context."""
import numpy as np
from scipy import ndimage

from src.dsp import EPS, features, score_mask


def transient_features(x):
    base = features(x)
    centers = base['centers']
    hop = base['hop']
    # Pre-emphasis highlights short, broadband changes. A 2 ms peak envelope
    # retains evidence that is diluted by the 8 ms analysis window.
    diff = np.diff(x, prepend=x[0])
    envelope = ndimage.uniform_filter1d(diff**2, size=hop, mode='reflect')[centers]
    short = np.maximum(envelope, 0.)
    floor = max(np.percentile(short, 60) * .03, EPS)
    # Compare with both sides, excluding the centre 20 ms. Using the stronger
    # side reduces the response to a sustained speech onset.
    padded = np.pad(short, (30, 30), mode='edge')
    window = np.lib.stride_tricks.sliding_window_view(padded, 61)
    flank = np.maximum(np.mean(window[:, :21], axis=1),
                       np.mean(window[:, -21:], axis=1))
    contrast = np.maximum(10 * np.log10((short + floor) / (flank + floor)), 0.)
    level = np.clip(10 * np.log10((short + floor) / (np.mean(short) + floor)), -30., 0.)
    return dict(base, contrast=contrast, level=level)


def transient_score(feat, contrast_weight, floor_weight):
    # Bound flux so a normalised spectrum change at very low energy cannot
    # dominate the score on its own.
    return (contrast_weight * feat['contrast'] / 6
            + np.maximum(feat['rise'], 0.) / 9
            + feat['bandwidth'] / 1700
            + np.minimum(feat['flux'] * 3, 1.)
            + floor_weight * feat['level'] / 12)


def detect_transient(x, settings, feat=None):
    """Return a sample mask from a development-selected score and threshold."""
    feat = transient_features(x) if feat is None else feat
    score = transient_score(feat, settings['contrast_weight'], settings['floor_weight'])
    return score_mask(len(x), feat, score, settings['threshold'])
