# Detection and Suppression of Short Impulsive Noise in Speech Recordings

## Background

Speech recordings are widely used in online meetings, lectures, interviews, voice messages, and video narration. However, recordings are often affected by sudden noises such as desk knocks, mouse clicks, microphone contact, dropped objects, or closing doors.

Although these noises are usually short, they can have much larger amplitudes than normal speech and may briefly mask important spoken information. They can also sound particularly distracting when the recording is played through headphones.

## The Problem

Short impulsive noise is different from continuous background noise such as fan or air-conditioning noise.

Continuous noise normally has relatively stable characteristics, while impulsive noise:

- appears suddenly;
- lasts for a short time;
- may have a high amplitude;
- may contain energy across a wide frequency range;
- occurs at unpredictable positions.

Conventional filters designed for stationary background noise may therefore be ineffective. A simple amplitude threshold may also incorrectly classify loud speech or plosive sounds as noise.

## Project Aim

This project aims to develop a Python audio signal processing program that can detect and suppress short impulsive noise in speech recordings while preserving the quality and intelligibility of the original speech.

A time-domain energy-threshold method will first be developed as a baseline. Time-frequency features will then be introduced to improve detection and reduce incorrect classifications.

## Research Question

> Can time-frequency features improve the detection and suppression of short impulsive noise compared with a time-domain energy-threshold method?

## Proposed Approach

The project will gradually investigate the following processing stages:

1. Load and preprocess a speech recording.
2. Divide the audio into short frames.
3. Calculate short-time energy.
4. analyse the signal using the Fast Fourier Transform (FFT) and Short-Time Fourier Transform (STFT).
5. Extract temporal and spectral features.
6. Detect frames containing impulsive noise.
7. Suppress detected noise using local attenuation or interpolation.
8. Reconstruct and evaluate the processed audio.

## Experimental Design

Clean speech recordings will be obtained from the Open Speech Repository. Short impulsive noises will initially be generated and inserted into the speech at known time positions.

This produces three signals:

- the original clean speech;
- speech containing impulsive noise;
- the processed speech.

Because the original speech and noise positions are known, the detection and suppression results can be evaluated objectively.

Later experiments may also use real impulsive sounds from public environmental audio datasets.
