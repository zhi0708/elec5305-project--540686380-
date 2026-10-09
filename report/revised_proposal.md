# Sparse Impulsive-Noise Removal: Classical Detection and Repair versus Neural Speech Enhancement

Leona Wang | SID: 540686380 | GitHub: zhi0708

Repository: https://github.com/zhi0708/elec5305-project--540686380-

Project site: https://zhi0708.github.io/elec5305-project--540686380-/

## Project overview

Speech recordings may contain brief clicks, desk knocks or keyboard events while most of the waveform remains undisturbed. A general enhancement model processes the complete recording, whereas a specialised detector can attempt to alter only corrupted intervals. The revised research question is: for sparse short-duration impulsive interference, how does event-localised DSP compare with pretrained neural enhancement in event restoration, preservation of clean speech and computational cost? The secondary question asks how interference overlapping genuine speech transients changes this trade-off.

## Background and motivation

Short impulses differ from stationary background noise because they arrive abruptly and can overlap information-bearing speech onsets. Historical predictive reconstruction [1], stationary wavelet methods [2] and HHT detection [3] demonstrate several approaches to this problem. Recent sparse adaptive filtering [5] motivates making localisation and collateral alteration central to evaluation. This project does not reproduce that advanced algorithm. It instead compares a transparent, manageable DSP implementation with the official pretrained GTCRN [4], connecting classical audio features to contemporary lightweight speech enhancement. No neural architecture development or model training is required.

## Methodology

The implementation uses Python, NumPy, SciPy, Matplotlib and CPU PyTorch. A reference speech file already present in the repository is resampled from 8 to 16 kHz for compatibility with GTCRN. The first thirty seconds are split into six disjoint five-second clips, with two development clips and four test clips. This is a time-segment split within one recording, so generalisation to independent speakers remains outside the demonstrated scope.

Known events are inserted at controlled locations. Two synthetic Gaussian/Hann bursts last five and twenty milliseconds. Three short real-event crops come from ESC-50: mouse click, door knock and keyboard typing. The generator retains the clean signal, event waveform, corrupted signal and exact insertion mask. Three event-to-local-speech RMS ratios (-6, 0 and +6 dB) probe weak through stronger disturbances. Three acoustic locations approximate silence, steady voiced speech and a speech transient. These locations are selected using energy, harmonicity and spectral change, without claiming phoneme-level alignment. The experiment contains ninety development mixtures and 180 test mixtures.

The baseline compares short-time energy with an adaptive local median. The stronger detector combines energy rise, spectral flux and spectral bandwidth using eight-millisecond analysis windows and two-millisecond hops. Spectral flux captures abrupt change; bandwidth captures spectral spread. Both can also respond to legitimate speech changes, making clean-input false positives a necessary control. Thresholds are selected on development data, including a clean-preservation penalty, and frozen before test evaluation.

Detected intervals are repaired by attenuation or linear interpolation. Samples outside predicted intervals remain exactly unchanged, although a false detection can still damage genuinely clean speech. The neural comparator uses the unmodified official DNS3 GTCRN checkpoint, pinned to a documented source commit. It processes precisely the same noisy signals with the official STFT settings and matched output length. No ground-truth mask, test-specific gain adjustment or reference-guided alignment is given to either tested method. An oracle-mask interpolation control is labelled diagnostic rather than a practical competitor.

## Evaluation and completed outcomes

Event precision, recall, F1 and onset error are reported by strength and acoustic overlap. Sample-level localisation is also retained because a small partial overlap can count as an event hit. Restoration RMSE is computed inside the true event; collateral RMSE is measured outside it. Whole-recording SI-SDR and STOI serve as secondary checks. Clean recordings are independently passed through both systems to measure false detections and unnecessary alteration. Real-time factor includes algorithm execution on the same CPU, excluding model loading and warm-up. Offline throughput is distinguished from causal latency.

The core implementation and experiment have been completed. The STFT detector has measured event F1 0.362, versus 0.080 for energy detection, but still misses many events. GTCRN gives greater mean event error reduction, while the STFT local methods alter clean regions less. The study therefore exposes a trade-off rather than assuming a winner. Five result figures cover detection, collateral error, overlap examples, clean false alarms and computational cost. Waveforms, spectrograms, audio examples, per-case CSVs, exact masks and reproduction commands accompany the code.

## Timeline and next steps

The original proposal's literature and initial implementation stages are complete. The present milestone provides working code, real measurements and GitHub-ready documentation. The remaining revision stage should prioritise detector recall, broader speech/event coverage and reconstruction quality. Phoneme labelling or LPC reconstruction may be considered only if the core scope remains manageable. A GUI, new neural model and formal listening experiment are not required. The repository includes a static project website, full technical report and a short demonstration so the work is reviewable without running the environment.

## References

[1] Vaseghi and Rayner, IEE Proceedings I, 1990. [2] Nongpiur and Shpak, JASA, 2013. [3] Medina, Coelho and Zao, IEEE/ACM TASLP, 2021, DOI: 10.1109/TASLP.2021.3086131. [4] Rong et al., ICASSP, 2024, DOI: 10.1109/ICASSP48485.2024.10448310. [5] Zhou et al., Signal Processing 241, 2026, DOI: 10.1016/j.sigpro.2025.110390. Full bibliographic entries and dataset attributions appear in the accompanying technical report.
