"""Build the English report, README and static website from measured results."""
import csv,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TITLE='Sparse Impulsive-Noise Removal: Classical Detection and Repair versus Neural Speech Enhancement'
REPO='https://github.com/zhi0708/elec5305-project--540686380-'
SITE='https://zhi0708.github.io/elec5305-project--540686380-/'
METHODS=['noisy','energy_interpolate','stft_attenuate','stft_interpolate','gtcrn']
NAMES={'noisy':'Unprocessed','energy_interpolate':'Energy + interpolation','stft_attenuate':'STFT + attenuation',
       'stft_interpolate':'STFT + interpolation','gtcrn':'GTCRN (DNS3 pretrained)'}


def main():
    s=json.loads((ROOT/'results/summary.json').read_text())
    res=s['restoration'];det=s['detection'];clean=s['clean_preservation']
    report_dir=ROOT/'report';report_dir.mkdir(exist_ok=True)
    table='| Method | Event RMSE | Clean-region RMSE | Event error reduction (dB) | SI-SDR (dB) | STOI | Median RTF |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in METHODS:
        r=res[m]
        table+=f"| {NAMES[m]} | {r['event_rmse_mean']:.6f} | {r['clean_rmse_mean']:.6f} | {r['event_error_reduction_db_mean']:.2f} | {r['si_sdr_db_mean']:.2f} | {r['stoi_mean']:.3f} | {r['rtf_median']:.5f} |\n"
    dt='| Detector | Precision | Recall | F1 | TP / FP / FN | Mean matched onset error (ms) |\n|---|---:|---:|---:|---|---:|\n'
    for m in ['energy','stft']:
        r=det[m];dt+=f"| {m.upper()} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']} / {r['fp']} / {r['fn']} | {r['onset_error_ms_mean']:.2f} |\n"
    cp='| Method | False events / 20 s | False events / minute | Mean clean RMSE | Modified clean samples |\n|---|---:|---:|---:|---:|\n'
    for m in ['energy_interpolate','stft_interpolate','gtcrn']:
        r=clean[m];fp=r['false_events'];cp+=f"| {NAMES[m]} | {fp if fp is not None else 'N/A'} | {fp*3 if fp is not None else 'N/A'} | {r['clean_rmse_mean']:.8f} | {r['changed_fraction_mean']*100:.4f}% |\n"
    detection_rows=list(csv.DictReader((ROOT/'results/detection_cases.csv').open()))
    grouped=[]
    for dimension in ['condition','event','level_db']:
        for value in sorted({r[dimension] for r in detection_rows}):
            for detector in ['energy','stft']:
                part=[r for r in detection_rows if r[dimension]==value and r['detector']==detector]
                counts={k:sum(int(r[k]) for r in part) for k in ['tp','fp','fn']}
                tp,fp,fn=(counts[k] for k in ['tp','fp','fn'])
                grouped.append(dict(dimension=dimension,value=value,detector=detector,**counts,
                                    precision=tp/(tp+fp) if tp+fp else 0,
                                    recall=tp/(tp+fn) if tp+fn else 0,
                                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
    with (ROOT/'results/detection_groups.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,list(grouped[0]),lineterminator="\n");w.writeheader();w.writerows(grouped)
    overlap='| Acoustic location | Detector | Precision | Recall | F1 |\n|---|---|---:|---:|---:|\n'
    for r in grouped:
        if r['dimension']=='condition':overlap+=f"| {r['value'].replace('_',' ')} | {r['detector']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |\n"
    refs='''1. S. V. Vaseghi and P. J. W. Rayner, "Detection and suppression of impulsive noise in speech communication systems," IEE Proceedings I, vol. 137, no. 1, pp. 38-46, 1990.
2. R. C. Nongpiur and D. J. Shpak, "Impulse-noise suppression in speech using the stationary wavelet transform," JASA, vol. 133, no. 2, pp. 866-879, 2013. https://doi.org/10.1121/1.4773264
3. C. Medina, R. Coelho and L. Zao, "Impulsive noise detection for speech enhancement in HHT domain," IEEE/ACM TASLP, vol. 29, pp. 2244-2253, 2021. https://doi.org/10.1109/TASLP.2021.3086131
4. X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu and J. Lu, "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources," ICASSP 2024. https://doi.org/10.1109/ICASSP48485.2024.10448310. Official implementation: https://github.com/Xiaobin-Rong/gtcrn
5. L. Zhou, H. Liu, L. Gan, Y. Zhou, M. Niedzwiecki and T.-K. Truong, "A novel sparse adaptive filter for suppressing impulsive disturbance in audio signals," Signal Processing, vol. 241, article 110390, April 2026. https://doi.org/10.1016/j.sigpro.2025.110390. Author code/data: https://github.com/minikatty/Lq_JSLMP
6. K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM Multimedia, 2015. Dataset and attribution: https://github.com/karolpiczak/ESC-50
7. C. H. Taal et al., "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech," IEEE TASLP, 2011. STOI implementation: https://github.com/mpariente/pystoi
'''
    report=f'''# {TITLE}

Leona Wang | SID 540686380 | GitHub: zhi0708 | ELEC5305

Repository: {REPO}

Project site: {SITE}

## Abstract

Sparse impulsive interference presents a preservation problem: removing a short event can damage considerably more speech than the event itself. This project compares event-localised DSP with an official pretrained GTCRN enhancer using controlled insertions into a reference speech recording. An energy detector and a three-feature STFT detector drive local attenuation or linear interpolation. The experiment uses 90 development mixtures and 180 held-out mixtures, covering five event templates, three interference levels and three acoustic locations. Exact insertion intervals are available for every case. The STFT detector achieves event F1 {det['stft']['f1']:.3f}, compared with {det['energy']['f1']:.3f} for energy detection, while retaining low recall. GTCRN reduces mean event reconstruction error more strongly ({res['gtcrn']['event_error_reduction_db_mean']:.2f} dB), but introduces greater clean-region alteration than the STFT local methods. These results support a restoration-versus-preservation trade-off, not a universal ranking of DSP and neural enhancement.

## 1. Research question and motivation

The research question is: for sparse short-duration impulsive interference in speech, how does event-localised DSP detection and repair compare with a modern pretrained neural speech enhancer in event restoration, speech preservation and computational cost? A secondary question concerns interference overlapping an existing speech transient.

Predictive reconstruction [1], stationary-wavelet suppression [2] and HHT detection [3] provide historical context for specialised impulsive-noise processing. The current project implements neither the wavelet nor HHT algorithms. Recent sparse adaptive filtering [5] motivates examining disturbance sparsity and localisation: its author repository discusses sparse optimisation and the consequences of altering undisturbed samples. It is used as motivation rather than reproduced or claimed as a baseline. GTCRN [4] supplies an accessible lightweight learned comparator with official pretrained weights. The project contribution is the controlled experiment, transparent student-developed detector and separation of event error from collateral alteration.

## 2. Data and controlled corruption

The existing repository contains one 33.62-second, mono 8 kHz speech recording. It is OSR_us_000_0010_8k.wav from the Open Speech Repository, verified against a byte-identical public mirror. Its source and SHA-256 are recorded in data/speech_manifest.json. It is treated as a controlled reference rather than a certified noise-free corpus. Only the first 30 seconds are used. They form six non-overlapping five-second clips: clips 1-2 are development data and clips 3-6 are test data. The split separates time segments, not speakers. Upsampling to 16 kHz meets GTCRN's input requirements but does not restore missing bandwidth. All speech clips share one fixed scaling that limits the original recording peak to 0.15; mixtures and outputs are never independently normalised.

Two synthetic templates are seeded Gaussian bursts with Hann envelopes, lasting 5 and 20 ms. Three real templates are cropped from ESC-50 [6]: door knock ({s['events'][2]['duration_ms']:.3f} ms), mouse click ({s['events'][3]['duration_ms']:.3f} ms) and keyboard event ({s['events'][4]['duration_ms']:.3f} ms). A 2 ms RMS envelope locates the strongest peak; the crop expands while above 10% of peak, with a 75 ms limit on each side, 2 ms guards and short boundary fades. Source files, crop times, individual attribution and file hashes are provided. There is only one extracted template per real class, so this tests three examples rather than environmental-event diversity.

Each template is added separately at a pause, a steady voiced candidate and a speech-transient candidate. Locations are derived from the reference waveform: minimum smoothed energy identifies the pause; autocorrelation with an energy gate and low-flux preference identifies steady voiced speech; large positive spectral flux identifies the transient. These are acoustic proxies, not human-verified vowels or phoneme-aligned /p/, /t/ and /k/ releases. The reference is used for constructing controlled cases and evaluating outputs, never supplied to a tested detector.

Event levels are -6, 0 and +6 dB. Level means event RMS relative to the RMS of a local speech window extending 50 ms on each side of the inserted interval. In near silence, that reference has a floor at 10% of whole-clip RMS. An insertion mask covers the exact half-open interval [start, end), including its fades. The generator retains clean speech, added event, corrupted speech and the mask. All case intervals are saved in the manifest; scripts/export_case.py materialises any case. Float arrays are used for measurements to avoid clipping or quantisation. The grid contains 2 x 5 x 3 x 3 = 90 development mixtures and 4 x 5 x 3 x 3 = 180 test mixtures. These are controlled variants of a small reference, not 270 independent utterances.

## 3. Detection and local restoration

The analysis uses 8 ms windows and 2 ms hops, with a Hann window for spectra and a 256-point real FFT. Short frames favour transient timing; zero padding samples the spectrum more densely but does not increase physical frequency resolution. Short windows improve temporal localisation at the expense of frequency discrimination. Energy describes local intensity, spectral flux captures positive frame-to-frame spectral change, and bandwidth describes spectral spread. Suppression is evaluated separately from speech preservation; LPC reconstruction remains a possible extension.

The energy baseline compares a local energy rise against a threshold. Its rise is 10 log10((frame_energy + floor)/(local_median_energy + floor)). The median spans 101 frames, approximately 202 ms; the floor is 1% of the recording's 60th-percentile frame energy. This adaptive baseline can detect sudden increases but can mistake normal speech onsets for interference. Its runtime excludes all FFT calculations.

The STFT method combines three features: positive energy rise, positive spectral flux and spectral bandwidth. Flux is the sum of squared positive changes between consecutive L2-normalised magnitude spectra. Its robust z-score uses the recording median and a scale equal to the maximum of 1.4826 times the median absolute deviation, 0.1 times its standard deviation, and a numerical floor. Bandwidth is the square root of the power-weighted frequency variance. The score is max(energy_rise,0)/6 + max(flux_z,0)/3 + bandwidth/1500. The coefficients are fixed heuristic scales, not learned from test labels. Their benefit is interpretability; no ablation is available to isolate each feature's contribution.

Candidate frames exceeding the threshold are mapped to sample intervals, including the frame's half-window support. Candidates longer than 150 ms are rejected. The same duration rule applies to both detectors. The development search considers seven energy thresholds and eight STFT thresholds. It maximises event F1 with false events on the two clean development clips counted 45 times, giving each clean clip the same weight as its 45 noisy variants. This makes preservation an explicit selection preference. The frozen thresholds are {s['thresholds']['energy']} dB energy rise and STFT score {s['thresholds']['stft']}. The test set is not used to choose either threshold.

Attenuation applies a gain of 0.2 inside the predicted event, with up to 2 ms entry and exit ramps. Interpolation replaces the predicted samples with a straight line connecting the preceding and following samples. At file boundaries, the missing neighbour is zero. Both methods preserve the input exactly outside the predicted mask. This guarantee concerns the predicted interval; false detections can still alter genuinely clean speech. An oracle interpolation control uses the true mask solely to diagnose reconstruction limitations and is excluded from detector performance and runtime comparison.

## 4. Neural comparison and evaluation protocol

GTCRN uses the official DNS3 checkpoint, unmodified model architecture and a pinned source commit. No model is trained or fine-tuned. The adapter follows the official 512-sample FFT, 256-sample hop and square-root Hann analysis/synthesis convention. Modern PyTorch complex STFT tensors are converted into the model's real/imaginary input representation and converted back for iSTFT. Output length is explicitly matched to input length. There is no case-specific gain correction, delay fitting or reference-guided enhancement.

Event detection uses one-to-one matching of predicted and true intervals with positive overlap. TP, FP, FN, precision, recall and F1 are pooled over test cases. A partial overlap counts as an event hit, so sample-level metrics are also stored to reveal poor interval coverage. Onset error is the absolute start-time difference for matched events only; it does not summarise misses. Reports separate pause, steady voiced and transient locations, and retain event type and strength in per-case CSVs.

Event RMSE compares output with the clean reference inside the true insertion interval. Collateral RMSE uses every sample outside that true interval, thereby counting false-positive repair and any neural alteration. Event error reduction is 10 log10(noisy event MSE / output event MSE); positive values mean improvement. Whole-file SI-SDR and STOI [7] are secondary checks, not substitutes for local measurements. Table values are arithmetic means across cases; median RTF is reported separately. A mean of dB improvements is not equivalent to a dB ratio of mean errors. Clean-input preservation is evaluated over all four test clips (20 seconds) with no inserted events.

Both methods run on the same {s['environment']['cpu']} CPU environment. Neural inference is CPU-only with one PyTorch thread. Model loading and one warm-up call are excluded. Local runtime includes detector computation and repair; neural runtime includes STFT, model and iSTFT. RTF is wall-clock processing time divided by five seconds of audio. It measures offline throughput, not causal latency: local median filtering and recording-wide feature normalisation use future context. No streaming implementation is claimed.

## 5. Measured results

### 5.1 Detection

{dt}

The STFT detector markedly increases precision and overall F1 but does not improve recall. Both methods miss most of the weak events. The STFT method has {det['stft']['fn']} missed events out of 180; it is a conservative detector, not a reliable general impulse remover. Its sample-level recall is only {det['stft']['sample_metrics']['recall']:.3f}, lower than event recall, because some detections overlap only a fraction of an inserted event. Inference from event F1 alone would therefore overstate useful localisation.

{overlap}

Figure 1 shows precision, recall and F1 against strength separately for these locations. Full grouped values also separate the five event templates. The strongest performance is template-dependent: the short synthetic click and real mouse click are detected more often than the cropped knock and keyboard event. This is consistent with the chosen spectral features favouring abrupt broad-spectrum change, but without an ablation the explanation remains a hypothesis. The transient condition is not uniquely hardest in this small experiment, so the experiment does not establish consonant-overlap conditions as the most difficult category.

### 5.2 Restoration, preservation and computation

{table}

GTCRN produces the lowest measured mean event RMSE and about {res['gtcrn']['event_error_reduction_db_mean']:.2f} dB mean event error reduction, compared with {res['stft_interpolate']['event_error_reduction_db_mean']:.2f} dB for STFT interpolation. However, STFT interpolation has far lower collateral RMSE ({res['stft_interpolate']['clean_rmse_mean']:.6f} versus {res['gtcrn']['clean_rmse_mean']:.6f}) and preserves global speech metrics more closely. GTCRN's mean whole-file STOI is lower than the unprocessed sparse-corruption input despite better local event repair. This shows why event removal alone does not establish whole-speech benefit. It may also reflect domain mismatch: the reference originates at 8 kHz and is upsampled, whereas the model is a general 16 kHz enhancer. The experiment cannot isolate domain mismatch from learned processing effects.

The energy baseline has particularly high collateral error because its false events repair many legitimate speech intervals. Locality alone does not guarantee preservation if the detector is wrong. STFT attenuation and interpolation yield modest average improvement and a zero median improvement: at least half of cases receive no useful repair. The oracle control yields mean event RMSE {res['oracle_interpolate']['event_rmse_mean']:.6f} and mean event error reduction {res['oracle_interpolate']['event_error_reduction_db_mean']:.2f} dB. Even perfect event location does not guarantee that a straight line reconstructs oscillatory speech. Its event RMSE can remain large on speech overlaps despite exactly zero collateral error.

Figure 2 displays event error against collateral error. Figure 3 uses the same 20 ms synthetic event waveform at the same absolute amplitude in all three locations; it is a separate demonstration from the local-RMS strength sweep. The fixed amplitude is set from the +6 dB steady-voiced example and is not rescaled in silence or at the transient. Waveform and spectrogram panels use identical inputs for all methods. Figure 5 shows the measured runtime trade-off. All measured median RTFs are below one, but this is an offline CPU result. GTCRN parameter tensors occupy {s['model']['parameter_bytes']:,} bytes; the checkpoint is {s['model']['checkpoint_bytes']:,} bytes. The model has {s['model']['parameters']:,} parameters including fixed ERB mappings, of which {s['model']['trainable_parameters']:,} are trainable. These are model-storage figures, not measured process peak memory; DSP stores no learned weights but does use waveform, frame and FFT work buffers.

### 5.3 Clean-input false alarms and preservation

{cp}

The STFT detector produces one false event over the clean test clips, versus 24 for the energy baseline. Figure 4 shows that STFT false alarm in clip 6: a small waveform change coincides with a sharp normalised spectral-flux peak, rather than an unusually loud pulse. This explains how abrupt spectral evolution can cross the threshold even at low absolute energy. The waveform has not been phoneme-labelled, so the event is described as a legitimate reference-speech change, not identified as /t/ or /k/. The clean no-op test also shows that GTCRN changes almost all samples; the changes are numerically measurable, not evidence that every altered sample sounds worse. Audio examples enable informal inspection, but no formal listening study or perceptual preference claim is made.

## 6. Limitations and conclusions

The study has one source recording, four test clips and three real-event templates. Development and test share recording conditions, and mixing variants are dependent. No speaker-generalisation claim, confidence interval based on independent utterances or significance test is justified. The original recording is attributed to the Open Speech Repository and verified against a byte-identical public mirror; it is not independently certified noise-free. Location proxies do not establish phoneme identity. Event matching permits partial overlap; sample-level metrics expose that limitation. The chosen detector prioritises avoiding collateral alteration and misses weak or less broadband events. Straight-line repair is weak for longer voiced intervals. Neither a learned impulse-specific model nor the recent sparse adaptive algorithm is tested, and peak memory and causal latency are not measured.

Within these controls, STFT features reduce false detections compared with energy-only detection, while pretrained GTCRN offers stronger local restoration and greater whole-waveform alteration. The strongest supported conclusion is that sparse corruption requires separate evaluation of event reconstruction, clean-region preservation and computation. A future study should expand source speakers and event templates, label consonant releases, and investigate either LPC reconstruction or detector-gated neural replacement. Neither extension is needed to interpret the completed core experiment.

## 7. Reproduction and implementation acknowledgement

Run: python -m pip install -r requirements.txt; python scripts/prepare_data.py; python -m pytest -q; python scripts/run_experiment.py; python scripts/build_deliverables.py. The run creates manifest intervals, frozen thresholds, per-case detection/restoration results, no-op tests, figures and exact example arrays. scripts/export_case.py exports any controlled input; scripts/enhance.py processes a user-supplied WAV without reference labels. See README for platform-specific environment and website instructions.

AI assistance was used in code implementation and documentation. Reported measurements were produced by executing the accompanying experiment code, not taken from expected outcomes or substituted from published GTCRN benchmark numbers.

## References

{refs}
'''
    (report_dir/'final_report.md').write_text(report)
    proposal=f'''# {TITLE}

Leona Wang | SID: 540686380 | GitHub: zhi0708

Repository: {REPO}

Project site: {SITE}

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

The core implementation and experiment have been completed. The STFT detector has measured event F1 {det['stft']['f1']:.3f}, versus {det['energy']['f1']:.3f} for energy detection, but still misses many events. GTCRN gives greater mean event error reduction, while the STFT local methods alter clean regions less. The study therefore exposes a trade-off rather than assuming a winner. Five result figures cover detection, collateral error, overlap examples, clean false alarms and computational cost. Waveforms, spectrograms, audio examples, per-case CSVs, exact masks and reproduction commands accompany the code.

## Timeline and next steps

The original proposal's literature and initial implementation stages are complete. The present milestone provides working code, real measurements and GitHub-ready documentation. The remaining revision stage should prioritise detector recall, broader speech/event coverage and reconstruction quality. Phoneme labelling or LPC reconstruction may be considered only if the core scope remains manageable. A GUI, new neural model and formal listening experiment are not required. The repository includes a static project website, full technical report and a short demonstration so the work is reviewable without running the environment.

## References

[1] Vaseghi and Rayner, IEE Proceedings I, 1990. [2] Nongpiur and Shpak, JASA, 2013. [3] Medina, Coelho and Zao, IEEE/ACM TASLP, 2021, DOI: 10.1109/TASLP.2021.3086131. [4] Rong et al., ICASSP, 2024, DOI: 10.1109/ICASSP48485.2024.10448310. [5] Zhou et al., Signal Processing 241, 2026, DOI: 10.1016/j.sigpro.2025.110390. Full bibliographic entries and dataset attributions appear in the accompanying technical report.
'''
    (report_dir/'revised_proposal.md').write_text(proposal)
    print('Revised proposal word count:',len(proposal.split()))
    readme=f'''# {TITLE}

**Leona Wang · SID 540686380 · ELEC5305**

> For sparse impulsive interference, should processing also be sparse? We compare transparent event detection and local repair with official pretrained GTCRN, measuring restoration, collateral alteration and computation separately.

## Start here

Open [the main experiment notebook](notebooks/impulsive_noise_project.ipynb) for a guided, executable walkthrough. Read [the technical report](report/final_report.pdf) for the full analysis. Supporting modules and generated measurements are organised below; the notebook is the main entry point.

## Experiment

- Controlled generator: clean reference, two synthetic and three ESC-50 event templates, exact insertion intervals and masks.
- Energy-only baseline and STFT energy/flux/bandwidth detector; thresholds chosen only on development clips.
- Local attenuation/interpolation, official pretrained DNS3 GTCRN, and diagnostic oracle interpolation.
- 90 development mixtures, 180 held-out mixtures, 20 seconds of clean no-op tests.
- Five requested result figures, supplementary spectrograms, locally generated listening examples, CSV measurements and technical report.

The test samples are variants of **one existing recording**, not independent speakers. Locations are **acoustic proxies**, not aligned phoneme labels. Detector recall remains low; limitations are reported alongside results.

## Main measured results

{dt}

{table}

GTCRN restores the event region more strongly; the conservative STFT local methods preserve the surrounding reference better. Neither result establishes that DSP or neural enhancement is universally superior. See the [full report](report/final_report.md), [revised proposal](report/revised_proposal.md) and [case-level results](results/).

![Event restoration versus clean-region alteration](docs/assets/figures/02_tradeoff.png)

## Reproduce (Python 3.12 recommended)

```bash
git clone {REPO}.git
cd elec5305-project--540686380-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m pytest -q
python scripts/run_experiment.py
python scripts/build_deliverables.py
```

On Windows, activate with `.venv\\Scripts\\activate`. No GPU is needed. Preparation downloads only three ESC-50 recordings and the official GTCRN repository/checkpoint. It checks the event hashes and pins GTCRN commit `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`. Data downloads require an internet connection. Measurements exclude first-call model warm-up and loading. The tested software/hardware and checkpoint hash are in [summary.json](results/summary.json). Hardware-specific RTFs may change on your machine; deterministic signal metrics should agree within normal numerical tolerance.

## Process your own recording

```bash
python scripts/enhance.py input.wav output.wav --method stft --repair interpolate
python scripts/enhance.py input.wav output_gtcrn.wav --method gtcrn
```

Local output also includes a JSON file of predicted sample intervals. Ground truth is not used. Thresholds are specific to this experiment and are not a validated general-purpose setting.

## Export a reproducible controlled input

```bash
python scripts/export_case.py clip3_synthetic_20ms_steady_voiced_+6dB
```

This writes clean.wav, noise.wav, noisy.wav, event_mask.npy and metadata.json. All 270 case IDs/intervals are listed in [case_manifest.json](results/case_manifest.json). A full run also generates three compact NPZ examples with exact floating-point signals and true masks; these derived binary files are not committed. The website examples use a separate fixed-amplitude experiment: precisely the same event samples are moved among three locations.

## Website and submission

- Main notebook: [notebooks/impulsive_noise_project.ipynb](notebooks/impulsive_noise_project.ipynb). Use the same file across updates.
- Listening examples and a short demonstration video are available on the project website.
- Website source: [docs/index.html](docs/index.html). Preview locally: `python -m http.server 8000 --directory docs`.
- Intended public URL: {SITE}
- [GitHub Actions workflow](.github/workflows/pages.yml) publishes `docs/` after an authorised push to main. In Settings → Pages, select **GitHub Actions** if needed. A deployment is only live after the workflow succeeds; verify the public URL before submitting.
- [Brief submission text](report/brief_submission.txt) is intended for use after the website is deployed.
- Existing `audio_exploration` is retained as the original initial experiment.

## Files

| Path | Purpose |
|---|---|
| notebooks/impulsive_noise_project.ipynb | Main experiment walkthrough, controls, recorded results and interpretation |
| src/dsp.py | Transparent features, detectors, exact support masks, local repair, metrics |
| src/neural.py | Official model adapter with modern complex STFT handling |
| scripts/prepare_data.py | Reproducible dataset/model setup |
| scripts/run_experiment.py | Frozen-threshold experiment and no-op controls |
| scripts/make_figures.py | Five measured result figures and supplementary spectrograms |
| results/ | Manifest, threshold search, per-case and grouped measurements, example arrays |
| report/ | English report, revised 700-1000 word proposal and submission description |
| docs/ | Static website, figures, audio examples and demonstration |
| tests/ | Ground-truth integrity, preservation and one-to-one scoring tests |

## Data provenance and attribution

Speech is `OSR_us_000_0010_8k.wav` from the [Open Speech Repository](https://www.voiptroubleshooter.com/open_speech/american.html). The inherited file is byte-identical to a public mirror of that file; its SHA-256 and verification details are in [data/speech_manifest.json](data/speech_manifest.json). OSR permits copying, modification and website incorporation with source attribution. It is a controlled reference, not certified noise-free speech. ESC-50 is CC BY-NC 3.0 as a dataset; individual records and attribution appear in [data/esc50/LICENSE.txt](data/esc50/LICENSE.txt) and [manifest.json](data/esc50/manifest.json). Used originals: Robinhood76 (door knock, CC BY-NC), MegaPenguin13 (trackpad click, CC0), Anton (typing, CC BY). Raw downloaded events and GTCRN files are excluded from git and fetched by setup; no licence is asserted over third-party materials.

## Academic and technical references

{refs}

## Acknowledgement

AI assistance was used for implementation and documentation. The displayed results come from executing the supplied code; pretrained model performance is measured here rather than copied from its authors' benchmark.
'''
    (ROOT/'README.md').write_text(readme)
    (report_dir/'brief_submission.txt').write_text(f'''Sparse Impulsive-Noise Removal: Classical Detection and Repair versus Neural Speech Enhancement

Project website: {SITE}
Repository: {REPO}

I have implemented a controlled impulsive-noise generator, energy and STFT-feature detectors, event-localised attenuation/interpolation, and inference with the official pretrained GTCRN model. The project includes synthetic and real ESC-50 events, exact insertion masks, frozen development-set thresholds and 180 held-out mixture evaluations. Results separately measure detection, event restoration, clean-region alteration, SI-SDR/STOI and CPU runtime, with clean-input false-positive controls. The website presents measured results, figures, limitations and reproduction instructions. Listening examples are generated locally. The comparison shows a trade-off between stronger local restoration with GTCRN and lower collateral alteration with conservative STFT-local repair.

Preparation note (remove before submitting): publish and verify the website URL first. This text is a submission draft, not confirmation of deployment.
''')
    build_site(s,table,dt,cp,overlap)


def md_table_to_html(table):
    lines=table.strip().splitlines();head=[x.strip() for x in lines[0].strip('|').split('|')]
    body=[[x.strip() for x in l.strip('|').split('|')] for l in lines[2:]]
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in r)+'</tr>' for r in body)+'</tbody></table></div>'


def build_site(s,table,dt,cp,overlap):
    stft=s['restoration']['stft_interpolate'];gt=s['restoration']['gtcrn']
    figs=[('01_detection.png','Detection depends on event strength and overlap.','Precision, recall and F1 are pooled separately at each strength and acoustic location. Weak disturbances are often missed.'),
          ('02_tradeoff.png','Removing an event and preserving speech are separate objectives.','The horizontal axis measures alteration outside the true inserted interval. The vertical axis measures restoration error inside it. Oracle interpolation is a diagnostic, not a tested deployable detector.'),
          ('03_overlap_waveforms.png','One identical event, three acoustic locations.','The same 20 ms waveform and absolute amplitude are moved between a pause, steady voiced speech and a transient. Shading marks the true event; grey curves are the clean reference.'),
          ('03_overlap_spectrograms.png','The same comparison in the time-frequency domain.','Spectrograms share display scales. Cyan boundaries identify the inserted interval. Acoustic categories are proxies, not phoneme labels.'),
          ('04_clean_false_positive.png','A spectral change in clean speech can still trigger detection.','The strongest STFT false alarm is in clip 6. A high normalised flux peak occurs at low energy. No inserted event exists, and no particular consonant identity is claimed.'),
          ('05_quality_runtime.png','Higher restoration quality has a computational cost.','RTF is measured on the same CPU and includes processing, excluding model loading and warm-up. It demonstrates offline throughput, not real-time streaming latency.')]
    gallery=''.join(f'<figure><img loading="lazy" src="assets/figures/{f}" alt="{html.escape(t)}"><figcaption><strong>{t}</strong> {c}</figcaption></figure>' for f,t,c in figs)
    audio=''
    for condition in ['silence','steady_voiced','speech_transient','real_mouse_click']:
        label='Real ESC-50 mouse click (transient overlap)' if condition=='real_mouse_click' else condition.replace('_',' ').title()
        audio+=f'<article class="audio-card"><h3>{label}</h3>'
        for name,label in [('clean','Clean reference'),('noisy','With the inserted event'),('stft_interpolate','STFT + interpolation'),('gtcrn','Pretrained GTCRN')]:
            audio+=f'<label>{label}</label><audio controls preload="none" src="assets/audio/{condition}/{name}.wav"></audio>'
        audio+='</article>'
    page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Controlled comparison of event-localised impulse repair and pretrained GTCRN, with measured restoration, collateral error and runtime.">
<title>Leona Wang | ELEC5305 Impulsive Noise</title>
<style>
:root{{--ink:#182f3b;--teal:#197b80;--cream:#f8f7f2;--muted:#556774;--line:#dce2e0;--white:#fff}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--cream);color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}}a{{color:var(--teal)}}.wrap{{max-width:1120px;margin:auto;padding:0 26px}}header{{padding:76px 0 54px;background:var(--ink);color:white}}.eyebrow{{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#a3d6d4}}h1{{font-size:clamp(32px,5vw,54px);line-height:1.14;max-width:980px;margin:18px 0 25px;font-weight:650;letter-spacing:-1.4px}}header p{{max-width:790px;color:#d5e0e3;font-size:19px}}.buttons{{display:flex;gap:12px;flex-wrap:wrap;margin-top:25px}}.buttons a{{padding:10px 17px;border:1px solid #67858f;border-radius:5px;color:white;text-decoration:none;font-size:14px}}nav{{position:sticky;top:0;background:#fffffff2;border-bottom:1px solid var(--line);z-index:5}}nav .wrap{{display:flex;gap:24px;padding-top:13px;padding-bottom:13px;overflow:auto}}nav a{{text-decoration:none;white-space:nowrap;font-size:14px}}section{{padding:45px 0;border-bottom:1px solid var(--line);scroll-margin-top:65px}}h2{{font-size:30px;line-height:1.25;margin:0 0 20px;letter-spacing:-.5px}}h3{{font-size:19px}}p{{max-width:930px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.stat,.audio-card{{background:white;border:1px solid var(--line);border-radius:8px;padding:22px}}.stat strong{{display:block;font-size:32px;color:var(--teal);line-height:1.2}}.stat span{{font-size:13px;color:var(--muted)}}.note{{border-left:4px solid var(--teal);padding:15px 20px;background:#eaf2ef;margin:24px 0}}.table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:6px;margin:24px 0}}table{{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f0}}figure{{margin:26px 0;background:white;border:1px solid var(--line);border-radius:8px;padding:15px}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:14px;color:var(--muted);padding:14px 8px 4px;line-height:1.6}}figcaption strong{{color:var(--ink)}}.audio-card label{{display:block;font-size:13px;color:var(--muted);margin-top:17px}}audio{{width:100%;margin-top:6px}}pre{{background:#172d39;color:#e5f3ef;padding:24px;border-radius:7px;overflow:auto;font-size:13px;line-height:1.8}}footer{{padding:30px 0;color:var(--muted);font-size:13px}}li{{margin:7px 0}}video{{width:100%;border-radius:8px;background:#182f3b}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}header{{padding:45px 0 35px}}.wrap{{padding:0 18px}}nav .wrap{{gap:18px}}section{{padding:32px 0}}}}
</style></head><body>
<header><div class="wrap"><div class="eyebrow">ELEC5305 · Audio signal processing · Measured experiment</div><h1>When only part of speech is corrupted, how much should we change?</h1><p>{TITLE}</p><p style="font-size:14px">Leona Wang · SID 540686380 · GitHub zhi0708</p><div class="buttons"><a href="{REPO}">Code &amp; reproduction</a><a href="{REPO}/blob/main/report/final_report.pdf">Read the technical report</a><a href="#listen">Listen to examples</a></div></div></header>
<nav aria-label="Page sections"><div class="wrap"><a href="#question">Question</a><a href="#design">Experiment</a><a href="#results">Results</a><a href="#figures">Figures</a><a href="#listen">Audio</a><a href="#limits">Limits</a><a href="#reproduce">Reproduce</a></div></nav>
<main class="wrap">
<section id="question"><h2>Sparse disturbance, two processing strategies.</h2><p>The project compares detecting and repairing short intervals with enhancing the complete waveform using official pretrained GTCRN. We measure three things separately: whether the event is restored, whether surrounding speech changes, and what the processing costs.</p><div class="grid"><div class="stat"><strong>180</strong><span>Held-out controlled mixtures; 90 development mixtures</span></div><div class="stat"><strong>{s['detection']['stft']['f1']:.3f}</strong><span>STFT event F1; energy baseline {s['detection']['energy']['f1']:.3f}</span></div><div class="stat"><strong>{gt['event_error_reduction_db_mean']:.2f} dB</strong><span>Mean GTCRN event-error reduction; not a whole-file quality score</span></div></div><div class="note">The main result is a trade-off: GTCRN repairs the event more strongly, while conservative STFT-local repair preserves the uncorrupted reference more closely. The detector still misses many weak events.</div></section>
<section id="design"><h2>Controlled inputs, frozen thresholds.</h2><p>Six non-overlapping five-second clips come from the repository's existing speech recording. Clips 1–2 set detector thresholds; clips 3–6 are held out. Every inserted event has a known waveform, location and support mask. All systems receive identical noisy inputs.</p><ul><li><strong>Events:</strong> 5 ms and 20 ms synthetic bursts; cropped ESC-50 door knock, mouse click and keyboard events.</li><li><strong>Levels:</strong> −6, 0 and +6 dB relative to local speech RMS, with a defined silence floor.</li><li><strong>Locations:</strong> pause, steady voiced candidate and speech-transient candidate; acoustic proxies, not phoneme alignment.</li><li><strong>Local DSP:</strong> adaptive energy detection, or energy/flux/bandwidth STFT detection, followed by attenuation/interpolation.</li><li><strong>Neural reference:</strong> official DNS3 GTCRN checkpoint, frozen weights, no training.</li><li><strong>Controls:</strong> clean-input false alarms and no-op preservation; true-mask interpolation as a diagnostic only.</li></ul><p>All measurements use floating-point signals, fixed shared scaling and exact true-event intervals. Local methods preserve samples outside predicted intervals, but false detections can still damage clean speech.</p></section>
<section id="results"><h2>What the experiment actually shows.</h2><h3>Event detection</h3>{md_table_to_html(dt)}<p>The STFT detector improves precision, not recall. Event matching permits partial interval overlap; sample-level recall is only {s['detection']['stft']['sample_metrics']['recall']:.3f}, revealing incomplete event coverage.</p>{md_table_to_html(overlap)}<h3>Restoration and collateral alteration</h3>{md_table_to_html(table)}<p>Errors are averaged across the held-out grid. GTCRN's stronger local repair comes with changes outside the event and lower whole-file STOI than the original sparse-corruption input. This is a result for this recording and checkpoint, not a general verdict on neural enhancement.</p><h3>Already-clean input</h3>{md_table_to_html(cp)}<p>Twenty seconds of reference input are processed without inserted noise. The STFT detector has one false event; the energy baseline has 24. Numerically changing samples is not proof of audible degradation.</p><p>Runtime uses the same {html.escape(s['environment']['cpu'])} CPU; median RTF excludes model loading and warm-up. RTF below one indicates offline throughput, not causal latency. GTCRN has {s['model']['parameters']:,} parameters including fixed mappings ({s['model']['trainable_parameters']:,} trainable); parameter storage is {s['model']['parameter_bytes']:,} bytes. Process peak memory was not measured.</p></section>
<section id="figures"><h2>Five questions, five result figures.</h2>{gallery}</section>
<section id="listen"><h2>Hear the same event in different speech contexts.</h2><p>These examples move one identical event at a fixed absolute amplitude. They are separate from the local-RMS strength sweep. Playback files use a shared signal scale; analytical measurements use the original floating-point arrays.</p><div class="grid">{audio}</div><h3>Short project demonstration</h3><video controls preload="metadata" src="assets/demo.mp4">Your browser does not support video. <a href="assets/demo.mp4">Download the demonstration</a>.</video><p>No formal listening experiment was performed. Examples are provided for informal inspection.</p></section>
<section id="limits"><h2>Limits that matter for interpreting these results.</h2><ul><li>One source recording, not independent speakers; the 180 mixtures are dependent controlled variants.</li><li>Speech originally sampled at 8 kHz is upsampled for a general 16 kHz neural model. Domain mismatch may affect the comparison.</li><li>Speech is OSR_us_000_0010_8k.wav from Open Speech Repository, verified against a byte-identical public mirror; it is not certified noise-free.</li><li>One real template per class; broad event generalisation has not been tested.</li><li>Acoustic location proxies cannot establish that a false alarm is /t/ or /k/.</li><li>Conservative detection misses weak events; interpolation does not reconstruct complex voiced speech well.</li><li>No impulse-specific neural model, new model training, streaming implementation or measured peak-memory comparison.</li></ul><p>The supported conclusion is that local restoration, collateral alteration and computation should be measured separately for sparse corruption. A larger corpus, verified consonant labels and one focused reconstruction extension are natural next steps.</p></section>
<section id="reproduce"><h2>Run and inspect the complete experiment.</h2><pre>python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m pytest -q
python scripts/run_experiment.py
python scripts/build_deliverables.py</pre><p><a href="{REPO}/tree/main/results">Per-case results and manifest</a> · <a href="{REPO}/blob/main/report/final_report.md">Full report source</a> · <a href="{REPO}/blob/main/report/revised_proposal.pdf">Revised proposal</a> · <a href="{REPO}/blob/main/README.md">Setup and attribution</a></p><p>Primary references: <a href="https://doi.org/10.1109/ICASSP48485.2024.10448310">GTCRN (ICASSP 2024)</a>, <a href="https://doi.org/10.1016/j.sigpro.2025.110390">Sparse adaptive filtering (2026)</a>, <a href="https://doi.org/10.1109/TASLP.2021.3086131">HHT impulsive detection (2021)</a>, and <a href="https://github.com/karolpiczak/ESC-50">ESC-50</a>. Full historical references are in the report.</p><p>ESC-50 source attribution: Robinhood76 (knock, CC BY-NC), MegaPenguin13 (click, CC0), Anton (typing, CC BY); dataset licence CC BY-NC 3.0. Official GTCRN is obtained from its authors' repository.</p></section>
</main><footer><div class="wrap">Leona Wang · ELEC5305 · AI-assisted implementation and documentation. All displayed numerical results were generated by the accompanying code.</div></footer></body></html>'''
    (ROOT/'docs/index.html').write_text(page)
    (ROOT/'docs/.nojekyll').touch()


if __name__=='__main__':main()
