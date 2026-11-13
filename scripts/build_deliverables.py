"""Build the English report, README and static website from measured results."""
import csv,html,json
import extended_results as extension
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
    extended = extension.load(ROOT/'results/extended_summary.json')
    overview = extension.headline(extended)
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
    refs += f'8. {extension.DATASET}\n'
    report=f'''# {TITLE}

Leona Wang | SID 540686380 | GitHub: zhi0708 | ELEC5305

[GitHub code repository]({REPO})

[GitHub Pages project website]({SITE})

## Abstract

A mouse click can interrupt a word while leaving the rest of the recording clear. This project compares repairing the detected part of the signal with processing the entire recording using pretrained GTCRN. The first experiment uses 90 development mixtures and 180 test mixtures with known noise locations. STFT detection raised event F1 from {det['energy']['f1']:.3f} to {det['stft']['f1']:.3f}, although many events were missed. {overview} GTCRN reduced more error inside the noisy intervals. Local repair changed less of the surrounding speech and took less time on the same CPU.

## 1. Research question and motivation

A short click in a lecture recording can cover part of a word. If the rest of the recording is clear, one option is to locate the click and repair only that interval. This depends on the detector being able to tell a click from a sudden change in speech. A neural speech enhancer can process the whole recording, including the parts that were already clear. The research question is how these approaches compare in the amount of error they remove from the noisy interval, the changes they make elsewhere, and the time they take to run.

## 2. Literature Review

### 2.1 Detecting and repairing sparse interference

Vaseghi and Rayner [1] use speech prediction to find samples that are difficult to explain from their neighbours. Prediction also offers a way to reconstruct the damaged samples. It depends on nearby speech providing a useful model of what was lost. Stationary-wavelet suppression [2] examines changes at several scales, and the Hilbert-Huang transform method [3] uses a representation adapted to the signal. The STFT gives this project a simpler way to inspect when the spectrum changes and how broadly its energy spreads. A stop-consonant release can still show both features, which is why the experiment includes clean speech and separately tests annotated stops.

Zhou et al. [5] study filters that account for impulsive disturbances being concentrated in a small part of the signal. Here, the known insertion interval makes it possible to check how much a method changes outside that part. Detection and repair are tested separately. If interpolation still makes an error when given the true interval, the error can arise from the repair itself even when the position is known.

### 2.2 General neural enhancement and evaluation

GTCRN [4] uses grouped temporal convolution and recurrence for speech enhancement. The authors provide a pretrained DNS3 checkpoint, which allows the comparison to use the same mixtures and CPU as the local methods. The model enhances the whole signal. Its training for general speech enhancement leaves open how well it handles a brief click. The experiment therefore compares each output with the reference both inside and outside the insertion. STOI [7] gives an additional estimate of intelligibility over the full recording.

Spectral flux describes the change between two neighbouring spectra. The detector uses it to find sudden changes, but the original version also responded to a sharp spectral change at low energy. An event mask records which samples the repair can change. The real-time factor (RTF) records how long processing takes relative to the length of the clip.

## 3. Data and controlled corruption

The existing repository contains one 33.62-second, mono 8 kHz speech recording. It is OSR_us_000_0010_8k.wav from the Open Speech Repository, verified against a byte-identical public mirror. Its source and SHA-256 are recorded in data/speech_manifest.json. It is treated as a controlled reference rather than a certified noise-free corpus. Only the first 30 seconds are used. They form six non-overlapping five-second clips: clips 1-2 are development data and clips 3-6 are test data. The split separates time segments, not speakers. Upsampling to 16 kHz meets GTCRN's input requirements but does not restore missing bandwidth. All speech clips share one fixed scaling that limits the original recording peak to 0.15; mixtures and outputs are never independently normalised.

Two synthetic templates are seeded Gaussian bursts with Hann envelopes, lasting 5 and 20 ms. Three real templates are cropped from ESC-50 [6]: door knock ({s['events'][2]['duration_ms']:.3f} ms), mouse click ({s['events'][3]['duration_ms']:.3f} ms) and keyboard event ({s['events'][4]['duration_ms']:.3f} ms). A 2 ms RMS envelope locates the strongest peak; the crop expands while above 10% of peak, with a 75 ms limit on each side, 2 ms guards and short boundary fades. Source files, crop times, individual attribution and file hashes are provided. There is only one extracted template per real class, so this tests three examples rather than environmental-event diversity.

Each template is added separately at a pause, a steady voiced candidate and a speech-transient candidate. Locations are derived from the reference waveform: minimum smoothed energy identifies the pause; autocorrelation with an energy gate and low-flux preference identifies steady voiced speech; large positive spectral flux identifies the transient. These are acoustic proxies, not human-verified vowels or phoneme-aligned /p/, /t/ and /k/ releases. The reference is used for constructing controlled cases and evaluating outputs, never supplied to a tested detector.

Event levels are -6, 0 and +6 dB. Level means event RMS relative to the RMS of a local speech window extending 50 ms on each side of the inserted interval. In near silence, that reference has a floor at 10% of whole-clip RMS. An insertion mask covers the exact half-open interval [start, end), including its fades. The generator retains clean speech, added event, corrupted speech and the mask. All case intervals are saved in the manifest; scripts/export_case.py materialises any case. Float arrays are used for measurements to avoid clipping or quantisation. The grid contains 2 x 5 x 3 x 3 = 90 development mixtures and 4 x 5 x 3 x 3 = 180 test mixtures. These are controlled variants of a small reference, not 270 independent utterances.

## 4. Detection and local restoration

The analysis uses 8 ms windows and 2 ms hops, with a Hann window for spectra and a 256-point real FFT. Short frames favour transient timing; zero padding samples the spectrum more densely but does not increase physical frequency resolution. Short windows improve temporal localisation at the expense of frequency discrimination. Energy describes local intensity, spectral flux captures positive frame-to-frame spectral change, and bandwidth describes spectral spread. Suppression is evaluated separately from speech preservation; LPC reconstruction remains a possible extension.

The energy baseline compares a local energy rise against a threshold. Its rise is 10 log10((frame_energy + floor)/(local_median_energy + floor)). The median spans 101 frames, approximately 202 ms; the floor is 1% of the recording's 60th-percentile frame energy. This adaptive baseline can detect sudden increases but can mistake normal speech onsets for interference. Its runtime excludes all FFT calculations.

The STFT method combines three features: positive energy rise, positive spectral flux and spectral bandwidth. Flux is the sum of squared positive changes between consecutive L2-normalised magnitude spectra. Its robust z-score uses the recording median and a scale equal to the maximum of 1.4826 times the median absolute deviation, 0.1 times its standard deviation, and a numerical floor. Bandwidth is the square root of the power-weighted frequency variance. The score is max(energy_rise,0)/6 + max(flux_z,0)/3 + bandwidth/1500. The coefficients are fixed heuristic scales, not learned from test labels. Their benefit is interpretability; no ablation is available to isolate each feature's contribution.

Candidate frames exceeding the threshold are mapped to sample intervals, including the frame's half-window support. Candidates longer than 150 ms are rejected. The same duration rule applies to both detectors. The development search considers seven energy thresholds and eight STFT thresholds. It maximises event F1 with false events on the two clean development clips counted 45 times, giving each clean clip the same weight as its 45 noisy variants. This makes preservation an explicit selection preference. The frozen thresholds are {s['thresholds']['energy']} dB energy rise and STFT score {s['thresholds']['stft']}. The test set is not used to choose either threshold.

Attenuation applies a gain of 0.2 inside the predicted event, with up to 2 ms entry and exit ramps. Interpolation replaces the predicted samples with a straight line connecting the preceding and following samples. At file boundaries, the missing neighbour is zero. Both methods preserve the input exactly outside the predicted mask. This guarantee concerns the predicted interval; false detections can still alter genuinely clean speech. An oracle interpolation control uses the true mask solely to diagnose reconstruction limitations and is excluded from detector performance and runtime comparison.

## 5. Neural comparison and evaluation protocol

GTCRN uses the official DNS3 checkpoint, unmodified model architecture and a pinned source commit. No model is trained or fine-tuned. The adapter follows the official 512-sample FFT, 256-sample hop and square-root Hann analysis/synthesis convention. Modern PyTorch complex STFT tensors are converted into the model's real/imaginary input representation and converted back for iSTFT. Output length is explicitly matched to input length. There is no case-specific gain correction, delay fitting or reference-guided enhancement.

Event detection uses one-to-one matching of predicted and true intervals with positive overlap. TP, FP, FN, precision, recall and F1 are pooled over test cases. A partial overlap counts as an event hit, so sample-level metrics are also stored to reveal poor interval coverage. Onset error is the absolute start-time difference for matched events only; it does not summarise misses. Reports separate pause, steady voiced and transient locations, and retain event type and strength in per-case CSVs.

Event RMSE compares output with the clean reference inside the true insertion interval. Collateral RMSE uses every sample outside that true interval, thereby counting false-positive repair and any neural alteration. Event error reduction is 10 log10(noisy event MSE / output event MSE); positive values mean improvement. Whole-file SI-SDR and STOI [7] are secondary checks, not substitutes for local measurements. Table values are arithmetic means across cases; median RTF is reported separately. A mean of dB improvements is not equivalent to a dB ratio of mean errors. Clean-input preservation is evaluated over all four test clips (20 seconds) with no inserted events.

Both methods run on the same {s['environment']['cpu']} CPU environment. Neural inference is CPU-only with one PyTorch thread. Model loading and one warm-up call are excluded. Local runtime includes detector computation and repair; neural runtime includes STFT, model and iSTFT. RTF is wall-clock processing time divided by five seconds of audio. It measures offline throughput, not causal latency: local median filtering and recording-wide feature normalisation use future context. No streaming implementation is claimed.

## 6. Measured results

### 6.1 Detection

{dt}

The STFT detector markedly increases precision and overall F1 but does not improve recall. Both methods miss most of the weak events. The STFT method has {det['stft']['fn']} missed events out of 180; it is a conservative detector, not a reliable general impulse remover. Its sample-level recall is only {det['stft']['sample_metrics']['recall']:.3f}, lower than event recall, because some detections overlap only a fraction of an inserted event. Inference from event F1 alone would therefore overstate useful localisation.

{overlap}

![Detection by strength and acoustic location](../docs/assets/figures/01_detection.png)

Figure 1 shows precision, recall and F1 against strength separately for these locations. Full grouped values also separate the five event templates. The strongest performance is template-dependent: the short synthetic click and real mouse click are detected more often than the cropped knock and keyboard event. This is consistent with the chosen spectral features favouring abrupt broad-spectrum change, but without an ablation the explanation remains a hypothesis. The transient condition is not uniquely hardest in this small experiment, so the experiment does not establish consonant-overlap conditions as the most difficult category.

### 6.2 Restoration, preservation and computation

{table}

GTCRN produces the lowest measured mean event RMSE and about {res['gtcrn']['event_error_reduction_db_mean']:.2f} dB mean event error reduction, compared with {res['stft_interpolate']['event_error_reduction_db_mean']:.2f} dB for STFT interpolation. However, STFT interpolation has far lower collateral RMSE ({res['stft_interpolate']['clean_rmse_mean']:.6f} versus {res['gtcrn']['clean_rmse_mean']:.6f}) and preserves global speech metrics more closely. GTCRN's mean whole-file STOI is lower than the unprocessed sparse-corruption input despite better local event repair. This shows why event removal alone does not establish whole-speech benefit. It may also reflect domain mismatch: the reference originates at 8 kHz and is upsampled, whereas the model is a general 16 kHz enhancer. The experiment cannot isolate domain mismatch from learned processing effects.

The energy baseline has particularly high collateral error because its false events repair many legitimate speech intervals. Locality alone does not guarantee preservation if the detector is wrong. STFT attenuation and interpolation yield modest average improvement and a zero median improvement: at least half of cases receive no useful repair. The oracle control yields mean event RMSE {res['oracle_interpolate']['event_rmse_mean']:.6f} and mean event error reduction {res['oracle_interpolate']['event_error_reduction_db_mean']:.2f} dB. Even perfect event location does not guarantee that a straight line reconstructs oscillatory speech. Its event RMSE can remain large on speech overlaps despite exactly zero collateral error.

![Restoration versus alteration outside the event](../docs/assets/figures/02_tradeoff.png)

Figure 2 displays event error against collateral error. The project website includes a separate waveform and spectrogram demonstration using an identical 20 ms event and fixed absolute amplitude across three locations. Table values describe the local-RMS sweep. All measured median RTFs are below one, but this is an offline CPU result. GTCRN parameter tensors occupy {s['model']['parameter_bytes']:,} bytes; the checkpoint is {s['model']['checkpoint_bytes']:,} bytes. The model has {s['model']['parameters']:,} parameters including fixed ERB mappings, of which {s['model']['trainable_parameters']:,} are trainable. These are model-storage figures, not measured process peak memory; DSP stores no learned weights but does use waveform, frame and FFT work buffers.

### 6.3 Clean-input false alarms and preservation

{cp}

The STFT detector produces one false event over the clean test clips, versus 24 for the energy baseline. The clean false-positive figure on the project website shows the STFT false alarm in clip 6: a small waveform change coincides with a sharp normalised spectral-flux peak, rather than an unusually loud pulse. This explains how abrupt spectral evolution can cross the threshold even at low absolute energy. The waveform has not been phoneme-labelled, so the event is described as a legitimate reference-speech change, not identified as /t/ or /k/. The clean no-op test also shows that GTCRN changes almost all samples; the changes are numerically measurable, not evidence that every altered sample sounds worse. Audio examples enable informal inspection, but no formal listening study or perceptual preference claim is made.

{extension.report_section(extended)}

## 8. Discussion and Conclusion

### 8.1 Why the two pipelines behave differently

The difference is easiest to see by measuring the noisy interval and the rest of the clip separately. On the held-out speakers, mean error reduction in the event was {extended['restoration']['gtcrn']['event_error_reduction_db_mean']:.2f} dB for GTCRN and {extended['restoration']['transient_interpolate']['event_error_reduction_db_mean']:.2f} dB for local interpolation. Outside the event, the RMSE was {extended['restoration']['gtcrn']['clean_rmse_mean']:.6f} for GTCRN and {extended['restoration']['transient_interpolate']['clean_rmse_mean']:.6f} for local repair. GTCRN estimates an enhanced signal throughout the recording [4], whereas the local method leaves the input untouched outside its predicted mask. Even with the true mask, a straight line can miss the oscillations of voiced speech. This gives a reason to test a repair based on speech prediction [1] while keeping the mask fixed. The smaller changes outside the event are consistent with the motivation for sparse processing [5].

### 8.2 What the detector change contributed

The keyboard example in Figure 3 shows how noise can survive the local method. The score stayed below the threshold at -6 dB. At +6 dB it crossed the threshold, but the predicted interval still missed part of the insertion. Retuning the old STFT threshold gave F1 0.154, and changing the score gave 0.475 on the same test set. The comparison supports the combined change to the score. It does not show how much each feature contributed. On clean speech, the original STFT still produced fewer false events than the short-context detector (6 versus 14). The improved detection rate therefore comes with a cost relative to the original conservative setting.

### 8.3 Scope and next experiment

For these recordings, GTCRN repaired more of the noise interval. The local method ran faster and changed less of the speech around it. This conclusion is based on three test speakers and six real disturbance recordings. Several mixtures reuse the same speech and noise, so the number of mixtures is larger than the number of independent recordings. The annotated test locates /p/, /t/ and /k/ intervals and uses a peak as an estimate of release timing. It measures detection in those intervals. A listening test would be needed to tell whether people prefer either output.

A focused next experiment would hold the predicted mask fixed and compare interpolation with prediction-based reconstruction [1] on voiced overlaps. Evaluating each method with both true and predicted masks would distinguish gains in repair from errors in localisation. Testing on additional speakers and event recordings would then check whether the result survives changes in source material.

## 9. Reproduction and implementation acknowledgement

Use the README commands to set up dependencies and reproduce both benchmarks. Run scripts/prepare_extended_data.py and scripts/run_extended_experiment.py for the speaker-disjoint evaluation before rebuilding the deliverables. The run creates manifest intervals, frozen thresholds, per-case detection/restoration results, no-op tests, figures and exact example arrays. scripts/export_case.py exports any controlled input; scripts/enhance.py processes a user-supplied WAV without reference labels. See README for platform-specific environment and website instructions.

AI assistance was used in code implementation and documentation. Reported measurements were produced by executing the accompanying experiment code, not taken from expected outcomes or substituted from published GTCRN benchmark numbers.

## References

{refs}
'''
    (report_dir/'final_report.md').write_text(report)
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

The original benchmark uses one Open Speech Repository recording. The extension uses speaker- and prompt-disjoint L2-ARCTIC utterances, three new ESC-50 templates and a separate check using manually annotated stop-consonant intervals.

## Speaker-disjoint results

{overview}

{extension.detection_table(extended)}

See [the technical report](report/final_report.md) for restoration, runtime, weak-event misses, and the annotated /p/, /t/ and /k/ comparison.

## Original benchmark results

{dt}

{table}

GTCRN restores the event region more strongly; the conservative STFT local methods preserve the surrounding reference better. Neither result establishes that DSP or neural enhancement is universally superior. See the [full report](report/final_report.md), [revised proposal](report/revised_proposal.md) and [case-level results](results/).

![Event restoration versus clean-region alteration](docs/assets/figures/02_tradeoff.png)

## Quick demo

From the repository root, prepare the Python environment and model once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_data.py
```

Run both methods on the included mouse-click example:

```bash
python scripts/demo.py
```

Open `demo_output/index.html` in your browser to play the noisy input, local DSP repair and GTCRN output on one page. The detected intervals are printed in seconds and saved to `demo_output/local.events.json` as sample indices. The example click occurs around 2.64 seconds. The demo uses the included audio and the prepared checkpoint.

To process a different WAV, or to run again into a fresh folder:

```bash
python scripts/demo.py my_recording.wav --output-dir my_comparison
python scripts/demo.py --output-dir recording_take
```

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
python scripts/prepare_extended_data.py
python scripts/run_extended_experiment.py
python scripts/plot_extended_case.py
python scripts/build_deliverables.py
python scripts/render_reports.py final_report
```

On Windows, activate with `.venv\\Scripts\\activate`. No GPU is needed. Preparation downloads only three ESC-50 recordings and the official GTCRN repository/checkpoint. It checks the event hashes and pins GTCRN commit `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`. Data downloads require an internet connection. The extended preparation additionally downloads five checksum-verified L2-ARCTIC speaker archives (about 1.5 GB total) and three ESC-50 fold-2 clips. The corpus archives stay local. Measurements exclude first-call model warm-up and loading. The tested software/hardware and checkpoint hash are in [summary.json](results/summary.json). Hardware-specific RTFs may change on your machine; deterministic signal metrics should agree within normal numerical tolerance.

## Process your own recording

```bash
python scripts/enhance.py input.wav output.wav --method stft --repair interpolate
python scripts/enhance.py input.wav output_gtcrn.wav --method gtcrn
python scripts/enhance.py input.wav output_transient.wav --method transient
```

Local output also includes a JSON file of predicted sample intervals. Ground truth is not used. Thresholds are specific to this experiment and are not a validated general-purpose setting.

## Export a reproducible controlled input

```bash
python scripts/export_case.py clip3_synthetic_20ms_steady_voiced_+6dB
```

This writes clean.wav, noise.wav, noisy.wav, event_mask.npy and metadata.json. All 270 case IDs/intervals are listed in [case_manifest.json](results/case_manifest.json). A full run also generates three compact NPZ examples with exact floating-point signals and true masks; these derived binary files are not committed. The website examples use a separate fixed-amplitude experiment: precisely the same event samples are moved among three locations.

## Website and submission

- Main notebook: [notebooks/impulsive_noise_project.ipynb](notebooks/impulsive_noise_project.ipynb). Use the same file across updates.
- Listening examples are available on the project website.
- Website source: [docs/index.html](docs/index.html). Preview locally: `python -m http.server 8000 --directory docs`.
- Intended public URL: {SITE}
- [GitHub Actions workflow](.github/workflows/pages.yml) publishes `docs/` after an authorised push to main. In Settings → Pages, select **GitHub Actions** if needed. A deployment is only live after the workflow succeeds; verify the public URL before submitting.
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

L2-ARCTIC: [corpus and annotation documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/), [pinned v5.0 mirror](https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/tree/cc02fd37197966e5de11ad737f7affd577c61992). Additional ESC-50 recordings: GodofPH (knock, CC0), esperri (mouse, CC0), SpazTastic (typing, CC Sampling+). Only derived numerical results are distributed for these new sources.

## Academic and technical references

{refs}

## Acknowledgement

AI assistance was used for implementation and documentation. The displayed results come from executing the supplied code; pretrained model performance is measured here rather than copied from its authors' benchmark.
'''
    (ROOT/'README.md').write_text(readme)
    build_site(s,table,dt,cp,overlap,extended)


def md_table_to_html(table):
    lines=table.strip().splitlines();head=[x.strip() for x in lines[0].strip('|').split('|')]
    body=[[x.strip() for x in l.strip('|').split('|')] for l in lines[2:]]
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in r)+'</tr>' for r in body)+'</tbody></table></div>'


def build_site(s,table,dt,cp,overlap,extended):
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
<header><div class="wrap"><div class="eyebrow">ELEC5305 · Audio signal processing · Measured experiment</div><h1>When only part of speech is corrupted, how much should we change?</h1><p>{TITLE}</p><p style="font-size:14px">Leona Wang · SID 540686380 · GitHub zhi0708</p><div class="buttons"><a href="{REPO}">Code &amp; reproduction</a><a href="final_report.pdf">Download the technical report</a><a href="#listen">Listen to examples</a></div></div></header>
<nav aria-label="Page sections"><div class="wrap"><a href="#question">Question</a><a href="#design">Experiment</a><a href="#extended">Speaker-disjoint results</a><a href="#results">Original results</a><a href="#figures">Figures</a><a href="#listen">Audio</a><a href="#limits">Limits</a><a href="#reproduce">Reproduce</a></div></nav>
<main class="wrap">
<section id="question"><h2>Sparse disturbance, two processing strategies.</h2><p>The project compares detecting and repairing short intervals with enhancing the complete waveform using official pretrained GTCRN. We measure three things separately: whether the event is restored, whether surrounding speech changes, and what the processing costs.</p><div class="grid"><div class="stat"><strong>180</strong><span>Held-out controlled mixtures; 90 development mixtures</span></div><div class="stat"><strong>{s['detection']['stft']['f1']:.3f}</strong><span>STFT event F1; energy baseline {s['detection']['energy']['f1']:.3f}</span></div><div class="stat"><strong>{gt['event_error_reduction_db_mean']:.2f} dB</strong><span>Mean GTCRN event-error reduction; not a whole-file quality score</span></div></div><div class="note">The main result is a trade-off: GTCRN repairs the event more strongly, while conservative STFT-local repair preserves the uncorrupted reference more closely. The detector still misses many weak events.</div></section>
<section id="design"><h2>Controlled inputs, frozen thresholds.</h2><p>Six non-overlapping five-second clips come from the repository's existing speech recording. Clips 1–2 set detector thresholds; clips 3–6 are held out. Every inserted event has a known waveform, location and support mask. All systems receive identical noisy inputs.</p><ul><li><strong>Events:</strong> 5 ms and 20 ms synthetic bursts; cropped ESC-50 door knock, mouse click and keyboard events.</li><li><strong>Levels:</strong> −6, 0 and +6 dB relative to local speech RMS, with a defined silence floor.</li><li><strong>Locations:</strong> pause, steady voiced candidate and speech-transient candidate; acoustic proxies, not phoneme alignment.</li><li><strong>Local DSP:</strong> adaptive energy detection, or energy/flux/bandwidth STFT detection, followed by attenuation/interpolation.</li><li><strong>Neural reference:</strong> official DNS3 GTCRN checkpoint, frozen weights, no training.</li><li><strong>Controls:</strong> clean-input false alarms and no-op preservation; true-mask interpolation as a diagnostic only.</li></ul><p>All measurements use floating-point signals, fixed shared scaling and exact true-event intervals. Local methods preserve samples outside predicted intervals, but false detections can still damage clean speech.</p></section>
<section id="extended"><h2>Evaluation on unseen speakers</h2><p>{extension.headline(extended)}</p><p>Two L2-ARCTIC speakers were used for development and three different speakers for testing. The test also includes three new ESC-50 event recordings. The retuned STFT comparison uses the same development data as the short-context detector.</p>{md_table_to_html(extension.detection_table(extended))}<h3>Restoration and computation</h3>{md_table_to_html(extension.quality_table(extended))}<h3>Annotated /p/, /t/ and /k/ intervals</h3>{md_table_to_html(extension.phone_table(extended))}<p>Clean overlap counts how often each detector touched a labelled phone with no injected event. Each corrupted token is tested at three levels.</p><figure><img loading="lazy" src="assets/figures/06_extended_miss.png" alt="Weak and strong versions of the same keyboard insertion"><figcaption>The -6 dB insertion is missed while the +6 dB version crosses the frozen threshold in the same voiced interval.</figcaption></figure></section>
<section id="results"><h2>Original single-recording benchmark</h2><h3>Event detection</h3>{md_table_to_html(dt)}<p>The STFT detector improves precision, not recall. Event matching permits partial interval overlap; sample-level recall is only {s['detection']['stft']['sample_metrics']['recall']:.3f}, revealing incomplete event coverage.</p>{md_table_to_html(overlap)}<h3>Restoration and collateral alteration</h3>{md_table_to_html(table)}<p>Errors are averaged across the held-out grid. GTCRN's stronger local repair comes with changes outside the event and lower whole-file STOI than the original sparse-corruption input. This is a result for this recording and checkpoint, not a general verdict on neural enhancement.</p><h3>Already-clean input</h3>{md_table_to_html(cp)}<p>Twenty seconds of reference input are processed without inserted noise. The STFT detector has one false event; the energy baseline has 24. Numerically changing samples is not proof of audible degradation.</p><p>Runtime uses the same {html.escape(s['environment']['cpu'])} CPU; median RTF excludes model loading and warm-up. RTF below one indicates offline throughput, not causal latency. GTCRN has {s['model']['parameters']:,} parameters including fixed mappings ({s['model']['trainable_parameters']:,} trainable); parameter storage is {s['model']['parameter_bytes']:,} bytes. Process peak memory was not measured.</p></section>
<section id="figures"><h2>Five questions, five result figures.</h2>{gallery}</section>
<section id="listen"><h2>Hear the same event in different speech contexts.</h2><p>These examples move one identical event at a fixed absolute amplitude. They are separate from the local-RMS strength sweep. Playback files use a shared signal scale; analytical measurements use the original floating-point arrays.</p><div class="grid">{audio}</div><p>No formal listening experiment was performed. Examples are provided for informal inspection.</p></section>
<section id="limits"><h2>Limits that matter for interpreting these results.</h2><ul><li>The original 180 mixtures use one source recording. The extension adds three independent test speakers, with dependent mixture variants within each utterance.</li><li>Speech originally sampled at 8 kHz is upsampled for a general 16 kHz neural model. Domain mismatch may affect the comparison.</li><li>Speech is OSR_us_000_0010_8k.wav from Open Speech Repository, verified against a byte-identical public mirror; it is not certified noise-free.</li><li>The extension has two real recordings per event class in total, so broad event generalisation remains untested.</li><li>The original location proxies are complemented by a separate annotated phone analysis. Burst peak timing within each phone is estimated.</li><li>Conservative detection misses weak events; interpolation does not reconstruct complex voiced speech well.</li><li>No impulse-specific neural model, new model training, streaming implementation or measured peak-memory comparison.</li></ul><p>The supported conclusion is that local restoration, collateral alteration and computation should be measured separately for sparse corruption. The benchmark covers a small speaker sample and uses simple local interpolation.</p></section>
<section id="reproduce"><h2>Run and inspect the complete experiment.</h2><pre>python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m pytest -q
python scripts/run_experiment.py
python scripts/prepare_extended_data.py
python scripts/run_extended_experiment.py
python scripts/plot_extended_case.py
python scripts/build_deliverables.py
python scripts/render_reports.py final_report</pre><p><a href="{REPO}/tree/main/results">Per-case results and manifest</a> · <a href="{REPO}/blob/main/report/final_report.md">Full report source</a> · <a href="{REPO}/blob/main/report/revised_proposal.pdf">Revised proposal</a> · <a href="{REPO}/blob/main/README.md">Setup and attribution</a></p><p>Primary references: <a href="https://doi.org/10.1109/ICASSP48485.2024.10448310">GTCRN (ICASSP 2024)</a>, <a href="https://doi.org/10.1016/j.sigpro.2025.110390">Sparse adaptive filtering (2026)</a>, <a href="https://doi.org/10.1109/TASLP.2021.3086131">HHT impulsive detection (2021)</a>, and <a href="https://github.com/karolpiczak/ESC-50">ESC-50</a>. Full historical references are in the report.</p><p>ESC-50 source attribution: Robinhood76 (knock, CC BY-NC), MegaPenguin13 (click, CC0), Anton (typing, CC BY); dataset licence CC BY-NC 3.0. Official GTCRN is obtained from its authors' repository.</p></section>
</main><footer><div class="wrap">Leona Wang · ELEC5305 · AI-assisted implementation and documentation. All displayed numerical results were generated by the accompanying code.</div></footer></body></html>'''
    (ROOT/'docs/index.html').write_text(page)
    (ROOT/'docs/.nojekyll').touch()


if __name__=='__main__':main()
