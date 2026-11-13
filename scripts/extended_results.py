"""Render measured tables for the report, README, website and notebook."""
import json

LABELS = {
    'energy': 'Energy (original threshold)',
    'stft': 'STFT (original threshold)',
    'stft_retuned': 'STFT (retuned)',
    'transient': 'Short-context detector',
}
DATASET = ('G. Zhao et al., "L2-ARCTIC: A Non-native English Speech Corpus," '
           'Interspeech 2018, pp. 2783–2787. https://doi.org/10.21437/Interspeech.2018-1110')


def table(headers, rows):
    lines = [headers, ['---']*len(headers), *rows]
    return '\n'.join('| ' + ' | '.join(map(str, line)) + ' |' for line in lines)


def detection_table(s):
    rows = []
    for method, label in LABELS.items():
        data = s['detection'][method]
        clean = s['clean_preservation'][method+'_interpolate']
        rows.append([label, f"{data['precision']:.3f}", f"{data['recall']:.3f}",
                     f"{data['f1']:.3f}", f"{data['sample_recall']:.3f}", clean['false_events']])
    return table(['Detector', 'Precision', 'Recall', 'F1', 'Sample recall', 'Clean false events'], rows)


def quality_table(s):
    rows = []
    for method, label in [('noisy', 'Unprocessed'), ('stft_interpolate', 'STFT (original)'),
                          ('stft_retuned_interpolate', 'STFT (retuned)'),
                          ('transient_interpolate', 'Short-context repair'), ('gtcrn', 'GTCRN')]:
        r = s['restoration'][method]
        rows.append([label, f"{r['event_error_reduction_db_mean']:.2f}", f"{r['clean_rmse_mean']:.6f}",
                     f"{r['si_sdr_db_mean']:.2f}", f"{r['stoi_mean']:.3f}", f"{r['rtf_median']:.5f}"])
    return table(['Pipeline', 'Event reduction (dB)', 'Outside-event RMSE', 'SI-SDR (dB)', 'STOI', 'Median CPU RTF'], rows)


def phone_table(s):
    rows = []
    for phone, measures in s['phone_detection'].items():
        row = [f'/{phone.lower()}/']
        for method in ('stft', 'stft_retuned', 'transient'):
            d = measures[method]
            c = s['phone_clean'][phone][method]
            row.extend([f"{d['tp']}/{d['tp']+d['fn']}", f"{c['overlapped']}/{c['tokens']}"])
        rows.append(row)
    return table(['Phone', 'Original hits', 'Original clean overlap', 'Retuned hits',
                  'Retuned clean overlap', 'Short-context hits', 'Short-context clean overlap'], rows)


def grouped_table(s):
    rows = []
    for dimension in ('level_db', 'condition'):
        for key, group in s['groups'][dimension].items():
            label = f'{int(key):+d} dB' if dimension=='level_db' else key.replace('_', ' ')
            counts = [f"{group[m]['tp']}/{group[m]['tp']+group[m]['fn']}"
                      for m in ('stft', 'stft_retuned', 'transient')]
            rows.append([label, *counts])
    group = s['unseen_template_detection']
    rows.append(['Unseen real templates', *[f"{group[m]['tp']}/{group[m]['tp']+group[m]['fn']}"
                  for m in ('stft', 'stft_retuned', 'transient')]])
    return table(['Condition', 'Original STFT hits', 'Retuned STFT hits', 'Short-context hits'], rows)


def headline(s):
    old, retuned, new = (s['detection'][m] for m in ('stft', 'stft_retuned', 'transient'))
    return (f"On {s['test_mixtures']:,} mixtures from three held-out speakers, event F1 was "
            f"{old['f1']:.3f} for the original STFT detector, {retuned['f1']:.3f} after "
            f"retuning STFT, and {new['f1']:.3f} for the short-context detector.")


def report_section(s):
    d = s['detection']['transient']
    comparison = s['detection']['stft_retuned']
    c = s['clean_preservation']
    config = s['settings']
    r = s['restoration']
    return f"""## 6. Speaker-disjoint evaluation

### 6.1 Protocol and detector

The extension uses L2-ARCTIC v5.0 [8]. The first six manually annotated utterances in sorted order for ASI and HKK form the development set. The next eight utterance IDs for each of BWC, TLV and YDCK form the test set. Neither speakers nor prompt IDs cross the split. Waveforms are resampled from 44.1 kHz to 16 kHz and each reference is scaled once to a peak of 0.15. The reference, mixture and outputs then share that scale. In development, the original five event templates yield {s['development_mixtures']} mixtures. Testing yields {s['test_mixtures']:,} mixtures, including {s['unseen_template_mixtures']} using three additional ESC-50 recordings from fold 2. Clean controls total {s['clean_seconds']:.1f} seconds. These are controlled variants of 24 test utterances from three speakers.

The short-context detector measures first-difference energy in 2 ms windows. It compares this energy with the larger mean of the two flanks, approximately 20–60 ms away. This downweights rises that continue into surrounding speech. Its score combines positive flank contrast in dB (scaled by 6), positive local energy rise (scaled by 9), bandwidth (scaled by 1700 Hz), capped flux min(3 × flux, 1), and an optional low-level penalty. Development selected contrast weight {config['contrast_weight']}, low-level penalty weight {config['floor_weight']}, and threshold {config['threshold']}. A threshold-only STFT control selected {config['stft_threshold']} from the original eight-value grid. Both searches used the same development clips and maximised event F1 with each clean false alarm weighted by 45. Test mixtures and phoneme labels did not select settings. The published energy and STFT thresholds of 9 and 5 serve as fixed transfer baselines.

### 6.2 Detection and restoration

{headline(s)} The short-context detector matched {d['tp']} events and missed {d['fn']}, with {d['fp']} unmatched predictions. Retuned STFT matched {comparison['tp']} and missed {comparison['fn']}, with {comparison['fp']} unmatched predictions. On the uncorrupted inputs, short-context detection produced {c['transient_interpolate']['false_events']} false events and retuned STFT produced {c['stft_retuned_interpolate']['false_events']}.

{detection_table(s)}

{grouped_table(s)}

A hit requires interval overlap. Sample recall measures how much of the true event was covered. The strength and location breakdown shows where the detector remains unreliable. Low-level insertions can have less short-window contrast than the surrounding speech. Similar local transients can occur naturally in speech, as the clean control below illustrates. Figure 6 illustrates a matched failure: in BWC arctic a0011, the detector missed the -6 dB keyboard insertion during steady speech but detected the +6 dB version at the same sample position. Even the hit covered only 289 of 381 inserted samples. The unseen-template row pools the three extra ESC-50 recordings, so it supports only a limited check of transfer to new disturbance examples.

{quality_table(s)}

GTCRN achieved {r['gtcrn']['event_error_reduction_db_mean']:.2f} dB mean event-error reduction, while short-context interpolation achieved {r['transient_interpolate']['event_error_reduction_db_mean']:.2f} dB. Their mean outside-event errors were {r['gtcrn']['clean_rmse_mean']:.6f} and {r['transient_interpolate']['clean_rmse_mean']:.6f}, respectively. Each CPU timing excludes model setup and one warm-up. Clean-region RMSE quantifies waveform alteration. Perceptual quality would require a listening study.

### 6.3 Manually annotated stops

The clean control checks whether each detector touches unambiguous P, T or K intervals in the corpus's manually checked phones tier. The paired insertion test uses the first eligible token of each phone per utterance and places a 20 ms burst entirely inside the phone interval, near its strongest 2 ms first-difference-energy peak. Tests use three levels per selected token. The peak is an acoustic proxy for the release time. Human annotation identifies the phone interval.

{phone_table(s)}

Clean overlap is counted per token, so one false event can touch more than one token. The targeted burst results measure detector sensitivity for a single synthetic disturbance shape and the annotated subset. The utterances repeat across strength variants, and the phoneme counts differ. The experiment does not support a population-level ranking of phonemes.

### 6.4 Limits and conclusion

The short-context detector improves event F1 over both STFT controls while still missing over half of the test events. Its clean false-event count falls between the two STFT controls, and GTCRN retains a substantial event-restoration advantage. The original OSR experiment established the restoration and preservation trade-off in one recording. The extension repeats that comparison on three unseen speakers and adds a threshold-only control and an annotated consonant test. The sampling of speakers, utterances and event sources remains small. Straight-line repair can remove real speech within the predicted intervals, and partial event coverage can leave interference audible. No formal listening test, streaming latency or peak-memory measurement is included.
"""


def load(path):
    return json.loads(path.read_text())
