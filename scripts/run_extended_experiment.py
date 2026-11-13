"""Evaluate on unseen speakers and annotate failure modes without test tuning."""
import csv
import io
import itertools
import json
import re
import time
import zipfile

import numpy as np
from scipy import ndimage
from pystoi import stoi

from run_experiment import (
    ROOT, FS, aggregate, crop_real_event, cpu_name, event_bank, make_cases,
    save_csv,
)
from prepare_extended_data import EXTRA_EVENTS
from src.dsp import detect, detection_counts, inject, prf, regions, repair, restoration_metrics
from src.neural import GTCRNEnhancer
from src.transient import detect_transient, transient_features

DEV_SPEAKERS = ('ASI', 'HKK')
TEST_SPEAKERS = ('BWC', 'TLV', 'YDCK')
PHONES = ('P', 'T', 'K')
METHODS = ('energy', 'stft', 'stft_retuned', 'transient')
LEVELS = (-6, 0, 6)
THRESHOLDS = {'energy': 9, 'stft': 5}


def phone_intervals(text):
    phones = text.split('name = "phones"', 1)[1].split('item [', 1)[0]
    pattern = r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'
    return [(round(float(a)*FS), round(float(b)*FS), label)
            for a, b, label in re.findall(pattern, phones) if label in PHONES]


def speech_clips():
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    folder = ROOT / 'data/l2_arctic'
    clips = []
    for speaker in DEV_SPEAKERS + TEST_SPEAKERS:
        development = speaker in DEV_SPEAKERS
        with zipfile.ZipFile(folder / f'{speaker}.zip') as archive:
            paths = sorted(p for p in archive.namelist() if '/annotation/' in p and p.endswith('.TextGrid'))
            # Disjoint speakers and disjoint prompt IDs. No waveform or outcomes
            # are used to choose utterances.
            for path in paths[:6] if development else paths[6:14]:
                utterance = path.rsplit('/', 1)[-1].removesuffix('.TextGrid')
                audio, rate = sf.read(io.BytesIO(archive.read(f'{speaker}/wav/{utterance}.wav')))
                divisor = gcd(rate, FS)
                x = resample_poly(audio, FS//divisor, rate//divisor)
                x *= .15 / np.max(np.abs(x))
                labels = phone_intervals(archive.read(path).decode('utf-8-sig'))
                clips.append(dict(id=f'{speaker}_{utterance}', speaker=speaker,
                                  utterance=utterance, x=x, labels=labels,
                                  split='development' if development else 'test'))
    assert not ({c['utterance'] for c in clips if c['split']=='development'}
                & {c['utterance'] for c in clips if c['split']=='test'})
    return clips


def fit_detector(clips, bank):
    noisy = []
    clean = []
    for clip in clips:
        clean.append((clip['x'], transient_features(clip['x'])))
        for case in make_cases([clip], bank):
            noisy.append((case['noisy'], case['truth'], transient_features(case['noisy'])))
    def measure(config, detector):
        tp = fp = fn = 0
        for x, truth, feat in noisy:
            prediction = detector(x, config, feat)
            result = detection_counts(prediction, truth)
            tp += result['tp']
            fp += result['fp']
            fn += result['fn']
        clean_fp = sum(len(regions(detector(x, config, feat))) for x, feat in clean)
        weighted_fp = fp + len(noisy) // len(clean) * clean_fp
        return dict(**config, tp=tp, fp=fp, fn=fn, clean_fp=clean_fp,
                    selection_f1=prf(tp, weighted_fp, fn)['f1'], **prf(tp, fp, fn))

    def rank(row):
        return row['selection_f1'], -row['clean_fp'], row['f1']

    stft_grid = [measure({'threshold': threshold},
                         lambda x, c, f: detect(x, 'stft', c['threshold'], feat=f))
                 for threshold in (1.5, 2., 2.5, 3., 4., 5., 6., 8.)]
    stft_best = max(stft_grid, key=rank)
    settings = [measure(dict(contrast_weight=contrast, floor_weight=floor, threshold=threshold),
                        detect_transient)
                for contrast, floor, threshold in itertools.product(
                    (0., .75, 1.5), (0., .5, 1.), (1.5, 2., 2.5, 3., 4., 5.))]
    best = max(settings, key=rank)
    config = {key: best[key] for key in ('contrast_weight', 'floor_weight', 'threshold')}
    return config, settings, stft_best['threshold'], stft_grid


def predict(x, method, config, feat=None):
    if method == 'transient':
        return detect_transient(x, config, feat)
    if method == 'stft_retuned':
        return detect(x, 'stft', config['stft_threshold'], feat=feat)
    return detect(x, method, THRESHOLDS[method], feat=feat)


def held_out_events():
    with (ROOT/'data/esc50/esc50.csv').open() as handle:
        events = [crop_real_event(r) for r in csv.DictReader(handle)
                  if r['filename'] in EXTRA_EVENTS]
    return [dict(event, name=event['name']+'_fold2') for event in events]


def stop_cases(clip, event):
    """Target the first eligible token per phone; use a peak as a release proxy."""
    x = clip['x']
    d = np.diff(x, prepend=x[0])
    energy = ndimage.uniform_filter1d(d**2, size=round(.002*FS))
    selected = set()
    for a, b, phone in clip['labels']:
        if phone in selected or b-a < len(event['x']) or a < .1*FS or b > len(x)-.1*FS:
            continue
        selected.add(phone)
        peak = a + int(np.argmax(energy[a:b]))
        start = int(np.clip(peak-len(event['x'])//2, a, b-len(event['x'])))
        for level in LEVELS:
            noisy, _, truth, _ = inject(x, event['x'], start, level)
            yield dict(id=f"{clip['id']}_{phone}_{level:+d}dB", clip=clip['id'],
                       phone=phone, level_db=level, clean=x, noisy=noisy, truth=truth,
                       start_sample=start, end_sample=start+len(event['x']),
                       phone_start_sample=a, phone_end_sample=b, peak_sample=peak)


def evaluate_case(case, config, neural):
    x, truth = case['noisy'], case['truth']
    detection, restoration = [], []
    outputs = {'noisy': (x, 0.)}
    for method in METHODS:
        start = time.perf_counter()
        mask = predict(x, method, config)
        output = repair(x, mask)
        outputs[method+'_interpolate'] = output, time.perf_counter()-start
        assert np.array_equal(output[~mask], x[~mask])
        detection.append(dict(detector=method, **detection_counts(mask, truth)))
    start = time.perf_counter()
    outputs['gtcrn'] = neural(x), time.perf_counter()-start
    for method, (output, elapsed) in outputs.items():
        restoration.append(dict(method=method, rtf=elapsed/(len(x)/FS),
                                stoi=float(stoi(case['clean'], output, FS)),
                                **restoration_metrics(case['clean'], output, truth, x)))
    return detection, restoration


def totals(rows, field='detector'):
    result = {}
    for method in sorted({r[field] for r in rows}):
        group = [r for r in rows if r[field] == method]
        counts = {k: sum(r[k] for r in group) for k in ('tp', 'fp', 'fn', 'sample_tp', 'sample_fp', 'sample_fn')}
        result[method] = dict(**counts, **prf(counts['tp'], counts['fp'], counts['fn']),
                              sample_recall=prf(counts['sample_tp'], counts['sample_fp'], counts['sample_fn'])['recall'])
    return result


def main():
    out = ROOT / 'results'
    clips = speech_clips()
    bank = event_bank()
    dev = [c for c in clips if c['split']=='development']
    test = [c for c in clips if c['split']=='test']
    config, grid, stft_threshold, stft_grid = fit_detector(dev, bank)
    config['stft_threshold'] = stft_threshold
    save_csv(out/'extended_stft_development.csv', stft_grid)
    save_csv(out/'extended_development.csv', grid)
    print('Frozen settings:', config, flush=True)
    neural = GTCRNEnhancer(ROOT/'third_party/gtcrn')
    neural(dev[0]['x'])
    detection, restoration, clean, phone_cases, phone_clean = [], [], [], [], []
    extra = held_out_events()
    stop_event = next(e for e in bank if e['name']=='synthetic_20ms')
    for clip in test:
        x = clip['x']
        zero = np.zeros(len(x), dtype=bool)
        for method in METHODS:
            mask = predict(x, method, config)
            clean.append(dict(clip=clip['id'], method=method+'_interpolate',
                              false_events=len(regions(mask)),
                              **restoration_metrics(x, repair(x, mask), zero)))
            for a, b, phone in clip['labels']:
                phone_clean.append(dict(clip=clip['id'], detector=method, phone=phone,
                                        start_sample=a, end_sample=b, overlapped=int(mask[a:b].any())))
        clean.append(dict(clip=clip['id'], method='gtcrn', false_events=None,
                          **restoration_metrics(x, neural(x), zero)))
        for case in make_cases([clip], bank + extra):
            info = {key: case[key] for key in ('id','clip','event','kind','condition','level_db','start_sample','end_sample')}
            info['unseen_template'] = case['event'].endswith('_fold2')
            d, r = evaluate_case(case, config, neural)
            detection.extend(dict(info, **row) for row in d)
            restoration.extend(dict(info, **row) for row in r)
        for case in stop_cases(clip, stop_event):
            info = {k: v for k, v in case.items() if k not in ('clean','noisy','truth')}
            d = [dict(detector=m, **detection_counts(predict(case['noisy'], m, config), case['truth']))
                 for m in METHODS]
            phone_cases.extend(dict(info, **row) for row in d)
        print('Evaluated', clip['id'], flush=True)
    for name, rows in [('detection', detection), ('restoration', restoration),
                       ('clean', clean), ('phone_cases', phone_cases), ('phone_clean', phone_clean)]:
        save_csv(out/f'extended_{name}.csv', rows)
    # The existing aggregator supplies identical restoration and clean metrics.
    shared = aggregate(restoration, detection, clean)
    summary = dict(
        settings=config, development_speakers=list(DEV_SPEAKERS), test_speakers=list(TEST_SPEAKERS),
        development_mixtures=len(dev)*len(bank)*9, test_mixtures=len(detection)//len(METHODS),
        unseen_template_mixtures=sum(r['unseen_template'] for r in detection)//len(METHODS),
        event_templates=[{k: v for k, v in event.items() if k != 'x'} for event in bank + extra],
        clean_seconds=sum(len(c['x']) for c in test)/FS, model=neural.metadata, cpu=cpu_name(),
        clips=[dict(id=c['id'], split=c['split'], seconds=len(c['x'])/FS) for c in clips],
        detection=totals(detection), restoration=shared['restoration'],
        clean_preservation=shared['clean_preservation'],
        unseen_template_detection=totals([r for r in detection if r['unseen_template']]),
        groups={key: {value: totals([r for r in detection if str(r[key])==value])
                      for value in sorted({str(r[key]) for r in detection})}
                for key in ('condition','level_db','event','clip')},
        phone_detection={p: totals([r for r in phone_cases if r['phone']==p]) for p in PHONES},
        phone_clean={p: {m: dict(tokens=sum(r['phone']==p and r['detector']==m for r in phone_clean),
                                 overlapped=sum(r['overlapped'] for r in phone_clean if r['phone']==p and r['detector']==m))
                         for m in METHODS} for p in PHONES})
    (out/'extended_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
