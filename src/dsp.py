"""Detection never receives the clean reference or the inserted-event mask."""
from dataclasses import dataclass
import numpy as np
from scipy import signal, ndimage
from scipy.optimize import linear_sum_assignment

FS = 16000
EPS = 1e-12


def regions(mask):
    """Half-open intervals [start, end), including boundary events."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def features(x, fs=FS):
    """8 ms Hann windows; 2 ms hops: favour transient time localisation."""
    length, hop = round(.008 * fs), round(.002 * fs)
    padded = np.pad(x, (length//2, length//2 + hop))
    frames = np.lib.stride_tricks.sliding_window_view(padded, length)[::hop]
    centers = np.arange(len(frames)) * hop
    keep = centers < len(x)
    frames, centers = frames[keep], centers[keep]
    power = np.abs(np.fft.rfft(frames * signal.windows.hann(length, sym=False), n=256))**2
    mag = np.sqrt(power)
    energy = np.mean(frames**2, axis=1)
    frequencies = np.fft.rfftfreq(256, 1/fs)
    norm = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + EPS)
    flux = np.r_[0., np.sum(np.maximum(np.diff(norm, axis=0), 0.)**2, axis=1)]
    mass = power.sum(axis=1) + EPS
    centroid = (power * frequencies).sum(axis=1) / mass
    bandwidth = np.sqrt((power * (frequencies[None] - centroid[:, None])**2).sum(axis=1) / mass)
    # Relative local energy rise, rather than a loudness-only global threshold.
    local = ndimage.median_filter(energy, size=101, mode='reflect')
    floor = max(np.percentile(energy, 60) * .01, EPS)
    rise = 10*np.log10((energy + floor)/(local + floor))
    flux_z = robust_z(flux)
    # Three interpretable features: energy rise, flux, broadband spectral spread.
    score = np.maximum(rise, 0)/6 + np.maximum(flux_z, 0)/3 + bandwidth/1500
    return dict(centers=centers, energy=energy, rise=rise, flux=flux,
                bandwidth=bandwidth, score=score, hop=hop, length=length)


def robust_z(x):
    m = np.median(x)
    scale = max(1.4826*np.median(np.abs(x-m)), np.std(x)*.1, EPS)
    return (x-m)/scale


def detect(x, detector, threshold, fs=FS, feat=None):
    f = (energy_features(x, fs) if detector == 'energy' else features(x, fs)) if feat is None else feat
    frame_mask = (f['rise'] if detector == 'energy' else f['score']) > threshold
    # Restrict candidates to <=150 ms. Do not discard isolated short clicks.
    mask = np.zeros(len(x), bool)
    for a,b in regions(frame_mask):
        start = max(0, int(f['centers'][a] - f['length']//2))
        end = min(len(x), int(f['centers'][b-1] + f['length']//2 + 1))
        if end-start <= round(.15*fs):
            mask[start:end] = True
    return mask


def energy_features(x, fs=FS):
    """Energy-only baseline: no FFT is charged to its runtime."""
    length, hop = round(.008 * fs), round(.002 * fs)
    padded = np.pad(x, (length//2, length//2 + hop))
    frames = np.lib.stride_tricks.sliding_window_view(padded, length)[::hop]
    centers = np.arange(len(frames)) * hop
    keep = centers < len(x)
    energy = np.mean(frames[keep]**2, axis=1)
    local = ndimage.median_filter(energy, size=101, mode='reflect')
    floor = max(np.percentile(energy, 60)*.01, EPS)
    rise = 10*np.log10((energy+floor)/(local+floor))
    return dict(centers=centers[keep], rise=rise, hop=hop, length=length)


def repair(x, mask, method='interpolate'):
    """The original waveform remains bit-identical outside the predicted mask."""
    y = np.array(x, copy=True)
    for a,b in regions(mask):
        n = b-a
        if method == 'attenuate':
            # Smooth 2 ms entry/exit ramps, 80% attenuation in the interior.
            gain = np.full(n, .2)
            ramp = min(32, n//2)
            if ramp:
                gain[:ramp] = np.linspace(1, .2, ramp)
                gain[-ramp:] = np.linspace(.2, 1, ramp)
            y[a:b] *= gain
        elif method == 'interpolate':
            left = x[a-1] if a > 0 else 0.
            right = x[b] if b < len(x) else 0.
            y[a:b] = np.linspace(left, right, n+2)[1:-1]
        else:
            raise ValueError(method)
    return y


def inject(clean, event, start, level_db, fs=FS):
    """Event RMS relative to local 100 ms speech RMS; exact support mask."""
    if start < 0 or start+len(event)>len(clean):
        raise ValueError('Event must fit completely in the recording')
    a,b = max(0,start-round(.05*fs)), min(len(clean),start+len(event)+round(.05*fs))
    # Silence reference is explicitly floored to 10% of whole-clip speech RMS.
    reference = max(np.sqrt(np.mean(clean[a:b]**2)), .1*np.sqrt(np.mean(clean**2)), EPS)
    scaled = event / (np.sqrt(np.mean(event**2))+EPS)*reference*10**(level_db/20)
    noise = np.zeros_like(clean); noise[start:start+len(event)] = scaled
    mask = np.zeros(len(clean), bool); mask[start:start+len(event)] = True
    return clean + noise, noise, mask, reference


def detection_counts(predicted, truth, fs=FS):
    """One-to-one event matching by interval overlap, no multiple TP credit."""
    p,t = regions(predicted), regions(truth)
    pairs=[]
    if p and t:
        overlap=np.array([[max(0,min(b,d)-max(a,c)) for c,d in t] for a,b in p])
        rows,cols=linear_sum_assignment(-overlap)
        pairs=[(i,j) for i,j in zip(rows,cols) if overlap[i,j]>0]
    errors=[abs(p[i][0]-t[j][0])/fs*1000 for i,j in pairs]
    return dict(tp=len(pairs), fp=len(p)-len(pairs), fn=len(t)-len(pairs),
                onset_error_ms=float(np.mean(errors)) if errors else None,
                sample_tp=int(np.sum(predicted & truth)),
                sample_fp=int(np.sum(predicted & ~truth)),
                sample_fn=int(np.sum(~predicted & truth)))


def prf(tp, fp, fn):
    return dict(precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def si_sdr(reference, estimate):
    r=reference-np.mean(reference); e=estimate-np.mean(estimate)
    projection=np.dot(e,r)/(np.dot(r,r)+EPS)*r
    return float(10*np.log10((np.dot(projection,projection)+EPS)/(np.sum((e-projection)**2)+EPS)))


def restoration_metrics(clean, output, truth, noisy=None):
    err=(output-clean)**2
    event_mse=float(np.mean(err[truth])) if np.any(truth) else None
    clean_rmse=float(np.sqrt(np.mean(err[~truth]))) if np.any(~truth) else None
    result=dict(event_rmse=np.sqrt(event_mse) if event_mse is not None else None,
                clean_rmse=clean_rmse, si_sdr_db=si_sdr(clean,output),
                changed_fraction=float(np.mean(np.abs(output-clean)>1e-9)))
    if noisy is not None and event_mse is not None:
        before=np.mean((noisy[truth]-clean[truth])**2)
        result['event_error_reduction_db']=float(10*np.log10((before+EPS)/(event_mse+EPS)))
    return result
