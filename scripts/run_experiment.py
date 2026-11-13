"""Reproduce all numbers, figures, exact masks and audio examples in one run."""
import argparse, csv, hashlib, json, platform, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import signal, ndimage
import soundfile as sf
from pystoi import stoi
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dsp import FS, features, detect, repair, inject, regions, prf, detection_counts, restoration_metrics
from src.neural import GTCRNEnhancer


def load_audio(path):
    x,fs=sf.read(path,dtype='float64',always_2d=True)
    x=x.mean(axis=1)
    if fs!=FS:
        from math import gcd
        g=gcd(fs,FS);x=signal.resample_poly(x,FS//g,fs//g)
    return x


def save_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,keys,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def clips_from_source():
    source=load_audio(ROOT/'audio/clean_speech.wav')
    # Fixed headroom; never independently normalise noisy/enhanced waveforms.
    source=source/(np.max(np.abs(source))+1e-12)*.15
    clips=[]
    for i in range(6):
        x=source[i*5*FS:(i+1)*5*FS]
        if len(x)!=5*FS: raise ValueError('Source must contain at least 30 seconds')
        clips.append(dict(id=f'clip{i+1}',split='development' if i<2 else 'test',
                          offset_seconds=i*5,x=x))
    return clips


def event_bank():
    rng=np.random.default_rng(5305)
    bank=[]
    for duration in [5,20]:
        n=round(duration*FS/1000)
        e=rng.normal(size=n)*np.hanning(n)
        bank.append(dict(name=f'synthetic_{duration}ms',x=e,kind='synthetic',duration_ms=duration))
    manifest=json.loads((ROOT/'data/esc50/manifest.json').read_text())
    bank.extend(crop_real_event(row) for row in manifest)
    return bank


def crop_real_event(row):
    """Use the same extraction procedure for original and held-out recordings."""
    x=load_audio(ROOT/'data/esc50'/row['filename'])
    envelope=np.sqrt(np.maximum(ndimage.uniform_filter1d(x*x,size=round(.002*FS)),0.))
    peak=int(np.argmax(envelope));floor=envelope[peak]*.10
    a=peak;b=peak+1
    while a>0 and envelope[a-1]>floor and peak-a<.075*FS: a-=1
    while b<len(x) and envelope[b]>floor and b-peak<.075*FS: b+=1
    # 2 ms guards and a short fade avoid hard-cut extraction clicks.
    a=max(0,a-round(.002*FS));b=min(len(x),b+round(.002*FS))
    e=x[a:b].copy();ramp=min(16,len(e)//2)
    e[:ramp]*=np.linspace(0,1,ramp);e[-ramp:]*=np.linspace(1,0,ramp)
    return dict(name=row['category'],x=e,kind='real',duration_ms=len(e)/FS*1000,
                     source=row['filename'],crop_start_seconds=a/FS,crop_end_seconds=b/FS)


def choose_locations(x, max_duration):
    """Acoustic proxies, not forced-aligned phoneme labels or test-detector tuning."""
    f=features(x);c=f['centers'];e=f['energy']
    valid=(c>round(.2*FS)) & (c<len(x)-max_duration-round(.2*FS))
    idx=np.flatnonzero(valid)
    # Lowest 100 ms energy region approximates an inter-word pause.
    smooth=ndimage.uniform_filter1d(e,size=51)
    silence=int(idx[np.argmin(smooth[idx])])
    # Steady voiced proxy: harmonicity, moderate/high energy, low flux.
    candidates=idx[e[idx]>np.percentile(e[idx],55)]
    harmonic=[]
    for j in candidates:
        a=int(c[j]);seg=x[a:a+640];seg=seg-np.mean(seg)
        ac=signal.correlate(seg,seg,mode='full',method='fft')[len(seg)-1:]
        harmonic.append(np.max(ac[40:321])/(ac[0]+1e-12)/(1+f['flux'][j]*4))
    vowel=int(candidates[np.argmax(harmonic)])
    transient_candidates=idx[(e[idx]>np.percentile(e[idx],40)) & (np.abs(c[idx]-c[vowel])>.15*FS)]
    transient=int(transient_candidates[np.argmax(f['flux'][transient_candidates])])
    return dict(silence=int(c[silence]),steady_voiced=int(c[vowel]),speech_transient=int(c[transient]))


def make_cases(clips,bank):
    cases=[]
    for clip in clips:
        loc=choose_locations(clip['x'],max(len(e['x']) for e in bank))
        clip['locations']=loc
        for event in bank:
            for condition,start in loc.items():
                for level in [-6,0,6]:
                    noisy,noise,mask,ref=inject(clip['x'],event['x'],start,level)
                    cases.append(dict(id=f"{clip['id']}_{event['name']}_{condition}_{level:+d}dB",
                                      clip=clip['id'],split=clip['split'],event=event['name'],
                                      kind=event['kind'],duration_ms=event['duration_ms'],condition=condition,
                                      level_db=level,start_sample=start,end_sample=start+len(event['x']),
                                      reference_rms=ref,clean=clip['x'],noisy=noisy,noise=noise,truth=mask))
    return cases


def calibrate(cases,clips):
    """Choose thresholds on development clips; clean inputs penalise false alarms."""
    dev=[c for c in cases if c['split']=='development']
    cached=[features(c['noisy']) for c in dev]
    clean=[c for c in clips if c['split']=='development']
    clean_features=[features(c['x']) for c in clean]
    table=[];best={}
    for detector,grid in [('energy',[3,6,9,12,15,18,21]),('stft',[1.5,2,2.5,3,4,5,6,8])]:
        candidates=[]
        for threshold in grid:
            tp=fp=fn=0
            for c,f in zip(dev,cached):
                m=detect(c['noisy'],detector,threshold,feat=f)
                r=detection_counts(m,c['truth']);tp+=r['tp'];fp+=r['fp'];fn+=r['fn']
            clean_fp=sum(len(regions(detect(c['x'],detector,threshold,feat=f))) for c,f in zip(clean,clean_features))
            # Weight each clean clip equally to all its 45 corrupted variants.
            score=prf(tp,fp+45*clean_fp,fn)['f1']
            row=dict(detector=detector,threshold=threshold,tp=tp,fp=fp,fn=fn,
                     clean_fp=clean_fp,selection_f1=score,**prf(tp,fp,fn))
            table.append(row);candidates.append(row)
        winner=max(candidates,key=lambda r:(r['selection_f1'],r['threshold']))
        best[detector]=winner['threshold']
    return best,table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results')
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    docs=ROOT/'docs/assets';docs.mkdir(parents=True,exist_ok=True)
    bank=event_bank();clips=clips_from_source();cases=make_cases(clips,bank)
    thresholds,calibration=calibrate(cases,clips)
    save_csv(out/'development_thresholds.csv',calibration)
    print('Frozen development thresholds:',thresholds,flush=True)
    neural=GTCRNEnhancer(ROOT/'third_party/gtcrn')
    neural(clips[0]['x']) # warm-up excluded; model loading excluded for all RTFs.
    test=[c for c in cases if c['split']=='test']
    rows=[];detections=[];clean_rows=[]
    for i,c in enumerate(test):
        outputs={'noisy':c['noisy']};timings={'noisy':0.}
        for detector in ['energy','stft']:
            tick=time.perf_counter();mask=detect(c['noisy'],detector,thresholds[detector]);dtime=time.perf_counter()-tick
            detections.append({k:c[k] for k in ['id','clip','event','kind','condition','duration_ms','level_db']} |
                              dict(detector=detector,predicted_events=len(regions(mask)),
                                   predicted_fraction=float(mask.mean()),**detection_counts(mask,c['truth'])))
            for method in (['interpolate'] if detector=='energy' else ['attenuate','interpolate']):
                tick=time.perf_counter();y=repair(c['noisy'],mask,method);rtime=time.perf_counter()-tick
                name=f'{detector}_{method}';outputs[name]=y;timings[name]=dtime+rtime
                assert np.array_equal(y[~mask],c['noisy'][~mask])
        tick=time.perf_counter();outputs['gtcrn']=neural(c['noisy']);timings['gtcrn']=time.perf_counter()-tick
        # Oracle mask is diagnostic only, never passed to the tested detectors.
        outputs['oracle_interpolate']=repair(c['noisy'],c['truth'])
        for name,y in outputs.items():
            if not np.all(np.isfinite(y)):raise RuntimeError(f'Nonfinite output: {name}')
            row={k:c[k] for k in ['id','clip','event','kind','condition','duration_ms','level_db']}
            row.update(method=name,**restoration_metrics(c['clean'],y,c['truth'],c['noisy']),
                       stoi=float(stoi(c['clean'],y,FS,extended=False)),
                       rtf=timings.get(name)/(len(y)/FS) if name in timings else None)
            rows.append(row)
        if (i+1)%30==0:print(f'Evaluated {i+1}/{len(test)} held-out mixtures',flush=True)
    for clip in [c for c in clips if c['split']=='test']:
        x=clip['x'];zero=np.zeros(len(x),bool)
        for detector in ['energy','stft']:
            tick=time.perf_counter();mask=detect(x,detector,thresholds[detector]);y=repair(x,mask)
            clean_rows.append(dict(clip=clip['id'],method=detector+'_interpolate',
                                  false_events=len(regions(mask)),false_events_per_minute=len(regions(mask))*12,
                                  rtf=(time.perf_counter()-tick)/5,**restoration_metrics(x,y,zero)))
        tick=time.perf_counter();y=neural(x)
        clean_rows.append(dict(clip=clip['id'],method='gtcrn',false_events=None,false_events_per_minute=None,
                              rtf=(time.perf_counter()-tick)/5,**restoration_metrics(x,y,zero)))
    save_csv(out/'detection_cases.csv',detections);save_csv(out/'restoration_cases.csv',rows)
    save_csv(out/'clean_preservation.csv',clean_rows)
    summary=aggregate(rows,detections,clean_rows)
    summary.update(thresholds=thresholds,test_mixtures=len(test),development_mixtures=len(cases)-len(test),
                   clean_test_seconds=20,model=neural.metadata,
                   environment=dict(python=sys.version,platform=platform.platform(),cpu=cpu_name(),
                                    numpy=np.__version__,scipy=__import__('scipy').__version__),
                   speech_sha256=hashlib.sha256((ROOT/'audio/clean_speech.wav').read_bytes()).hexdigest(),
                   speech_source='Open Speech Repository, OSR_us_000_0010_8k.wav; byte-identical mirror verified; see data/speech_manifest.json.',
                   events=[{k:v for k,v in e.items() if k!='x'} for e in bank],
                   clips=[{k:v for k,v in c.items() if k!='x'} for c in clips])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    manifest=[]
    for c in cases:
        manifest.append({k:v for k,v in c.items() if k not in ['clean','noisy','noise','truth']})
    (out/'case_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    example_cases=generate_examples(clips,bank,thresholds,neural,out,docs)
    from make_figures import make_figures
    make_figures(out,docs,rows,detections,clips,thresholds,example_cases)
    print('Completed experiment. Summary:',out/'summary.json',flush=True)


def cpu_name():
    path=Path('/proc/cpuinfo')
    if path.exists():
        return next((line.split(':',1)[1].strip() for line in path.read_text().splitlines()
                     if line.startswith('model name')),platform.machine())
    return platform.processor() or platform.machine()


def generate_examples(clips,bank,thresholds,neural,out,docs):
    """Same physical event amplitude at each location, separate from RMS sweep."""
    clip=next(c for c in clips if c['id']=='clip3')
    event=next(e for e in bank if e['name']=='synthetic_20ms')
    base=next(c for c in make_cases([clip],[event]) if c['condition']=='steady_voiced' and c['level_db']==6)
    fixed_event=base['noise'][base['start_sample']:base['end_sample']].copy()
    examples=[]
    for condition,start in clip['locations'].items():
        clean=clip['x'];noise=np.zeros_like(clean);noise[start:start+len(fixed_event)]=fixed_event
        truth=np.zeros(len(clean),bool);truth[start:start+len(fixed_event)]=True
        noisy=clean+noise
        c=dict(clean=clean,noisy=noisy,noise=noise,truth=truth,condition=condition,
               start_sample=start,end_sample=start+len(fixed_event),duration_ms=20)
        outputs={'noisy':noisy}
        for detector in ['energy','stft']:
            mask=detect(noisy,detector,thresholds[detector])
            for method in (['interpolate'] if detector=='energy' else ['attenuate','interpolate']):
                outputs[f'{detector}_{method}']=repair(noisy,mask,method)
        outputs['gtcrn']=neural(noisy)
        outputs['oracle_interpolate']=repair(noisy,truth)
        examples.append((c,outputs))
        path=docs/'audio'/condition;path.mkdir(parents=True,exist_ok=True)
        for name,y in {'clean':clean,**outputs}.items():sf.write(path/f'{name}.wav',y,FS,subtype='PCM_16')
        np.savez_compressed(out/f'example_{condition}.npz',clean=clean,noise=noise,noisy=noisy,
                            event_mask=truth,sample_rate=FS,
                            **{n:y for n,y in outputs.items() if n!='noisy'})
    real=next(c for c in make_cases([clip],bank) if c['event']=='mouse_click' and c['condition']=='speech_transient' and c['level_db']==6)
    mask=detect(real['noisy'],'stft',thresholds['stft'])
    path=docs/'audio/real_mouse_click';path.mkdir(parents=True,exist_ok=True)
    for name,y in dict(clean=real['clean'],noisy=real['noisy'],stft_interpolate=repair(real['noisy'],mask),gtcrn=neural(real['noisy'])).items():
        sf.write(path/f'{name}.wav',y,FS,subtype='PCM_16')
    return examples


def aggregate(rows,detections,clean_rows):
    summary={'restoration':{},'detection':{},'clean_preservation':{}}
    for name in sorted(set(r['method'] for r in rows)):
        part=[r for r in rows if r['method']==name]
        values={}
        for key in ['event_rmse','clean_rmse','event_error_reduction_db','si_sdr_db','stoi','rtf']:
            v=[r[key] for r in part if r.get(key) is not None]
            if v: values[key+'_mean']=float(np.mean(v));values[key+'_median']=float(np.median(v))
        summary['restoration'][name]=values
    for detector in ['energy','stft']:
        part=[r for r in detections if r['detector']==detector]
        counts={k:sum(r[k] for r in part) for k in ['tp','fp','fn','sample_tp','sample_fp','sample_fn']}
        err=[r['onset_error_ms'] for r in part if r['onset_error_ms'] is not None]
        summary['detection'][detector]=dict(**counts,**prf(counts['tp'],counts['fp'],counts['fn']),
                                          onset_error_ms_mean=float(np.mean(err)) if err else None,
                                          sample_metrics=prf(counts['sample_tp'],counts['sample_fp'],counts['sample_fn']))
    for name in sorted(set(r['method'] for r in clean_rows)):
        part=[r for r in clean_rows if r['method']==name]
        summary['clean_preservation'][name]=dict(clean_rmse_mean=float(np.mean([r['clean_rmse'] for r in part])),
                                               changed_fraction_mean=float(np.mean([r['changed_fraction'] for r in part])),
                                               false_events=sum(r['false_events'] or 0 for r in part) if name!='gtcrn' else None)
    return summary


if __name__=='__main__':main()
