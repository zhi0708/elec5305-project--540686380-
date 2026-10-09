"""Five measured result figures, built from measured case-level results."""
from pathlib import Path
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from src.dsp import FS, features, detect, regions, prf

COLORS={'noisy':'#8c8c8c','energy_interpolate':'#e39c35','stft_attenuate':'#9457ad',
        'stft_interpolate':'#197b80','gtcrn':'#3763cc','oracle_interpolate':'#a5a5a5'}
LABELS={'noisy':'Unprocessed','energy_interpolate':'Energy + interpolation',
        'stft_attenuate':'STFT + attenuation','stft_interpolate':'STFT + interpolation',
        'gtcrn':'Pretrained GTCRN','oracle_interpolate':'Oracle interpolation (diagnostic)'}
CONDITIONS=['silence','steady_voiced','speech_transient']


def make_figures(out,assets,rows,detections,clips,thresholds,examples):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    figdir=assets/'figures';figdir.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight')
        target = figdir/name
        temporary = target.with_suffix('.png.tmp')
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(target)
        plt.close(fig)

    fig,ax=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
    for ci,condition in enumerate(CONDITIONS):
        for di,detector in enumerate(['energy','stft']):
            values={metric:[] for metric in ['precision','recall','f1']}
            for level in [-6,0,6]:
                group=[r for r in detections if r['condition']==condition and r['level_db']==level and r['detector']==detector]
                counts={k:sum(r[k] for r in group) for k in ['tp','fp','fn']};m=prf(**counts)
                for metric in values:values[metric].append(m[metric])
            for mi,metric in enumerate(values):
                ax[ci,mi].plot([-6,0,6],values[metric],'-o',label=detector.upper(),color=['#e39c35','#197b80'][di])
                ax[ci,mi].set_ylim(-.03,1.03);ax[ci,mi].grid(alpha=.2)
                ax[ci,mi].set_title(f'{condition.replace("_"," ").title()} | {metric.upper()}')
    for a in ax[-1]:a.set_xlabel('Event / local speech RMS ratio (dB)')
    ax[0,0].legend();fig.suptitle('1. Held-out event detection: strength and overlap condition',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'01_detection.png')

    fig,ax=plt.subplots(figsize=(9,6))
    for name in COLORS:
        group=[r for r in rows if r['method']==name]
        if not group:continue
        xx=np.mean([r['clean_rmse'] for r in group]);yy=np.mean([r['event_rmse'] for r in group])
        ax.scatter(max(xx,1e-8),yy,s=110,color=COLORS[name],label=LABELS[name])
    ax.set_xscale('log');ax.set_xlabel('Mean RMSE outside true event (log scale; zero shown at 1e-8)')
    ax.set_ylabel('Mean RMSE inside true event');ax.grid(alpha=.2)
    ax.set_title('2. Restoration versus collateral alteration\nLower and further left is better; oracle is not a deployed method')
    ax.legend(fontsize=9,loc='best');fig.tight_layout();save(fig,'02_tradeoff.png')

    fig,ax=plt.subplots(4,3,figsize=(14,10),sharex='col')
    names=['noisy','stft_attenuate','stft_interpolate','gtcrn']
    for ci,condition in enumerate(CONDITIONS):
        c,outputs=next((c,o) for c,o in examples if c['condition']==condition)
        a=max(0,c['start_sample']-round(.10*FS));b=min(len(c['clean']),c['end_sample']+round(.15*FS))
        t=(np.arange(a,b)-c['start_sample'])/FS*1000
        for mi,name in enumerate(names):
            ax[mi,ci].plot(t,c['clean'][a:b],color='#333',lw=.8,alpha=.6,label='Clean reference')
            ax[mi,ci].plot(t,outputs[name][a:b],color=COLORS[name],lw=.8,label=LABELS[name])
            ax[mi,ci].axvspan(0,c['duration_ms'],color='#e07060',alpha=.14)
            ax[mi,ci].grid(alpha=.15)
            if mi==0:ax[mi,ci].set_title(condition.replace('_',' ').title())
            if ci==0:ax[mi,ci].set_ylabel(LABELS[name]+'\nAmplitude')
        ax[-1,ci].set_xlabel('Time from inserted event onset (ms)')
    fig.suptitle('3. Identical 20 ms event at three locations (fixed waveform and absolute level)',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'03_overlap_waveforms.png')

    fig,ax=plt.subplots(4,3,figsize=(13,10),sharex='col',sharey=True)
    for ci,condition in enumerate(CONDITIONS):
        c,outputs=next((c,o) for c,o in examples if c['condition']==condition)
        a=max(0,c['start_sample']-round(.15*FS));b=min(len(c['clean']),c['end_sample']+round(.2*FS))
        for mi,name in enumerate(['clean','noisy','stft_interpolate','gtcrn']):
            x=c['clean'] if name=='clean' else outputs[name]
            f,t,z=signal.stft(x[a:b],FS,nperseg=128,noverlap=96,boundary=None)
            img=ax[mi,ci].pcolormesh(t*1000+(a-c['start_sample'])/FS*1000,f/1000,
                                     20*np.log10(np.abs(z)+1e-8),cmap='magma',vmin=-90,vmax=-25,shading='auto')
            ax[mi,ci].axvline(0,color='cyan',lw=.6);ax[mi,ci].axvline(c['duration_ms'],color='cyan',lw=.6)
            if mi==0:ax[mi,ci].set_title(condition.replace('_',' '))
            if ci==0:ax[mi,ci].set_ylabel((name if name=='clean' else LABELS[name])+'\nkHz')
        ax[-1,ci].set_xlabel('Time from event onset (ms)')
    fig.suptitle('Supplement: same inputs and outputs in the time-frequency domain',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'03_overlap_spectrograms.png')

    candidates=[]
    for clip in [c for c in clips if c['split']=='test']:
        mask=detect(clip['x'],'stft',thresholds['stft']);f=features(clip['x'])
        for a,b in regions(mask):
            center=(a+b)//2;j=np.argmin(abs(f['centers']-center))
            candidates.append((float(f['score'][j]),clip,a,b))
    candidates.sort(key=lambda z:z[0],reverse=True)
    if candidates:
        _,clip,a,b=candidates[0];title='4. Strongest false alarm on clean speech (no inserted noise)'
    else:
        clip=[c for c in clips if c['split']=='test'][0];f=features(clip['x'])
        a=int(f['centers'][np.argmax(f['score'])]);b=a+128;title='4. Highest-scoring clean candidate (no false alarm at frozen threshold)'
    x=clip['x'];f=features(x);lo=max(0,a-1600);hi=min(len(x),b+1600)
    t=np.arange(lo,hi)/FS;ft=f['centers']/FS;keep=(ft>=lo/FS)&(ft<hi/FS)
    fig,ax=plt.subplots(5,1,figsize=(11,10),sharex=True)
    ax[0].plot(t,x[lo:hi],lw=.8);ax[0].set_ylabel('Waveform')
    freqs,times,z=signal.stft(x[lo:hi],FS,nperseg=128,noverlap=96,boundary=None)
    ax[1].pcolormesh(times+lo/FS,freqs/1000,20*np.log10(abs(z)+1e-8),cmap='magma',shading='auto');ax[1].set_ylabel('Frequency (kHz)')
    ax[2].plot(ft[keep],f['energy'][keep]);ax[2].set_ylabel('Frame energy')
    ax[3].plot(ft[keep],f['flux'][keep]);ax[3].set_ylabel('Spectral flux')
    ax[4].plot(ft[keep],f['score'][keep]);ax[4].axhline(thresholds['stft'],ls='--',color='r',label='Frozen threshold')
    ax[4].set_ylabel('STFT score');ax[4].set_xlabel(f'Time in {clip["id"]} (seconds)');ax[4].legend()
    for aplot in ax:aplot.axvspan(a/FS,b/FS,alpha=.14,color='#e07060')
    fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.96));save(fig,'04_clean_false_positive.png')

    fig,ax=plt.subplots(figsize=(9,6))
    for name in ['energy_interpolate','stft_attenuate','stft_interpolate','gtcrn']:
        group=[r for r in rows if r['method']==name]
        rtf=np.median([r['rtf'] for r in group]);quality=np.mean([r['event_error_reduction_db'] for r in group])
        ax.scatter(rtf,quality,color=COLORS[name],s=110,label=LABELS[name])
    ax.axhline(0,ls='--',color='#777',lw=.8);ax.set_xscale('log');ax.grid(alpha=.2)
    ax.set_xlabel('Median real-time factor (same CPU, model loading/warm-up excluded)')
    ax.set_ylabel('Mean event-region error reduction (dB; higher is better)')
    ax.set_title('5. Quality versus processing cost\nIncludes detection and repair, or neural STFT/model/iSTFT')
    ax.legend(fontsize=9);fig.tight_layout();save(fig,'05_quality_runtime.png')
