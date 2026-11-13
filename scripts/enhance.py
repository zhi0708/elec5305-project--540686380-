"""Process an arbitrary WAV using frozen development-set detector thresholds."""
import argparse,json,sys
from pathlib import Path
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dsp import FS,detect,repair,regions
from src.neural import GTCRNEnhancer
from src.transient import detect_transient
from run_experiment import load_audio


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--method',choices=['energy','stft','transient','gtcrn'],default='stft')
    p.add_argument('--repair',choices=['interpolate','attenuate'],default='interpolate')
    a=p.parse_args();x=load_audio(a.input)
    if a.method=='gtcrn':
        y=GTCRNEnhancer(ROOT/'third_party/gtcrn')(x)
    else:
        if a.method == 'transient':
            settings = json.loads((ROOT/'results/extended_summary.json').read_text())['settings']
            mask = detect_transient(x, settings)
        else:
            thresholds = json.loads((ROOT/'results/summary.json').read_text())['thresholds']
            settings = {'threshold': thresholds[a.method]}
            mask = detect(x, a.method, settings['threshold'])
        y = repair(x, mask, a.repair)
        metadata=dict(sample_rate=FS,detector=a.method,**settings,
                      predicted_intervals_samples=[[int(s),int(e)] for s,e in regions(mask)])
        a.output.with_suffix('.events.json').parent.mkdir(parents=True,exist_ok=True)
        a.output.with_suffix('.events.json').write_text(json.dumps(metadata,indent=2)+'\n')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(a.output,y,FS,subtype='FLOAT')
    print(a.output)


if __name__=='__main__':main()
