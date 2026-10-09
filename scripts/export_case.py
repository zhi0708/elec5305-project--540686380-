"""Materialise any manifest case into clean/noise/noisy WAVs and an exact mask."""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
from run_experiment import ROOT,FS,clips_from_source,event_bank,make_cases


def main():
    p=argparse.ArgumentParser();p.add_argument('case_id');p.add_argument('--output',type=Path,default=ROOT/'data/cases')
    args=p.parse_args()
    cases=make_cases(clips_from_source(),event_bank())
    matches=[c for c in cases if c['id']==args.case_id]
    if not matches:raise SystemExit('Unknown case ID; see results/case_manifest.json')
    c=matches[0];out=args.output/c['id'];out.mkdir(parents=True,exist_ok=True)
    for name in ['clean','noise','noisy']:sf.write(out/f'{name}.wav',c[name],FS,subtype='FLOAT')
    np.save(out/'event_mask.npy',c['truth'])
    (out/'metadata.json').write_text(json.dumps({k:v for k,v in c.items() if k not in ['clean','noise','noisy','truth']},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
