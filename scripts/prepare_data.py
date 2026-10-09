"""Fetch three small ESC-50 records and a pinned official GTCRN checkout."""
import csv, hashlib, io, json, subprocess
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
GTCRN_COMMIT='502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73'
GTCRN_SOURCE_SHA256='20bf0610702506fa45cd543454b84bbeb862d73c7c20a3f8d820fdc04da6099b'
GTCRN_CHECKPOINT_SHA256='a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79'
EXPECTED_HASHES={
    '1-101336-A-30.wav':'e738266e395e378102f74e0920b05ff5937543788420d5b2a6b8d3df8bdb3910',
    '1-118206-A-31.wav':'4461fa26f119749f99a4762be421d0d64e774960be7eef9780812d48580e48fb',
    '1-137-A-32.wav':'d68e4716f74f073be8f78f5fe8e45daf7c09b0ec0575b21f57989d803536abb3',
}


def fetch(url,path):
    if not path.exists():
        response=requests.get(url,timeout=90);response.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=ROOT/'data/esc50';root.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/karolpiczak/ESC-50/master/'
    fetch(base+'meta/esc50.csv',root/'esc50.csv')
    fetch(base+'LICENSE',root/'LICENSE.txt')
    wanted=['1-118206-A-31.wav','1-101336-A-30.wav','1-137-A-32.wav']
    records=[]
    for row in csv.DictReader(io.StringIO((root/'esc50.csv').read_text())):
        if row['filename'] in wanted:
            row['url']=base+'audio/'+row['filename']
            row['sha256']=fetch(row['url'],root/row['filename']);records.append(row)
            if row['sha256']!=EXPECTED_HASHES[row['filename']]:
                raise RuntimeError(f"ESC-50 content changed: {row['filename']}")
    if len(records)!=3: raise RuntimeError('Missing ESC-50 records')
    (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    gt=ROOT/'third_party/gtcrn'
    if not gt.exists():
        subprocess.run(['git','clone','https://github.com/Xiaobin-Rong/gtcrn.git',str(gt)],check=True)
    if (gt/'.git').exists():
        subprocess.run(['git','-C',str(gt),'checkout',GTCRN_COMMIT],check=True)
    # The download archive can contain the exact source/weights without git history.
    if hashlib.sha256((gt/'gtcrn.py').read_bytes()).hexdigest()!=GTCRN_SOURCE_SHA256:
        raise RuntimeError('GTCRN source differs from the pinned official version')
    if hashlib.sha256((gt/'checkpoints/model_trained_on_dns3.tar').read_bytes()).hexdigest()!=GTCRN_CHECKPOINT_SHA256:
        raise RuntimeError('GTCRN checkpoint differs from the official DNS3 weights')
    print('Prepared ESC-50 recordings and official pretrained GTCRN.')


if __name__=='__main__':main()
