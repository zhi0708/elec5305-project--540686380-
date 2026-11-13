"""Download the pinned speaker archives and the extra ESC-50 test events."""
import hashlib
from pathlib import Path
import shutil

import requests

from prepare_data import ROOT, fetch

REVISION = 'cc02fd37197966e5de11ad737f7affd577c61992'
ARCHIVES = {
    'ASI': '302dc53af870acb8378c2c8205062fb7af1b3ad8ebc61826f87faf7d3dc4e8ff',
    'HKK': '6a2fc1b0ced8c65cfc231a57050016866d66981efe3b9d48c5c0fbcec121e088',
    'BWC': 'd3b5f304647968babd69468c3fcf691eaac4fc4dd3a55c595ee9cb47cd8e5c2f',
    'TLV': '86a0bfee50347f6c0cd506deb7d88fad7922155172699f921e20b66395a21461',
    'YDCK': '314ce2580c30508096efb4fab9e287bbb4c7db9cd89bcae53d90afca60417ce3',
}
EXTRA_EVENTS = {
    '2-104168-A-32.wav': '62f9d3cd5d576e5dd6f956fe11d5c50fb04f054c38e32597a16e19021536943d',
    '2-114254-A-30.wav': '5717c5c7a3bce7f38c9c444ef54f4c702cca53a6e368a6dc5b1cfb8388f7f9eb',
    '2-119139-A-31.wav': 'ddb156148e7cd6fc096f0d6192f4bfbad9c744be7c98c8021f8767e069c79bbf',
}


def download_archive(speaker, expected):
    folder = ROOT / 'data/l2_arctic'
    folder.mkdir(exist_ok=True)
    path = folder / f'{speaker}.zip'
    temporary = path.with_suffix('.part')
    if not path.exists():
        url = f'https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/resolve/{REVISION}/{speaker}.zip'
        with requests.get(url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                shutil.copyfileobj(response.raw, output)
        temporary.replace(path)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != expected:
        raise ValueError(f'Archive checksum mismatch: {path}')


def main():
    for speaker, expected in ARCHIVES.items():
        download_archive(speaker, expected)
        print(f'Verified L2-ARCTIC speaker {speaker}', flush=True)
    for filename, expected in EXTRA_EVENTS.items():
        path = ROOT / 'data/esc50' / filename
        url = f'https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}'
        if fetch(url, path) != expected:
            raise ValueError(f'Event checksum mismatch: {path}')


if __name__ == '__main__':
    main()
