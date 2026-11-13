"""Run both enhancement methods and create a local listening page."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import soundfile as sf

from enhance import ROOT
from run_experiment import load_audio

SAMPLE = ROOT / 'docs/assets/audio/real_mouse_click/noisy.wav'
MODEL = ROOT / 'third_party/gtcrn'
TRACKS = (
    ('input.wav', 'Noisy input'),
    ('local.wav', 'Local DSP repair'),
    ('gtcrn.wav', 'Pretrained GTCRN'),
)


def listening_page():
    tracks = '\n'.join(
        f'<section><h2>{html.escape(label)}</h2>'
        f'<audio controls preload="metadata" src="{name}"></audio></section>'
        for name, label in TRACKS
    )
    return f'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech enhancement comparison</title>
<style>
body {{ max-width: 42rem; margin: 3rem auto; padding: 0 1.25rem;
        font: 1rem/1.5 system-ui, sans-serif; color: #17313d; background: #f7faf9; }}
section {{ margin: 1rem 0; padding: .75rem 1rem; background: white; border: 1px solid #dce6e2; }}
h2 {{ font-size: 1rem; margin: 0 0 .5rem; }}
audio {{ width: 100%; }}
</style>
<h1>Speech enhancement comparison</h1>
<p>Play the same recording before and after each method.</p>
{tracks}
</html>
'''


def run(input_path, destination):
    x = load_audio(input_path)
    from src.dsp import FS

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as workspace:
        staging = Path(workspace) / 'results'
        staging.mkdir()
        # Preserve floating-point input for processing and use PCM for playback.
        source = Path(workspace) / 'source.wav'
        sf.write(source, x, FS, subtype='DOUBLE')
        sf.write(staging / 'input.wav', x, FS, subtype='PCM_16')
        for method, name, label in (
            ('transient', 'local.wav', 'Local DSP repair'),
            ('gtcrn', 'gtcrn.wav', 'Pretrained GTCRN'),
        ):
            output = Path(workspace) / name
            print(f'Running {label}...', flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / 'scripts/enhance.py'), str(source),
                 str(output), '--method', method],
                check=True, stdout=subprocess.DEVNULL,
            )
            restored, rate = sf.read(output)
            sf.write(staging / name, restored, rate, subtype='PCM_16')
        metadata = (Path(workspace) / 'local.events.json').read_text(encoding='utf-8')
        (staging / 'local.events.json').write_text(metadata, encoding='utf-8')
        (staging / 'index.html').write_text(listening_page(), encoding='utf-8')
        staging.rename(destination)

    intervals = json.loads(metadata)['predicted_intervals_samples']
    print(f'Detected intervals: {len(intervals)}')
    for start, end in intervals[:5]:
        print(f'  {start / FS:.3f} to {end / FS:.3f} seconds')
    print(f'Listening page: {destination / "index.html"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, default=SAMPLE, help='Input WAV')
    parser.add_argument('--output-dir', type=Path, default=Path('demo_output'),
                        help='New directory for the WAV files, event intervals and listening page')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input WAV not found: {args.input}')
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f'Output directory already exists: {args.output_dir}. Choose a new --output-dir.')
    if not (MODEL / 'gtcrn.py').is_file() or not (MODEL / 'checkpoints/model_trained_on_dns3.tar').is_file():
        parser.error('GTCRN files are missing. Run python scripts/prepare_data.py first.')
    run(args.input.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
