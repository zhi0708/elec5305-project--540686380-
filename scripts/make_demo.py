"""Run the project CLI and assemble a captioned video from its real output."""
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dsp import repair
from src.transient import detect_transient

INPUT = Path('docs/assets/audio/real_mouse_click/noisy.wav')
OUTPUT = ROOT/'docs/assets/demo.mp4'
ENVIRONMENT = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')


def frame(path, title, lines, terminal=False):
    canvas = Image.new('RGB', (1280, 720), '#101d29')
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype('DejaVuSans.ttf', 27)
    heading = ImageFont.truetype('DejaVuSans-Bold.ttf', 40)
    mono = ImageFont.truetype('DejaVuSansMono.ttf', 21)
    draw.rectangle((0, 0, 1280, 10), fill='#68d9c5')
    draw.text((55, 43), 'ELEC5305  /  SPARSE IMPULSIVE NOISE', font=regular, fill='#68d9c5')
    draw.text((55, 102), title, font=heading, fill='#ffffff')
    for i, line in enumerate(lines):
        draw.text((58, 195+i*41), line, font=mono if terminal else regular, fill='#dbe8f1')
    canvas.save(path)


def run_command(arguments):
    process = subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True,
                             capture_output=True, text=True, env=ENVIRONMENT)
    command = '$ ' + shlex.join(['python', *arguments])
    import textwrap
    return textwrap.wrap(command, width=89, subsequent_indent='  ') + process.stdout.splitlines()


def main():
    summary = json.loads((ROOT/'results/extended_summary.json').read_text())
    detection = summary['detection']
    restoration = summary['restoration']
    with TemporaryDirectory(prefix='demo-', dir=ROOT) as folder:
        temp = Path(folder)
        relative = temp.relative_to(ROOT)
        local = relative/'local.wav'
        neural = relative/'gtcrn.wav'
        # Execute the same entry point and pretrained checkpoint used for the project.
        local_log = run_command(['scripts/enhance.py', str(INPUT), str(local), '--method', 'transient'])
        neural_log = run_command(['scripts/enhance.py', str(INPUT), str(neural), '--method', 'gtcrn'])
        mask = json.loads((ROOT/local.with_suffix('.events.json')).read_text())
        rate = mask['sample_rate']
        intervals = mask['predicted_intervals_samples']
        assert intervals and all(sf.info(ROOT/p).frames == sf.info(ROOT/INPUT).frames for p in (local, neural))
        start, end = intervals[0]
        clips = []

        def scene(title, lines, duration, wav=None, terminal=False):
            index = len(clips)
            image = temp/f'{index:02}.png'
            video = temp/f'{index:02}.mp4'
            frame(image, title, lines, terminal)
            cmd = ['ffmpeg','-nostdin','-y','-loglevel','error','-loop','1','-framerate','20','-i',str(image)]
            cmd += ['-i',str(wav)] if wav else ['-f','lavfi','-i','anullsrc=r=16000:cl=mono']
            cmd += ['-t',str(duration),'-pix_fmt','yuv420p','-c:v','libx264','-threads','2',
                    '-preset','fast','-crf','25','-c:a','aac','-ar','16000','-ac','1','-b:a','64k',str(video)]
            subprocess.run(cmd, check=True)
            clips.append(video)

        scene('Repairing a click in speech', [
            'Leona Wang  |  Local DSP repair versus pretrained GTCRN',
            'Input: speech with a real ESC-50 mouse click at a known position.',
            'Compare event repair, alteration outside the event, and CPU cost.',
        ], 5)
        scene('The local processing code', [
            'src/transient.py', *inspect.getsource(detect_transient).splitlines()[-3:],
            'src/dsp.py',
            next(line for line in inspect.getsource(repair).splitlines() if 'np.linspace(left, right' in line).strip(),
            'A short-context score locates candidate samples.',
            'Interpolation joins the surrounding samples across that interval.',
        ], 8, terminal=True)
        scene('Run 1  /  Local detection and repair', [
            *local_log, '', f'Actual mask: samples [{start}, {end}) at {rate} Hz',
            f'Detected interval: {start/rate:.3f} to {end/rate:.3f} seconds.',
            'Output: one repaired WAV and a JSON file containing the event mask.',
        ], 8, terminal=True)
        scene('Run 2  /  Pretrained GTCRN', [
            *neural_log, '', 'The same noisy WAV is processed end to end.',
            'The model estimates enhanced STFT frames, then reconstructs the WAV.',
        ], 8, terminal=True)
        for label, wav in [('Input with mouse click', INPUT), ('Local repair output', local), ('GTCRN output', neural)]:
            scene(label, ['Listen near 2.64 seconds.', 'All three WAVs use the same playback gain.'], 5, ROOT/wav)
        scene('Results on 1,728 held-out mixtures', [
            f"Retuned STFT event F1: {detection['stft_retuned']['f1']:.3f}",
            f"Short-context event F1: {detection['transient']['f1']:.3f}",
            f"Event error reduction: local {restoration['transient_interpolate']['event_error_reduction_db_mean']:.2f} dB / GTCRN {restoration['gtcrn']['event_error_reduction_db_mean']:.2f} dB.",
            'Local repair changed less speech outside the noise interval.',
            'The weakest level still produced 475 misses out of 576.',
        ], 8)
        listing = temp/'concat.txt'
        listing.write_text(''.join(f"file '{clip.name}'\n" for clip in clips))
        subprocess.run(['ffmpeg','-nostdin','-y','-loglevel','error','-f','concat','-safe','0',
                        '-i',str(listing),'-c','copy','-movflags','+faststart',str(OUTPUT)], check=True)
    print(OUTPUT)


if __name__ == '__main__':
    main()
