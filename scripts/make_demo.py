"""Create a 40-second captioned audio/figure demonstration, without narration."""
import subprocess,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
TEMP=ROOT/'data/demo_work'
def font(size):
    for name in ['DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/System/Library/Fonts/Supplemental/Arial.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default(size=size)


def main():
    TEMP.mkdir(parents=True,exist_ok=True)
    s=json.loads((ROOT/'results/summary.json').read_text())
    normal=font(27);small=font(20)
    cards=[]
    def card(title,lines,figure=None):
        canvas=Image.new('RGB',(1280,720),'#f8f7f2');draw=ImageDraw.Draw(canvas)
        draw.rectangle((0,0,1280,95),fill='#182f3b');draw.text((40,24),title,font=normal,fill='white')
        if figure:
            image=Image.open(ROOT/'docs/assets/figures'/figure).convert('RGB');image.thumbnail((1170,495))
            canvas.paste(image,((1280-image.width)//2,110))
            for j,line in enumerate(lines):draw.text((40,626+j*32),line,font=small,fill='#182f3b')
        else:
            for j,line in enumerate(lines):draw.text((55,155+j*65),line,font=normal,fill='#182f3b')
        path=TEMP/f'card{len(cards):02}.png';canvas.save(path);cards.append(path);return path
    clips=[]
    intro=card('ELEC5305 | Leona Wang | SID 540686380',[
        'Sparse impulsive noise: local DSP repair versus pretrained GTCRN',
        'Same inputs. Exact event masks. Frozen development thresholds.',
        '180 held-out mixtures; synthetic + real clicks, knocks and typing.',
        'Measure event restoration, collateral alteration and CPU runtime.',
        'One source recording: controlled results, not speaker generalisation.'])
    clips.append((intro,None))
    for name,title in [('clean','Listen: clean reference'),('noisy','Listen: identical inserted 20 ms event'),
                       ('stft_interpolate','Listen: STFT detection + local interpolation'),('gtcrn','Listen: official pretrained GTCRN')]:
        path=card(title,['Speech-transient location; every example uses the same signal scale.',
                        'Audio playback is an illustration, not a formal listening test.'],figure='03_overlap_waveforms.png')
        clips.append((path,ROOT/'docs/assets/audio/speech_transient'/f'{name}.wav'))
    p=card('Results | event restoration versus clean-region alteration',[
        f"GTCRN: {s['restoration']['gtcrn']['event_error_reduction_db_mean']:.2f} dB mean event-error reduction; larger collateral error.",
        'Conservative STFT-local repair preserves more clean speech but misses weak events.'],figure='02_tradeoff.png');clips.append((p,None))
    p=card('Results | false alarms and computational cost',[
        'Clean input: 1 STFT false event versus 24 energy false events in 20 seconds.',
        'RTF is measured on the same CPU; loading and warm-up are excluded.'],figure='05_quality_runtime.png');clips.append((p,None))
    p=card('Conclusion | preserving clean speech matters',[
        'Better local event removal does not guarantee better whole-file quality.',
        f"STFT event F1: {s['detection']['stft']['f1']:.3f}; low recall remains a major limitation.",
        'GTCRN is a general enhancer, not trained specifically for impulses.',
        'All code, masks, tables and limitations accompany the report.',
        'Next step: broader data, verified speech labels, improved local repair.']);clips.append((p,None))
    outputs=[]
    for i,(img,wav) in enumerate(clips):
        video=TEMP/f'segment{i:02}.mp4'
        cmd=['ffmpeg','-nostdin','-y','-loglevel','error','-loop','1','-i',str(img)]
        if wav:cmd+=['-i',str(wav)]
        else:cmd+=['-f','lavfi','-i','anullsrc=r=16000:cl=mono']
        cmd+=['-t','5','-vf','format=yuv420p','-r','20','-c:v','libx264','-preset','fast','-crf','23',
              '-c:a','aac','-ar','16000','-ac','1','-b:a','64k',str(video)]
        subprocess.run(cmd,check=True,stdin=subprocess.DEVNULL)
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
                                               '-of','default=noprint_wrappers=1:nokey=1',str(video)]))
        if duration<4.9:raise RuntimeError('Incomplete demonstration segment')
        outputs.append(video)
    listing=TEMP/'concat.txt';listing.write_text(''.join(f"file '{p.name}'\n" for p in outputs))
    output=ROOT/'docs/assets/demo.mp4'
    subprocess.run(['ffmpeg','-nostdin','-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),
                    '-c','copy','-movflags','+faststart',str(output)],check=True)
    print(output)


if __name__=='__main__':main()
