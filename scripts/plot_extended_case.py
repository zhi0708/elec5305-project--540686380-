"""Plot the matched low- and high-level keyboard example discussed in the report."""
import matplotlib.pyplot as plt
import numpy as np

from run_extended_experiment import ROOT, FS, held_out_events, speech_clips
from run_experiment import event_bank, make_cases
from src.transient import detect_transient, transient_features, transient_score
from extended_results import load


def main():
    settings = load(ROOT/'results/extended_summary.json')['settings']
    clip = next(c for c in speech_clips() if c['id']=='BWC_arctic_a0011')
    cases = make_cases([clip], event_bank()+held_out_events())
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True)
    for ax, level in zip(axes, (-6, 6)):
        case = next(c for c in cases if c['event']=='keyboard_typing_fold2'
                    and c['condition']=='steady_voiced' and c['level_db']==level)
        noisy = case['noisy']
        feat = transient_features(noisy)
        score = transient_score(feat, settings['contrast_weight'], settings['floor_weight'])
        start, end = case['start_sample'], case['end_sample']
        relative = (feat['centers']-start)*1000/FS
        keep = (relative >= -35) & (relative <= 65)
        hit = np.any(detect_transient(noisy, settings, feat) & case['truth'])
        ax.axvspan(0, (end-start)*1000/FS, color='#c3e3dc', label='True insertion interval')
        ax.axhline(settings['threshold'], color='#a24730', ls='--', label='Frozen threshold')
        ax.plot(relative[keep], score[keep], color='#174f72', label='Detector score')
        ax.set_title(f'{level:+d} dB  |  {"detected" if hit else "missed"}', loc='left')
        ax.set_ylabel('Score')
    axes[-1].set_xlabel('Time from insertion onset (ms)')
    axes[0].legend(loc='upper right', fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(ROOT/'docs/assets/figures/06_extended_miss.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
