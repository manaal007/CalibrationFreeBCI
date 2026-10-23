"""Cue-aligned source imagery PSD; historical version is in the original archive."""
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrationfreebci.data import load_bnci, CHANNELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--out', type=Path, default=Path('runs/imagery_psd.png'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    x, metadata, _ = load_bnci([args.subject], args.data_root)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, channel in zip(axes, ['C3', 'C4']):
        for label, condition in [(1, 'left'), (2, 'right')]:
            frequencies, power = welch(x[metadata.label == label, CHANNELS.index(channel)],
                                       fs=250, nperseg=500, axis=-1)
            keep = (frequencies >= 8) & (frequencies <= 30)
            ax.plot(frequencies[keep], 10*np.log10(np.maximum(power.mean(axis=0)[keep]*1e12, 1e-20)), label=condition)
        ax.set(title=channel, xlabel='Frequency (Hz)', ylabel='PSD (dB re 1 microvolt squared/Hz)')
        ax.legend()
    fig.suptitle(f'BNCI participant {args.subject}: 0.5-2.5 s after imagery cue, both sessions')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
