import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .image_ops import bicubic, known_context, read_mask, read_rgb, upscale_mask
from .metrics import write_csv

METRICS = ('psnr_db', 'lpips', 'clip_iqa', 'psnr_roi_db', 'seconds')
LABELS = ('PSNR, дБ', 'LPIPS', 'CLIP-IQA', 'PSNR общей ROI, дБ', 'Время, с')


def mean(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None and not math.isnan(float(row[key]))]
    return float(np.mean(values)) if values else None


def report(data, out, rows):
    degradations = list(dict.fromkeys(row['degradation'] for row in rows))
    names = list(dict.fromkeys(row['mask'] for row in rows))
    ids = list(dict.fromkeys(row['image'] for row in rows))
    summary = []
    for degradation in degradations:
        for name in names:
            group = [row for row in rows if row['degradation'] == degradation and row['mask'] == name]
            summary.append({'degradation': degradation, 'mask': name, 'n': len(group),
                            **{metric: mean(group, metric) for metric in METRICS},
                            'outside_max_error': max((row['outside_max_error'] for row in group
                                                      if row['outside_max_error'] is not None), default=None)})
    write_csv(out / 'summary.csv', summary)
    pairs = []
    for identifier in ids:
        for degradation in degradations:
            group = {row['mask']: row for row in rows if row['image'] == identifier and row['degradation'] == degradation}
            if 'ones' in group and 'mixed' in group:
                full, mixed = group['ones']['psnr_roi_db'], group['mixed']['psnr_roi_db']
                pairs.append({'image': identifier, 'degradation': degradation, 'ones_roi_db': full,
                              'mixed_roi_db': mixed, 'gain_roi_db': mixed - full})
    if pairs:
        write_csv(out / 'roi_comparison.csv', pairs)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    for metric, label in zip(METRICS, LABELS):
        if not any(row[metric] is not None and math.isfinite(row[metric]) for row in summary):
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            vals = [next(row[metric] for row in summary if row['degradation'] == deg and row['mask'] == name)
                    for deg in degradations]
            vals = [v if v is not None and math.isfinite(v) else np.nan for v in vals]
            ax.bar(np.arange(len(degradations)) + (i - (len(names)-1)/2) * width, vals, width, label=name)
        ax.set_xticks(np.arange(len(degradations)), degradations)
        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f'{metric}.png', dpi=130)
        plt.close(fig)
    for identifier in ids:
        ref = read_rgb(data / 'hr' / f'{identifier}.png')
        fig, axes = plt.subplots(len(degradations), len(names) + 4, figsize=(2.6*(len(names)+4), 2.7*len(degradations)), squeeze=False)
        for i, deg in enumerate(degradations):
            lr = read_rgb(data / 'lq' / deg / f'{identifier}.png')
            mask = upscale_mask(read_mask(data / 'masks/mixed.png', lr.shape[:2]))
            items = [('HR', ref), ('Bicubic', bicubic(lr)), ('ROI', mask),
                     ('Mixed context', known_context(lr, read_mask(data / 'masks/mixed.png', lr.shape[:2]), ref))]
            items += [(name, read_rgb(out / 'images' / deg / identifier / f'{name}.png')) for name in names]
            for ax, (name, image) in zip(axes[i], items):
                ax.imshow(image, cmap='gray', vmin=0, vmax=1) if image.ndim == 2 else ax.imshow(image)
                ax.set_title(f'{deg}\n{name}')
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(figures / f'{identifier}.png', dpi=120)
        plt.close(fig)
