import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .image_ops import bicubic, circular_mask, known_context, read_rgb, upscale_mask
from .metrics import write_csv
from .study import METHODS

METRICS = ('psnr_db', 'lpips', 'clip_iqa', 'psnr_roi_db', 'psnr_boundary_db', 'seconds')
LABELS = ('PSNR, dB (higher)', 'LPIPS (lower)', 'CLIP-IQA (higher)', 'ROI PSNR, dB (higher)',
          'Boundary PSNR, dB (higher)', 'Time, s')
CONTRASTS = ('mixed_vs_post', 'repaint_vs_post', 'repaint_vs_mixed')


def mean(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None and not math.isnan(float(row[key]))]
    return float(np.mean(values)) if values else None


def finite(value):
    return value if value is not None and math.isfinite(value) else np.nan


def study_report(data, out, rows, pairs):
    degradations = list(dict.fromkeys(r['degradation'] for r in rows))
    areas = sorted({r['area_percent'] for r in pairs})
    ids = list(dict.fromkeys(r['image'] for r in rows))
    summary, paired_summary = [], []
    for deg in degradations:
        subset = [r for r in rows if r['degradation'] == deg]
        for name in dict.fromkeys(r['mask'] for r in subset):
            group = [r for r in subset if r['mask'] == name]
            summary.append({'degradation': deg, 'mask': name, 'method': group[0]['method'],
                            'area_percent': group[0]['area_percent'], 'coverage': mean(group, 'coverage'),
                            'n': len(group), **{key: mean(group, key) for key in METRICS},
                            'outside_max_error': max((r['outside_max_error'] for r in group
                                                      if r['outside_max_error'] is not None), default=None)})
        for area in areas:
            group = [r for r in pairs if r['degradation'] == deg and r['area_percent'] == area]
            record = {'degradation': deg, 'area_percent': area, 'n': len(group)}
            for key in ('psnr_roi_db', 'psnr_boundary_db', 'lpips', 'clip_iqa', 'seconds'):
                for contrast in CONTRASTS:
                    field = f'{contrast}_{key}'
                    record[field] = mean(group, field)
            for contrast in CONTRASTS:
                field = f'{contrast}_psnr_roi_db'
                values = [r[field] for r in group if math.isfinite(r[field])]
                record[f'{contrast}_wins'] = sum(v > 0 for v in values)
                record[f'{contrast}_valid'] = len(values)
            paired_summary.append(record)
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'paired_summary.csv', paired_summary)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    for key, label in zip(METRICS, LABELS):
        if not any(math.isfinite(finite(r[key])) for r in summary):
            continue
        fig, axes = plt.subplots(1, len(degradations), figsize=(4.2 * len(degradations), 4), squeeze=False, sharey=True)
        for ax, deg in zip(axes[0], degradations):
            for method in METHODS:
                values = [next(r[key] for r in summary if r['degradation'] == deg and r['method'] == method and r['area_percent'] == a) for a in areas]
                ax.plot(areas, [finite(v) for v in values], marker='o', label=method)
            ax.set_title(deg)
            ax.set_xticks(areas)
            ax.set_xlabel('Unknown area, %')
            ax.grid(alpha=.2)
        axes[0, 0].set_ylabel(label)
        axes[0, -1].legend()
        fig.tight_layout()
        fig.savefig(figures / f'{key}.png', dpi=130)
        plt.close(fig)
    for key, label in (('psnr_roi_db', 'Delta ROI PSNR, dB'), ('psnr_boundary_db', 'Delta boundary PSNR, dB')):
        fig, axes = plt.subplots(1, len(degradations), figsize=(4.2 * len(degradations), 4), squeeze=False, sharey=True)
        for ax, deg in zip(axes[0], degradations):
            for contrast in CONTRASTS:
                values = [next(r[f'{contrast}_{key}'] for r in paired_summary if r['degradation'] == deg and r['area_percent'] == a) for a in areas]
                ax.plot(areas, [finite(v) for v in values], marker='o', label=contrast.replace('_', ' '))
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_title(deg)
            ax.set_xticks(areas)
            ax.set_xlabel('Unknown area, %')
            ax.grid(alpha=.2)
        axes[0, 0].set_ylabel(label)
        axes[0, -1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f'gain_{key}.png', dpi=130)
        plt.close(fig)
    for identifier in ids:
        ref = read_rgb(data / 'hr' / f'{identifier}.png')
        for deg in degradations:
            lr = read_rgb(data / 'lq' / deg / f'{identifier}.png')
            path = out / 'images' / deg / identifier
            full = read_rgb(path / 'ones.png')
            fig, axes = plt.subplots(len(areas), 8, figsize=(20, 2.7 * len(areas)), squeeze=False)
            for row_axes, area in zip(axes, areas):
                mask = circular_mask(lr.shape[:2], area)
                items = [('HR', ref), ('Bicubic', bicubic(lr)), ('Ones', full),
                         (f'Mask {area}%', upscale_mask(mask)), ('Context', known_context(lr, mask, ref))]
                items += [(method, read_rgb(path / f'{method}_{area}.png')) for method in METHODS]
                for ax, (name, image) in zip(row_axes, items):
                    ax.imshow(image, cmap='gray', vmin=0, vmax=1) if image.ndim == 2 else ax.imshow(image)
                    ax.set_title(name)
                    ax.axis('off')
            fig.suptitle(f'{identifier} — {deg}')
            fig.tight_layout()
            fig.savefig(figures / f'{identifier}_{deg}.png', dpi=115)
            plt.close(fig)
