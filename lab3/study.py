import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, __version__ as pillow_version

from .image_ops import circular_mask, composite, read_rgb, save_image, upscale_mask
from .metrics import QualityMetrics, preservation, psnr, write_csv

ROOT = Path(__file__).resolve().parent
METHODS = ('post_blend', 'mixed', 'repaint')


def inner_boundary(mask, width=8):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    interior = np.asarray(image.filter(ImageFilter.MinFilter(2 * width + 1))) != 0
    return mask & ~interior


def run_study(data, out, ids, degradations, areas, device, seed=1234, repeats=2, psnr_only=False):
    if not areas or len(set(areas)) != len(areas) or repeats < 2:
        raise ValueError('Use distinct mask areas and at least two repeats')
    from .model import ResShift
    from .study_report import study_report

    out.mkdir(parents=True, exist_ok=True)
    model = ResShift(device)
    quality = None if psnr_only else QualityMetrics(device)
    rows, pairs = [], []
    metadata = {'study': True, 'ids': list(ids), 'degradations': list(degradations),
                'areas_percent': list(areas), 'seed': seed, 'device': device, 'repeats': repeats,
                'boundary_width_hr_px': 8, 'python': platform.python_version(), 'numpy': np.__version__,
                'pillow': pillow_version, 'psnr_only': psnr_only}
    import torch
    metadata['torch'] = torch.__version__
    revision = ROOT / 'weights/revision.txt'
    if revision.exists():
        metadata['resshift_revision'] = revision.read_text().strip()
    (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')

    for identifier in ids:
        ref = read_rgb(data / 'hr' / f'{identifier}.png')
        for degradation in degradations:
            lr = read_rgb(data / 'lq' / degradation / f'{identifier}.png')
            if ref.shape != (*upscale_mask(np.ones(lr.shape[:2], bool)).shape, 3):
                raise ValueError('HR/LR size mismatch')

            def record(method, area, prediction, mask, elapsed, blend_seconds=None):
                name = method if area is None else f'{method}_{area}'
                extra = quality.score(prediction, ref) if quality else {'lpips': None, 'clip_iqa': None}
                row = {'image': identifier, 'degradation': degradation, 'mask': name,
                       'method': method, 'area_percent': area, 'coverage': float(mask.mean()),
                       'psnr_db': psnr(prediction, ref), **extra,
                       'psnr_roi_db': psnr(prediction, ref, mask) if area is not None else None,
                       'psnr_mask_db': psnr(prediction, ref, mask),
                       'psnr_boundary_db': psnr(prediction, ref, inner_boundary(mask)) if area is not None else None,
                       'outside_max_error': preservation(prediction, ref, mask), 'seconds': elapsed,
                       'blend_seconds': blend_seconds}
                rows.append(row)
                save_image(out / 'images' / degradation / identifier / f'{name}.png', prediction)
                write_csv(out / 'metrics.csv', rows)
                print(identifier, degradation, name, f"PSNR={row['psnr_db']:.3f}",
                      f"seconds={elapsed:.3f}", flush=True)
                return row

            started = time.perf_counter()
            empty = ref.copy()
            record('zeros', None, empty, np.zeros(ref.shape[:2], bool), time.perf_counter() - started)
            started = time.perf_counter()
            full = model.restore(lr, np.ones(lr.shape[:2], bool), seed=seed)
            full_seconds = time.perf_counter() - started
            record('ones', None, full, np.ones(ref.shape[:2], bool), full_seconds)
            for area in areas:
                mask_lr = circular_mask(lr.shape[:2], area)
                mask = upscale_mask(mask_lr)
                save_image(out / 'masks' / f'{area}.png', mask_lr.astype(np.uint8) * 255)
                started = time.perf_counter()
                post = composite(full, ref, mask)
                blend_seconds = time.perf_counter() - started
                post_row = record('post_blend', area, post, mask, full_seconds + blend_seconds, blend_seconds)
                started = time.perf_counter()
                mixed = model.restore(lr, mask_lr, hr=ref, seed=seed)
                mixed_row = record('mixed', area, mixed, mask, time.perf_counter() - started)
                started = time.perf_counter()
                repaint = model.restore(lr, mask_lr, hr=ref, seed=seed, repeats=repeats)
                repaint_row = record('repaint', area, repaint, mask, time.perf_counter() - started)
                pair = {'image': identifier, 'degradation': degradation, 'area_percent': area,
                        'coverage': float(mask.mean()), 'ones_roi_db': psnr(full, ref, mask)}
                for key in ('psnr_roi_db', 'psnr_boundary_db', 'lpips', 'clip_iqa', 'seconds'):
                    for name, row in zip(METHODS, (post_row, mixed_row, repaint_row)):
                        pair[f'{name}_{key}'] = row[key]
                    for name, first, second in (('mixed_vs_post', mixed_row, post_row),
                                                ('repaint_vs_post', repaint_row, post_row),
                                                ('repaint_vs_mixed', repaint_row, mixed_row)):
                        a, b = first[key], second[key]
                        pair[f'{name}_{key}'] = None if a is None or b is None else a - b
                pairs.append(pair)
                write_csv(out / 'roi_comparison.csv', pairs)
    study_report(data, out, rows, pairs)
    from .ci_report import experiment_report, write_html, write_markdown
    write_markdown(out / 'report.md', experiment_report(out), sorted((out / 'figures').glob('*.png')))
    write_html(out)
    return rows, pairs
