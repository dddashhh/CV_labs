import argparse
import json
import time
import platform
from pathlib import Path

import numpy as np

from .image_ops import bicubic, known_context, read_mask, read_rgb, save_image, upscale_mask
from .metrics import QualityMetrics, preservation, psnr, write_csv
from .prepare_data import DEGRADATIONS, IDS, prepare_official

ROOT = Path(__file__).resolve().parent
MASKS = ('zeros', 'ones', 'mixed')


def run(data, out, ids, degradations, mask_names, device, seed, baseline_only=False, psnr_only=False):
    out.mkdir(parents=True, exist_ok=True)
    if baseline_only:
        mask_names = ('bicubic_baseline',)
    model = None
    if not baseline_only and any(name != 'zeros' for name in mask_names):
        from .model import ResShift
        model = ResShift(device)
    quality = None if psnr_only else QualityMetrics(device)
    rows = []
    for identifier in ids:
        ref = read_rgb(data / 'hr' / f'{identifier}.png')
        for degradation in degradations:
            lr = read_rgb(data / 'lq' / degradation / f'{identifier}.png')
            base = bicubic(lr)
            if ref.shape != base.shape:
                raise ValueError('HR/LR size mismatch')
            roi = upscale_mask(read_mask(data / 'masks/mixed.png', lr.shape[:2]))
            for name in mask_names:
                mask_lr = (np.ones(lr.shape[:2], dtype=bool) if baseline_only else
                           read_mask(data / 'masks' / f'{name}.png', lr.shape[:2]))
                mask = upscale_mask(mask_lr)
                started = time.perf_counter()
                if baseline_only:
                    prediction = bicubic(lr)
                elif not mask_lr.any():
                    prediction = known_context(lr, mask_lr, ref)
                else:
                    prediction = model.restore(lr, mask_lr, hr=None if mask_lr.all() else ref, seed=seed)
                elapsed = time.perf_counter() - started
                extra = quality.score(prediction, ref) if quality else {'lpips': None, 'clip_iqa': None}
                rows.append({'image': identifier, 'degradation': degradation, 'mask': name,
                             'coverage': float(mask.mean()), 'psnr_db': psnr(prediction, ref), **extra,
                             'psnr_roi_db': psnr(prediction, ref, roi),
                             'psnr_mask_db': psnr(prediction, ref, mask),
                             'outside_max_error': preservation(prediction, ref, mask),
                             'seconds': elapsed})
                save_image(out / 'images' / degradation / identifier / f'{name}.png', prediction)
                write_csv(out / 'metrics.csv', rows)
                row = rows[-1]
                values = ' '.join(f'{key}={row[key]:.3f}' for key in
                                  ('psnr_db', 'lpips', 'clip_iqa', 'psnr_roi_db', 'seconds')
                                  if row[key] is not None)
                print(identifier, degradation, name, values, flush=True)
    metadata = {'ids': list(ids), 'degradations': list(degradations), 'masks': list(mask_names),
                'seed': seed, 'device': device, 'known_source': 'none' if baseline_only else 'hr_outside_mask', 'baseline_only': baseline_only, 'psnr_only': psnr_only,
                'python': platform.python_version(), 'numpy': np.__version__}
    from PIL import __version__ as pillow_version
    metadata['pillow'] = pillow_version
    if model is not None or quality is not None:
        import torch
        metadata['torch'] = torch.__version__
    revision = ROOT / 'weights/revision.txt'
    if model is not None and revision.exists():
        metadata['resshift_revision'] = revision.read_text().strip()
    (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from .report import report
    report(data, out, rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ids', nargs='+', default=IDS)
    parser.add_argument('--degradations', nargs='+', choices=DEGRADATIONS, default=DEGRADATIONS)
    parser.add_argument('--masks', nargs='+', choices=MASKS, default=MASKS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--study', action='store_true')
    parser.add_argument('--areas', nargs='+', type=int, choices=(10, 25, 50, 75), default=(10, 25, 50, 75))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--psnr-only', action='store_true')
    args = parser.parse_args()
    if args.study and (args.baseline_only or args.repeats < 2 or len(set(args.areas)) != len(args.areas)):
        parser.error('Study requires distinct areas, repeats >= 2 and no baseline-only flag')
    out = args.out or ROOT / ('results/study' if args.study else 'results/baseline' if args.baseline_only else 'results/experiment')
    data = args.data if args.data is not None else prepare_official(ROOT / 'data', args.ids, args.cache)
    if args.study:
        from .study import run_study
        run_study(data, out, args.ids, args.degradations, args.areas, args.device,
                  args.seed, args.repeats, args.psnr_only)
    else:
        run(data, out, args.ids, args.degradations, args.masks, args.device, args.seed,
            args.baseline_only, args.psnr_only)


if __name__ == '__main__':
    main()
