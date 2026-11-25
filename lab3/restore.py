import argparse
from pathlib import Path

from .image_ops import composite, known_context, read_mask, read_rgb, save_image, upscale_mask

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--hr', type=Path)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--method', choices=('mixed', 'post_blend', 'repaint'), default='mixed')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    lr = read_rgb(args.image)
    mask = read_mask(args.mask, lr.shape[:2])
    hr = read_rgb(args.hr) if args.hr and not mask.all() else None
    context = known_context(lr, mask, hr)
    if mask.any():
        from .model import ResShift
        model = ResShift(args.device)
        if args.method == 'post_blend':
            full = model.restore(lr, np.ones_like(mask), seed=args.seed)
            result = composite(full, context, upscale_mask(mask))
        else:
            result = model.restore(lr, mask, hr=hr, seed=args.seed,
                                   repeats=args.repeats if args.method == 'repaint' else 1)
    else:
        result = context
    save_image(args.out, result)


if __name__ == '__main__':
    main()
