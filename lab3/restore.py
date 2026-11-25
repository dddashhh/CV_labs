import argparse
from pathlib import Path

from .image_ops import known_context, read_mask, read_rgb, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--hr', type=Path)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    lr = read_rgb(args.image)
    mask = read_mask(args.mask, lr.shape[:2])
    hr = read_rgb(args.hr) if args.hr and not mask.all() else None
    context = known_context(lr, mask, hr)
    if mask.any():
        from .model import ResShift
        result = ResShift(args.device).restore(lr, mask, hr=hr, seed=args.seed)
    else:
        result = context
    save_image(args.out, result)


if __name__ == '__main__':
    main()
