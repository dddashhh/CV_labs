import argparse
import io
import zipfile
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .image_ops import masks, save_image
from .setup_model import download

ROOT = Path(__file__).resolve().parent
URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'
ARCHIVES = {'hr': 'DIV2K_valid_HR.zip', 'bicubic': 'DIV2K_valid_LR_bicubic_X4.zip',
            'unknown': 'DIV2K_valid_LR_unknown_X4.zip'}
IDS = ('0801', '0804', '0810', '0829', '0855')
DEGRADATIONS = ('bicubic', 'unknown', 'blur_noise_jpeg', 'severe')


def read_zip(archive, filename):
    candidates = [name for name in archive.namelist() if Path(name).name == filename and '__MACOSX' not in name]
    if len(candidates) != 1:
        raise ValueError(f'Expected one file: {filename}')
    with Image.open(io.BytesIO(archive.read(candidates[0]))) as image:
        return np.array(image.convert('RGB'))


def crop_pair(hr, low):
    h, w = low.shape[:2]
    if h < 64 or w < 64 or hr.shape[0] < h * 4 or hr.shape[1] < w * 4:
        raise ValueError('Invalid paired image sizes')
    y, x = (h - 64) // 2, (w - 64) // 2
    return hr[y * 4:y * 4 + 256, x * 4:x * 4 + 256].copy(), low[y:y + 64, x:x + 64].copy()


def degrade(hr, seed, severe=False):
    rng = np.random.default_rng(seed)
    blurred = Image.fromarray(hr).filter(ImageFilter.GaussianBlur(radius=5.0 if severe else 1.7))
    low = blurred.resize((64, 64), Image.Resampling.BICUBIC)
    values = np.asarray(low, dtype=np.float32) + rng.normal(0, 25 if severe else 8, (64, 64, 3))
    noisy = Image.fromarray(np.clip(values, 0, 255).round().astype(np.uint8))
    buffer = io.BytesIO()
    noisy.save(buffer, format='JPEG', quality=15 if severe else 45)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.array(image.convert('RGB'))


def prepare(out, reader, ids=IDS):
    for identifier in ids:
        hr = reader('hr', identifier)
        reference = None
        for name in ('bicubic', 'unknown'):
            crop, low = crop_pair(hr, reader(name, identifier))
            if reference is not None and not np.array_equal(reference, crop):
                raise ValueError('Different HR crop for paired degradations')
            reference = crop
            save_image(out / 'lq' / name / f'{identifier}.png', low)
        save_image(out / 'hr' / f'{identifier}.png', reference)
        save_image(out / 'lq/blur_noise_jpeg' / f'{identifier}.png', degrade(reference, 1000 + int(identifier)))
        save_image(out / 'lq/severe' / f'{identifier}.png', degrade(reference, 1000 + int(identifier), severe=True))
    for name, mask in masks((64, 64)).items():
        save_image(out / 'masks' / f'{name}.png', mask.astype(np.uint8) * 255)


def prepare_official(out, ids=IDS, cache=None):
    with ExitStack() as stack:
        if cache is None:
            cache = Path(stack.enter_context(TemporaryDirectory(prefix='div2k-')))
        archives = {}
        for key, name in ARCHIVES.items():
            print(f'DIV2K: {name}', flush=True)
            archives[key] = stack.enter_context(zipfile.ZipFile(download(URL + name, cache / name)))

        def reader(kind, identifier):
            suffix = '' if kind == 'hr' else 'x4'
            return read_zip(archives[kind], f'{identifier}{suffix}.png')

        prepare(out, reader, ids)
    print('Data:', out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'data')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--ids', nargs='+', default=IDS)
    args = parser.parse_args()
    prepare_official(args.out, args.ids, args.cache)


if __name__ == '__main__':
    main()
