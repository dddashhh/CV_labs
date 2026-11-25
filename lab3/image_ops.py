import numpy as np
from PIL import Image

SCALE = 4


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def read_mask(path, shape):
    with Image.open(path) as image:
        return validate_mask(np.array(image.convert('L')), shape)


def validate_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != tuple(shape) or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('Expected a binary mask at LR resolution')
    return mask != 0


def masks(shape):
    return {'zeros': np.zeros(shape, bool), 'ones': np.ones(shape, bool), 'mixed': circular_mask(shape, 50)}


def circular_mask(shape, percent):
    if not 0 < percent < 100:
        raise ValueError('Expected a mask area between 0 and 100 percent')
    h, w = shape
    y, x = np.ogrid[:h, :w]
    return ((y + 0.5 - h / 2) / h) ** 2 + ((x + 0.5 - w / 2) / w) ** 2 <= percent / (100 * np.pi)


def upscale_mask(mask):
    return np.repeat(np.repeat(mask, SCALE, axis=0), SCALE, axis=1)


def bicubic(lr):
    h, w = lr.shape[:2]
    return np.array(Image.fromarray(lr).resize((w * SCALE, h * SCALE), Image.Resampling.BICUBIC))


def composite(restored, base, mask):
    if restored.shape != base.shape or mask.shape != base.shape[:2]:
        raise ValueError('Incompatible output shapes')
    return np.where(mask[..., None], restored, base)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def known_context(lr, mask, hr):
    mask = validate_mask(mask, lr.shape[:2])
    base = bicubic(lr)
    if mask.all():
        return base
    if hr is None or hr.shape != base.shape or hr.dtype != np.uint8:
        raise ValueError('An aligned RGB uint8 HR image is required outside the mask')
    return composite(base, hr, upscale_mask(mask))
