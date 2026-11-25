import numpy as np
import pytest

from lab3.image_ops import bicubic, composite, known_context, masks, upscale_mask, validate_mask
from lab3.metrics import preservation, psnr
from lab3.prepare_data import crop_pair, degrade


def test_masks():
    result = masks((64, 64))
    assert not result['zeros'].any()
    assert result['ones'].all()
    assert 0.48 < result['mixed'].mean() < 0.52
    assert upscale_mask(result['mixed']).shape == (256, 256)
    assert upscale_mask(result['mixed']).mean() == result['mixed'].mean()


@pytest.mark.parametrize('mask', [np.full((3, 3), 2), np.zeros((4, 3)), np.full((3, 3), np.nan)])
def test_invalid_mask(mask):
    with pytest.raises(ValueError):
        validate_mask(mask, (3, 3))


def test_composite():
    base = np.zeros((8, 8, 3), np.uint8)
    restored = np.full_like(base, 255)
    mask = upscale_mask(np.array([[True, False], [False, True]]))
    result = composite(restored, base, mask)
    assert np.array_equal(result[mask], restored[mask])
    assert np.array_equal(result[~mask], base[~mask])
    assert preservation(result, base, mask) == 0


def test_psnr():
    zero = np.zeros((4, 4, 3), np.uint8)
    white = np.full_like(zero, 255)
    assert psnr(zero, white) == 0
    assert np.isinf(psnr(zero, zero))
    assert np.isnan(psnr(zero, white, np.zeros((4, 4), bool)))
    assert np.isclose(psnr(zero, np.full_like(zero, 10)), 20*np.log10(25.5))


def test_crop_and_degrade():
    lr = np.arange(80*84*3, dtype=np.uint8).reshape(80, 84, 3)
    hr = np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1)
    high, low = crop_pair(hr, lr)
    assert high.shape == (256, 256, 3)
    assert low.shape == (64, 64, 3)
    assert np.array_equal(high[::4, ::4], low)
    assert bicubic(low).shape == high.shape
    assert np.array_equal(degrade(high, 7), degrade(high, 7))
    assert not np.array_equal(degrade(high, 7), degrade(high, 8))


def test_known_context_does_not_expose_hidden_hr():
    lr = np.full((4, 4, 3), 10, np.uint8)
    hr = np.full((16, 16, 3), 200, np.uint8)
    mask = masks(lr.shape[:2])['mixed']
    mask_hr = upscale_mask(mask)
    context = known_context(lr, mask, hr)
    altered = hr.copy()
    altered[mask_hr] = 100
    assert np.array_equal(context, known_context(lr, mask, altered))
    assert np.array_equal(context[~mask_hr], hr[~mask_hr])
    assert np.array_equal(context[mask_hr], bicubic(lr)[mask_hr])
    assert np.array_equal(known_context(lr, np.ones_like(mask), None), bicubic(lr))
    assert np.array_equal(known_context(lr, np.zeros_like(mask), hr), hr)
    with pytest.raises(ValueError):
        known_context(lr, mask, None)


def test_severe_degradation():
    hr = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    severe = degrade(hr, 7, severe=True)
    assert severe.shape == (64, 64, 3)
    assert np.array_equal(severe, degrade(hr, 7, severe=True))
    assert not np.array_equal(severe, degrade(hr, 7))
