import os

import numpy as np
import pytest

from lab3.image_ops import bicubic, read_rgb, read_mask, upscale_mask


@pytest.mark.skipif(os.getenv('LAB3_REAL_MODEL') != '1', reason='Pretrained weights not requested')
def test_pretrained_cpu(div2k_data):
    from lab3.metrics import QualityMetrics
    from lab3.model import ResShift

    hr = read_rgb(div2k_data / 'hr/0801.png')
    lr = read_rgb(div2k_data / 'lq/severe/0801.png')
    mask = {name: read_mask(div2k_data / f'masks/{name}.png', lr.shape[:2])
            for name in ('zeros', 'ones', 'mixed')}
    base = bicubic(lr)
    model = ResShift('cpu')
    empty = model.restore(lr, mask['zeros'], hr=hr)
    full = model.restore(lr, mask['ones'])
    mixed = model.restore(lr, mask['mixed'], hr=hr)
    assert np.array_equal(empty, hr)
    assert full.shape == mixed.shape == hr.shape
    assert np.array_equal(mixed[~upscale_mask(mask['mixed'])], hr[~upscale_mask(mask['mixed'])])
    assert not np.array_equal(full, base)
    changed = hr.copy()
    changed[upscale_mask(mask['mixed'])] ^= 255
    assert np.array_equal(mixed, model.restore(lr, mask['mixed'], hr=changed))
    del model
    metrics = QualityMetrics('cpu')
    scores = metrics.score(mixed, hr)
    assert np.isfinite(scores['lpips'])
    assert 0 <= scores['clip_iqa'] <= 1
