import numpy as np

from lab3.image_ops import read_mask, read_rgb
from lab3.main import run
from lab3.prepare_data import DEGRADATIONS, degrade


def test_official_div2k(div2k_data, tmp_path):
    hr = read_rgb(div2k_data / 'hr/0801.png')
    assert hr.shape == (256, 256, 3)
    for name in DEGRADATIONS:
        lr = read_rgb(div2k_data / f'lq/{name}/0801.png')
        assert lr.shape == (64, 64, 3)
        assert lr.dtype == np.uint8
    severe = read_rgb(div2k_data / 'lq/severe/0801.png')
    assert np.array_equal(severe, degrade(hr, 1801, severe=True))
    for name in ('zeros', 'ones', 'mixed'):
        mask = read_mask(div2k_data / f'masks/{name}.png', (64, 64))
        assert mask.shape == (64, 64)
    rows = run(div2k_data, tmp_path / 'zero', ['0801'], DEGRADATIONS,
               ['zeros'], 'cpu', 1234, False, True)
    assert len(rows) == len(DEGRADATIONS)
    assert all(row['outside_max_error'] == 0 for row in rows)
    assert all(np.isinf(row['psnr_db']) for row in rows)
