import numpy as np

from lab3.image_ops import bicubic, masks, read_rgb, save_image
from lab3.main import run
from lab3.prepare_data import degrade


def test_baseline_pipeline(tmp_path):
    data, out = tmp_path / 'data', tmp_path / 'out'
    rng = np.random.default_rng(1)
    hr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    save_image(data / 'hr/0801.png', hr)
    save_image(data / 'lq/blur_noise_jpeg/0801.png', degrade(hr, 1))
    for name, mask in masks((64, 64)).items():
        save_image(data / 'masks' / f'{name}.png', mask.astype(np.uint8)*255)
    rows = run(data, out, ['0801'], ['blur_noise_jpeg'], ['zeros'], 'cpu', 1234, True, True)
    assert len(rows) == 1
    assert rows[0]['outside_max_error'] is None
    assert rows[0]['mask'] == 'bicubic_baseline'
    assert np.array_equal(read_rgb(out / 'images/blur_noise_jpeg/0801/bicubic_baseline.png'), bicubic(degrade(hr, 1)))
    assert rows[0]['lpips'] is None
    assert (out / 'metrics.csv').is_file()
    assert (out / 'figures/0801.png').is_file()
    assert (out / 'summary.csv').is_file()

    zero_out = tmp_path / 'zero'
    rows = run(data, zero_out, ['0801'], ['blur_noise_jpeg'], ['zeros'], 'cpu', 1234, False, True)
    assert rows[0]['outside_max_error'] == 0
    assert np.isinf(rows[0]['psnr_db'])
    assert np.array_equal(read_rgb(zero_out / 'images/blur_noise_jpeg/0801/zeros.png'), hr)
