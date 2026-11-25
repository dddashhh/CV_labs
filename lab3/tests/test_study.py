import numpy as np
import torch

from lab3.image_ops import circular_mask, masks, read_rgb, save_image, upscale_mask
from lab3.metrics import psnr
from lab3.sampler import forward_resample, known_path, sample_masked
from lab3.study import inner_boundary, run_study
from lab3.ci_report import experiment_report, write_html
from lab3.tests.test_sampler import FakeDiffusion, setup


class ResamplingDiffusion(FakeDiffusion):
    def p_sample(self, model, state, condition, t, **kwargs):
        result = super().p_sample(model, state, condition, t, **kwargs)
        result['pred_xstart'] = state * .2
        return result


def test_nested_mask_areas():
    previous = np.zeros((64, 64), bool)
    for area in (10, 25, 50, 75):
        mask = circular_mask(previous.shape, area)
        assert abs(mask.mean() - area / 100) < .01
        assert np.all(mask[previous])
        assert upscale_mask(mask).mean() == mask.mean()
        boundary = inner_boundary(upscale_mask(mask))
        assert boundary.any() and np.all(upscale_mask(mask)[boundary])
        previous = mask
    assert np.array_equal(circular_mask((64, 64), 50), masks((64, 64))['mixed'])


def test_forward_resample_formula():
    state = torch.tensor([2.0, 3.0])
    condition = torch.tensor([4.0, 2.0])
    clean = torch.tensor([0.0, 1.0])
    noise = torch.tensor([1.0, -1.0])
    expected = torch.tensor([4.0, 2.25])
    assert torch.allclose(forward_resample(state, condition, clean, .25, 2., noise), expected)


def test_resampling_schedule_and_projection():
    condition, known, lr = setup()
    mask = torch.zeros_like(condition, dtype=torch.bool)
    mask[..., :2] = True
    diffusion = ResamplingDiffusion()
    path = known_path(diffusion, known, condition, 43)
    rng = torch.random.get_rng_state()
    result = sample_masked(diffusion, None, condition, lr, mask, known, 42, repeats=2)
    assert [t for t, _ in diffusion.inputs] == [2, 2, 1, 1, 0]
    for t, state in diffusion.inputs:
        assert torch.equal(state[~mask], path[t + 1][~mask])
    assert torch.equal(result[~mask], known[~mask])
    assert torch.equal(result, sample_masked(ResamplingDiffusion(), None, condition, lr, mask, known, 42, repeats=2))
    assert torch.equal(rng, torch.random.get_rng_state())
    original = sample_masked(ResamplingDiffusion(), None, condition, lr, mask, known, 42)
    assert not torch.equal(result[mask], original[mask])


def test_resampling_does_not_change_full_or_empty_controls():
    condition, known, lr = setup()
    full = torch.ones_like(condition, dtype=torch.bool)
    diffusion = ResamplingDiffusion()
    output = sample_masked(diffusion, None, condition, lr, full, None, 42, repeats=3)
    assert len(diffusion.inputs) == 3
    assert torch.equal(output, sample_masked(ResamplingDiffusion(), None, condition, lr, full, None, 42))
    assert torch.equal(known, sample_masked(diffusion, None, condition, lr, ~full, known, 42, repeats=3))
    assert len(diffusion.inputs) == 3


def test_study_pipeline(monkeypatch, tmp_path):
    from lab3 import model

    calls = []

    class FakeModel:
        def __init__(self, device):
            pass

        def restore(self, lr, mask, hr=None, seed=1234, repeats=1):
            calls.append((mask.copy(), hr is not None, repeats))
            pred = np.full((256, 256, 3), 100 + repeats * 10, np.uint8)
            if hr is not None:
                pred[~upscale_mask(mask)] = hr[~upscale_mask(mask)]
            return pred

    monkeypatch.setattr(model, 'ResShift', FakeModel)
    data, out = tmp_path / 'data', tmp_path / 'out'
    hr = np.full((256, 256, 3), 160, np.uint8)
    save_image(data / 'hr/0801.png', hr)
    save_image(data / 'lq/severe/0801.png', np.full((64, 64, 3), 90, np.uint8))
    rows, pairs = run_study(data, out, ['0801'], ['severe'], [25, 50], 'cpu', psnr_only=True)
    assert len(rows) == 8 and len(pairs) == 2 and len(calls) == 5
    assert calls[0][1] is False and calls[0][0].all()
    assert [c[2] for c in calls] == [1, 1, 2, 1, 2]
    assert all(r['outside_max_error'] == 0 for r in rows if r['method'] != 'ones')
    full = read_rgb(out / 'images/severe/0801/ones.png')
    for pair in pairs:
        area = pair['area_percent']
        region = upscale_mask(circular_mask((64, 64), area))
        post = read_rgb(out / f'images/severe/0801/post_blend_{area}.png')
        assert np.array_equal(post[region], full[region])
        assert np.array_equal(post[~region], hr[~region])
        assert pair['post_blend_psnr_roi_db'] == psnr(full, hr, region)
        assert pair['mixed_vs_post_psnr_roi_db'] == 0
        assert pair['repaint_vs_mixed_psnr_roi_db'] > 0
    for name in ('summary.csv', 'paired_summary.csv', 'roi_comparison.csv', 'figures/gain_psnr_roi_db.png', 'figures/0801_severe.png', 'report.md', 'report.html'):
        assert (out / name).is_file()
    assert 'RePaint − Mixed' in experiment_report(out)
    write_html(out)
    assert 'RePaint − Mixed' in (out / 'report.html').read_text()
