import numpy as np
import torch
import torch.nn.functional as F

from lab3.image_ops import masks, upscale_mask
from lab3.model import ResShift


class ContextDiffusion:
    etas = np.array([0.1, 0.4, 0.9])
    kappa = 2.0
    num_timesteps = 3

    def __init__(self):
        self.encoded = []

    def encode_first_stage(self, image, autoencoder, up_sample=False):
        self.encoded.append(image.clone())
        if up_sample:
            return image
        return F.avg_pool2d(image, 4) + image.mean()

    def decode_first_stage(self, latent, autoencoder):
        return F.interpolate(latent, scale_factor=4, mode='nearest')

    def prior_sample(self, condition, noise):
        return condition + noise * 0.1

    def p_sample(self, model, state, condition, t, **kwargs):
        return {'sample': state * 0.5 + state.mean() * 0.5}


def make_model():
    model = ResShift.__new__(ResShift)
    model.device = torch.device('cpu')
    model.model = None
    model.autoencoder = None
    model.diffusion = ContextDiffusion()
    return model


def test_hr_context_used_without_hidden_pixels():
    lr = np.full((64, 64, 3), 90, np.uint8)
    hr = np.full((256, 256, 3), 180, np.uint8)
    mask = masks(lr.shape[:2])['mixed']
    region = upscale_mask(mask)
    model = make_model()
    result = model.restore(lr, mask, hr=hr)
    hidden = hr.copy()
    hidden[region] = 0
    assert np.array_equal(result, model.restore(lr, mask, hr=hidden))
    assert torch.equal(model.diffusion.encoded[1], model.diffusion.encoded[3])
    changed = hr.copy()
    changed[~region] = 20
    different = model.restore(lr, mask, hr=changed)
    assert not np.array_equal(result[region], different[region])
    assert np.array_equal(result[~region], hr[~region])
    assert np.array_equal(different[~region], changed[~region])


def test_full_mask_does_not_encode_hr():
    lr = np.full((64, 64, 3), 90, np.uint8)
    hr = np.full((256, 256, 3), 180, np.uint8)
    mask = masks(lr.shape[:2])
    model = make_model()
    assert np.array_equal(model.restore(lr, mask['zeros'], hr=hr), hr)
    assert model.diffusion.encoded == []
    first = model.restore(lr, mask['ones'])
    second = model.restore(lr, mask['ones'], hr=hr)
    assert np.array_equal(first, second)
    assert len(model.diffusion.encoded) == 2
