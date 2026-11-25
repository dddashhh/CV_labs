import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from .image_ops import composite, known_context, upscale_mask, validate_mask
from .sampler import latent_mask, sample_masked

ROOT = Path(__file__).resolve().parent
MODEL_FILE = 'resshift_realsrx4_s15_v2.pth'
VAE_FILE = 'autoencoder_vq_f4.pth'


def instantiate(config):
    module, name = config.target.rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(**dict(config.params))


def load_weights(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)


def tensor(image, device):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].to(device, torch.float32) / 127.5 - 1


class ResShift:
    def __init__(self, device='cpu', root=ROOT):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Use cpu or cuda')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        repo = root / 'ResShift'
        config_path = repo / 'configs/realsr_swinunet_realesrgan256.yaml'
        for path in (config_path, root / 'weights' / MODEL_FILE, root / 'weights' / VAE_FILE):
            if not path.is_file():
                raise FileNotFoundError(f'{path}; run python -m lab3.setup_model')
        from omegaconf import OmegaConf

        sys.path.insert(0, str(repo))
        for name in ('models.unet', 'ldm.modules.diffusionmodules.model', 'ldm.modules.attention'):
            importlib.import_module(name).XFORMERS_IS_AVAILBLE = False
        config = OmegaConf.load(config_path)
        config.model.params.use_fp16 = False
        config.autoencoder.params.ddconfig.attn_type = 'vanilla'
        self.model = instantiate(config.model)
        load_weights(self.model, root / 'weights' / MODEL_FILE)
        self.model.eval().requires_grad_(False).to(self.device)
        self.autoencoder = instantiate(config.autoencoder)
        load_weights(self.autoencoder, root / 'weights' / VAE_FILE)
        self.autoencoder.eval().requires_grad_(False).to(self.device)
        self.diffusion = instantiate(config.diffusion)

    @torch.inference_mode()
    def restore(self, lr, mask, hr=None, seed=1234):
        mask = validate_mask(mask, lr.shape[:2])
        context = known_context(lr, mask, hr)
        if not mask.any():
            return context
        if lr.shape != (64, 64, 3):
            raise ValueError('ResShift expects an RGB 64 x 64 LR crop in this experiment')
        condition = tensor(lr, self.device)
        z_lr = self.diffusion.encode_first_stage(condition, self.autoencoder, up_sample=True)
        known = None if mask.all() else self.diffusion.encode_first_stage(
            tensor(context, self.device), self.autoencoder, up_sample=False)
        mask_hr = upscale_mask(mask)
        mask_z = latent_mask(mask_hr, z_lr.shape[-2:], self.device)
        z = sample_masked(self.diffusion, self.model, z_lr, condition, mask_z, known, seed)
        prediction = self.diffusion.decode_first_stage(z, self.autoencoder)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Non-finite model output')
        pixels = ((prediction[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        return composite(pixels, context, mask_hr)
