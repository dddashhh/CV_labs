import math

import torch
import torch.nn.functional as F


def latent_mask(mask_hr, shape, device):
    mask = torch.as_tensor(mask_hr, dtype=torch.float32, device=device)[None, None]
    return F.adaptive_max_pool2d(mask, shape).bool()


def known_path(diffusion, known, condition, seed):
    generator = torch.Generator(device=known.device).manual_seed(seed)
    previous = 0.0
    state = known
    path = [known]
    for eta in diffusion.etas:
        eta = float(eta)
        delta = eta - previous
        noise = torch.randn(known.shape, generator=generator, device=known.device, dtype=known.dtype)
        state = state + delta * (condition - known) + diffusion.kappa * math.sqrt(delta) * noise
        path.append(state)
        previous = eta
    return path


@torch.inference_mode()
def sample_masked(diffusion, model, condition, lr, mask, known, seed=1234):
    full = bool(mask.all())
    if not full and (known is None or known.shape != condition.shape):
        raise ValueError('Known latent is required for a non-full mask')
    if not mask.any():
        return known.clone()
    devices = [condition.device.index] if condition.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if condition.is_cuda:
            with torch.cuda.device(condition.device):
                torch.cuda.manual_seed(seed)
        state = diffusion.prior_sample(condition, torch.randn_like(condition))
        path = None if full else known_path(diffusion, known, condition, (seed + 1) % (2**63))
        if path is not None:
            state = torch.where(mask, state, path[-1])
        for i in reversed(range(diffusion.num_timesteps)):
            t = torch.full((condition.shape[0],), i, device=condition.device, dtype=torch.long)
            state = diffusion.p_sample(model, state, condition, t, clip_denoised=False,
                                       model_kwargs={'lq': lr})['sample']
            if path is not None:
                state = torch.where(mask, state, path[i])
        return state
