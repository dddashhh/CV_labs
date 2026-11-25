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


def forward_resample(state, condition, clean_estimate, delta, kappa, noise):
    return state + delta * (condition - clean_estimate) + kappa * math.sqrt(delta) * noise


@torch.inference_mode()
def sample_masked(diffusion, model, condition, lr, mask, known, seed=1234, repeats=1):
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
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
            count = repeats if path is not None and i > 0 else 1
            for attempt in range(count):
                output = diffusion.p_sample(model, state, condition, t, clip_denoised=False,
                                            model_kwargs={'lq': lr})
                state = output['sample']
                if path is not None:
                    state = torch.where(mask, state, path[i])
                if attempt + 1 < count:
                    delta = float(diffusion.etas[i] - diffusion.etas[i - 1])
                    state = forward_resample(state, condition, output['pred_xstart'], delta,
                                             diffusion.kappa, torch.randn_like(state))
                    state = torch.where(mask, state, path[i + 1])
        return state
