import numpy as np
import torch

from lab3.sampler import known_path, latent_mask, sample_masked


class FakeDiffusion:
    etas = np.array([0.1, 0.4, 0.9])
    kappa = 2.0
    num_timesteps = 3

    def __init__(self):
        self.inputs = []

    def prior_sample(self, condition, noise):
        return condition + noise

    def p_sample(self, model, state, condition, t, **kwargs):
        self.inputs.append((int(t[0]), state.clone()))
        return {'sample': state + torch.randn_like(state) + 1}


def setup():
    condition = torch.zeros(1, 3, 4, 4)
    known = torch.full_like(condition, 2)
    lr = condition.clone()
    return condition, known, lr


def test_empty_mask_skips_all_steps():
    condition, known, lr = setup()
    diffusion = FakeDiffusion()
    out = sample_masked(diffusion, None, condition, lr, torch.zeros_like(condition, dtype=torch.bool), known)
    assert torch.equal(out, known)
    assert diffusion.inputs == []


def test_full_mask_matches_original_loop():
    condition, known, lr = setup()
    diffusion = FakeDiffusion()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = diffusion.prior_sample(condition, torch.randn_like(condition))
        for i in reversed(range(3)):
            expected = diffusion.p_sample(None, expected, condition, torch.tensor([i]))['sample']
    actual = sample_masked(FakeDiffusion(), None, condition, lr, torch.ones_like(condition, dtype=torch.bool), None, 42)
    assert torch.equal(actual, expected)


def test_mixed_mask_projects_before_every_reverse_step():
    condition, known, lr = setup()
    mask = torch.zeros_like(condition, dtype=torch.bool)
    mask[..., :2] = True
    diffusion = FakeDiffusion()
    path = known_path(diffusion, known, condition, 43)
    result = sample_masked(diffusion, None, condition, lr, mask, known, 42)
    assert [t for t, _ in diffusion.inputs] == [2, 1, 0]
    for t, state in diffusion.inputs:
        assert torch.equal(state[~mask], path[t+1][~mask])
    assert torch.equal(result[~mask], known[~mask])
    assert not torch.equal(result[mask], known[mask])
    repeat = sample_masked(FakeDiffusion(), None, condition, lr, mask, known, 42)
    assert torch.equal(result, repeat)


def test_latent_mask():
    mask = np.zeros((8, 8), bool)
    mask[0, 0] = True
    result = latent_mask(mask, (2, 2), 'cpu')
    assert result.shape == (1, 1, 2, 2)
    assert result.sum() == 1
    assert result[0, 0, 0, 0]


def test_rng_is_restored():
    condition, known, lr = setup()
    state = torch.random.get_rng_state()
    sample_masked(FakeDiffusion(), None, condition, lr, torch.ones_like(condition, dtype=torch.bool), known, 42)
    assert torch.equal(torch.random.get_rng_state(), state)
