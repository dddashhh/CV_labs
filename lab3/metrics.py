import csv
import math

import numpy as np


def psnr(prediction, reference, mask=None):
    if prediction.shape != reference.shape:
        raise ValueError('Different image shapes')
    delta = prediction.astype(np.float64) - reference.astype(np.float64)
    if mask is not None:
        delta = delta[mask]
    if delta.size == 0:
        return float('nan')
    mse = float(np.mean(delta ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def preservation(prediction, base, mask):
    delta = np.abs(prediction.astype(np.int16) - base.astype(np.int16))[~mask]
    return int(delta.max()) if delta.size else None


class QualityMetrics:
    def __init__(self, device):
        import lpips
        import piq
        import torch

        self.torch = torch
        self.device = device
        self.lpips = lpips.LPIPS(net='alex', verbose=False).eval().to(device)
        self.clip = piq.CLIPIQA(data_range=1.0).eval().to(device)

    def tensor(self, image):
        return self.torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(self.device, self.torch.float32) / 255

    def score(self, prediction, reference):
        with self.torch.inference_mode():
            pred, ref = self.tensor(prediction), self.tensor(reference)
            return {'lpips': float(self.lpips(pred, ref, normalize=True).item()),
                    'clip_iqa': float(self.clip(pred).item())}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


