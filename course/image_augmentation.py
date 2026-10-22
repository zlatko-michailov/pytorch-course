"""Train-only, label-preserving hypotheses introduced in session 24."""

import torch
from torch.nn import functional as F


def make_augment(seed=2400):
    generator = torch.Generator().manual_seed(seed)

    def augment(images):
        padded = F.pad(images, (2, 2, 2, 2), value=-1.0)
        offsets = torch.randint(0, 5, (len(images), 2), generator=generator)
        crops = []
        for index, (top, left) in enumerate(offsets.tolist()):
            crop = padded[index, :, top:top + 28, left:left + 28]
            if torch.rand((), generator=generator).item() < 0.5:
                crop = crop.flip(dims=(-1,))
            crops.append(crop)
        return torch.stack(crops)

    return augment
