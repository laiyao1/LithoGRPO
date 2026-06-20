"""Reference shot counter (LithoBench baseline).

`ShotCounter` reproduces the manufacturability metric used by LithoBench: every
connected component of the mask is decomposed into axis-aligned rectangles with
the `adaptive-boxes` (adabox) decomposition, and the total rectangle count is
the number of e-beam shots.

This is accurate but slow; it is used here only as the ground-truth reference
against which the fast counter (`shotcount.fast.FastShotCounter`) is validated.
"""

import numpy as np
import torch
import cv2

from adabox import proc, tools

REALTYPE = torch.float32


class ShotCounter:
    """Reference shot counter using adabox rectangular decomposition.

    Args:
        thresh: binarization threshold (unused for already-binary masks; kept
                for API compatibility with the LithoBench counter).
        device: torch device for the input tensor.
    """

    def __init__(self, thresh=0.5, device=None):
        self._thresh = thresh
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, mask, target=None, scale=1, shape=(512, 512)):
        """Return the number of shots needed to write `mask`.

        Args:
            mask:  2D array / tensor of {0, 1}. Resized (nearest) to `shape`.
            shape: working resolution (default 512x512). LithoBench uses 256x256.
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        image = torch.nn.functional.interpolate(
            mask[None, None, :, :], size=shape, mode="nearest")[0, 0]
        image = image.detach().cpu().numpy().astype(np.uint8)

        comps, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        rectangles = []
        for label in range(1, comps):
            ys, xs = np.where(labels == label)
            pixels = np.stack([ys, xs, np.zeros_like(ys)], axis=1)
            x_data = np.unique(np.sort(pixels[:, 0]))
            y_data = np.unique(np.sort(pixels[:, 1]))
            if x_data.shape[0] == 1 or y_data.shape[0] == 1:
                rectangles.append(
                    tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max()))
                continue
            (rects, sep) = proc.decompose(pixels, 4)
            rectangles.extend(rects)
        return len(rectangles)
