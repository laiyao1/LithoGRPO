"""Fast shot counting via maximal-rectangle cover.

`FastShotCounter` estimates the number of e-beam shots (rectangles) needed to
write a binary mask. It first enumerates all maximal all-ones rectangles, prunes
those that are fully contained in another, and then solves a minimum
set-cover ILP so that every foreground pixel is covered by at least one chosen
rectangle. The number of selected rectangles is the shot count.

This is the >130x-faster manufacturability estimator described in the LithoGRPO
paper. It preserves the ranking of the slower reference counter
(`shotcount.reference.ShotCounter`) while being cheap enough to use inside the
RL reward loop.
"""

from collections import defaultdict
from typing import List, Sequence

import numpy as np
import torch
from numba import njit
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

REALTYPE = torch.float32


@njit
def check_containment_optimized(y1_s, x1_s, y2_s, x2_s, areas, n):
    """Greedy containment pruning, in descending area order (numba-accelerated)."""
    keep_flag = np.ones(n, dtype=np.bool_)
    keep_list = np.empty(n, dtype=np.int32)
    keep_count = 0

    for i in range(n):
        cur_y1, cur_x1 = y1_s[i], x1_s[i]
        cur_y2, cur_x2 = y2_s[i], x2_s[i]

        contained = False
        for j in range(keep_count):
            k = keep_list[j]
            if (y1_s[k] <= cur_y1 and cur_y2 <= y2_s[k] and
                    x1_s[k] <= cur_x1 and cur_x2 <= x2_s[k]):
                contained = True
                break

        if not contained:
            keep_list[keep_count] = i
            keep_count += 1
        else:
            keep_flag[i] = False

    return keep_flag


class FastShotCounter:
    """Fast shot counter based on maximal-rectangle cover + ILP set cover.

    Args:
        scan_line: use scan-line segment constraints to keep the ILP compact.
                   Default True.
    """

    def __init__(self, scan_line=True, device=None):
        self.scan_line = scan_line
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def rle_rows(self, img):
        """Run-length encode the foreground runs of each row."""
        runs = []
        H, W = img.shape
        for r in range(H):
            c = 0
            while c < W:
                while c < W and img[r, c] == 0:
                    c += 1
                s = c
                while c < W and img[r, c] == 1:
                    c += 1
                if s < c:
                    runs.append((r, s, c - 1))
        return runs

    def prune_rectangles_numba(self, rects: Sequence[Sequence[int]]) -> List[List[int]]:
        """Remove rectangles contained in another, greedily by area (numba)."""
        n = len(rects)
        if n <= 1:
            return [list(r) for r in rects]

        arr = np.asarray(rects, dtype=np.int32)
        y1, x1, h, w = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        y2 = y1 + h
        x2 = x1 + w

        areas = h * w
        order = np.argsort(-areas)

        y1_s = y1[order].astype(np.int32)
        x1_s = x1[order].astype(np.int32)
        y2_s = y2[order].astype(np.int32)
        x2_s = x2[order].astype(np.int32)

        keep_flag = check_containment_optimized(y1_s, x1_s, y2_s, x2_s, areas, n)
        kept_indices = order[keep_flag]
        return arr[kept_indices].tolist()

    def maximal_rectangles(self, img):
        """Enumerate maximal all-ones rectangles via the histogram/stack method."""
        H, W = img.shape
        h = np.zeros(W, int)
        rects = set()
        for r in range(H):
            h = np.where(img[r] == 1, h + 1, 0)
            stack = []
            for c in range(W + 1):
                cur = h[c] if c < W else 0
                last = c
                while stack and cur < stack[-1][1]:
                    pos, height = stack.pop()
                    area_left = stack[-1][0] + 1 if stack else 0
                    rects.add((r - height + 1, area_left, height, c - area_left))
                    last = pos
                stack.append((last, cur))
        return list(rects)

    def run(self, mask, shape=(512, 512)):
        """Return the number of shots needed to write `mask`.

        Args:
            mask:  2D array / tensor of {0, 1}. Resized (nearest) to `shape`.
            shape: working resolution for the cover (default 512x512).
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        mask = torch.nn.functional.interpolate(
            mask[None, None, :, :], size=shape, mode="nearest")[0, 0]

        if mask.device.type != 'cpu':
            mask = mask.detach().cpu().numpy().astype(np.uint8)
        else:
            mask = mask.detach().numpy().astype(np.uint8)

        R = self.maximal_rectangles(mask)
        R = self.prune_rectangles_numba(R)

        H, W = shape

        prob = LpProblem('rect_cover', LpMinimize)
        x = LpVariable.dicts('x', range(len(R)), 0, 1, cat='Binary')

        row_rects = defaultdict(list)
        for i, (y, x0, h, w) in enumerate(R):
            for r in range(y, y + h):
                row_rects[r].append(i)

        if not self.scan_line:
            for r, s, e in self.rle_rows(mask):
                rects_in_row = row_rects[r]
                for c in range(s, e + 1):
                    prob += lpSum(x[i] for i in rects_in_row
                                  if R[i][1] <= c < R[i][1] + R[i][3]) >= 1
        else:
            row_runs = defaultdict(list)
            for r, s, e in self.rle_rows(mask):
                row_runs[r].append((s, e))

            for r in range(H):
                rects_in_row = row_rects[r]
                if not rects_in_row:
                    continue

                cuts = {R[i][1] for i in rects_in_row} | {R[i][1] + R[i][3] for i in rects_in_row}
                cuts |= {col for (s, e) in row_runs[r] for col in (s, e)}
                segments = sorted(cuts)

                for a, b in zip(segments, segments[1:]):
                    c0 = a
                    if mask[r, c0] == 0:
                        continue
                    prob += lpSum(x[i] for i in rects_in_row
                                  if R[i][1] <= c0 < R[i][1] + R[i][3]) >= 1

        prob += lpSum(x.values())
        prob.solve(PULP_CBC_CMD(timeLimit=300, msg=False))
        answer = [R[i] for i in range(len(R)) if x[i].value() == 1]
        return len(answer)
