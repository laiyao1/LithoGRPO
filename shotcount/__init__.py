"""Shot counting for mask manufacturability evaluation.

- `FastShotCounter`: the fast (>130x) maximal-rectangle + ILP cover estimator
  introduced in LithoGRPO.
- `ShotCounter`: the slower adabox-decomposition reference (LithoBench baseline).
"""

from .fast import FastShotCounter
from .reference import ShotCounter

__all__ = ["FastShotCounter", "ShotCounter"]
