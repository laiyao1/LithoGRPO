"""Tensor types and device selected by the original workflow."""

import torch

REALTYPE = torch.float32
COMPLEXTYPE = torch.complex64
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CROP_RATIO = 0.0
