# Dataset-specific RLFT parameters

The release includes one RLFT YAML profile per benchmark. These are training
hyperparameters, not trained model weights. The original stage-1/2 configurations
are not substitutes for checkpoints, and stage-1/2 training code is not included
in this release.

| Benchmark | Epochs | Learning rate | Target batch | Generations per target | Target mini-batch | SDE steps | Noise level | Evaluate every batches | Save every batches | Stage-2 initialization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MetalSet | 1000 | 2e-6 | 10 | 6 | 2 | 1 | 0.1 | 1 | 50 | MetalSet, epoch 24 |
| ViaSet | 1000 | 2e-6 | 10 | 6 | 2 | 1 | 0.1 | 1 | 50 | ViaSet, epoch 0 |
| StdContact | 50 | 2e-6 | 10 | 6 | 2 | 1 | 0.1 | 5 | 50 | StdContact, epoch 99 |
| StdMetal | 50 | 2e-6 | 10 | 6 | 2 | 1 | 0.1 | 10 | 25 | MetalSet, historical epoch 49 |

All four profiles use `base_channels: 64`, a 512x512 model input, 2048x2048
physics metrics, gradient clipping at 1.0, and equal weights of 1.0 for shot count,
L2, PVB, and EPE. The fast shot counter operates at 256x256. See
[the RLFT workflow](../docs/RLFT.md) for reward normalization and learning-rate
scheduling.

StdMetal's historical configuration initializes from a **MetalSet** checkpoint;
it does not imply an independently pretrained StdMetal model. Its epoch-49
checkpoint comes from a different SFT run than the current 25-epoch MetalSet SFT
profile. Preserve the historical file if available, or explicitly update
`load_path` in the selected YAML with the intended replacement. No trained checkpoint was available
to verify those initialization files when preparing this release.

Select a profile with `--benchmark MetalSet`, `--benchmark ViaSet`,
`--benchmark StdContact`, or `--benchmark StdMetal`. Alternatively, pass the YAML
path via `--config`. The original checkpoint and W&B metadata retain the run configuration.

The original YAML values are preserved, including legacy fields that the active
loop does not use. No new noise parameters are added. The learning-rate schedule
remains defined in code. Mini-batches are weighted by their target counts; see
the documented implementation fixes. `grpo_beta` must remain zero because this
objective has no reference-policy KL term.
