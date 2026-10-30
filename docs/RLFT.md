# Stage 3: RL fine-tuning with implementation bug fixes

This release packages the original RLFT training loop and inference workflow,
with the implementation bug fixes described below. The original colored-noise
generation, SDE drift, and surrogate score formula are retained. Stage-1/2
training scripts, trained checkpoints, datasets, and kernel assets are external
inputs. Use a checkpoint matching `base_channels: 64` in the original configuration.

## Dependencies and assets

Run from the repository root with the original CUDA/PyTorch environment, or
install the dependencies in a compatible CUDA environment:

```bash
pip install -r requirements.txt
mkdir -p external
git clone https://github.com/shelljane/lithobench.git external/lithobench
cp config/data_config.example.yaml config/data_config.yaml
```

Set `data_path` in the local `config/data_config.yaml` to the parent directory of
the MetalSet/ViaSet dataset folders. The original loader uses matched GLP,
target PNG, and pixelILT PNG files, creates its 90/10 split, and extracts up to
12 validation layouts. The local data configuration is ignored by Git.

The external LithoBench checkout must contain `benchmark/`, `kernel/kernels/*.pt`,
and `kernel/scales/*.pt`. If those assets need a separate download, follow the
upstream instructions. `config/lithosimple.txt` points to this external kernel
directory. For an existing checkout, set `LITHOGRPO_LITHOBENCH_ROOT` to that
checkout's root. `LITHOGRPO_DATA_CONFIG` can select another local data YAML.
The optional `cvpr` inference path still expects the original external demo
images under `demo/`; those images are not bundled.

## Training

Place the original SFT checkpoint at the `load_path` in the chosen YAML and run:

```bash
python train_litho_grpo.py --benchmark MetalSet
```

The supported YAML profiles are MetalSet, ViaSet, StdContact, and StdMetal.
Use `--config config/train_config_litho_grpo_metalset.yaml` to select a file
explicitly. [The parameter table](../config/README.md) describes their differences.
Original learning rates, batch/group sizes, epoch counts, reward weights, and
checkpoint/output paths are retained.

W&B calls and the original project/metric names are preserved. Use the original
account setup for online tracking, or set `WANDB_MODE=offline` to record locally
without uploading metrics. No account credentials are bundled.

The training loop uses the benchmark-layout loader, named `test_loader` in the
code, for GRPO updates. Validation and benchmark metrics are evaluated
separately. Reward normalization is initialized from the benchmark evaluation,
and the best checkpoint is selected by the benchmark score.

## Original sampling and implementation fixes

Reward values and their metric records remain in generation order, including
after the training loop shuffles targets. The group-relative loss therefore
pairs each advantage with the score of the same sampled mask.

The original FFT low-pass colored noise, cutoff of 0.02, per-sample measured
standard-deviation normalization, scalar noise schedule, and SDE drift formula
are unchanged. No spectral floor or covariance-based drift is introduced.
The sampler retains the original pixel-mean diagonal-Gaussian surrogate score
for the correlated, per-sample-normalized exploration noise. This surrogate
approximates the transition log density and preserves the original score formula.

For multi-step score gradients, each sampled state is detached before the next
model call, holding the state fixed when differentiating that transition's score.
Sampled actions are also held fixed. This corrects an extra gradient path without
changing forward sample values, trajectories, or surrogate score values for the
same model and random draws. Fixed-trajectory recomputation respects the explicit
`training` gradient mode.

The objective keeps the original score scale:

```text
advantage = (reward - group_mean) / max(group_sample_std, 1e-4)
step_score = mean_pixels(original_diagonal_gaussian_surrogate)
loss = -mean(stop_gradient(clip(advantage, -5, 5)) * sum_t(step_score_t))
```

Each step score is already a pixel mean; the training call site sums over time
without dividing by image dimensions again. This implementation takes one
optimizer step per fresh rollout batch. It does not implement trajectory reuse,
PPO clipping, or reference-policy KL; `grpo_beta` must be zero. Other legacy YAML
fields such as `use_amp` retain their original inactive behavior.

Mini-batch losses are weighted by their target counts, so a smaller final
mini-batch does not receive extra weight. The scalar loss is no longer clamped
to a constant at 100. Parameter gradient clipping is retained, and nonfinite
rewards/scores/loss/gradients raise before applying an optimizer update. Local
loss/reward summaries and the epoch denominator follow the actual optimized
batches. At least two generations per group are required. The original optimizer,
warmup/cosine schedule, frozen convolution, DataParallel behavior, val/test
evaluation, checkpoint schema, and best-model selection remain. No fixed training
seed, final/latest checkpoint, or resume mode is added.

Training initialization strips a saved DataParallel `module.` prefix before
loading the unwrapped model, then applies the current multi-GPU wrapper. This
supports loading the same weights on a single GPU. Full-checkpoint loading is
explicit for compatibility with the NumPy scalar metadata in legacy experiment
files, matching the original inference loader; load only trusted checkpoints.

These implementation fixes can change optimization updates and subsequent
training results. They do not change the original noise/SDE sampling scheme.
Existing model parameter names remain compatible and ODE sampling equations are
unchanged. Morphology is disabled by default as described below.
Historical training results have not been reproduced by a full retraining run.

## Inference

The original argument names and options are retained:

```bash
python infer_litho.py --benchmark MetalSet \
  --checkpoint_path checkpoints/v1.1-cfg-litho-grpo-MetalSet-bilinear-interpolation-no-smooth/MiniUnet_grpo_step_epoch999_batch0.pth \
  --step 1
```

Use `--sde --noise_level 0.03` for the original SDE evaluation path, `--step` for
multiple integration steps, `--no_fast_shot` for the reference counter, and
`--save_images` for detailed visualizations. Morphological postprocessing is
disabled by default (`--smooth_kernel 1`). Pass a value greater than 1, such as
`--smooth_kernel 7`, to enable it explicitly. The separate fixed 3x3 average-pooling
step is retained in both reward evaluation and inference.

ODE inference runs once. SDE inference runs five stochastic repeats: the main
`results_*.csv` reports their per-layout means. The additional
`all_results_*.csv` contains individual draws, means, and `MinShots` rows. Each
`MinShots` row selects the repeat with the lowest shot count for that layout and
reports the other metrics from that same repeat. The mean and minimum-shot
summaries are labeled separately in the output.

The production inference/training code retains its original CUDA assumptions,
including synchronization and GPU memory diagnostics. CPU-only execution of
the full workflow is not a supported replacement for the original CUDA run.

## Cleanup and verification scope

Cleanup removes comments/docstrings containing experiment notes, diagnostic
`print` statements, unused imports, standalone dependency demos, and three
uncalled helpers: `compute_ppo_grpo_loss`, `soft_clear_border`, and
`generate_gaussian_noise`. The active `recompute_log_probs_from_trajectory`
branch and its API are retained. The unused global training simulator instance
is removed; physics evaluation still constructs the original simulators.

The only asset adaptations are public relative configuration paths, an external
LithoBench root override, and a public fallback example for the local data YAML.
Numerical behavior outside the documented implementation fixes and morphology
default is preserved.
Local validation scripts and generated data are not shipped.

Full CUDA training with the original stage-2 checkpoint and real dataset has not
yet been rerun for this release. Local checks compare noise draws, SDE samples,
trajectories, and surrogate score values against the original source with fixed
seeds; verify fixed-trajectory surrogate gradients with finite differences; and
exercise real-model/real-kernel integration at reduced resolution. CPU validation
adapts CUDA-only diagnostics and physics resolution; it does not establish full
CUDA training behavior, trained-model quality, or paper metrics. Online W&B/API
logging is not live-verified. Test scripts and smoke artifacts stay outside the
publication repository.
