"""LithoGRPO RLFT with reward alignment and implementation bug fixes."""

import math
import os
import time
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import wandb
import yaml
from model_litho import MiniUnet
from lithobench.dataset import loadersILT_GRPO
from lithobench import evaluate
from infer_litho import ilt_evaluate, run_sample_batch_sde, TraditionalMorphology


def calculate_reward_litho(
    masks_pred,
    targets,
    indices,
    device="cuda",
    smooth_kernel=1,
    use_shot=True,
    fast_shot=True,
    shot_weight=1.0,
    l2_weight=2.0,
    pvb_weight=2.0,
    epe_weight=2.0,
    norm_stats=None,
    use_dynamic_norm=True,
    wandb=None,
):
    assert smooth_kernel == 1
    traditional_smoother = TraditionalMorphology(
        kernel_size=smooth_kernel, operation="opening-closing"
    )
    batch_size = masks_pred.shape[0]
    if len(masks_pred.shape) == 4:
        masks_pred = masks_pred.squeeze(1)
    if len(targets.shape) == 4:
        targets = targets.squeeze(1)
    per_image_metrics = {}
    sample_metrics = []
    if batch_size < 1 or len(targets) != batch_size or len(indices) != batch_size:
        raise ValueError(
            "Masks, targets, and indices must have matching nonempty batches"
        )
    for i in range(batch_size):
        mask_i = masks_pred[i]
        target_i = targets[i]
        idx = int(indices[i])
        mask_binary = mask_i.clone()
        mask_binary[mask_binary > 0.5] = 1.0
        mask_binary[mask_binary <= 0.5] = 0.0
        mask_smooth = traditional_smoother(mask_binary)
        mask_smooth = (
            F.avg_pool2d(
                mask_smooth.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask = mask_smooth.detach()
        target_cpu = target_i.detach()
        target_np = target_cpu.cpu().numpy()
        (l2s, pvbs, epes) = evaluate.evalRaw([mask], [target_np])
        l2 = l2s[0]
        pvb = pvbs[0]
        epe = epes[0]
        if use_shot:
            if fast_shot:
                shots = evaluate.fast_shots([mask])
            else:
                shots = evaluate.shots([mask])
            shot = shots[0]
        else:
            shot = 0.0
        sample_metrics.append((l2, pvb, epe, shot))
        if not all(math.isfinite(float(value)) for value in (l2, pvb, epe, shot)):
            raise FloatingPointError("Nonfinite physics metrics")
        if idx not in per_image_metrics:
            per_image_metrics[idx] = {"l2": [], "pvb": [], "epe": [], "shot": []}
        per_image_metrics[idx]["l2"].append(l2)
        per_image_metrics[idx]["pvb"].append(pvb)
        per_image_metrics[idx]["epe"].append(epe)
        per_image_metrics[idx]["shot"].append(shot)
    if norm_stats is None and use_dynamic_norm:
        per_image_means = {"l2": [], "pvb": [], "epe": [], "shot": []}
        for idx, metrics in per_image_metrics.items():
            per_image_means["l2"].append(np.mean(metrics["l2"]))
            per_image_means["pvb"].append(np.mean(metrics["pvb"]))
            per_image_means["epe"].append(np.mean(metrics["epe"]))
            per_image_means["shot"].append(np.mean(metrics["shot"]))
        norm_stats = {
            "l2": max(np.mean(per_image_means["l2"]), 1.0),
            "pvb": max(np.mean(per_image_means["pvb"]), 1.0),
            "epe": max(np.mean(per_image_means["epe"]), 0.001),
            "shot": max(np.mean(per_image_means["shot"]), 1.0),
        }
    elif norm_stats is None:
        norm_stats = {"shot": 600.0, "l2": 250000.0, "pvb": 35000.0, "epe": 80.0}
    if any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in norm_stats.values()
    ):
        raise ValueError("Reward normalization values must be finite and positive")
    rewards = []
    all_metrics = []
    total_shot_contrib = 0
    total_l2_contrib = 0
    total_pvb_contrib = 0
    total_epe_contrib = 0
    for l2, pvb, epe, shot in sample_metrics:
        shot_norm = np.clip(shot / norm_stats["shot"], 0, 2.0)
        l2_norm = np.clip(l2 / norm_stats["l2"], 0, 2.0)
        pvb_norm = np.clip(pvb / norm_stats["pvb"], 0, 2.0)
        epe_norm = np.clip(epe / norm_stats["epe"], 0, 2.0)
        shot_contrib = shot_weight * shot_norm
        l2_contrib = l2_weight * l2_norm
        pvb_contrib = pvb_weight * pvb_norm
        epe_contrib = epe_weight * epe_norm
        total_shot_contrib += shot_contrib
        total_l2_contrib += l2_contrib
        total_pvb_contrib += pvb_contrib
        total_epe_contrib += epe_contrib
        reward = -(shot_contrib + l2_contrib + pvb_contrib + epe_contrib)
        reward = np.clip(reward, -50, 0)
        rewards.append(reward)
        all_metrics.append(
            {
                "l2": l2,
                "pvb": pvb,
                "epe": epe,
                "shot": shot,
                "shot_contrib": shot_contrib,
                "l2_contrib": l2_contrib,
                "pvb_contrib": pvb_contrib,
                "epe_contrib": epe_contrib,
                "score": shot_contrib + l2_contrib + pvb_contrib + epe_contrib,
            }
        )
    total = (
        total_shot_contrib + total_l2_contrib + total_pvb_contrib + total_epe_contrib
    )
    total = max(float(total), 1e-12)
    total_weight = shot_weight + l2_weight + pvb_weight + epe_weight
    if (
        not math.isfinite(total_weight)
        or total_weight <= 0
        or min(shot_weight, l2_weight, pvb_weight, epe_weight) < 0
    ):
        raise ValueError(
            "Reward weights must be finite, nonnegative, and have a positive sum"
        )
    expected_shot = shot_weight / total_weight * 100
    expected_l2 = l2_weight / total_weight * 100
    expected_pvb = pvb_weight / total_weight * 100
    expected_epe = epe_weight / total_weight * 100
    actual = [
        total_shot_contrib / total * 100,
        total_l2_contrib / total * 100,
        total_pvb_contrib / total * 100,
        total_epe_contrib / total * 100,
    ]
    expected = [expected_shot, expected_l2, expected_pvb, expected_epe]
    max_deviation = max((abs(a - e) for (a, e) in zip(actual, expected)))
    if wandb is not None:
        wandb.log(
            {
                "reward/norm_shot": norm_stats["shot"],
                "reward/norm_l2": norm_stats["l2"],
                "reward/norm_pvb": norm_stats["pvb"],
                "reward/norm_epe": norm_stats["epe"],
                "reward_contrib/shot": total_shot_contrib / total * 100,
                "reward_contrib/l2": total_l2_contrib / total * 100,
                "reward_contrib/pvb": total_pvb_contrib / total * 100,
                "reward_contrib/epe": total_epe_contrib / total * 100,
                "reward_contrib/shot_deviation": abs(
                    total_shot_contrib / total * 100 - expected_shot
                ),
                "reward_contrib/l2_deviation": abs(
                    total_l2_contrib / total * 100 - expected_l2
                ),
                "reward_contrib/pvb_deviation": abs(
                    total_pvb_contrib / total * 100 - expected_pvb
                ),
                "reward_contrib/epe_deviation": abs(
                    total_epe_contrib / total * 100 - expected_epe
                ),
                "reward_contrib/max_deviation": max_deviation,
            }
        )
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    return (rewards, all_metrics)


def compute_grpo_loss(
    rewards, log_probs, num_gen, min_std=0.0001, advantage_clip=5.0, verbose=True
):
    """On-policy group-relative score loss; log_probs must match reward order.

    One fresh rollout batch is used for one optimizer update, with no PPO reuse
    or reference-policy KL. Scores are sums of the original pixel-mean
    diagonal-Gaussian surrogates, not exact colored-noise log densities.
    """
    if (
        num_gen < 2
        or rewards.ndim != 1
        or rewards.shape != log_probs.shape
        or rewards.numel() == 0
        or rewards.numel() % num_gen
    ):
        raise ValueError(
            "Expected aligned 1D rewards/scores with at least two samples per group"
        )
    if not torch.isfinite(rewards).all() or not torch.isfinite(log_probs).all():
        raise FloatingPointError("Nonfinite rewards or log probabilities")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be finite and positive")
    if advantage_clip is not None and (
        not math.isfinite(advantage_clip) or advantage_clip <= 0
    ):
        raise ValueError("advantage_clip must be finite and positive, or None")
    grouped = rewards.detach().reshape(-1, num_gen)
    group_std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / group_std.clamp_min(
        min_std
    )
    before_clip = advantages.clone()
    if advantage_clip is not None:
        advantages = advantages.clamp(-advantage_clip, advantage_clip)
    clipped = (advantages != before_clip).sum().item()
    policy_loss = -(advantages.flatten() * log_probs).mean()
    if not torch.isfinite(policy_loss):
        raise FloatingPointError("Nonfinite GRPO loss")
    chosen = grouped.max(dim=1).values
    rejected = grouped.min(dim=1).values
    stats = {
        "policy_loss": policy_loss.item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "advantage_min": advantages.min().item(),
        "advantage_max": advantages.max().item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
        "chosen_reward": chosen.mean().item(),
        "rejected_reward": rejected.mean().item(),
        "reward_gap": (chosen - rejected).mean().item(),
        "log_prob_mean": log_probs.mean().item(),
        "log_prob_std": log_probs.std().item(),
        "log_prob_min": log_probs.min().item(),
        "log_prob_max": log_probs.max().item(),
        "num_small_std_groups": (group_std < min_std).sum().item(),
        "num_clipped_advantages": clipped,
        "pct_clipped": clipped / advantages.numel() * 100,
    }
    return policy_loss, stats


def train_one_batch_grpo(
    model,
    optimizer,
    batch,
    num_generations,
    mini_batch_size,
    device,
    sde_steps,
    noise_level,
    max_grad_norm,
    grpo_beta=0.0,
    use_shot=True,
    fast_shot=True,
    shot_weight=1.0,
    l2_weight=1.0,
    pvb_weight=1.0,
    epe_weight=1.0,
    norm_stats=None,
    wandb=None,
    epoch=None,
    batch_idx=None,
    total_batches=None,
    global_step=None,
):
    if grpo_beta != 0:
        raise ValueError(
            "This on-policy objective has no reference-policy KL; grpo_beta must be zero"
        )
    if mini_batch_size < 1 or num_generations < 2 or len(batch["data"]) < 1:
        raise ValueError(
            "Batch sizes must be positive and num_generations must be at least two"
        )
    optimizer.zero_grad()
    batch_size = len(batch["data"])
    num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
    total_loss = 0.0
    total_reward = 0.0
    shuffle_indices = torch.randperm(batch_size)
    batch["data"] = batch["data"][shuffle_indices]
    batch["raw_target"] = batch["raw_target"][shuffle_indices]
    batch["index"] = batch["index"][shuffle_indices]
    per_image_metrics = {}
    accumulated_grpo_stats = {
        "policy_loss": [],
        "reward_mean": [],
        "reward_std": [],
        "advantage_mean": [],
        "advantage_std": [],
        "chosen_reward": [],
        "rejected_reward": [],
        "log_prob_mean": [],
        "log_prob_min": [],
        "log_prob_max": [],
    }
    for mini_idx in range(num_mini_batches):
        start_img = mini_idx * mini_batch_size
        end_img = min(start_img + mini_batch_size, batch_size)
        current_mini_size = end_img - start_img
        mini_targets = batch["data"][start_img:end_img].to(device)
        mini_raw_targets_cpu = batch["raw_target"][start_img:end_img]
        mini_indices = batch["index"][start_img:end_img]
        if len(mini_targets.shape) == 3:
            mini_targets = mini_targets.unsqueeze(1)
            mini_raw_targets_cpu = mini_raw_targets_cpu.unsqueeze(1)
        targets_repeated = mini_targets.repeat_interleave(num_generations, dim=0)
        raw_targets_repeated = mini_raw_targets_cpu.repeat_interleave(
            num_generations, dim=0
        ).to(device)
        indices_repeated = []
        for idx in mini_indices:
            indices_repeated.extend([idx.item()] * num_generations)
        (generated_images, all_masks, step_log_probs) = run_sample_batch_sde(
            model,
            targets_repeated,
            sde_steps,
            indices_repeated,
            noise_level=noise_level,
            save_path=None,
            training=True,
        )
        # Each step already returns a pixel-mean score; preserve its original scale.
        sample_log_probs = torch.stack(step_log_probs, dim=0).sum(dim=0)
        with torch.no_grad():
            (rewards, metrics_list) = calculate_reward_litho(
                masks_pred=generated_images.detach(),
                targets=raw_targets_repeated.squeeze(1),
                indices=indices_repeated,
                device=device,
                use_shot=use_shot,
                fast_shot=fast_shot,
                shot_weight=shot_weight,
                l2_weight=l2_weight,
                pvb_weight=pvb_weight,
                epe_weight=epe_weight,
                use_dynamic_norm=True,
                wandb=wandb,
                norm_stats=norm_stats,
                smooth_kernel=1,
            )
            for sample_idx, (idx, metrics) in enumerate(
                zip(indices_repeated, metrics_list)
            ):
                if idx not in per_image_metrics:
                    per_image_metrics[idx] = {
                        "l2": [],
                        "pvb": [],
                        "epe": [],
                        "shot": [],
                        "score": [],
                    }
                per_image_metrics[idx]["l2"].append(metrics["l2"])
                per_image_metrics[idx]["pvb"].append(metrics["pvb"])
                per_image_metrics[idx]["epe"].append(metrics["epe"])
                per_image_metrics[idx]["shot"].append(metrics["shot"])
                per_image_metrics[idx]["score"].append(metrics["score"])
            del raw_targets_repeated
        mini_rewards = rewards.view(current_mini_size, num_generations)
        (mini_loss, grpo_stats) = compute_grpo_loss(
            rewards, sample_log_probs, num_gen=num_generations
        )
        if not torch.isfinite(mini_loss):
            raise FloatingPointError("Nonfinite GRPO loss; optimizer was not updated")
        for key in accumulated_grpo_stats:
            if key in grpo_stats:
                accumulated_grpo_stats[key].append(grpo_stats[key])
        mini_weight = current_mini_size / batch_size
        scaled_loss = mini_loss * mini_weight
        scaled_loss.backward()
        total_loss += mini_loss.item() * mini_weight
        total_reward += mini_rewards.mean().item() * mini_weight
        del generated_images, all_masks, step_log_probs, sample_log_probs
        del rewards, mini_rewards, mini_loss, scaled_loss
        del targets_repeated
        if (
            torch.cuda.memory_allocated()
            / torch.cuda.get_device_properties(0).total_memory
            > 0.85
        ):
            torch.cuda.empty_cache()
    unclipped_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            unclipped_norm += param_norm.item() ** 2
    unclipped_norm = unclipped_norm**0.5
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    clipped_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clipped_norm += param_norm.item() ** 2
    clipped_norm = clipped_norm**0.5
    clip_ratio = unclipped_norm / max_grad_norm
    was_clipped = unclipped_norm > max_grad_norm
    optimizer.step()
    per_image_avg_metrics = {"l2": [], "pvb": [], "epe": [], "shot": [], "score": []}
    per_image_std_metrics = {"l2": [], "pvb": [], "epe": [], "shot": [], "score": []}
    for idx, metrics in per_image_metrics.items():
        per_image_avg_metrics["l2"].append(np.mean(metrics["l2"]))
        per_image_avg_metrics["pvb"].append(np.mean(metrics["pvb"]))
        per_image_avg_metrics["epe"].append(np.mean(metrics["epe"]))
        per_image_avg_metrics["shot"].append(np.mean(metrics["shot"]))
        per_image_avg_metrics["score"].append(np.mean(metrics["score"]))
        per_image_std_metrics["l2"].append(np.std(metrics["l2"]))
        per_image_std_metrics["pvb"].append(np.std(metrics["pvb"]))
        per_image_std_metrics["epe"].append(np.std(metrics["epe"]))
        per_image_std_metrics["shot"].append(np.std(metrics["shot"]))
        per_image_std_metrics["score"].append(np.std(metrics["score"]))
    final_avg_metrics = {
        "l2": np.mean(per_image_avg_metrics["l2"]),
        "pvb": np.mean(per_image_avg_metrics["pvb"]),
        "epe": np.mean(per_image_avg_metrics["epe"]),
        "shot": np.mean(per_image_avg_metrics["shot"]),
        "score": np.mean(per_image_avg_metrics["score"]),
    }
    metrics_std = {
        "l2_std": np.mean(per_image_std_metrics["l2"]),
        "pvb_std": np.mean(per_image_std_metrics["pvb"]),
        "epe_std": np.mean(per_image_std_metrics["epe"]),
        "shot_std": np.mean(per_image_std_metrics["shot"]),
    }
    if wandb is not None:
        wandb.log(
            {
                "train/batch_loss": total_loss,
                "train/policy_loss": total_loss,
                "train/reward_mean": total_reward,
                "train/reward_std": np.mean(accumulated_grpo_stats["reward_std"]),
                "train/advantage_mean": np.mean(
                    accumulated_grpo_stats["advantage_mean"]
                ),
                "train/advantage_std": np.mean(accumulated_grpo_stats["advantage_std"]),
                "train/chosen_reward": np.mean(accumulated_grpo_stats["chosen_reward"]),
                "train/rejected_reward": np.mean(
                    accumulated_grpo_stats["rejected_reward"]
                ),
                "train/log_prob_mean": np.mean(accumulated_grpo_stats["log_prob_mean"]),
                "train/log_prob_min": np.mean(accumulated_grpo_stats["log_prob_min"]),
                "train/log_prob_max": np.mean(accumulated_grpo_stats["log_prob_max"]),
                "train/l2": final_avg_metrics["l2"],
                "train/pvb": final_avg_metrics["pvb"],
                "train/epe": final_avg_metrics["epe"],
                "train/shot": final_avg_metrics["shot"],
                "train/score": final_avg_metrics["score"],
                "train/l2_std": metrics_std["l2_std"],
                "train/pvb_std": metrics_std["pvb_std"],
                "train/epe_std": metrics_std["epe_std"],
                "train/shot_std": metrics_std["shot_std"],
                "train/grad_norm_unclipped": unclipped_norm,
                "train/grad_norm_clipped": clipped_norm,
                "train/grad_clip_ratio": clip_ratio,
                "train/grad_was_clipped": 1.0 if was_clipped else 0.0,
                "train/num_images": len(per_image_metrics),
                "epoch": epoch + (batch_idx + 1) / total_batches,
                "global_step": global_step,
            }
        )
        if len(per_image_avg_metrics["l2"]) > 1:
            wandb.log(
                {
                    "train/l2_distribution": wandb.Histogram(
                        per_image_avg_metrics["l2"]
                    ),
                    "train/pvb_distribution": wandb.Histogram(
                        per_image_avg_metrics["pvb"]
                    ),
                    "train/epe_distribution": wandb.Histogram(
                        per_image_avg_metrics["epe"]
                    ),
                    "train/shot_distribution": wandb.Histogram(
                        per_image_avg_metrics["shot"]
                    ),
                }
            )
    avg_loss = total_loss
    avg_reward = total_reward
    return (avg_loss, avg_reward, final_avg_metrics)


def to_serializable(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for (k, v) in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return str(obj)


def load_grpo_model(base_channels, load_path, device):
    """Load trusted experiment checkpoints across single/multi-GPU runs.

    Legacy checkpoints include NumPy scalar metadata and therefore need the
    original full-checkpoint loading behavior on recent PyTorch versions.
    """
    model = MiniUnet(base_channels)
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in checkpoint["model"].items()
        }
        model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model.to(device)


def train(config: str):
    config = yaml.load(open(config, "rb"), Loader=yaml.FullLoader)
    base_channels = config.get("base_channels", 16)
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 10)
    lr_adjust_epoch = config.get("lr_adjust_epoch", 25)
    batch_print_interval = config.get("batch_print_interval", 10)
    checkpoint_save_interval = config.get("checkpoint_save_interval", 1)
    save_path = config.get("save_path", "./checkpoints")
    device = config.get("device", "cuda")
    learning_rate = float(config.get("lr", 1e-05))
    load_path = config.get("load_path", None)
    Benchmark = config.get("benchmark", "MetalSet")
    num_generations = config.get("num_generations", 20)
    max_batch_per_step = config.get("max_batch_per_step", 8)
    sde_steps = config.get("sde_steps", 2)
    noise_level = config.get("noise_level", 0.1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    grpo_beta = config.get("grpo_beta", 0.0)
    if grpo_beta != 0:
        raise ValueError(
            "This objective has no reference-policy KL; grpo_beta must be zero"
        )
    use_shot = config.get("use_shot", True)
    fast_shot = config.get("fast_shot", True)
    shot_weight = config.get("shot_weight", 1.0)
    l2_weight = config.get("l2_weight", 0.0)
    pvb_weight = config.get("pvb_weight", 0.0)
    epe_weight = config.get("epe_weight", 0.0)
    mini_batch_size = config.get("mini_batch_size", 2)
    eval_interval_batches = config.get("eval_interval_batches", 50)
    save_interval_batches = config.get("save_interval_batches", 100)
    ImageSize = (512, 512)
    NJobs = 4
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    (train_loader, val_loader, test_loader, gradient_accumulation_steps) = (
        loadersILT_GRPO(
            Benchmark,
            targets,
            ImageSize,
            batch_size,
            NJobs,
            num_generations=num_generations,
            max_batch_per_step=max_batch_per_step,
        )
    )
    val_targets = []
    max_val_images = 12
    for batch_idx, batch in enumerate(val_loader):
        if isinstance(batch, dict):
            batch_targets = batch["raw_target"]
        elif isinstance(batch, (list, tuple)):
            batch_targets = batch[1]
        if isinstance(batch_targets, torch.Tensor):
            batch_targets_np = batch_targets.cpu().numpy()
        else:
            batch_targets_np = batch_targets
        for i in range(batch_targets_np.shape[0]):
            if len(val_targets) >= max_val_images:
                break
            img = batch_targets_np[i]
            if img.ndim == 3:
                img = img[0]
            val_targets.append(img)
        if len(val_targets) >= max_val_images:
            break
        if (batch_idx + 1) % 10 == 0:
            pass
    model = load_grpo_model(base_channels, load_path, device)
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    for param in actual_model.conv_in.parameters():
        param.requires_grad = False
    with torch.no_grad():
        (val_l2s, val_pvbs, val_epes, val_shots) = ilt_evaluate(
            model,
            val_targets,
            step=1,
            finetune=False,
            folder=f"grpo_eval_init_val",
            size=ImageSize[0],
            sde_test=False,
            noise_level=noise_level,
            smooth_kernel=1,
        )
        (test_l2s, test_pvbs, test_epes, test_shots) = ilt_evaluate(
            model,
            targets,
            step=1,
            finetune=False,
            folder=f"grpo_eval_init_test",
            size=ImageSize[0],
            sde_test=False,
            noise_level=noise_level,
            smooth_kernel=1,
        )
    initial_val_shot = np.mean(val_shots)
    initial_test_shot = np.mean(test_shots)
    initial_val_l2 = np.mean(val_l2s)
    initial_test_l2 = np.mean(test_l2s)
    initial_val_pvb = np.mean(val_pvbs)
    initial_test_pvb = np.mean(test_pvbs)
    initial_val_epe = np.mean(val_epes)
    initial_test_epe = np.mean(test_epes)
    initial_val_shot = max(initial_val_shot, 0.1)
    initial_test_shot = max(initial_test_shot, 0.1)
    initial_val_l2 = max(initial_val_l2, 0.1)
    initial_test_l2 = max(initial_test_l2, 0.1)
    initial_val_pvb = max(initial_val_pvb, 0.1)
    initial_test_pvb = max(initial_test_pvb, 0.1)
    initial_val_epe = max(initial_val_epe, 0.1)
    initial_test_epe = max(initial_test_epe, 0.1)
    initial_val_score = (
        np.mean(val_l2s) / initial_val_l2 * l2_weight
        + np.mean(val_pvbs) / initial_val_pvb * pvb_weight
        + np.mean(val_epes) / initial_val_epe * epe_weight
        + np.mean(val_shots) / initial_val_shot * shot_weight
    )
    initial_test_score = (
        np.mean(test_l2s) / initial_test_l2 * l2_weight
        + np.mean(test_pvbs) / initial_test_pvb * pvb_weight
        + np.mean(test_epes) / initial_test_epe * epe_weight
        + np.mean(test_shots) / initial_test_shot * shot_weight
    )
    wandb.init(
        project="rf-litho-grpo",
        config=config,
        name=f"grpo_{time.strftime('%Y%m%d_%H%M%S')}_{Benchmark}",
    )
    wandb.log(
        {
            "eval/val_L2": np.mean(val_l2s),
            "eval/val_PVB": np.mean(val_pvbs),
            "eval/val_EPE": np.mean(val_epes),
            "eval/val_Shot": np.mean(val_shots),
            "eval/val_total_score": initial_val_score,
            "eval/test_L2": np.mean(test_l2s),
            "eval/test_PVB": np.mean(test_pvbs),
            "eval/test_EPE": np.mean(test_epes),
            "eval/test_Shot": np.mean(test_shots),
            "eval/test_total_score": initial_test_score,
            "eval/global_step": 0,
        }
    )
    model.train()
    warmup_epochs = int(0.05 * epochs)
    min_lr_ratio = 0.05
    total_epochs = epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(0.01, epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

    scheduler = LambdaLR(optimizer, lr_lambda)
    os.makedirs(save_path, exist_ok=True)
    best_val_score = initial_val_score
    best_test_score = initial_test_score
    best_global_step = 0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_metrics = []
        norm_stats = {
            "shot": initial_test_shot,
            "l2": initial_test_l2,
            "pvb": initial_test_pvb,
            "epe": initial_test_epe,
        }
        for batch_idx, batch in enumerate(test_loader):
            global_step += 1
            (loss, avg_reward, metrics) = train_one_batch_grpo(
                model=model,
                optimizer=optimizer,
                batch=batch,
                num_generations=num_generations,
                mini_batch_size=mini_batch_size,
                device=device,
                sde_steps=sde_steps,
                noise_level=noise_level,
                max_grad_norm=max_grad_norm,
                grpo_beta=grpo_beta,
                use_shot=use_shot,
                fast_shot=fast_shot,
                shot_weight=shot_weight,
                l2_weight=l2_weight,
                pvb_weight=pvb_weight,
                epe_weight=epe_weight,
                wandb=wandb,
                norm_stats=norm_stats,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=len(test_loader),
                global_step=global_step,
            )
            wandb.log({"train/global_step": global_step})
            epoch_loss += loss
            epoch_reward += avg_reward
            epoch_metrics.append(metrics)
            if global_step % eval_interval_batches == 0:
                model.eval()
                with torch.no_grad():
                    (val_l2s, val_pvbs, val_epes, val_shots) = ilt_evaluate(
                        model,
                        val_targets,
                        step=1,
                        finetune=False,
                        folder=None,
                        size=ImageSize[0],
                        sde_test=False,
                        noise_level=noise_level,
                        smooth_kernel=1,
                    )
                    (test_l2s, test_pvbs, test_epes, test_shots) = ilt_evaluate(
                        model,
                        targets,
                        step=1,
                        finetune=False,
                        folder=None,
                        size=ImageSize[0],
                        sde_test=False,
                        noise_level=noise_level,
                        smooth_kernel=1,
                    )
                current_val_score = (
                    np.mean(val_l2s) / initial_val_l2 * l2_weight
                    + np.mean(val_pvbs) / initial_val_pvb * pvb_weight
                    + np.mean(val_epes) / initial_val_epe * epe_weight
                    + np.mean(val_shots) / initial_val_shot * shot_weight
                )
                current_test_score = (
                    np.mean(test_l2s) / initial_test_l2 * l2_weight
                    + np.mean(test_pvbs) / initial_test_pvb * pvb_weight
                    + np.mean(test_epes) / initial_test_epe * epe_weight
                    + np.mean(test_shots) / initial_test_shot * shot_weight
                )
                wandb.log(
                    {
                        "eval/val_L2": np.mean(val_l2s),
                        "eval/val_PVB": np.mean(val_pvbs),
                        "eval/val_EPE": np.mean(val_epes),
                        "eval/val_Shot": np.mean(val_shots),
                        "eval/val_total_score": current_val_score,
                        "eval/test_L2": np.mean(test_l2s),
                        "eval/test_PVB": np.mean(test_pvbs),
                        "eval/test_EPE": np.mean(test_epes),
                        "eval/test_Shot": np.mean(test_shots),
                        "eval/test_total_score": current_test_score,
                        "eval/global_step": global_step,
                        "eval/epoch": epoch + (batch_idx + 1) / len(test_loader),
                        "eval/current_lr": optimizer.param_groups[0]["lr"],
                    }
                )
                if current_test_score < best_test_score:
                    improvement = best_test_score - current_test_score
                    best_test_score = current_test_score
                    best_global_step = global_step
                    save_dict = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "val_shot": np.mean(val_shots),
                        "val_l2": np.mean(val_l2s),
                        "val_pvb": np.mean(val_pvbs),
                        "val_epe": np.mean(val_epes),
                        "test_shot": np.mean(test_shots),
                        "test_l2": np.mean(test_l2s),
                        "test_pvb": np.mean(test_pvbs),
                        "test_epe": np.mean(test_epes),
                        "config": to_serializable(config),
                        "test_score": current_test_score,
                    }
                    torch.save(
                        save_dict, os.path.join(save_path, "MiniUnet_grpo_best.pth")
                    )
                    wandb.log(
                        {
                            "eval/best_test_score": best_test_score,
                            "eval/current_test_score": current_test_score,
                            "eval/best_global_step": best_global_step,
                        }
                    )
                model.train()
            if global_step % save_interval_batches == 0:
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "config": to_serializable(config),
                }
                torch.save(
                    save_dict,
                    os.path.join(
                        save_path,
                        f"MiniUnet_grpo_step_epoch{epoch}_batch{batch_idx}.pth",
                    ),
                )
        num_batches = len(test_loader)
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_reward = epoch_reward / num_batches
        avg_epoch_metrics = {
            "l2": np.mean([m["l2"] for m in epoch_metrics]),
            "pvb": np.mean([m["pvb"] for m in epoch_metrics]),
            "epe": np.mean([m["epe"] for m in epoch_metrics]),
            "shot": np.mean([m["shot"] for m in epoch_metrics]),
        }
        wandb.log(
            {
                "epoch/loss": avg_epoch_loss,
                "epoch/reward": avg_epoch_reward,
                "epoch/train_l2": avg_epoch_metrics["l2"],
                "epoch/train_pvb": avg_epoch_metrics["pvb"],
                "epoch/train_epe": avg_epoch_metrics["epe"],
                "epoch/train_shot": avg_epoch_metrics["shot"],
                "epoch/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "epoch/global_step": global_step,
                "epoch/best_test_score": best_test_score,
            }
        )
        scheduler.step()
    wandb.finish()


def main():
    """Choose an original per-benchmark configuration without editing source."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--benchmark",
        default="MetalSet",
        choices=("MetalSet", "ViaSet", "StdContact", "StdMetal"),
    )
    args = parser.parse_args()
    config = (
        args.config
        or Path("config") / f"train_config_litho_grpo_{args.benchmark.lower()}.yaml"
    )
    train(str(config))


if __name__ == "__main__":
    main()
