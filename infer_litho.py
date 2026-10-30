"""Original ODE/SDE inference with fixed-trajectory score-gradient handling."""

import argparse
import math
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model_litho import MiniUnet
from rectified_flow import RectifiedFlow
from lithobench import evaluate
from pycommon.settings import CROP_RATIO


device = "cuda" if torch.cuda.is_available() else "cpu"
COLOR_BACKGROUND = (255, 255, 255)
COLOR_MASK = (58, 99, 81)
COLOR_IMAGE = (244, 180, 0)
COLOR_BORDER = (0, 0, 0)


class TraditionalMorphology:
    def __init__(self, kernel_size=5, operation="opening-closing"):
        self.kernel_size = kernel_size
        self.operation = operation
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    def __call__(self, x):
        original_shape = x.shape
        device = x.device
        if len(x.shape) == 4:
            x_np = x.squeeze().cpu().numpy()
        else:
            x_np = x.cpu().numpy()
        x_np = (x_np * 255).astype(np.uint8)
        if self.operation == "opening":
            result = cv2.morphologyEx(x_np, cv2.MORPH_OPEN, self.kernel)
        elif self.operation == "closing":
            result = cv2.morphologyEx(x_np, cv2.MORPH_CLOSE, self.kernel)
        elif self.operation == "opening-closing":
            result = cv2.morphologyEx(x_np, cv2.MORPH_OPEN, self.kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.kernel)
        elif self.operation == "closing-opening":
            result = cv2.morphologyEx(x_np, cv2.MORPH_CLOSE, self.kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
        result = torch.from_numpy(result).float() / 255.0
        result = result.reshape(original_shape).to(device)
        return result


def run_batch(model, x_batch, step, indices, save_path=None):
    rf = RectifiedFlow()
    batch_size = x_batch.shape[0]
    with torch.no_grad():
        dt = 1.0 / step
        x_0 = x_batch.to(device)
        x_t = torch.randn_like(x_0).to(device)
        for j in range(step):
            t = j * dt
            t_batch = torch.full((batch_size,), t, device=device)
            input_x = torch.cat([x_0, x_t], dim=1)
            v_pred = model(x=input_x, t=t_batch)
            x_t = rf.euler(x_t, v_pred, dt)
            if j % 10 == 0 and save_path:
                for i in range(batch_size):
                    x_tmp = torch.sigmoid((x_t[i, 0] - 0.5) * 10)
                    img = (x_tmp.detach().cpu().numpy() * 255).astype("uint8")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(
                        os.path.join(save_path, f"{indices[i]}_step{j}.png"), img
                    )
        x_t = torch.sigmoid((x_t - 0.5) * 10)
        if save_path:
            for i in range(batch_size):
                img = (x_t[i, 0].detach().cpu().numpy() * 255).astype("uint8")
                cv2.imwrite(os.path.join(save_path, f"{indices[i]}_final.png"), img)
        torch.cuda.synchronize()
        return x_t.squeeze(1)


def generate_colored_noise(
    tensor_shape: tuple,
    cutoff_sigma: float,
    device: torch.device,
    generator: torch.Generator = None,
) -> tuple:
    """Apply the original FFT filter and per-sample standard-deviation scaling."""
    B, C, H, W = tensor_shape
    white_noise = torch.randn(tensor_shape, generator=generator, device=device)
    freqs_h = torch.fft.fftfreq(H, device=device)
    freqs_w = torch.fft.fftfreq(W, device=device)
    mesh_h, mesh_w = torch.meshgrid(freqs_h, freqs_w, indexing="ij")
    freq_distances_sq = mesh_h**2 + mesh_w**2
    rfft_freq_distances_sq = freq_distances_sq[:, : W // 2 + 1]
    filter_mask = torch.exp(-rfft_freq_distances_sq / (2 * cutoff_sigma**2))
    filter_mask[0, 0] = 1.0
    fft_white_noise = torch.fft.rfft2(white_noise, norm="ortho")
    fft_filtered_noise = fft_white_noise * filter_mask
    unscaled_colored_noise = torch.fft.irfft2(
        fft_filtered_noise, s=(H, W), norm="ortho"
    )
    std_dev = torch.std(
        unscaled_colored_noise,
        dim=tuple(range(1, unscaled_colored_noise.ndim)),
        keepdim=True,
    )
    std_dev = torch.where(std_dev == 0, torch.tensor(1.0, device=device), std_dev)
    colored_noise = unscaled_colored_noise / std_dev
    log_std = torch.log(std_dev)
    return (colored_noise, white_noise, filter_mask, log_std)


def run_sample_batch_sde(
    model,
    x_batch,
    step,
    indices,
    save_path=None,
    noise_level=0.7,
    training=False,
    fixed_samples=None,
    fixed_noise_trajectory=None,
):
    """Use the original SDE and pixel-mean diagonal-Gaussian surrogate scores."""
    if step < 1 or not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("SDE steps and noise_level must be positive")
    batch_size = x_batch.shape[0]
    all_log_probs = []
    if fixed_samples is not None:
        with torch.set_grad_enabled(training):
            return recompute_log_probs_from_trajectory(
                model, x_batch, fixed_samples, step, noise_level, fixed_noise_trajectory
            )
    context_manager = torch.enable_grad() if training else torch.no_grad()
    noise_trajectory = [] if training else None
    x_trajectory = [] if training else None
    with context_manager:
        x_0 = x_batch.to(device)
        x_t = torch.randn_like(x_0).to(device)
        if training:
            x_trajectory.append(x_t.detach().clone())
        for j in range(step):
            # Score gradients condition on the sampled state at each step.
            x_t = x_t.detach()
            t = j / step
            dt = 1.0 / step
            t_batch = torch.tensor([t]).to(device).expand(batch_size)
            if j == 0:
                t_safe = 0.8 if step == 1 else dt
                t_batch_safe = torch.tensor([t_safe]).to(device).expand(batch_size)
                std_dev_t = torch.sqrt((1 - t_batch_safe) / t_batch_safe) * noise_level
            else:
                t_batch_safe = (
                    torch.tensor([t + dt if step == 1 else t])
                    .to(device)
                    .expand(batch_size)
                )
                std_dev_t = torch.sqrt((1 - t_batch_safe) / t_batch_safe) * noise_level
            std_dev_t = std_dev_t.to(device).view(batch_size, 1, 1, 1)
            input_x = torch.cat([x_0, x_t], dim=1)
            v_pred = model(x=input_x, t=t_batch)
            x_t_mean = (
                x_t
                + x_t * std_dev_t**2 / (2 * (1 - t)) * dt
                + v_pred * dt
                + v_pred * std_dev_t**2 / (2 * (1 - t)) * t * dt
            )
            variance_noise, _, _, _ = generate_colored_noise(
                tensor_shape=(batch_size, 1, x_t.shape[2], x_t.shape[3]),
                cutoff_sigma=0.02,
                device=device,
            )
            if training:
                noise_trajectory.append(variance_noise.detach().clone())
            sde_std = std_dev_t * math.sqrt(dt)
            x_t_next = x_t_mean + sde_std * variance_noise
            if training:
                log_prob_per_pixel = (
                    -((x_t_next.detach() - x_t_mean) ** 2) / (2 * sde_std**2 + 1e-08)
                    - torch.log(sde_std + 1e-08)
                    - 0.5 * math.log(2 * math.pi)
                )
                log_prob = log_prob_per_pixel.mean(dim=[1, 2, 3])
                all_log_probs.append(log_prob)
            x_t = x_t_next
            if training:
                x_trajectory.append(x_t.detach().clone())
            if j % 10 == 0 and save_path:
                for i in range(batch_size):
                    x_tmp = torch.sigmoid((x_t[i, 0] - 0.5) * 10)
                    img = (x_tmp.detach().cpu().numpy() * 255).astype("uint8")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(
                        os.path.join(save_path, f"{indices[i]}_step{j}_sde.png"), img
                    )
        x_t = torch.sigmoid((x_t - 0.5) * 10)
        if save_path is not None:
            for i in range(batch_size):
                img = (x_t[i, 0].detach().cpu().numpy() * 255).astype("uint8")
                cv2.imwrite(os.path.join(save_path, f"{indices[i]}_final.png"), img)
        if training:
            return (
                x_t.squeeze(1),
                {"x_trajectory": x_trajectory, "noise_trajectory": noise_trajectory},
                all_log_probs,
            )
        else:
            return (x_t.squeeze(1), [], all_log_probs)


def recompute_log_probs_from_trajectory(
    model, x_batch, final_samples, step, noise_level, trajectory_info
):
    """Recompute the original surrogate scores while holding the trajectory fixed."""
    if (
        trajectory_info is None
        or len(trajectory_info.get("x_trajectory", [])) != step + 1
    ):
        raise ValueError("A fixed trajectory must contain step+1 states")
    batch_size = x_batch.shape[0]
    all_log_probs = []
    x_trajectory = trajectory_info["x_trajectory"]
    x_0 = x_batch.to(device)
    for j in range(step):
        t = j / step
        dt = 1.0 / step
        t_batch = torch.tensor([t]).to(device).expand(batch_size)
        if j == 0:
            t_safe = 0.8 if step == 1 else dt
            t_batch_safe = torch.tensor([t_safe]).to(device).expand(batch_size)
            std_dev_t = torch.sqrt((1 - t_batch_safe) / t_batch_safe) * noise_level
        else:
            t_batch_safe = (
                torch.tensor([t + dt if step == 1 else t]).to(device).expand(batch_size)
            )
            std_dev_t = torch.sqrt((1 - t_batch_safe) / t_batch_safe) * noise_level
        std_dev_t = std_dev_t.to(device).view(batch_size, 1, 1, 1)
        x_t = x_trajectory[j].to(device).detach()
        x_t_next = x_trajectory[j + 1].to(device).detach()
        input_x = torch.cat([x_0, x_t], dim=1)
        v_pred = model(x=input_x, t=t_batch)
        x_t_mean = (
            x_t
            + x_t * std_dev_t**2 / (2 * (1 - t)) * dt
            + v_pred * dt
            + v_pred * std_dev_t**2 / (2 * (1 - t)) * t * dt
        )
        sde_std = std_dev_t * math.sqrt(dt)
        log_prob_per_pixel = (
            -((x_t_next - x_t_mean) ** 2) / (2 * sde_std**2 + 1e-08)
            - torch.log(sde_std + 1e-08)
            - 0.5 * math.log(2 * math.pi)
        )
        log_prob = log_prob_per_pixel.mean(dim=[1, 2, 3])
        all_log_probs.append(log_prob)
    return (final_samples, {}, all_log_probs)


def ilt_evaluate(
    model,
    targets,
    step=50,
    finetune=False,
    folder="images",
    shot=True,
    fast_shot=True,
    size=512,
    sde_test=False,
    noise_level=0.7,
    smooth_kernel=1,
    save_images=False,
    demo_input=False,
    output_time=False,
):
    traditional_smoother = TraditionalMorphology(
        kernel_size=smooth_kernel, operation="opening-closing"
    )
    if shot:
        pass
    num_gpus = torch.cuda.device_count()
    batch_size = num_gpus * 1
    if folder is not None and (not os.path.exists(folder)):
        os.makedirs(folder, exist_ok=True)
    paramsInit = []
    masksInit = []
    runtimes = []
    shotruntimes = []
    for batch_start in range(0, len(targets), batch_size):
        batch_end = min(batch_start + batch_size, len(targets))
        batch_targets = targets[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        batch_x = []
        for target in batch_targets:
            target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
            x = F.interpolate(
                target_tensor[None, None, :, :],
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
            batch_x.append(x)
        batch_x = torch.cat(batch_x, dim=0)
        runtime = time.time()
        with torch.no_grad():
            if sde_test:
                (batch_params, _, _) = run_sample_batch_sde(
                    model,
                    batch_x,
                    step=step,
                    indices=batch_indices,
                    save_path=None,
                    noise_level=noise_level,
                )
            else:
                batch_params = run_batch(
                    model, batch_x, step=step, indices=batch_indices, save_path=None
                )
        runtime = (time.time() - runtime) / len(batch_targets)
        for i in range(batch_params.shape[0]):
            params0 = batch_params[i].cpu()
            mask0 = params0.clone()
            mask0[mask0 > 0.5] = 1.0
            mask0[mask0 <= 0.5] = 0.0
            if folder is not None:
                mask0_np = (mask0 * 255).squeeze().numpy().astype("uint8")
                cv2.imwrite(f"{folder}/rfilt_mask0_{batch_start + i + 1}.png", mask0_np)
            if smooth_kernel > 1:
                mask0 = traditional_smoother(mask0.to(device)).cpu()
            mask0_tensor = mask0.unsqueeze(0).unsqueeze(0).to(device)
            mask0 = F.avg_pool2d(mask0_tensor, kernel_size=3, stride=1, padding=1).cpu()
            if folder is not None:
                mask0_smooth_np = (mask0 * 255).squeeze().numpy().astype("uint8")
                cv2.imwrite(
                    f"{folder}/rfilt_mask0_smooth_{batch_start + i + 1}.png",
                    mask0_smooth_np,
                )
            mask0 = mask0.squeeze(0).squeeze(0)
            paramsInit.append(params0)
            masksInit.append(mask0)
            runtimes.append(runtime)
    masksInit_device = [m.to(device) for m in masksInit]
    (l2s, pvbs, epes) = evaluate.evalRaw(
        masksInit_device, targets, method="model_ilt", save_images=save_images
    )
    if shot:
        (shots, shotruntimes) = (
            evaluate.fast_shots(masksInit_device, timer=True)
            if fast_shot
            else evaluate.shots(masksInit_device, timer=True)
        )
        shotruntimes = [r for r in shotruntimes if r is not None]
    else:
        shots = [-1 for _ in range(len(masksInit))]
        shotruntimes = [-1 for _ in range(len(masksInit))]
    del masksInit_device
    torch.cuda.empty_cache()
    for idx in range(len(targets)):
        if save_images:
            mask_array = (masksInit[idx] * 255).detach().cpu().numpy().astype(np.uint8)
            crop_ratio = CROP_RATIO
            (height, width) = mask_array.shape
            crop_h = int(height * crop_ratio)
            crop_w = int(width * crop_ratio)
            mask_array = mask_array[crop_h : height - crop_h, crop_w : width - crop_w]
            (height, width) = mask_array.shape
            colored_mask = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
            colored_mask[mask_array == 255] = COLOR_MASK
            border_size = 0
            bordered_image = cv2.copyMakeBorder(
                colored_mask,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=COLOR_BORDER,
            )
            save_path = f"{folder}/mask0_{Benchmark}_{idx}_l2_{l2s[idx]:.0f}_pvb_{pvbs[idx]:.0f}_epe_{epes[idx]:.0f}_shot_{shots[idx]:.0f}_kernel_{smooth_kernel}.png"
            cv2.imwrite(save_path, bordered_image)
    if demo_input:
        masksInit = [None for _ in range(len(targets))]
        for idx in range(len(targets)):
            masksInit[idx] = torch.tensor(targets[idx], dtype=torch.float32)
            masksInit[idx] = masksInit[idx].to(device)
            masksInit[idx] = masksInit[idx]
        (demo_l2s, demo_pvbs, demo_epes) = evaluate.evalRaw(
            masksInit, targets, method="model_ilt_demo", save_images=save_images
        )
        for idx in range(len(targets)):
            mask_array = (
                (masksInit[idx].squeeze(0).squeeze(0) * 255)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            crop_ratio = CROP_RATIO
            (height, width) = mask_array.shape
            crop_h = int(height * crop_ratio)
            crop_w = int(width * crop_ratio)
            mask_array = mask_array[crop_h : height - crop_h, crop_w : width - crop_w]
            (height, width) = mask_array.shape
            colored_mask = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
            colored_mask[mask_array == 255] = COLOR_MASK
            border_size = 0
            bordered_image = cv2.copyMakeBorder(
                colored_mask,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=COLOR_BORDER,
            )
            save_path = f"{folder}/demo_input_{Benchmark}_mask_{idx}_l2_{l2s[idx]:.0f}_pvb_{pvbs[idx]:.0f}_epe_{epes[idx]:.0f}_shot_{shots[idx]:.0f}.png"
            cv2.imwrite(save_path, bordered_image)
    if not finetune:
        if output_time:
            return (l2s, pvbs, epes, shots, runtimes, shotruntimes)
        return (l2s, pvbs, epes, shots)
    (l2s, pvbs, epes, masks) = evaluate.finetuneFast(paramsInit, masksInit, targets)
    if shot:
        shots = evaluate.fast_shots(masks) if fast_shot else evaluate.shots(masks)
    else:
        shots = [-1 for _ in range(len(masks))]
    for idx in range(len(targets)):
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            cv2.imwrite(
                f"{folder}/rfilt_mask1_{idx + 1}.png",
                (masks[idx] * 255).detach().cpu().numpy(),
            )
    return (l2s, pvbs, epes, shots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1, help="number of steps")
    parser.add_argument(
        "--noise_level", type=float, default=0.03, help="noise level for sde sampling"
    )
    parser.add_argument("--sde", action="store_true", help="use sde sampling")
    parser.add_argument(
        "--smooth_kernel",
        type=int,
        default=1,
        help="morphology kernel size; 1 disables morphology (default)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="MetalSet", help="benchmark dataset name"
    )
    parser.add_argument(
        "--no_fast_shot", action="store_true", help="use fast shot counting method"
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument(
        "--save_images", action="store_true", help="whether to save output images"
    )
    args = parser.parse_args()
    output_csv = f"results_{args.benchmark}_step{args.step}_fastshot_{not args.no_fast_shot}_kernel_{args.smooth_kernel}_{'-'.join(args.checkpoint_path.split('/')[-2:]).replace('.pth', '')}.csv"
    model = MiniUnet(base_channels=64)
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint["model"]
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    has_module_prefix = any((k.startswith("module.") for k in state_dict.keys()))
    if has_module_prefix:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_state_dict = {f"module.{k}": v for (k, v) in state_dict.items()}
            model = torch.nn.DataParallel(model)
            model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    Benchmark = args.benchmark
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    all_results = {
        i: {
            "shots": [],
            "l2": [],
            "pvb": [],
            "epe": [],
            "runtime": [],
            "shot_runtime": [],
        }
        for i in range(len(targets))
    }
    if not args.sde:
        repeat_num = 1
    else:
        repeat_num = 5
    steps = [args.step]
    fast_shot = not args.no_fast_shot
    csv_data = []
    for step in steps:
        for repeat in range(repeat_num):
            (l2s, pvbs, epes, shots, runtimes, shotruntimes) = ilt_evaluate(
                model,
                targets,
                step=step,
                finetune=False,
                folder=f"trivial_rf_step{step}",
                shot=True,
                fast_shot=fast_shot,
                sde_test=args.sde,
                noise_level=args.noise_level,
                smooth_kernel=args.smooth_kernel,
                save_images=args.save_images,
                demo_input=args.benchmark == "cvpr",
                output_time=True,
            )
            for i in range(len(targets)):
                all_results[i]["shots"].append(shots[i])
                all_results[i]["l2"].append(l2s[i])
                all_results[i]["pvb"].append(pvbs[i])
                all_results[i]["epe"].append(epes[i])
                all_results[i]["runtime"].append(runtimes[i])
                all_results[i]["shot_runtime"].append(shotruntimes[i])
    import csv

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(
            ["Target_ID", "L2", "PVB", "EPE", "Shots", "Runtime(s)", "Shot_Runtime(s)"]
        )
        for target_id in range(len(targets)):
            row = [
                target_id + 1,
                f"{np.mean(all_results[target_id]['l2']):.2f}",
                f"{np.mean(all_results[target_id]['pvb']):.2f}",
                f"{np.mean(all_results[target_id]['epe']):.2f}",
                f"{np.mean(all_results[target_id]['shots']):.2f}",
                f"{np.mean(all_results[target_id]['runtime']):.4f}",
                f"{np.mean(all_results[target_id]['shot_runtime']):.4f}",
            ]
            writer.writerow(row)
        all_l2 = [np.mean(all_results[i]["l2"]) for i in range(len(targets))]
        all_pvb = [np.mean(all_results[i]["pvb"]) for i in range(len(targets))]
        all_epe = [np.mean(all_results[i]["epe"]) for i in range(len(targets))]
        all_shots = [np.mean(all_results[i]["shots"]) for i in range(len(targets))]
        all_runtimes = [np.mean(all_results[i]["runtime"]) for i in range(len(targets))]
        all_shot_runtimes = [
            np.mean(all_results[i]["shot_runtime"]) for i in range(len(targets))
        ]
        writer.writerow(
            [
                "Average",
                f"{np.mean(all_l2):.2f}",
                f"{np.mean(all_pvb):.2f}",
                f"{np.mean(all_epe):.2f}",
                f"{np.mean(all_shots):.2f}",
                f"{np.mean(all_runtimes):.4f}",
                f"{np.mean(all_shot_runtimes):.4f}",
            ]
        )
    if args.sde:
        output_path = f"all_results_{Benchmark}_{args.noise_level}_step{step}_kernel{args.smooth_kernel}.csv"
        with open(output_path, "w") as f:
            f.write("Testcase,Step,Shots,L2,PVBand,EPE\n")
            for i in range(len(targets)):
                for j in range(repeat_num):
                    f.write(
                        f"{i + 1},{step},{all_results[i]['shots'][j]},{all_results[i]['l2'][j]},{all_results[i]['pvb'][j]},{all_results[i]['epe'][j]}\n"
                    )
            for i in range(len(targets)):
                f.write(
                    f"Avg{i + 1},{step},{np.mean(all_results[i]['shots'])},{np.mean(all_results[i]['l2'])},{np.mean(all_results[i]['pvb'])},{np.mean(all_results[i]['epe'])}\n"
                )
            for i in range(len(targets)):
                min_idx = np.argmin(all_results[i]["shots"])
                f.write(
                    f"MinShots{i + 1},{step},{np.min(all_results[i]['shots'])},{all_results[i]['l2'][min_idx]},{all_results[i]['pvb'][min_idx]},{all_results[i]['epe'][min_idx]}\n"
                )
    else:
        pass
