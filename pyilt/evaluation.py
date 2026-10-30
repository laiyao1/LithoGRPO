import numpy as np
import torch
import torch.nn.functional as func
from pycommon.settings import *
import pylitho.exact as lithosim
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

COLOR_BACKGROUND = (255, 255, 255)
COLOR_BINARY_NOM = (244, 180, 0)
COLOR_BORDER = (0, 0, 0)


class Basic:
    def __init__(
        self,
        litho=lithosim.LithoSim("./config/lithosimple.txt"),
        thresh=0.5,
        device=DEVICE,
        method="Unknown",
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device
        self._method = method

    def run(self, mask, target, scale=1, folder="images", save_images=True, idx=0):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            (printedNom, printedMax, printedMin) = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
            self.l2loss = l2loss
            self.pvband = pvband
            self.binaryNom = binaryNom
            if save_images:
                self._save_images(
                    printedNom,
                    binaryNom,
                    binaryMax,
                    binaryMin,
                    folder,
                    idx,
                    target=target,
                )
        return (l2loss.item(), pvband.item())

    def _save_images(
        self, printedNom, binaryNom, binaryMax, binaryMin, folder, idx, target=None
    ):
        if not os.path.exists(folder):
            os.mkdir(folder)
        printedNom_np = printedNom.detach().cpu().numpy()
        binaryNom_np = (binaryNom * 255).detach().cpu().numpy().astype(np.uint8)
        self._save_heatmap(
            printedNom_np,
            os.path.join(
                folder,
                f"printedNom_heatmap_{self._method}_{idx}_l2_{self.l2loss:.0f}_pvb_{self.pvband:.0f}.png",
            ),
            target=target,
        )
        self._save_binary_image(
            binaryNom_np,
            os.path.join(
                folder,
                f"binaryNom_image_{self._method}_{idx}_l2_{self.l2loss:.0f}_pvb_{self.pvband:.0f}.png",
            ),
            target=target,
        )
        self._save_pvband_image(
            binaryMax=binaryMax,
            binaryMin=binaryMin,
            path=os.path.join(
                folder,
                f"pvband_image_{self._method}_{idx}_l2_{self.l2loss:.0f}_pvb_{self.pvband:.0f}.png",
            ),
            target=target,
        )
        if target is not None:
            self._save_target_image(
                target,
                os.path.join(folder, f"target_image_{self._method}_{idx}.png"),
                border_size=0,
            )

    def draw_epe(
        self, binary_data, target, sample_points, epe_results, path, border_size=0
    ):
        COLOR_BACKGROUND = (255, 255, 255)
        COLOR_BINARY_NOM = (0, 165, 255)
        COLOR_BORDER = (0, 0, 0)
        COLOR_CONTOUR = (0, 0, 0)
        COLOR_PASS = (0, 255, 0)
        COLOR_FAIL = (0, 0, 255)
        crop_ratio = CROP_RATIO
        (height, width) = binary_data.shape
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)
        binary_data = binary_data[crop_h : height - crop_h, crop_w : width - crop_w]
        (height, width) = binary_data.shape
        colored_image = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
        colored_image[binary_data == 255] = COLOR_BINARY_NOM
        if target is not None:
            target_np = target
            target_np = target_np[
                crop_h : target_np.shape[0] - crop_h,
                crop_w : target_np.shape[1] - crop_w,
            ]
            target_contour = (target_np > 0.5).astype(np.uint8)
            (contours, _) = cv2.findContours(
                target_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            dash_length = 5
            gap_length = 10
            for contour in contours:
                points = contour.reshape(-1, 2)
                points = np.vstack([points, points[0]])
                accumulated_distance = 0
                is_drawing = True
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    segment_length = np.linalg.norm(np.array(pt2) - np.array(pt1))
                    remaining_length = segment_length
                    current_pos = np.array(pt1, dtype=float)
                    direction = (
                        (np.array(pt2) - np.array(pt1)) / segment_length
                        if segment_length > 0
                        else np.array([0, 0])
                    )
                    while remaining_length > 0:
                        if is_drawing:
                            draw_length = min(
                                dash_length - accumulated_distance, remaining_length
                            )
                            end_pos = current_pos + direction * draw_length
                            cv2.line(
                                colored_image,
                                tuple(current_pos.astype(int)),
                                tuple(end_pos.astype(int)),
                                COLOR_CONTOUR,
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )
                            accumulated_distance += draw_length
                            current_pos = end_pos
                            remaining_length -= draw_length
                            if accumulated_distance >= dash_length:
                                is_drawing = False
                                accumulated_distance = 0
                        else:
                            skip_length = min(
                                gap_length - accumulated_distance, remaining_length
                            )
                            current_pos += direction * skip_length
                            accumulated_distance += skip_length
                            remaining_length -= skip_length
                            if accumulated_distance >= gap_length:
                                is_drawing = True
                                accumulated_distance = 0
        if sample_points.shape[0] > 0:
            sample_points_np = sample_points.cpu().numpy()
            sample_points_np[:, 0] -= crop_h
            sample_points_np[:, 1] -= crop_w
            epe_results_np = epe_results.cpu().numpy()
            marker_size = 5
            for i in range(sample_points_np.shape[0]):
                (y, x) = (int(sample_points_np[i, 0]), int(sample_points_np[i, 1]))
                if 0 <= y < height and 0 <= x < width:
                    if epe_results_np[i] == 1:
                        cv2.circle(colored_image, (x, y), marker_size, COLOR_PASS, 2)
                    else:
                        cv2.line(
                            colored_image,
                            (x - marker_size, y - marker_size),
                            (x + marker_size, y + marker_size),
                            COLOR_FAIL,
                            2,
                            lineType=cv2.LINE_AA,
                        )
                        cv2.line(
                            colored_image,
                            (x - marker_size, y + marker_size),
                            (x + marker_size, y - marker_size),
                            COLOR_FAIL,
                            2,
                            lineType=cv2.LINE_AA,
                        )
        bordered_image = cv2.copyMakeBorder(
            colored_image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=COLOR_BORDER,
        )
        cv2.imwrite(path, bordered_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def _save_pvband_image(
        self, binaryMax, binaryMin, path, border_size=0, target=None
    ):
        COLOR_BACKGROUND = (255, 255, 255)
        COLOR_MAX_ONLY = (50, 50, 200)
        COLOR_OVERLAP = (0, 200, 255)
        COLOR_MIN_ONLY = (100, 0, 150)
        COLOR_BORDER = (0, 0, 0)
        COLOR_CONTOUR = (0, 0, 0)
        binaryMax = binaryMax.detach().cpu().numpy().astype(np.uint8)
        binaryMin = binaryMin.detach().cpu().numpy().astype(np.uint8)
        (height, width) = binaryMax.shape
        crop_ratio = CROP_RATIO
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)
        binaryMax = binaryMax[crop_h : height - crop_h, crop_w : width - crop_w]
        binaryMin = binaryMin[crop_h : height - crop_h, crop_w : width - crop_w]
        (height, width) = binaryMax.shape
        colored_image = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
        min_mask = binaryMin == 1
        colored_image[min_mask] = COLOR_MIN_ONLY
        max_mask = binaryMax == 1
        colored_image[max_mask] = COLOR_MAX_ONLY
        overlap_mask = (binaryMax == 1) & (binaryMin == 1)
        colored_image[overlap_mask] = COLOR_OVERLAP
        max_only_count = np.sum((binaryMax == 1) & (binaryMin == 0))
        min_only_count = np.sum((binaryMin == 1) & (binaryMax == 0))
        overlap_count = np.sum(overlap_mask)
        if target is not None:
            target_np = target.cpu().numpy()
            target_np = target_np[
                crop_h : target_np.shape[0] - crop_h,
                crop_w : target_np.shape[1] - crop_w,
            ]
            target_contour = (target_np > 0.5).astype(np.uint8)
            (contours, _) = cv2.findContours(
                target_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            dash_length = 5
            gap_length = 10
            for contour in contours:
                points = contour.reshape(-1, 2)
                points = np.vstack([points, points[0]])
                accumulated_distance = 0
                is_drawing = True
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    segment_length = np.linalg.norm(np.array(pt2) - np.array(pt1))
                    remaining_length = segment_length
                    current_pos = np.array(pt1, dtype=float)
                    direction = (
                        (np.array(pt2) - np.array(pt1)) / segment_length
                        if segment_length > 0
                        else np.array([0, 0])
                    )
                    while remaining_length > 0:
                        if is_drawing:
                            draw_length = min(
                                dash_length - accumulated_distance, remaining_length
                            )
                            end_pos = current_pos + direction * draw_length
                            cv2.line(
                                colored_image,
                                tuple(current_pos.astype(int)),
                                tuple(end_pos.astype(int)),
                                COLOR_CONTOUR,
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )
                            accumulated_distance += draw_length
                            current_pos = end_pos
                            remaining_length -= draw_length
                            if accumulated_distance >= dash_length:
                                is_drawing = False
                                accumulated_distance = 0
                        else:
                            skip_length = min(
                                gap_length - accumulated_distance, remaining_length
                            )
                            current_pos += direction * skip_length
                            accumulated_distance += skip_length
                            remaining_length -= skip_length
                            if accumulated_distance >= gap_length:
                                is_drawing = True
                                accumulated_distance = 0
        bordered_image = cv2.copyMakeBorder(
            colored_image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=COLOR_BORDER,
        )
        cv2.imwrite(path, bordered_image)

    def _save_heatmap(self, data, path, border_size=0, target=None, draw_contour=False):
        crop_ratio = CROP_RATIO
        (height, width) = data.shape
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)
        data = data[crop_h : height - crop_h, crop_w : width - crop_w]
        cmap = plt.cm.hot.reversed()
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap=cmap, interpolation="nearest")
        if draw_contour and target is not None:
            target = target[crop_h : height - crop_h, crop_w : width - crop_w]
            plt.contour(
                target,
                levels=[0.5],
                colors="black",
                linestyles="dashed",
                linewidths=1.5,
            )
        plt.axis("off")
        plt.tight_layout()
        temp_path = path.replace(".png", "_temp.png")
        plt.savefig(temp_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        with Image.open(temp_path) as img:
            bordered_img = ImageOps.expand(img, border=border_size, fill="black")
            bordered_img.save(path)
        os.remove(temp_path)

    def _save_binary_image(
        self, binary_data, path, border_size=0, target=None, draw_contour=True
    ):
        COLOR_BACKGROUND = (255, 255, 255)
        COLOR_BINARY_NOM = (0, 165, 255)
        COLOR_BORDER = (0, 0, 0)
        COLOR_CONTOUR = (0, 0, 0)
        crop_ratio = CROP_RATIO
        (height, width) = binary_data.shape
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)
        binary_data = binary_data[crop_h : height - crop_h, crop_w : width - crop_w]
        (height, width) = binary_data.shape
        colored_image = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
        colored_image[binary_data == 255] = COLOR_BINARY_NOM
        if draw_contour and target is not None:
            target_np = target.cpu().numpy()
            target_np = target_np[
                crop_h : target_np.shape[0] - crop_h,
                crop_w : target_np.shape[1] - crop_w,
            ]
            target_contour = (target_np > 0.5).astype(np.uint8)
            (contours, _) = cv2.findContours(
                target_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            dash_length = 5
            gap_length = 10
            for contour in contours:
                points = contour.reshape(-1, 2)
                points = np.vstack([points, points[0]])
                accumulated_distance = 0
                is_drawing = True
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    segment_length = np.sqrt(dx * dx + dy * dy)
                    if segment_length == 0:
                        continue
                    ux = dx / segment_length
                    uy = dy / segment_length
                    remaining_length = segment_length
                    (current_x, current_y) = (float(pt1[0]), float(pt1[1]))
                    while remaining_length > 0.5:
                        if is_drawing:
                            draw_length = min(
                                dash_length - accumulated_distance, remaining_length
                            )
                            end_x = current_x + ux * draw_length
                            end_y = current_y + uy * draw_length
                            cv2.line(
                                colored_image,
                                (int(current_x), int(current_y)),
                                (int(end_x), int(end_y)),
                                COLOR_CONTOUR,
                                thickness=3,
                                lineType=cv2.LINE_AA,
                            )
                            accumulated_distance += draw_length
                            (current_x, current_y) = (end_x, end_y)
                            remaining_length -= draw_length
                            if accumulated_distance >= dash_length:
                                is_drawing = False
                                accumulated_distance = 0
                        else:
                            skip_length = min(
                                gap_length - accumulated_distance, remaining_length
                            )
                            current_x += ux * skip_length
                            current_y += uy * skip_length
                            accumulated_distance += skip_length
                            remaining_length -= skip_length
                            if accumulated_distance >= gap_length:
                                is_drawing = True
                                accumulated_distance = 0
        bordered_image = cv2.copyMakeBorder(
            colored_image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=COLOR_BORDER,
        )
        cv2.imwrite(path, bordered_image)

    def _save_target_image(self, target, path, border_size=0):
        COLOR_BACKGROUND = (255, 255, 255)
        COLOR_TARGET = (0, 165, 255)
        COLOR_BORDER = (0, 0, 0)
        if torch.is_tensor(target):
            target_np = target.cpu().numpy()
        else:
            target_np = target
        crop_ratio = CROP_RATIO
        (height, width) = target_np.shape
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)
        target_np = target_np[crop_h : height - crop_h, crop_w : width - crop_w]
        (height, width) = target_np.shape
        target_binary = (target_np > 0.5).astype(np.uint8) * 255
        colored_image = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
        colored_image[target_binary == 255] = COLOR_TARGET
        bordered_image = cv2.copyMakeBorder(
            colored_image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=COLOR_BORDER,
        )
        cv2.imwrite(path, bordered_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def sim(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            (printedNom, printedMax, printedMin) = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return (mask, binaryNom)


EPE_CONSTRAINT = 15
EPE_CHECK_INTERVEL = 40
MIN_EPE_CHECK_LENGTH = 80
EPE_CHECK_START_INTERVEL = 40


def boundaries(target, dtype=REALTYPE, device=DEVICE):
    boundary = torch.zeros_like(target)
    corner = torch.zeros_like(target)
    vertical = torch.zeros_like(target)
    horizontal = torch.zeros_like(target)
    padded = func.pad(target[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper = padded[2:, 1:-1] == 1
    lower = padded[:-2, 1:-1] == 1
    left = padded[1:-1, :-2] == 1
    right = padded[1:-1, 2:] == 1
    upperleft = padded[2:, :-2] == 1
    upperright = padded[2:, 2:] == 1
    lowerleft = padded[:-2, :-2] == 1
    lowerright = padded[:-2, 2:] == 1
    boundary = target == 1
    boundary[
        upper & lower & left & right & upperleft & upperright & lowerleft & lowerright
    ] = False
    padded = func.pad(boundary[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper = padded[2:, 1:-1] == 1
    lower = padded[:-2, 1:-1] == 1
    left = padded[1:-1, :-2] == 1
    right = padded[1:-1, 2:] == 1
    center = padded[1:-1, 1:-1] == 1
    vertical = center.clone()
    vertical[left & right] = False
    vsites = vertical.nonzero()
    vindices = np.lexsort(
        (vsites[:, 0].detach().cpu().numpy(), vsites[:, 1].detach().cpu().numpy())
    )
    vsites = vsites[vindices]
    vstart = torch.cat(
        (
            torch.tensor([True], device=vsites.device),
            vsites[:, 0][1:] != vsites[:, 0][:-1] + 1,
        )
    )
    vend = torch.cat(
        (
            vsites[:, 0][1:] != vsites[:, 0][:-1] + 1,
            torch.tensor([True], device=vsites.device),
        )
    )
    vstart = vsites[(vstart == True).nonzero()[:, 0], :]
    vend = vsites[(vend == True).nonzero()[:, 0], :]
    vposes = torch.stack((vstart, vend), axis=2)
    horizontal = center.clone()
    horizontal[upper & lower] = False
    hsites = horizontal.nonzero()
    hindices = np.lexsort(
        (hsites[:, 1].detach().cpu().numpy(), hsites[:, 0].detach().cpu().numpy())
    )
    hsites = hsites[hindices]
    hstart = torch.cat(
        (
            torch.tensor([True], device=hsites.device),
            hsites[:, 1][1:] != hsites[:, 1][:-1] + 1,
        )
    )
    hend = torch.cat(
        (
            hsites[:, 1][1:] != hsites[:, 1][:-1] + 1,
            torch.tensor([True], device=hsites.device),
        )
    )
    hstart = hsites[(hstart == True).nonzero()[:, 0], :]
    hend = hsites[(hend == True).nonzero()[:, 0], :]
    hposes = torch.stack((hstart, hend), axis=2)
    return (vposes.float(), hposes.float())


def check(image, sample, target, direction):
    if direction == "v":
        if (
            target[sample[0, 0].long(), sample[0, 1].long() + 1] == 1
            and target[sample[0, 0].long(), sample[0, 1].long() - 1] == 0
        ):
            inner = sample + torch.tensor(
                [0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]
        elif (
            target[sample[0, 0].long(), sample[0, 1].long() + 1] == 0
            and target[sample[0, 0].long(), sample[0, 1].long() - 1] == 1
        ):
            inner = sample + torch.tensor(
                [0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]
    if direction == "h":
        if (
            target[sample[0, 0].long() + 1, sample[0, 1].long()] == 1
            and target[sample[0, 0].long() - 1, sample[0, 1].long()] == 0
        ):
            inner = sample + torch.tensor(
                [EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]
        elif (
            target[sample[0, 0].long() + 1, sample[0, 1].long()] == 0
            and target[sample[0, 0].long() - 1, sample[0, 1].long()] == 1
        ):
            inner = sample + torch.tensor(
                [-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]
    return (inner, outer)


def epecheck(mask, target, vposes, hposes):
    inner = 0
    outer = 0
    epeMap = torch.zeros_like(target)
    vioMap = torch.zeros_like(target)
    all_sample_points = []
    for idx in range(vposes.shape[0]):
        center = (vposes[idx, :, 0] + vposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)
        if vposes[idx, 0, 1] - vposes[idx, 0, 0] <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            (v_in_site, v_out_site) = check(mask, sample, target, "v")
        else:
            sampleY = torch.cat(
                (
                    torch.arange(
                        vposes[idx, 0, 0] + EPE_CHECK_START_INTERVEL,
                        center[0, 0] + 1,
                        step=EPE_CHECK_INTERVEL,
                    ),
                    torch.arange(
                        vposes[idx, 0, 1] - EPE_CHECK_START_INTERVEL,
                        center[0, 0],
                        step=-EPE_CHECK_INTERVEL,
                    ),
                )
            ).unique()
            sample = vposes[idx, :, 0].repeat(sampleY.shape[0], 1)
            sample[:, 0] = sampleY
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            (v_in_site, v_out_site) = check(mask, sample, target, "v")
        all_sample_points.append(sample)
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1
    for idx in range(hposes.shape[0]):
        center = (hposes[idx, :, 0] + hposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)
        if hposes[idx, 1, 1] - hposes[idx, 1, 0] <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            (v_in_site, v_out_site) = check(mask, sample, target, "h")
        else:
            sampleX = torch.cat(
                (
                    torch.arange(
                        hposes[idx, 1, 0] + EPE_CHECK_START_INTERVEL,
                        center[0, 1] + 1,
                        step=EPE_CHECK_INTERVEL,
                    ),
                    torch.arange(
                        hposes[idx, 1, 1] - EPE_CHECK_START_INTERVEL,
                        center[0, 1],
                        step=-EPE_CHECK_INTERVEL,
                    ),
                )
            ).unique()
            sample = hposes[idx, :, 0].repeat(sampleX.shape[0], 1)
            sample[:, 1] = sampleX
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            (v_in_site, v_out_site) = check(mask, sample, target, "h")
        all_sample_points.append(sample)
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1
    if len(all_sample_points) > 0:
        all_sample_points = torch.cat(all_sample_points, dim=0)
    else:
        all_sample_points = torch.empty((0, 2), device=target.device)
    return (inner, outer, vioMap, epeMap, all_sample_points)


class EPEChecker:
    def __init__(
        self,
        litho=lithosim.LithoSim("./config/lithosimple.txt"),
        thresh=0.5,
        device=DEVICE,
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            (printedNom, printedMax, printedMin) = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            (vposes, hposes) = boundaries(target)
            (epeIn, epeOut, _, _, _) = epecheck(binaryNom, target, vposes, hposes)
        return (epeIn, epeOut)

    def vioMap(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            (printedNom, printedMax, printedMin) = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            (vposes, hposes) = boundaries(target)
            (epeIn, epeOut, vioMap, _, _) = epecheck(binaryNom, target, vposes, hposes)
        return (epeIn, epeOut, vioMap)

    def runWithSamples(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            (printedNom, printedMax, printedMin) = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            (vposes, hposes) = boundaries(target)
            (epeIn, epeOut, vioMap, epeMap, sample_points) = epecheck(
                binaryNom, target, vposes, hposes
            )
            epe_results = torch.ones(sample_points.shape[0], device=self._device)
            if sample_points.shape[0] > 0:
                violation_mask = (
                    vioMap[sample_points[:, 0].long(), sample_points[:, 1].long()] == 1
                )
                epe_results[violation_mask] = 0
        return (epeIn, epeOut, vioMap, epeMap, sample_points, epe_results)


import cv2
from adabox import proc, tools
from typing import List, Sequence


class Bit1D:
    __slots__ = ("n", "t")

    def __init__(self, n: int):
        self.n = n
        self.t = np.zeros(n + 1, dtype=np.int32)

    def update(self, idx: int, val: int):
        while idx <= self.n:
            if val > self.t[idx]:
                self.t[idx] = val
            idx += idx & -idx

    def query(self, idx: int) -> int:
        res = 0
        while idx:
            if self.t[idx] > res:
                res = self.t[idx]
            idx &= idx - 1
        return res


class Bit2D:
    __slots__ = ("ny", "nx", "rows")

    def __init__(self, ny: int, nx: int):
        self.ny = ny
        self.nx = nx
        self.rows: list[Bit1D | None] = [None] * (ny + 1)

    def _ensure(self, i):
        if self.rows[i] is None:
            self.rows[i] = Bit1D(self.nx)

    def update(self, iy: int, ix: int, val: int):
        while iy <= self.ny:
            self._ensure(iy)
            self.rows[iy].update(ix, val)
            iy += iy & -iy

    def query(self, iy: int, ix: int) -> int:
        res = 0
        while iy:
            r = self.rows[iy]
            if r is not None:
                v = r.query(ix)
                if v > res:
                    res = v
            iy &= iy - 1
        return res


from numba import njit


@njit
def check_containment_optimized(y1_s, x1_s, y2_s, x2_s, areas, n):
    keep_flag = np.ones(n, dtype=np.bool_)
    keep_list = np.empty(n, dtype=np.int32)
    keep_count = 0
    for i in range(n):
        (cur_y1, cur_x1) = (y1_s[i], x1_s[i])
        (cur_y2, cur_x2) = (y2_s[i], x2_s[i])
        contained = False
        for j in range(keep_count):
            k = keep_list[j]
            if (
                y1_s[k] <= cur_y1
                and cur_y2 <= y2_s[k]
                and (x1_s[k] <= cur_x1)
                and (cur_x2 <= x2_s[k])
            ):
                contained = True
                break
        if not contained:
            keep_list[keep_count] = i
            keep_count += 1
        else:
            keep_flag[i] = False
    return keep_flag


class FastShotCounter:
    def __init__(self, nlogn=False, scan_line=True):
        self.nlogn = nlogn
        self.scan_line = scan_line

    def rle_rows(self, img):
        runs = []
        (H, W) = img.shape
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

    def prune_rectangles(self, rects: Sequence[Sequence[int]]) -> List[List[int]]:
        n = len(rects)
        if n <= 1:
            return [list(r) for r in rects]
        a = np.asarray(rects, dtype=np.int32)
        (y1, x1, h, w) = (a[:, 0], a[:, 1], a[:, 2], a[:, 3])
        y2 = y1 + h
        x2 = x1 + w
        order = np.lexsort((-y2, -x2, y1, x1))
        (y1, y2, x2) = (y1[order], y2[order], x2[order])
        ys = np.unique(y1)
        xs = np.unique(x2)
        (ny, nx) = (len(ys), len(xs))
        iy = np.searchsorted(ys, y1) + 1
        ix = (nx - np.searchsorted(xs, x2)).astype(np.int32)
        bit = Bit2D(ny, nx)
        keep = np.empty(n, dtype=bool)
        for i in range(n):
            if bit.query(iy[i], ix[i]) >= y2[i]:
                keep[i] = False
            else:
                keep[i] = True
                bit.update(iy[i], ix[i], int(y2[i]))
        return a[order][keep].tolist()

    def prune_rectangles_numba(self, rects: Sequence[Sequence[int]]) -> List[List[int]]:
        n = len(rects)
        if n <= 1:
            return [list(r) for r in rects]
        arr = np.asarray(rects, dtype=np.int32)
        (y1, x1, h, w) = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
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
        (H, W) = img.shape
        h = np.zeros(W, int)
        rects = set()
        for r in range(H):
            h = np.where(img[r] == 1, h + 1, 0)
            stack = []
            for c in range(W + 1):
                cur = h[c] if c < W else 0
                last = c
                while stack and cur < stack[-1][1]:
                    (pos, height) = stack.pop()
                    area_left = stack[-1][0] + 1 if stack else 0
                    rects.add((r - height + 1, area_left, height, c - area_left))
                    last = pos
                stack.append((last, cur))
        return list(rects)

    def run(self, mask, shape=(512, 512)):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        mask = torch.nn.functional.interpolate(
            mask[None, None, :, :], size=shape, mode="nearest"
        )[0, 0]
        if mask.device.type != "cpu":
            mask = mask.detach().cpu().numpy().astype(np.uint8)
        R = self.maximal_rectangles(mask)
        start_time = time.time()
        if self.nlogn:
            R = self.prune_rectangles(R)
        else:
            R = self.prune_rectangles_numba(R)
        end_time = time.time()
        (H, W) = shape
        prob = LpProblem("rect_cover", LpMinimize)
        x = LpVariable.dicts("x", range(len(R)), 0, 1, cat="Binary")
        row_rects = defaultdict(list)
        for i, (y, x0, h, w) in enumerate(R):
            for r in range(y, y + h):
                row_rects[r].append(i)
        if not self.scan_line:
            for r, s, e in self.rle_rows(mask):
                rects_in_row = row_rects[r]
                for c in range(s, e + 1):
                    prob += (
                        lpSum(
                            (
                                x[i]
                                for i in rects_in_row
                                if R[i][1] <= c < R[i][1] + R[i][3]
                            )
                        )
                        >= 1
                    )
        else:
            row_runs = defaultdict(list)
            for r, s, e in self.rle_rows(mask):
                row_runs[r].append((s, e))
            for r in range(H):
                rects_in_row = row_rects[r]
                if not rects_in_row:
                    continue
                cuts = {R[i][1] for i in rects_in_row} | {
                    R[i][1] + R[i][3] for i in rects_in_row
                }
                cuts |= {col for (s, e) in row_runs[r] for col in (s, e)}
                segments = sorted(cuts)
                for a, b in zip(segments, segments[1:]):
                    c0 = a
                    if mask[r, c0] == 0:
                        continue
                    prob += (
                        lpSum(
                            (
                                x[i]
                                for i in rects_in_row
                                if R[i][1] <= c0 < R[i][1] + R[i][3]
                            )
                        )
                        >= 1
                    )
        prob += lpSum(x.values())
        start_time = time.time()
        prob.solve(PULP_CBC_CMD(timeLimit=300, msg=False))
        end_time = time.time()
        answer = [R[i] for i in range(len(R)) if x[i].value() == 1]
        return len(answer)


class ShotCounter:
    def __init__(
        self,
        litho=lithosim.LithoSim("./config/lithosimple.txt"),
        thresh=0.5,
        device=DEVICE,
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target=None, scale=1, shape=(512, 512)):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        image = torch.nn.functional.interpolate(
            mask[None, None, :, :], size=shape, mode="nearest"
        )[0, 0]
        image = image.detach().cpu().numpy().astype(np.uint8)
        (comps, labels, stats, centroids) = cv2.connectedComponentsWithStats(image)
        rectangles = []
        for label in range(1, comps):
            pixels = []
            for idx in range(labels.shape[0]):
                for jdx in range(labels.shape[1]):
                    if labels[idx, jdx] == label:
                        pixels.append([idx, jdx, 0])
            pixels = np.array(pixels)
            x_data = np.unique(np.sort(pixels[:, 0]))
            y_data = np.unique(np.sort(pixels[:, 1]))
            if x_data.shape[0] == 1 or y_data.shape[0] == 1:
                rectangles.append(
                    tools.Rectangle(
                        x_data.min(), x_data.max(), y_data.min(), y_data.max()
                    )
                )
                continue
            (rects, sep) = proc.decompose(pixels, 4)
            rectangles.extend(rects)
        return len(rectangles)
