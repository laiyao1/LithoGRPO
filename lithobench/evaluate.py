import os
import time
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycommon.settings import *
import pycommon.glp as glp
import pylitho.exact as litho
import pyilt.simpleilt as simpleilt
import pyilt.curvmulti as curvmulti
import pyilt.evaluation as evaluation


def getTargets(samples=10, dataset="MetalSet"):
    asset_root = os.environ.get("LITHOGRPO_LITHOBENCH_ROOT", "./external/lithobench")
    results = []
    if dataset == "MetalSet":
        if samples is None:
            samples = 10
        for idx in range(1, 1 + samples):
            design = glp.Design(
                f"{asset_root}/benchmark/ICCAD2013/M1_test{idx}.glp", down=1
            )
            design.center(2048, 2048, 384, 384)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset == "ViaSet":
        filenames = glob.glob(f"{asset_root}/benchmark/OpenROAD/*.glp")
        filenames.sort()
        if samples is None:
            samples = 10
        for idx in range(1, 1 + samples):
            design = glp.Design(filenames[idx - 1], down=1)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset == "StdMetal":
        filenames = glob.glob(f"{asset_root}/benchmark/StdMetal/*.glp")
        filenames.sort()
        if samples is None:
            samples = len(filenames)
        for idx in range(1, 1 + samples):
            design = glp.Design(filenames[idx - 1], down=1)
            design.center(2048, 2048, 384, 384)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset == "StdContact":
        filenames = glob.glob(f"{asset_root}/benchmark/StdContact/*.glp")
        filenames.sort()
        if samples is None:
            samples = len(filenames)
        for idx in range(1, 1 + samples):
            design = glp.Design(filenames[idx - 1], down=1)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset.lower() == "cvpr":
        cvpr_image = cv2.imread("demo/CVPR_2048x2048.png", cv2.IMREAD_GRAYSCALE) / 255.0
        cv_image = cv2.imread("demo/CV_2048x2048.png", cv2.IMREAD_GRAYSCALE) / 255.0
        results.append(cvpr_image)
        results.append(cv_image)
    return results


def evalRaw(masksInit, targets, method="Unknown", save_images=False):
    sim = litho.LithoSim("./config/lithosimple.txt")
    test = evaluation.Basic(sim, 0.5, method=method)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    all_samples = []
    for idx in range(len(targets)):
        mask0 = masksInit[idx]
        mask0 = F.interpolate(mask0[None, None, :, :], size=(2048, 2048))[0, 0]
        mask0[mask0 > 0.5] = 1.0
        mask0[mask0 <= 0.5] = 0.0
        (l20, pvb0) = test.run(
            mask0, targets[idx], scale=1, idx=idx, save_images=save_images
        )
        (epeIn0, epeOut0, vioMap, epeMap, sample_points, epe_results) = (
            epeCheck.runWithSamples(mask0, targets[idx], scale=1)
        )
        epe0 = epeIn0 + epeOut0
        l2s.append(l20)
        pvbs.append(pvb0)
        epes.append(epe0)
        all_samples.append(sample_points)
        if save_images:
            binaryNom_np = (test.binaryNom * 255).cpu().numpy().astype(np.uint8)
            test.draw_epe(
                binary_data=binaryNom_np,
                target=targets[idx],
                sample_points=sample_points,
                epe_results=epe_results,
                path=f"images/epe_visualization_{method}_{idx}_l2_{l20:.0f}_pvb_{pvb0:.0f}_epe_{epe0:.0f}.png",
                border_size=0,
            )
    return (l2s, pvbs, epes)


def shots(masks, targets=None, timer=False):
    shotCount = evaluation.ShotCounter(None, 0.5)
    shots = []
    if timer:
        times = []
    for idx in range(len(masks)):
        mask = masks[idx]
        mask = F.interpolate(mask[None, None, :, :], size=(2048, 2048))[0, 0]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        start_time = time.time()
        shot = shotCount.run(mask, shape=(256, 256))
        end_time = time.time()
        shots.append(shot)
        if timer:
            times.append(end_time - start_time)
    if timer:
        return (shots, times)
    return shots


def process_single_mask(idx, mask, shotCount):
    mask = F.interpolate(mask[None, None, :, :], size=(2048, 2048))[0, 0]
    mask[mask > 0.5] = 1.0
    mask[mask <= 0.5] = 0.0
    shot = shotCount.run(mask, shape=(256, 256))
    return (idx, shot)


import time


def fast_shots(masks, targets=None, timer=False):
    shotCount = evaluation.FastShotCounter()
    shots = []
    if timer:
        times = []
    for idx in range(len(masks)):
        mask = masks[idx]
        mask = F.interpolate(mask[None, None, :, :], size=(2048, 2048))[0, 0]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        start_time = time.time()
        shot = shotCount.run(mask, shape=(256, 256))
        end_time = time.time()
        shots.append(shot)
        if timer:
            times.append(end_time - start_time)
    if timer:
        return (shots, times)
    return shots


def finetuneSimple(paramsInit, masksInit, targets):
    cfg = simpleilt.SimpleCfg("./config/simpleilt.txt")
    sim = litho.LithoSim("./config/lithosimple.txt")
    solver = simpleilt.SimpleILT(cfg, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    for idx in range(len(targets)):
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)
        params0 = F.interpolate(params0[None, None, :, :], size=(2048, 2048))[0, 0]
        mask0 = F.interpolate(mask0[None, None, :, :], size=(2048, 2048))[0, 0]
        mask0[mask0 > 0.5] = 1.0
        mask0[mask0 <= 0.5] = 0.0
        (l2, pvb, bestParams, bestMask) = solver.solve(target, params0)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0
        (l2, pvb) = test.run(bestMask, target, scale=1)
        (epeIn, epeOut) = epeCheck.run(bestMask, target, scale=1)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
    return (l2s, pvbs, epes)


def finetuneFast(paramsInit, masksInit, targets):
    cfgMid = curvmulti.CurvILTCfg("./config/curvilt512.txt")
    cfgHigh = curvmulti.CurvILTCfg("./config/curvilt1024.txt")
    cfgMid._config["Iterations"] = 200
    cfgHigh._config["Iterations"] = 25
    sim = litho.LithoSim("./config/lithosimple.txt")
    solverMid = curvmulti.CurvILT(cfgMid, sim, multigpu=False)
    solverHigh = curvmulti.CurvILT(cfgHigh, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    masks = []
    for idx in range(len(targets)):
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)
        params0Mid = F.interpolate(mask0[None, None, :, :], size=(512, 512))[0, 0]
        targetMid = F.interpolate(target[None, None, :, :], size=(512, 512))[0, 0]
        (l2, pvb, bestParams, bestMask) = solverMid.solve(targetMid, params0Mid)
        params0High = F.interpolate(bestParams[None, None, :, :], size=(1024, 1024))[
            0, 0
        ]
        targetHigh = F.interpolate(target[None, None, :, :], size=(1024, 1024))[0, 0]
        (l2, pvb, bestParams, bestMask) = solverHigh.solve(targetHigh, params0High)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0
        (l2, pvb) = test.run(bestMask, target, scale=2)
        (epeIn, epeOut) = epeCheck.run(bestMask, target, scale=2)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        masks.append(bestMask)
    return (l2s, pvbs, epes, masks)


def finetune1024(paramsInit, masksInit, targets):
    cfgHigh = curvmulti.CurvILTCfg("./config/curvilt1024.txt")
    cfgHigh._config["StepSize"] = 1.0
    cfgHigh._config["Iterations"] = 150
    sim = litho.LithoSim("./config/lithosimple.txt")
    solverHigh = curvmulti.CurvILT(cfgHigh, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    masks = []
    for idx in range(len(targets)):
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)
        params0High = F.interpolate(mask0[None, None, :, :], size=(1024, 1024))[0, 0]
        targetHigh = F.interpolate(target[None, None, :, :], size=(1024, 1024))[0, 0]
        (l2, pvb, bestParams, bestMask) = solverHigh.solve(targetHigh, params0High)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0
        (l2, pvb) = test.run(bestMask, target, scale=2)
        (epeIn, epeOut) = epeCheck.run(bestMask, target, scale=2)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        masks.append(bestMask)
    return (l2s, pvbs, epes, masks)


def finetune(paramsInit, masksInit, targets):
    cfgLow = curvmulti.CurvILTCfg("./config/curvilt256.txt")
    cfgMid = curvmulti.CurvILTCfg("./config/curvilt512.txt")
    cfgHigh = curvmulti.CurvILTCfg("./config/curvilt1024.txt")
    sim = litho.LithoSim("./config/lithosimple.txt")
    solverLow = curvmulti.CurvILT(cfgLow, sim, multigpu=False)
    solverMid = curvmulti.CurvILT(cfgMid, sim, multigpu=False)
    solverHigh = curvmulti.CurvILT(cfgHigh, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    masks = []
    for idx in range(len(targets)):
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)
        params0Low = F.interpolate(mask0[None, None, :, :], size=(256, 256))[0, 0]
        targetLow = F.interpolate(target[None, None, :, :], size=(256, 256))[0, 0]
        (l2, pvb, bestParams, bestMask) = solverLow.solve(targetLow, params0Low)
        params0Mid = F.interpolate(bestParams[None, None, :, :], size=(512, 512))[0, 0]
        targetMid = F.interpolate(target[None, None, :, :], size=(512, 512))[0, 0]
        (l2, pvb, bestParams, bestMask) = solverMid.solve(targetMid, params0Mid)
        params0High = F.interpolate(bestParams[None, None, :, :], size=(1024, 1024))[
            0, 0
        ]
        targetHigh = F.interpolate(target[None, None, :, :], size=(1024, 1024))[0, 0]
        (l2, pvb, bestParams, bestMask) = solverHigh.solve(targetHigh, params0High)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0
        (l2, pvb) = test.run(bestMask, target, scale=2)
        (epeIn, epeOut) = epeCheck.run(bestMask, target, scale=2)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        masks.append(bestMask)
    return (l2s, pvbs, epes, masks)
