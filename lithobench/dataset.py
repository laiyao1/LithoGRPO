import os
import glob
import math
import random
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

config_file = os.environ.get("LITHOGRPO_DATA_CONFIG", "config/data_config.yaml")
if not os.path.isfile(config_file):
    config_file = "config/data_config.example.yaml"
with open(config_file, "rb") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
data_path = config["data_path"]


def filesMaskOpt(folder):
    folderGLP = os.path.join(folder, "glp")
    folderTarget = os.path.join(folder, "target")
    folderPixel = os.path.join(folder, "pixelILT")
    filesGLP = glob.glob(folderGLP + "/*.glp")
    filesTarget = glob.glob(folderTarget + "/*.png")
    filesPixel = glob.glob(folderPixel + "/*.png")
    basefunc = lambda x: os.path.basename(x)[:-4]
    setGLP = set(map(basefunc, filesGLP))
    setTarget = set(map(basefunc, filesTarget))
    setPixel = set(map(basefunc, filesPixel))
    basenames = setGLP & setTarget & setPixel
    filesGLP = list(filter(lambda x: basefunc(x) in basenames, filesGLP))
    filesTarget = list(filter(lambda x: basefunc(x) in basenames, filesTarget))
    filesPixel = list(filter(lambda x: basefunc(x) in basenames, filesPixel))
    filesGLP = sorted(filesGLP, key=basefunc)
    filesTarget = sorted(filesTarget, key=basefunc)
    filesPixel = sorted(filesPixel, key=basefunc)
    return (filesGLP, filesTarget, filesPixel)


def filesLithoSim(folder, binarized=True):
    folderPixel = os.path.join(folder, "pixelILT")
    folderLitho = os.path.join(folder, "litho")
    folderResist = (
        os.path.join(folder, "printed") if binarized else os.path.join(folder, "resist")
    )
    filesPixel = glob.glob(folderPixel + "/*.png")
    filesLitho = glob.glob(folderLitho + "/*.png")
    filesResist = glob.glob(folderResist + "/*.png")
    basefunc = lambda x: os.path.basename(x)[:-4]
    setPixel = set(map(basefunc, filesPixel))
    setLitho = set(map(basefunc, filesLitho))
    setResist = set(map(basefunc, filesResist))
    basenames = setPixel & setLitho & setResist
    filesPixel = list(filter(lambda x: basefunc(x) in basenames, filesPixel))
    filesLitho = list(filter(lambda x: basefunc(x) in basenames, filesLitho))
    filesResist = list(filter(lambda x: basefunc(x) in basenames, filesResist))
    filesPixel = sorted(filesPixel, key=basefunc)
    filesLitho = sorted(filesLitho, key=basefunc)
    filesResist = sorted(filesResist, key=basefunc)
    return (filesPixel, filesLitho, filesResist)


class DataILT_Test(Dataset):
    def __init__(self, targets, size=(1024, 1024)):
        super().__init__()
        self._size = size
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()
        elif isinstance(targets, list):
            if len(targets) > 0 and isinstance(targets[0], np.ndarray):
                targets = [torch.from_numpy(t).float() for t in targets]
        if isinstance(targets, torch.Tensor):
            if targets.ndim == 2:
                self._targets = [targets]
            elif targets.ndim == 3:
                self._targets = [targets[i] for i in range(targets.shape[0])]
            else:
                raise ValueError(f"Unexpected tensor shape: {targets.shape}")
        elif isinstance(targets, list):
            self._targets = targets
        else:
            raise ValueError(f"Unexpected targets type: {type(targets)}")
        processed_targets = []
        for target in self._targets:
            if not isinstance(target, torch.Tensor):
                target = torch.from_numpy(target).float()
            if target.shape != size:
                target = F.interpolate(
                    target[None, None, :, :],
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            processed_targets.append(target)
        self._raw_targets = self._targets
        self._targets = processed_targets

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        index = int(index)
        target = self._targets[index]
        target = target[None, :, :]
        return {
            "target": target,
            "mask": None,
            "raw_target": self._raw_targets[index].numpy(),
            "index": index,
        }

    def __len__(self):
        return len(self._targets)


class DataILT(Dataset):
    def __init__(
        self,
        filesTarget,
        filesPixel,
        crop=False,
        size=(1024, 1024),
        cache=False,
        return_dict=False,
    ):
        super().__init__()
        self._filesTarget = filesTarget
        self._filesPixel = filesPixel
        self._crop = crop
        self._size = size
        self._cache = cache
        self._return_dict = return_dict
        self._imagesTarget = []
        self._imagesPixel = []
        if self._cache:
            for filename in tqdm(self._filesTarget):
                raw_target = self._loadImageRaw(filename)
                resized_target = self._resizeImage(raw_target)
                self._imagesTarget.append(resized_target)
            for filename in tqdm(self._filesPixel):
                self._imagesPixel.append(self._loadImage(filename))

    def __getitem__(self, index):
        target = self._loadTarget(index)
        mask = self._loadMask(index)
        if self._return_dict:
            raw_target = self._loadImageRaw(self._filesTarget[index])
        target = target[None, :, :]
        mask = mask[None, :, :]
        if self._crop:
            padX = self._size[0] // 8
            padY = self._size[1] // 8
            startX = random.randint(0, 2 * padX - 1)
            startY = random.randint(0, 2 * padY - 1)
            target = F.pad(target.unsqueeze(0), (padX, padX, padY, padY))
            mask = F.pad(mask.unsqueeze(0), (padX, padX, padY, padY))
            target = target[
                0, :, startX : startX + self._size[0], startY : startY + self._size[1]
            ]
            mask = mask[
                0, :, startX : startX + self._size[0], startY : startY + self._size[1]
            ]
            if random.randint(0, 1) == 1:
                target = target.flip(1)
                mask = mask.flip(1)
            if random.randint(0, 1) == 1:
                target = target.flip(2)
                mask = mask.flip(2)
        if self._return_dict:
            return {
                "target": target,
                "mask": mask,
                "raw_target": raw_target.numpy(),
                "index": index,
            }
        else:
            return (target, mask)

    def __len__(self):
        return len(self._filesTarget)

    def _loadImageRaw(self, filename):
        image = cv2.imread(filename)
        if len(image.shape) > 2:
            image = (
                torch.tensor(image[:, :, 0], dtype=torch.float32, device="cpu") / 255.0
            )
        else:
            image = torch.tensor(image, dtype=torch.float32, device="cpu") / 255.0
        return image

    def _resizeImage(self, image):
        if image.shape != self._size:
            image = F.interpolate(image[None, None, :, :], self._size)[0, 0]
        return image

    def _loadImage(self, filename):
        image = cv2.imread(filename)
        if len(image.shape) > 2:
            image = (
                torch.tensor(image[:, :, 0], dtype=torch.float32, device="cpu") / 255.0
            )
        else:
            image = torch.tensor(image, dtype=torch.float32, device="cpu") / 255.0
        image = F.interpolate(image[None, None, :, :], self._size)[0, 0]
        return image

    def _loadTarget(self, index):
        if self._cache:
            return self._imagesTarget[index]
        else:
            return self._loadImage(self._filesTarget[index])

    def _loadMask(self, index):
        if self._cache:
            return self._imagesPixel[index]
        else:
            return self._loadImage(self._filesPixel[index])


class DataLitho(Dataset):
    def __init__(
        self,
        filesMask,
        filesLitho,
        filesResist,
        crop=False,
        size=(1024, 1024),
        cache=False,
    ):
        super().__init__()
        assert len(filesMask) == len(filesLitho) and len(filesMask) == len(
            filesResist
        ), f"File count mismatch: {len(filesMask)}/{len(filesLitho)}/{len(filesResist)}"
        self._filesMask = filesMask
        self._filesLitho = filesLitho
        self._filesResist = filesResist
        self._crop = crop
        self._size = size
        self._cache = cache
        self._imagesMask = []
        self._imagesLitho = []
        self._imagesResist = []
        if self._cache:
            for filename in tqdm(self._filesMask):
                self._imagesMask.append(self._loadImage(filename))
            for filename in tqdm(self._filesLitho):
                self._imagesLitho.append(self._loadImage(filename))
            for filename in tqdm(self._filesResist):
                self._imagesResist.append(self._loadImage(filename))

    def __getitem__(self, index):
        mask = self._loadMask(index)
        litho = self._loadLitho(index)
        resist = self._loadResist(index)
        mask = mask[None, :, :]
        litho = litho[None, :, :]
        resist = resist[None, :, :]
        if self._crop:
            padX = self._size[0] // 8
            padY = self._size[1] // 8
            startX = random.randint(0, 2 * padX - 1)
            startY = random.randint(0, 2 * padY - 1)
            mask = F.pad(mask.unsqueeze(0), (padX, padX, padY, padY))
            litho = F.pad(litho.unsqueeze(0), (padX, padX, padY, padY))
            resist = F.pad(resist.unsqueeze(0), (padX, padX, padY, padY))
            mask = mask[
                0, :, startX : startX + self._size[0], startY : startY + self._size[1]
            ]
            litho = litho[
                0, :, startX : startX + self._size[0], startY : startY + self._size[1]
            ]
            resist = resist[
                0, :, startX : startX + self._size[0], startY : startY + self._size[1]
            ]
            if random.randint(0, 1) == 1:
                mask = mask.flip(1)
                litho = litho.flip(1)
                resist = resist.flip(1)
            if random.randint(0, 1) == 1:
                mask = mask.flip(2)
                litho = litho.flip(2)
                resist = resist.flip(2)
        return (mask, litho, resist)

    def __len__(self):
        return len(self._filesMask)

    def _loadImage(self, filename):
        image = cv2.imread(filename)
        if len(image.shape) > 2:
            image = (
                torch.tensor(image[:, :, 0], dtype=torch.float32, device="cpu") / 255.0
            )
        image = F.interpolate(image[None, None, :, :], self._size)[0, 0]
        return image

    def _loadMask(self, index):
        return (
            self._imagesMask[index]
            if self._cache
            else self._loadImage(self._filesMask[index])
        )

    def _loadLitho(self, index):
        return (
            self._imagesLitho[index]
            if self._cache
            else self._loadImage(self._filesLitho[index])
        )

    def _loadResist(self, index):
        return (
            self._imagesResist[index]
            if self._cache
            else self._loadImage(self._filesResist[index])
        )


def maskopt(
    basedir,
    sizeImage=(512, 512),
    ratioTrain=0.9,
    cache=False,
    targets=None,
    return_dict=False,
):
    (filesGLP, filesTarget, filesMask) = filesMaskOpt(basedir)
    numFiles = len(filesGLP)
    numTrain = round(numFiles * ratioTrain)
    trainTarget = filesTarget[:numTrain]
    trainMask = filesMask[:numTrain]
    testTarget = filesTarget[numTrain:]
    testMask = filesMask[numTrain:]
    train = DataILT(
        trainTarget,
        trainMask,
        crop=True,
        size=sizeImage,
        cache=cache,
        return_dict=return_dict,
    )
    test = DataILT(
        testTarget,
        testMask,
        crop=False,
        size=sizeImage,
        cache=cache,
        return_dict=return_dict,
    )
    if targets is not None:
        test_final = DataILT_Test(targets, size=sizeImage)
        return (train, test, test_final)
    else:
        return (train, test)


def lithosim(basedir, sizeImage=(512, 512), ratioTrain=0.9, cache=False):
    (filesPixel, filesLitho, filesResist) = filesLithoSim(basedir)
    numFiles = len(filesPixel)
    numTrain = round(numFiles * ratioTrain)
    trainPixel = filesPixel[:numTrain]
    trainLitho = filesLitho[:numTrain]
    trainResist = filesResist[:numTrain]
    testPixel = filesPixel[numTrain:]
    testLitho = filesLitho[numTrain:]
    testResist = filesResist[numTrain:]
    train = DataLitho(
        trainPixel, trainLitho, trainResist, crop=True, size=sizeImage, cache=cache
    )
    test = DataLitho(
        testPixel, testLitho, testResist, crop=False, size=sizeImage, cache=cache
    )
    return (train, test)


def loadersILT(
    benchmark, image_size, batch_size, njobs, drop_last=False, debug_samples=None
):
    (trainset, valset) = maskopt(
        f"{data_path}/{benchmark}",
        sizeImage=image_size,
        ratioTrain=0.9,
        cache=False,
        return_dict=False,
    )
    if debug_samples is not None:
        trainset = Subset(trainset, list(range(debug_samples)))
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=False,
    )
    return (train_loader, val_loader)


def calculate_gradient_accumulation_steps(
    batch_size, num_generations, max_batch_per_step, round_up=True
):
    total_images = batch_size * num_generations
    if round_up:
        gradient_accumulation_steps = math.ceil(total_images / max_batch_per_step)
    else:
        gradient_accumulation_steps = math.floor(total_images / max_batch_per_step)
    actual_images_per_step = total_images / gradient_accumulation_steps
    if actual_images_per_step != int(actual_images_per_step):
        pass
    return gradient_accumulation_steps


def grpo_collate_fn(batch):
    if isinstance(batch[0], dict):
        result = {
            "data": torch.stack([item["target"] for item in batch]),
            "index": torch.tensor([item["index"] for item in batch]),
            "raw_target": torch.stack(
                [torch.from_numpy(item["raw_target"]) for item in batch]
            ),
        }
    else:
        result = {
            "data": torch.stack([item[0] for item in batch]),
            "index": torch.arange(len(batch)),
        }
    return result


def loadersILT_GRPO(
    benchmark,
    targets,
    image_size,
    batch_size,
    njobs,
    num_generations=4,
    max_batch_per_step=12,
    drop_last=False,
    debug_samples=None,
):
    if benchmark == "StdContact" or benchmark == "StdMetal":
        benchmark = "MetalSet"
    (trainset, valset, testset) = maskopt(
        f"{data_path}/{benchmark}",
        sizeImage=image_size,
        ratioTrain=0.9,
        cache=False,
        targets=targets,
        return_dict=True,
    )
    if debug_samples is not None:
        trainset = Subset(trainset, list(range(debug_samples)))
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=drop_last,
        collate_fn=grpo_collate_fn,
    )
    val_loader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=False,
        collate_fn=grpo_collate_fn,
    )
    test_loader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=False,
        collate_fn=grpo_collate_fn,
    )
    gradient_accumulation_steps = calculate_gradient_accumulation_steps(
        batch_size=batch_size,
        num_generations=num_generations,
        max_batch_per_step=max_batch_per_step,
        round_up=True,
    )
    return (train_loader, val_loader, test_loader, gradient_accumulation_steps)


def loadersLitho(benchmark, image_size, batch_size, njobs, drop_last=False):
    (trainset, valset) = lithosim(
        f"{data_path}/{benchmark}", sizeImage=image_size, ratioTrain=0.9, cache=False
    )
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=False,
    )
    return (train_loader, val_loader)


def loadersAllILT(benchmark, image_size, batch_size, njobs, drop_last=False):
    basedir = f"{data_path}/{benchmark}"
    (filesGLP, filesTarget, filesMask) = filesMaskOpt(basedir)
    data = DataILT(filesTarget, filesMask, crop=False, size=image_size, cache=False)
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=drop_last,
    )
    return loader


def loadersAllLitho(benchmark, image_size, batch_size, njobs, drop_last=False):
    basedir = f"{data_path}/{benchmark}"
    (filesPixel, filesLitho, filesResist) = filesLithoSim(basedir)
    data = DataLitho(
        filesPixel, filesLitho, filesResist, crop=False, size=image_size, cache=False
    )
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=njobs,
        shuffle=False,
        drop_last=drop_last,
    )
    return loader
