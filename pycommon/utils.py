"""LithoBench configuration loading with an external asset-root override."""

import os
from pathlib import Path


def parseConfig(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
    results = {}
    for line in lines:
        splited = line.strip().split()
        if len(splited) >= 2:
            key = splited[0]
            value = splited[1]
            results[key] = value
    if "KernelDir" in results:
        root = os.environ.get("LITHOGRPO_LITHOBENCH_ROOT")
        if root:
            results["KernelDir"] = str(Path(root).resolve() / "kernel")
    return results
