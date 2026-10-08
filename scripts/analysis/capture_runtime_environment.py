#!/usr/bin/env python3
"""Capture the exact software and CUDA environment used for an experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import pandas
import sklearn
import torch
import torchvision


def command_output(*command: str, default: str | None = None) -> str:
    """Return command stdout, with an explicit fallback for optional metadata."""

    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if default is None:
            raise
        return default


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu_name = None
    driver_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            driver_version = command_output(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ).splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError):
            driver_version = None

    payload = {
        "git_commit": command_output(
            "git", "rev-parse", "HEAD", default="unavailable (source archive)"
        ),
        "git_status_short": command_output(
            "git", "status", "--short", default=""
        ).splitlines(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_prefix": sys.prefix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "additional_packages": {
            distribution: package_version(distribution)
            for distribution in (
                "timm",
                "monai",
                "scipy",
                "SimpleITK",
                "nibabel",
                "scikit-image",
                "tensorboard",
                "swanlab",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
