import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset


class ScaleIntensity:
    """Lightweight replacement for MONAI ScaleIntensity used by cached arrays."""

    def __init__(self, minv=0.0, maxv=1.0, eps=1e-8):
        self.minv = float(minv)
        self.maxv = float(maxv)
        self.eps = float(eps)

    def __call__(self, arr):
        arr = arr.astype(np.float32, copy=False)
        src_min = float(arr.min())
        src_max = float(arr.max())
        if src_max - src_min <= self.eps:
            return np.full_like(arr, self.minv, dtype=np.float32)
        arr = (arr - src_min) / (src_max - src_min)
        return arr * (self.maxv - self.minv) + self.minv


class Medical_Dataset(Dataset):
    def __init__(
        self,
        mode,
        csv_path,
        roi_size=(256, 256, 128),
        margin=12,
        return_mask=False,
        roi_mask_dropout_prob=0.0,
        roi_mask_jitter_voxels=0,
        target_shape=None,
        spatial_aug_policy="full",
    ):
        self.mode = mode
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.return_mask = bool(return_mask)
        self.roi_mask_dropout_prob = float(roi_mask_dropout_prob)
        self.roi_mask_jitter_voxels = int(roi_mask_jitter_voxels)
        self.target_shape = self._normalize_target_shape(target_shape)
        self.spatial_aug_policy = str(spatial_aug_policy)
        # Avoid MONAI Compose RNG initialization here; current pipeline only
        # uses deterministic intensity scaling, so a single transform is enough.
        self.intensity_scale = ScaleIntensity(minv=0.0, maxv=1.0)
        if self.roi_mask_dropout_prob < 0.0 or self.roi_mask_dropout_prob > 1.0:
            raise ValueError("roi_mask_dropout_prob must be in [0, 1].")
        if self.roi_mask_jitter_voxels < 0:
            raise ValueError("roi_mask_jitter_voxels must be non-negative.")
        if self.spatial_aug_policy not in {
            "full",
            "hw_only",
            "slice_xy",
            "xy_flip_only",
            "none",
        }:
            raise ValueError(
                "spatial_aug_policy must be one of: full, hw_only, slice_xy, "
                "xy_flip_only, none."
            )

    def __len__(self):
        return len(self.data)

    def _normalize_target_shape(self, target_shape):
        if target_shape is None:
            return None
        if isinstance(target_shape, str):
            parts = target_shape.replace("x", ",").split(",")
            target_shape = tuple(int(item.strip()) for item in parts if item.strip())
        else:
            target_shape = tuple(int(item) for item in target_shape)
        if len(target_shape) != 3 or any(item <= 0 for item in target_shape):
            raise ValueError("target_shape must contain three positive spatial dimensions.")
        return target_shape

    def _ensure_channel_first(self, arr):
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr

    def _resize_spatial(self, arr, mode):
        if self.target_shape is None or tuple(arr.shape[-3:]) == self.target_shape:
            return arr.astype(np.float32, copy=False)
        tensor = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)
        if mode == "nearest":
            resized = F.interpolate(tensor, size=self.target_shape, mode=mode)
        else:
            resized = F.interpolate(
                tensor,
                size=self.target_shape,
                mode=mode,
                align_corners=False,
            )
        return resized.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _resolve_mask_path(self, row):
        for column in ("mask_npy_path", "mask_path", "roi_path"):
            if column in row and isinstance(row[column], str) and row[column].strip():
                return row[column]
        if self.return_mask:
            raise ValueError(
                "return_mask=True requires one of mask_npy_path, mask_path, or roi_path in the CSV."
            )
        return None

    def _load_mask(self, row):
        mask_path = self._resolve_mask_path(row)
        if mask_path is None:
            return None
        mask = np.load(mask_path)
        mask = self._ensure_channel_first(mask)
        mask = self._resize_spatial(mask, mode="nearest")
        return (mask > 0).astype(np.float32)

    def _spatial_flip_axes(self):
        if self.spatial_aug_policy == "full":
            return (1, 2, 3)
        if self.spatial_aug_policy == "hw_only":
            # Legacy name retained for old experiment reproducibility. Cached
            # arrays are [C, X, Y, Z], so this policy flips Y and Z and MUST NOT
            # be used when the Z acquisition order must be preserved.
            return (2, 3)
        if self.spatial_aug_policy in {"slice_xy", "xy_flip_only"}:
            return (1, 2)
        return ()

    def _random_flip(self, arr):
        # Cached volume layout is [C, X, Y, Z]. The final axis preserves the
        # ordered sweep/depth direction used by the 2.5D experts.
        for axis in self._spatial_flip_axes():
            if torch.rand(1).item() < 0.5:
                arr = np.flip(arr, axis=axis).copy()
        return arr

    def _random_rotate90(self, arr):
        if self.spatial_aug_policy not in {"full", "slice_xy"}:
            return arr
        if torch.rand(1).item() < 0.5:
            k = int(torch.randint(1, 4, (1,)).item())
            # Keep tensor shape unchanged by rotating only across equal-sized axes.
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        return arr

    def _random_flip_pair(self, image, mask):
        for axis in self._spatial_flip_axes():
            if torch.rand(1).item() < 0.5:
                image = np.flip(image, axis=axis).copy()
                if mask is not None:
                    mask = np.flip(mask, axis=axis).copy()
        return image, mask

    def _random_rotate90_pair(self, image, mask):
        if self.spatial_aug_policy not in {"full", "slice_xy"}:
            return image, mask
        if torch.rand(1).item() < 0.5:
            k = int(torch.randint(1, 4, (1,)).item())
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask

    def _random_gaussian_noise(self, arr, std=0.03):
        if torch.rand(1).item() < 0.3:
            noise = torch.randn(arr.shape, dtype=torch.float32).numpy() * std
            arr = arr + noise
        return arr

    def _random_intensity_scale_shift(self, arr):
        if torch.rand(1).item() < 0.5:
            scale = float(torch.empty(1).uniform_(0.9, 1.1).item())
            shift = float(torch.empty(1).uniform_(-0.08, 0.08).item())
            arr = arr * scale + shift
        return arr

    def _random_gamma(self, arr):
        if torch.rand(1).item() < 0.3:
            gamma = float(torch.empty(1).uniform_(0.85, 1.15).item())
            arr = np.power(np.clip(arr, 0.0, 1.0), gamma, dtype=np.float32)
        return arr

    def _random_coarse_dropout(self, arr):
        if torch.rand(1).item() >= 0.25:
            return arr
        _, size_x, size_y, size_z = arr.shape
        fill_value = float(arr.mean())
        num_regions = int(torch.randint(1, 3, (1,)).item())
        for _ in range(num_regions):
            crop_x = max(4, int(size_x * float(torch.empty(1).uniform_(0.08, 0.18).item())))
            crop_y = max(8, int(size_y * float(torch.empty(1).uniform_(0.08, 0.18).item())))
            crop_z = max(8, int(size_z * float(torch.empty(1).uniform_(0.08, 0.18).item())))
            start_x = int(torch.randint(0, max(1, size_x - crop_x + 1), (1,)).item())
            start_y = int(torch.randint(0, max(1, size_y - crop_y + 1), (1,)).item())
            start_z = int(torch.randint(0, max(1, size_z - crop_z + 1), (1,)).item())
            arr[
                :,
                start_x:start_x + crop_x,
                start_y:start_y + crop_y,
                start_z:start_z + crop_z,
            ] = fill_value
        return arr

    def _apply_train_aug(self, arr):
        arr = self._random_flip(arr)
        arr = self._random_rotate90(arr)
        arr = self._random_gaussian_noise(arr)
        arr = self._random_intensity_scale_shift(arr)
        arr = self._random_gamma(arr)
        arr = self._random_coarse_dropout(arr)
        return np.clip(arr, 0.0, 1.0)

    def _shift_mask(self, mask, shifts):
        shifted = np.zeros_like(mask)
        src_slices = []
        dst_slices = []
        for axis, shift in enumerate(shifts, start=1):
            size = mask.shape[axis]
            if shift >= 0:
                src_start, src_end = 0, size - shift
                dst_start, dst_end = shift, size
            else:
                src_start, src_end = -shift, size
                dst_start, dst_end = 0, size + shift
            if src_start >= src_end:
                return shifted
            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))
        shifted[(slice(None), *dst_slices)] = mask[(slice(None), *src_slices)]
        return shifted

    def _random_roi_mask_dropout_jitter(self, mask):
        if mask is None:
            return None
        if self.roi_mask_dropout_prob > 0.0:
            if torch.rand(1).item() < self.roi_mask_dropout_prob:
                return np.zeros_like(mask, dtype=np.float32)
        if self.roi_mask_jitter_voxels > 0:
            max_shift = self.roi_mask_jitter_voxels
            shifts = [
                int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                for _ in range(3)
            ]
            if any(shift != 0 for shift in shifts):
                mask = self._shift_mask(mask, shifts)
        return (mask > 0).astype(np.float32)

    def _apply_train_aug_pair(self, image, mask):
        image, mask = self._random_flip_pair(image, mask)
        image, mask = self._random_rotate90_pair(image, mask)
        mask = self._random_roi_mask_dropout_jitter(mask)
        image = self._random_gaussian_noise(image)
        image = self._random_intensity_scale_shift(image)
        image = self._random_gamma(image)
        image = self._random_coarse_dropout(image)
        return np.clip(image, 0.0, 1.0), mask

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = np.load(row["npy_path"])
        img = self._ensure_channel_first(img)
        img = self.intensity_scale(img)
        img = self._resize_spatial(img, mode="trilinear")

        mask = self._load_mask(row)
        if self.mode == "train":
            if mask is None:
                img = self._apply_train_aug(img)
            else:
                img, mask = self._apply_train_aug_pair(img, mask)

        img = torch.as_tensor(img).float()
        label = torch.as_tensor(int(row["label"]), dtype=torch.long)

        if not self.return_mask:
            return img, label

        if mask is None:
            raise ValueError("Mask loading returned None while return_mask=True.")

        return {
            "image": img,
            "mask": torch.as_tensor(mask).float(),
            "label": label,
        }
