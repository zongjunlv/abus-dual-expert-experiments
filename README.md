# ABUS dual-expert experiments

This repository contains the experiment code for a volumetric R3D-18 expert,
a tri-planar ConvNeXt-Tiny 2.5D attention-MIL expert, and their fixed
equal-logit fusion for tumor classification in automated breast ultrasound.
It also includes the preprocessing, repeated-evaluation, ablation, and
statistical-analysis programs used in the study.

No ABUS images, labels, patient-level records, split membership files,
predictions, checkpoints, logs, or manuscript files are included. Obtain and
use each dataset under its own access terms. In particular, publishing this
code does not grant access to the private cohort.

## Environment

The experiments were run with Python 3.10, PyTorch 1.13.0 (CUDA 11.7),
torchvision 0.14.0, and MONAI 1.3.0. Use a compatible CUDA host and install
the matching PyTorch/torchvision wheels before installing the remaining
dependencies from `requirements.txt`. The model backbones initialize from
standard torchvision weights, which must be available locally or downloaded
through torchvision on the first run.

## Data contract

The training and evaluation CSVs require `npy_path` and binary `label`
columns (`0` benign, `1` malignant). A `case_id` column is recommended for
paired analyses. Cached arrays are float32, min-max scaled to `[0,1]`, with
layout `[C,X,Y,Z]` and target shape `[1,128,128,64]`. The model uses images,
not segmentation masks; TDSC masks are used only for offline cropping.

Prepare the public TDSC image cache with:

```bash
python utils/pre_cache_binary_tdsc.py \
  --root /path/to/TDSC --target image \
  --target-shape 128 128 64 --workers 1
```

The study's exact fold membership and raw data are intentionally not in this
code-only repository. `scripts/experiments/materialize_tdsc_public_exact_splits.py`
can materialize the exact roles when an authorized membership CSV is supplied.
For a new repeated-CV protocol, use
`scripts/experiments/build_repeated_cv_protocol.py` with your own case CSV.
A newly generated protocol is not the same set of paper folds.

## Train and evaluate

Run `python scripts/train.py --help` and `python scripts/test.py --help` for all
options. The primary model names are `r3d` and
`convnext_triplanar_25d_mil`; the five component controls are selected using
the corresponding `convnext_triplanar_25d_mil_*` names. R3D training uses
`--spatial-aug-policy xy_flip_only` and tri-planar MIL uses
`--spatial-aug-policy slice_xy`. Both preserve Z-order.

```bash
python scripts/train.py \
  --train-csv /path/to/train.csv --val-csv /path/to/selection_val.csv \
  --model r3d --seed 3407 --epochs 80 --batch-size 28 \
  --lr 1e-4 --backbone-lr-scale 0.05 \
  --warmup-epochs 5 --freeze-backbone-epochs 5 --patience 10 \
  --target-shape 128,128,64 --spatial-aug-policy xy_flip_only \
  --disable-selection-threshold-search \
  --save-path outputs/checkpoints/r3d/best_model.pth \
  --log-dir outputs/logs/r3d --swanlab-mode disabled
```

The full public TDSC clean-room queue is
`scripts/remote/run_aim_tdsc_public_cleanroom_queue.sh`. It requires an
authorized exact membership file, materialized split CSVs, a CUDA GPU,
and a configured Conda environment. Check the GPU for active compute jobs
before running it. The queue writes generated artifacts under `outputs/`,
which must remain local and are excluded from version control.

The `scripts/analysis/` programs compute paired summaries from completed
predictions. The source code alone does not reproduce the paper's numerical
results without the underlying datasets and recorded split assignments.
