import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.train import (
    resolve_log_paths,
    resolve_run_name,
    resolve_tensorboard_log_dir,
    safe_tensorboard_log,
)
from trainer.checkpoints import WeightedTeacherEnsemble
from trainer.logging import default_swanlab_mode, normalize_swanlab_mode
from trainer.runner import validate_loaded_gate_config
from trainer.trainer import Trainer
from trainer.setup import build_hard_negative_batch_sampler
from utils.batch import batch_to_device, forward_model
from utils.evaluator import evaluate_model
from utils.losses import (
    AsymmetricLoss,
    CrossEntropyWithFocalPenalty,
    CrossEntropyWithConfidencePenalty,
    HardNegativeLogitPenaltyLoss,
    PairwiseAUCMarginLoss,
)
from utils.sampling import (
    CohortBalancedBatchSampler,
    LabelBalancedBatchSampler,
    WeightedReplacementBatchSampler,
)


class ImageOnlyModel(torch.nn.Module):
    def forward(self, image):
        return image.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1).repeat(1, 2)


class MaskAwareModel(torch.nn.Module):
    supports_roi_mask = True

    def forward(self, image, roi_mask=None):
        assert roi_mask is not None
        logits = (image * roi_mask).mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return logits.repeat(1, 2)




class TupleOutputModel(torch.nn.Module):
    aux_loss_weight = 0.2

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        feature = image.mean(dim=(1, 2, 3, 4)) * self.scale
        logits = torch.stack((-feature, feature), dim=1)
        view_logits = logits.unsqueeze(1).repeat(1, 2, 1)
        return logits, view_logits


class GateTupleOutputModel(torch.nn.Module):
    gate_temperature = 1.0

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.gate_logits = torch.nn.Parameter(torch.tensor([[1.0], [-1.0]]))

    def forward(self, image):
        feature = image.mean(dim=(1, 2, 3, 4)) * self.scale
        logits = torch.stack((-feature, feature), dim=1)
        source_logits = torch.stack((logits, -logits), dim=1)
        gate_logits = self.gate_logits.unsqueeze(0).expand(image.shape[0], -1, -1)
        return logits, source_logits, gate_logits


class BatchNormGateModel(GateTupleOutputModel):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm3d(1)

    def forward(self, image):
        return super().forward(self.bn(image))


class BufferedGateConfigModel(torch.nn.Module):
    residual_reliability_gate = True

    def gate_hyperparameters(self):
        return 0.25, 2.0

class MaskBatchDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        label = torch.tensor(index % 2, dtype=torch.long)
        image = torch.ones(1, 4, 4, 4) * (index + 1)
        mask = torch.ones(1, 4, 4, 4)
        return {"image": image, "mask": mask, "label": label}


class DistinctMaskBatchDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        label = torch.tensor(index % 2, dtype=torch.long)
        image = torch.ones(1, 4, 4, 4) * (1 + 2 * index)
        mask = torch.ones(1, 4, 4, 4) * (2 + 2 * index)
        return {"image": image, "mask": mask, "label": label}


class MaskAwareLinear(torch.nn.Module):
    supports_roi_mask = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image, roi_mask=None):
        assert roi_mask is not None
        feature = (image * roi_mask).mean(dim=(1, 2, 3, 4)) * self.scale
        return torch.stack((-feature, feature), dim=1)


class RecordingMaskAwareLinear(MaskAwareLinear):
    def forward(self, image, roi_mask=None):
        self.last_image = image.detach().clone()
        self.last_mask = roi_mask.detach().clone()
        return super().forward(image, roi_mask=roi_mask)


class FixedLogitTeacher(torch.nn.Module):
    supports_roi_mask = True

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, image, roi_mask=None):
        assert roi_mask is not None
        return self.logits.unsqueeze(0).repeat(image.shape[0], 1)


class FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flush_count += 1


def test_batch_to_device_accepts_legacy_tuple():
    image = torch.ones(2, 1, 4, 4, 4)
    label = torch.tensor([0, 1])
    moved = batch_to_device((image, label), torch.device("cpu"))
    assert set(moved) == {"image", "label", "mask"}
    assert moved["image"].shape == (2, 1, 4, 4, 4)
    assert moved["label"].tolist() == [0, 1]
    assert moved["mask"] is None


def test_batch_to_device_accepts_dict_with_mask():
    batch = {
        "image": torch.ones(2, 1, 4, 4, 4),
        "label": torch.tensor([1, 0]),
        "mask": torch.ones(2, 1, 4, 4, 4),
    }
    moved = batch_to_device(batch, torch.device("cpu"))
    assert moved["mask"].shape == (2, 1, 4, 4, 4)
    assert moved["label"].tolist() == [1, 0]


def test_forward_model_keeps_legacy_model_compatible():
    image = torch.ones(2, 1, 4, 4, 4)
    output = forward_model(ImageOnlyModel(), image=image, mask=torch.ones_like(image))
    assert output.shape == (2, 2)


def test_forward_model_passes_mask_to_mask_aware_model():
    image = torch.ones(2, 1, 4, 4, 4)
    mask = torch.ones_like(image)
    output = forward_model(MaskAwareModel(), image=image, mask=mask)
    assert output.shape == (2, 2)




def test_trainer_validate_uses_primary_logits_from_tuple_output():
    model = TupleOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))
    dataloader = DataLoader(MaskBatchDataset(), batch_size=2)

    loss = trainer.validate(dataloader)

    assert isinstance(loss, float)

def test_trainer_accepts_dict_batches_with_masks():
    model = MaskAwareLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))
    dataloader = DataLoader(MaskBatchDataset(), batch_size=2)

    loss = trainer.train(dataloader)

    assert isinstance(loss, float)


def test_trainer_mixup_combines_images_masks_and_label_losses():
    model = RecordingMaskAwareLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        torch.device("cpu"),
        mixup_alpha=0.4,
        mixup_prob=1.0,
    )
    trainer._sample_mixup_params = lambda image: (
        0.25,
        torch.tensor([1, 0], device=image.device),
    )
    dataloader = DataLoader(DistinctMaskBatchDataset(), batch_size=2)

    loss = trainer.train(dataloader)

    assert isinstance(loss, float)
    assert torch.allclose(
        model.last_image[:, 0, 0, 0, 0],
        torch.tensor([2.5, 1.5]),
    )
    assert torch.allclose(
        model.last_mask[:, 0, 0, 0, 0],
        torch.tensor([3.5, 2.5]),
    )


def test_trainer_mixup_supervised_loss_is_weighted_pair_loss():
    model = MaskAwareLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.float32)
    label_a = torch.tensor([0, 1], dtype=torch.long)
    label_b = torch.tensor([1, 0], dtype=torch.long)

    loss = trainer._supervised_loss(logits, label_a, label_b, lam=0.25)

    expected = 0.25 * criterion(logits, label_a) + 0.75 * criterion(logits, label_b)
    assert torch.allclose(loss, expected)




def test_trainer_auxiliary_supervised_loss_repeats_labels_for_view_logits():
    model = MaskAwareLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))
    logits = torch.tensor([[1.0, -0.2], [-0.3, 0.9]], dtype=torch.float32)
    view_logits = torch.tensor(
        [
            [[1.0, -0.2], [0.8, -0.1], [0.9, -0.3]],
            [[-0.3, 0.9], [-0.1, 0.7], [-0.2, 0.8]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    loss = trainer._auxiliary_supervised_loss(view_logits, logits, labels)

    expected = criterion(
        view_logits.reshape(6, 2),
        labels.repeat_interleave(3),
    )
    assert torch.allclose(loss, expected)


def test_trainer_view_consistency_loss_is_zero_for_identical_view_logits():
    logits = torch.tensor([[1.0, -0.2], [-0.3, 0.9]], dtype=torch.float32)
    view_logits = logits.unsqueeze(1).repeat(1, 3, 1)

    loss = Trainer._view_consistency_loss(logits, view_logits)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_oracle_gate_target_prefers_source_with_lower_detached_ce():
    model = GateTupleOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        gate_loss_weight=1.0,
        gate_oracle_temperature=0.25,
    )
    source_logits = torch.tensor(
        [[[3.0, -1.0], [-1.0, 3.0]], [[-1.0, 3.0], [3.0, -1.0]]],
        requires_grad=True,
    )
    gate_logits = torch.zeros(2, 2, 1, requires_grad=True)
    labels = torch.tensor([0, 1])

    loss, oracle_weights = trainer._gate_supervision_loss(
        gate_logits,
        source_logits,
        labels,
    )

    assert oracle_weights.argmax(dim=1).tolist() == [0, 0]
    assert not oracle_weights.requires_grad
    loss.backward()
    assert gate_logits.grad is not None
    assert source_logits.grad is None


def test_oracle_gate_target_supports_mixup_labels():
    model = GateTupleOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    source_logits = torch.tensor([[[3.0, -1.0], [-1.0, 3.0]]])
    gate_logits = torch.zeros(1, 2, 1)

    _, mostly_class_zero = trainer._gate_supervision_loss(
        gate_logits,
        source_logits,
        torch.tensor([0]),
        torch.tensor([1]),
        lam=0.75,
    )
    _, mostly_class_one = trainer._gate_supervision_loss(
        gate_logits,
        source_logits,
        torch.tensor([0]),
        torch.tensor([1]),
        lam=0.25,
    )

    assert mostly_class_zero.argmax(dim=1).item() == 0
    assert mostly_class_one.argmax(dim=1).item() == 1


def test_oracle_gate_target_respects_residual_floor_and_normalization():
    model = GateTupleOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        gate_alpha=0.5,
        gate_oracle_temperature=1e-6,
    )
    source_logits = torch.tensor(
        [[[1000.0, -1000.0], [-1000.0, 1000.0]]],
        dtype=torch.float16,
    )
    gate_logits = torch.tensor(
        [[[1000.0], [-1000.0]]],
        dtype=torch.float16,
        requires_grad=True,
    )

    loss, oracle_weights = trainer._gate_supervision_loss(
        gate_logits,
        source_logits,
        torch.tensor([0]),
    )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(oracle_weights).all()
    assert torch.allclose(oracle_weights.sum(dim=1), torch.ones(1))
    assert torch.all(oracle_weights >= 0.25)


@pytest.mark.parametrize(
    "gate_shape",
    [(2, 2), (2, 3, 1), (2, 2, 2)],
)
def test_oracle_gate_loss_rejects_invalid_gate_logit_shapes(gate_shape):
    model = GateTupleOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"))

    with pytest.raises(ValueError, match="gate_logits"):
        trainer._gate_supervision_loss(
            torch.zeros(gate_shape),
            torch.zeros(2, 2, 2),
            torch.tensor([0, 1]),
        )


def test_loaded_gate_config_must_match_cli_and_trainer_uses_model_values():
    model = BufferedGateConfigModel()

    validate_loaded_gate_config(model, gate_alpha=0.25, gate_temperature=2.0)
    with pytest.raises(ValueError, match="gate alpha"):
        validate_loaded_gate_config(model, gate_alpha=0.5, gate_temperature=2.0)
    with pytest.raises(ValueError, match="gate temperature"):
        validate_loaded_gate_config(model, gate_alpha=0.25, gate_temperature=1.0)

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model.register_parameter("dummy", parameter)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        gate_alpha=0.9,
        gate_temperature=0.1,
    )
    assert trainer.gate_alpha == 0.25
    assert trainer.gate_temperature == 2.0


def test_trainer_keeps_batchnorm_in_eval_and_optionally_freezes_affine():
    model = BatchNormGateModel()
    Trainer.configure_batchnorm(model, freeze_affine=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        freeze_batchnorm=True,
        freeze_batchnorm_affine=True,
    )
    dataloader = DataLoader(MaskBatchDataset(), batch_size=2)

    trainer.train(dataloader)

    assert not model.bn.training
    assert not model.bn.weight.requires_grad
    assert not model.bn.bias.requires_grad

def test_trainer_roi_consistency_dropout_mask_zeros_all_samples():
    model = MaskAwareLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        torch.device("cpu"),
        roi_consistency_weight=0.1,
        roi_consistency_dropout_prob=1.0,
    )
    mask = torch.ones(2, 1, 4, 4, 4)

    consistency_mask = trainer._make_consistency_mask(mask)

    assert torch.count_nonzero(consistency_mask).item() == 0


def test_trainer_roi_consistency_loss_is_zero_for_identical_logits():
    logits = torch.tensor([[1.0, -0.2], [-0.3, 0.8]], dtype=torch.float32)

    loss = Trainer._consistency_loss(logits, logits.clone())

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_ce_focal_penalty_zero_weight_matches_smoothed_weighted_ce():
    logits = torch.tensor([[1.8, -0.7], [-0.2, 1.1], [0.4, 0.6]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([0.75, 1.25], dtype=torch.float32)
    criterion = CrossEntropyWithFocalPenalty(
        weight=weights,
        label_smoothing=0.1,
        focal_weight=0.0,
        focal_gamma=1.5,
    )

    loss = criterion(logits, targets)

    expected = F.cross_entropy(
        logits,
        targets,
        weight=weights,
        label_smoothing=0.1,
    )
    assert torch.allclose(loss, expected)


def test_ce_focal_penalty_adds_hard_example_term():
    logits = torch.tensor([[0.2, 0.1], [0.0, 0.3], [1.2, -0.4]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    base = CrossEntropyWithFocalPenalty(focal_weight=0.0, focal_gamma=1.0)
    focal = CrossEntropyWithFocalPenalty(focal_weight=0.25, focal_gamma=1.0)

    assert focal(logits, targets) > base(logits, targets)


def test_weighted_teacher_ensemble_blends_probabilities_and_passes_masks():
    teacher_a = FixedLogitTeacher([0.0, 2.0])
    teacher_b = FixedLogitTeacher([1.5, -0.5])
    ensemble = WeightedTeacherEnsemble(
        [(teacher_a, 0.25), (teacher_b, 0.75)],
        probability_blend=True,
    )
    image = torch.ones(2, 1, 4, 4, 4)
    mask = torch.ones_like(image)

    logits = forward_model(ensemble, image=image, mask=mask)

    expected_probs = (
        0.25 * torch.softmax(teacher_a.logits, dim=0)
        + 0.75 * torch.softmax(teacher_b.logits, dim=0)
    )
    assert logits.shape == (2, 2)
    assert torch.allclose(torch.softmax(logits, dim=1), expected_probs.repeat(2, 1))


def test_evaluator_accepts_dict_batches_with_masks():
    model = MaskAwareLinear()
    dataloader = DataLoader(MaskBatchDataset(), batch_size=2)

    metrics = evaluate_model(
        model,
        dataloader,
        torch.device("cpu"),
        verbose=False,
        show_details=False,
        return_outputs=True,
    )

    assert len(metrics) == 11


def test_cohort_balanced_sampler_yields_one_sample_per_cohort_per_batch():
    cohorts = ["TDSC", "TDSC", "LSH", "LSH", "LSH"]
    sampler = CohortBalancedBatchSampler(
        cohorts=cohorts,
        batch_size=2,
        num_batches=6,
        seed=3407,
    )

    batches = list(iter(sampler))

    assert len(batches) == 6
    for batch in batches:
        assert len(batch) == 2
        assert {cohorts[index] for index in batch} == {"TDSC", "LSH"}


def test_cohort_balanced_sampler_requires_batch_size_divisible_by_cohort_count():
    cohorts = ["TDSC", "TDSC", "LSH", "LSH"]

    with pytest.raises(ValueError, match="divisible"):
        CohortBalancedBatchSampler(cohorts=cohorts, batch_size=3)


def test_cohort_balanced_sampler_uses_baseline_step_count_by_default():
    cohorts = ["TDSC", "TDSC", "LSH", "LSH", "LSH"]
    sampler = CohortBalancedBatchSampler(cohorts=cohorts, batch_size=2)

    assert len(sampler) == len(cohorts) // 2


def test_label_balanced_sampler_yields_one_sample_per_label_per_batch():
    labels = [0, 0, 1, 1, 1]
    sampler = LabelBalancedBatchSampler(
        labels=labels,
        batch_size=2,
        num_batches=6,
        seed=3407,
    )

    batches = list(iter(sampler))

    assert len(batches) == 6
    for batch in batches:
        assert len(batch) == 2
        assert {labels[index] for index in batch} == {0, 1}


def test_weighted_replacement_sampler_respects_zero_weights():
    sampler = WeightedReplacementBatchSampler(
        sample_weights=[0.0, 1.0, 0.0],
        batch_size=2,
        num_batches=3,
        seed=3407,
    )

    batches = list(iter(sampler))

    assert batches == [[1, 1], [1, 1], [1, 1]]


def test_hard_negative_sampler_boosts_mined_train_cases(tmp_path):
    class DummyDataset:
        def __init__(self):
            self.data = pd.DataFrame(
                {
                    "label": [0, 0, 1],
                    "data_path": [
                        "dataset/train/imagesTr_origin/DATA_001.nii.gz",
                        "dataset/train/imagesTr_origin/DATA_002.nii.gz",
                        "dataset/train/imagesTr_origin/DATA_003.nii.gz",
                    ],
                    "npy_path": [
                        "dataset/train/imagesTr_cache/0001.npy",
                        "dataset/train/imagesTr_cache/0002.npy",
                        "dataset/train/imagesTr_cache/0003.npy",
                    ],
                }
            )

        def __len__(self):
            return len(self.data)

    mining_csv = tmp_path / "hard_negative_cases.csv"
    pd.DataFrame(
        {
            "case_id": ["DATA_002.nii"],
            "is_hard_negative": [1],
        }
    ).to_csv(mining_csv, index=False)
    args = SimpleNamespace(
        hard_negative_mining_csv=str(mining_csv),
        hard_negative_sampling_factor=5.0,
        hard_negative_case_column="case_id",
        hard_negative_flag_column="is_hard_negative",
        batch_size=1,
    )

    sampler = build_hard_negative_batch_sampler(args, DummyDataset(), seed=3407)

    assert sampler.hard_negative_indices == [1]
    assert sampler.sample_weights.tolist() == [1.0, 5.0, 1.0]


def test_asymmetric_loss_uses_class_specific_focusing_for_each_sample():
    inputs = torch.tensor(
        [
            [2.0, -0.5],
            [0.1, 1.4],
            [1.2, 0.3],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    criterion = AsymmetricLoss(gamma_pos=0.5, gamma_neg=2.0, reduction="none")

    loss = criterion(inputs, targets)

    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    gammas = torch.where(
        targets == 1,
        torch.full_like(pt, 0.5),
        torch.full_like(pt, 2.0),
    )
    expected = torch.pow(1.0 - pt, gammas) * ce_loss
    assert torch.allclose(loss, expected)


def test_confidence_penalty_loss_preserves_weighted_smoothed_ce_when_disabled():
    inputs = torch.tensor(
        [
            [1.5, -0.2],
            [-0.1, 0.7],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([0.8, 1.2], dtype=torch.float32)
    criterion = CrossEntropyWithConfidencePenalty(
        weight=weights,
        label_smoothing=0.05,
        penalty_weight=0.0,
    )

    loss = criterion(inputs, targets)

    expected = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05,
    )(inputs, targets)
    assert torch.allclose(loss, expected)


def test_confidence_penalty_is_larger_for_overconfident_logits():
    targets = torch.tensor([0, 1], dtype=torch.long)
    criterion = CrossEntropyWithConfidencePenalty(penalty_weight=1.0)
    overconfident_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    uniform_logits = torch.zeros_like(overconfident_logits)

    overconfident_gap = criterion(overconfident_logits, targets) - F.cross_entropy(
        overconfident_logits,
        targets,
    )
    uniform_gap = criterion(uniform_logits, targets) - F.cross_entropy(uniform_logits, targets)

    assert overconfident_gap > uniform_gap


def test_hard_negative_penalty_is_noop_when_disabled():
    inputs = torch.tensor(
        [
            [0.2, 1.0],
            [1.2, -0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    base_loss = torch.nn.CrossEntropyLoss()
    criterion = HardNegativeLogitPenaltyLoss(base_loss, penalty_weight=0.0)

    assert torch.allclose(criterion(inputs, targets), base_loss(inputs, targets))


def test_hard_negative_penalty_targets_negative_positive_margins():
    inputs = torch.tensor(
        [
            [0.0, 0.6],
            [0.8, 0.1],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    base_loss = torch.nn.CrossEntropyLoss()
    criterion = HardNegativeLogitPenaltyLoss(
        base_loss,
        penalty_weight=0.5,
        margin=0.0,
    )

    loss = criterion(inputs, targets)
    expected_penalty = torch.tensor([0.6 ** 2, 0.0]).mean() * 0.5
    expected = base_loss(inputs, targets) + expected_penalty

    assert torch.allclose(loss, expected)


def test_pairwise_auc_margin_penalty_is_noop_without_class_pairs():
    inputs = torch.tensor(
        [
            [0.2, 1.0],
            [0.1, 0.5],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 1], dtype=torch.long)
    base_loss = torch.nn.CrossEntropyLoss()
    criterion = PairwiseAUCMarginLoss(base_loss, penalty_weight=0.5)

    assert torch.allclose(criterion(inputs, targets), base_loss(inputs, targets))


def test_pairwise_auc_margin_penalty_scores_positive_above_negative():
    inputs = torch.tensor(
        [
            [0.0, 0.8],
            [0.8, 0.0],
            [0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 0, 0], dtype=torch.long)
    base_loss = torch.nn.CrossEntropyLoss()
    criterion = PairwiseAUCMarginLoss(
        base_loss,
        penalty_weight=0.25,
        margin=0.1,
    )

    loss = criterion(inputs, targets)
    margins = inputs[:, 1] - inputs[:, 0]
    expected_penalty = F.softplus(
        0.1 - (margins[0].unsqueeze(0) - margins[1:])
    ).mean() * 0.25
    expected = base_loss(inputs, targets) + expected_penalty

    assert torch.allclose(loss, expected)


def test_tensorboard_run_names_match_existing_log_names(tmp_path):
    assert resolve_run_name("resnet34", 3407, multi_seed=False) == "resnet34"
    assert resolve_run_name("resnet34", 3407, multi_seed=True) == "resnet34_seed3407"
    assert (
        resolve_run_name("resnet34", 3407, multi_seed=True, suffix="fold2")
        == "resnet34_seed3407_fold2"
    )

    _, csv_path, plot_path = resolve_log_paths(
        tmp_path / "logs",
        "resnet34",
        3407,
        multi_seed=True,
        suffix="fold2",
    )
    tensorboard_dir = resolve_tensorboard_log_dir(
        None,
        tmp_path / "logs",
        "resnet34",
        3407,
        multi_seed=True,
        suffix="fold2",
    )

    assert csv_path.name == "train_log_resnet34_seed3407_fold2.csv"
    assert plot_path.name == "train_curves_resnet34_seed3407_fold2.png"
    assert tensorboard_dir == tmp_path / "logs" / "tensorboard" / "resnet34_seed3407_fold2"


def test_tensorboard_logging_skips_non_finite_scalars():
    writer = FakeSummaryWriter()
    safe_tensorboard_log(
        writer,
        {
            "epoch": 3,
            "train_loss": 0.4,
            "val_loss": math.nan,
            "auc": 0.82,
            "selection_score": math.inf,
            "constraints_met": 1,
        },
    )

    assert ("Loss/train", 0.4, 3) in writer.scalars
    assert ("Metrics/auc", 0.82, 3) in writer.scalars
    assert ("Selection/constraints_met", 1.0, 3) in writer.scalars
    assert not any(tag == "Loss/val" for tag, _, _ in writer.scalars)
    assert not any(tag == "Selection/score" for tag, _, _ in writer.scalars)
    assert writer.flush_count == 1


def test_swanlab_mode_uses_shared_server_safe_defaults(monkeypatch):
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    assert default_swanlab_mode() == "disabled"

    monkeypatch.setenv("SWANLAB_MODE", "online")
    assert default_swanlab_mode() == "cloud"
    assert normalize_swanlab_mode("online") == "cloud"
