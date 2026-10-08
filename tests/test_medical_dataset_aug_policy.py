import pytest

from data.medical_dataset import Medical_Dataset


def _dataset(tmp_path, policy):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("npy_path,label\nmissing.npy,0\n")
    return Medical_Dataset(
        mode="train",
        csv_path=str(csv_path),
        spatial_aug_policy=policy,
    )


@pytest.mark.parametrize(
    ("policy", "expected_axes"),
    [
        ("full", (1, 2, 3)),
        ("hw_only", (2, 3)),
        ("slice_xy", (1, 2)),
        ("xy_flip_only", (1, 2)),
        ("none", ()),
    ],
)
def test_spatial_aug_policy_controls_flip_axes(tmp_path, policy, expected_axes):
    dataset = _dataset(tmp_path, policy)
    assert dataset._spatial_flip_axes() == expected_axes


def test_spatial_aug_policy_rejects_unknown_policy(tmp_path):
    with pytest.raises(ValueError, match="spatial_aug_policy"):
        _dataset(tmp_path, "depth_only")


def test_xy_flip_only_preserves_z_order_and_disables_rotation(tmp_path, monkeypatch):
    import numpy as np
    import torch

    dataset = _dataset(tmp_path, "xy_flip_only")
    volume = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    monkeypatch.setattr(torch, "rand", lambda *_args, **_kwargs: torch.tensor([0.0]))

    flipped = dataset._random_flip(volume.copy())
    expected = np.flip(np.flip(volume, axis=1), axis=2)
    assert np.array_equal(flipped, expected)
    assert np.array_equal(dataset._random_rotate90(flipped.copy()), flipped)


def test_slice_xy_rotation_never_changes_z_slice_membership(tmp_path, monkeypatch):
    import numpy as np
    import torch

    dataset = _dataset(tmp_path, "slice_xy")
    volume = np.broadcast_to(np.arange(4), (1, 3, 3, 4)).copy()
    monkeypatch.setattr(torch, "rand", lambda *_args, **_kwargs: torch.tensor([0.0]))
    monkeypatch.setattr(torch, "randint", lambda *_args, **_kwargs: torch.tensor([1]))

    augmented = dataset._random_rotate90(volume)
    assert np.array_equal(augmented, volume)
