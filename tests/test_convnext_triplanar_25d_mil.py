import pytest
import torch

from models.model import ConvNeXtTriPlanar25DMILClassifier


def _build_model(**kwargs):
    defaults = dict(
        num_classes=2,
        pretrained=False,
        num_slices_per_plane=2,
        embed_dim=24,
        num_heads=6,
        num_layers=1,
        feedforward_dim=48,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return ConvNeXtTriPlanar25DMILClassifier(**defaults)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [(4, (2, 3, 2, 3)), (3, (2, 3, 2, 4)), (2, (2, 3, 3, 4))],
)
def test_plane_instance_shapes_and_neighbor_order(axis, expected):
    model = _build_model()
    volume = torch.arange(24, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    instances = model._select_plane_instances(volume, axis)
    assert instances.shape == expected
    assert torch.equal(instances[0, 1], volume.select(axis, 0)[0, 0])


def test_cross_plane_aggregation_uses_plane_identity():
    torch.manual_seed(7)
    model = _build_model().eval()
    features = torch.randn(1, 3, 2, model.feature_dim)
    with torch.no_grad():
        original = model._aggregate_triplanar_features(features)
        swapped = model._aggregate_triplanar_features(features[:, [1, 0, 2]])
    assert not torch.allclose(original, swapped)


def test_forward_is_finite_and_strictly_reloadable():
    torch.manual_seed(11)
    model = _build_model().eval()
    image = torch.rand(1, 1, 32, 32, 32)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()

    reloaded = _build_model().eval()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(logits, reloaded(image))


def test_head_warmup_and_parameter_groups_cover_model_once():
    model = _build_model()
    model.set_backbone_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    groups = model.get_param_groups(3e-5, 0.1, 1e-2)
    ids = [{id(parameter) for parameter in group["params"]} for group in groups]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0] | ids[1] == {id(parameter) for parameter in model.parameters()}
