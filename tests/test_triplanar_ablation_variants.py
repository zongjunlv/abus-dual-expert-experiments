"""Single-factor TriMIL controls; run in the server's existing lzj_env."""

from io import BytesIO

import pytest
import torch
from torch import nn

from models import model as model_module
from models.model import ConvNeXtTriPlanar25DMILClassifier
from models.model_factory import MODEL_CHOICES, TRIPLANAR_ABLATION_MODELS, build_model
from models.registry import ModelRegistry


class _StubConvNeXt(nn.Module):
    """Small deterministic encoder that needs no downloaded ImageNet weights."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.GELU())
        self.classifier = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 2))

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=(-2, -1)))


@pytest.fixture(autouse=True)
def _stub_backbone(monkeypatch):
    monkeypatch.setattr(model_module.tv_models, "convnext_tiny", lambda **kwargs: _StubConvNeXt())


def _build(ablation="full", **kwargs):
    options = dict(
        num_classes=2,
        pretrained=False,
        num_slices_per_plane=4,
        embed_dim=24,
        num_heads=6,
        num_layers=2,
        feedforward_dim=48,
        dropout=0.0,
        ablation=ablation,
    )
    options.update(kwargs)
    return ConvNeXtTriPlanar25DMILClassifier(**options)


@pytest.mark.parametrize("ablation", ["full", *TRIPLANAR_ABLATION_MODELS.values()])
def test_variants_forward_shape_and_strict_checkpoint_reload(ablation):
    torch.manual_seed(19)
    model = _build(ablation).eval()
    volume = torch.rand(2, 1, 4, 5, 6)
    with torch.no_grad():
        logits = model(volume)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()

    checkpoint = BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = _build(ablation).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(logits, reloaded(volume), rtol=0, atol=0)


@pytest.mark.parametrize("axis", (4, 3, 2))
def test_center_only_repeats_each_center_in_all_three_raw_channels(axis):
    model = _build("center_only")
    volume = torch.arange(120, dtype=torch.float32).reshape(1, 1, 4, 5, 6)
    instances = model._select_plane_instances(volume, axis)
    assert instances.shape[:2] == (4, 3)
    assert torch.equal(instances[:, 0], instances[:, 1])
    assert torch.equal(instances[:, 1], instances[:, 2])
    centers = model._uniform_slice_indices(volume.shape[axis], volume.device)
    for i, center in enumerate(centers):
        assert torch.equal(instances[i, 1], volume.select(axis, int(center))[0, 0])
    assert model.ablation_config["num_instances"] == 12
    assert model.ablation_config["neighbor_offsets"] == (0, 0, 0)


def test_mean_pool_replaces_cls_transformer_but_preserves_projection_norm_head():
    model = _build("mean_pool").eval()
    features = torch.randn(2, 3, 4, model.feature_dim)
    projected = model.projection(features)
    expected = model.sequence_norm(
        (
            projected + model.position_embedding.unsqueeze(1) + model.plane_embedding
        ).flatten(1, 2).mean(dim=1)
    )
    with torch.no_grad():
        torch.testing.assert_close(model._aggregate_triplanar_features(features), expected)
    assert model.cls_token is None
    assert isinstance(model.sequence_encoder, nn.Identity)
    assert model.position_embedding is not None and model.plane_embedding is not None
    assert not any(key.startswith("sequence_encoder.") for key in model.state_dict())
    assert "cls_token" not in model.state_dict()
    assert model.classifier.in_features == model.projection.out_features
    assert "constant offset" in model.ablation_config["mean_pool_embedding_effect"]


def test_mean_pool_is_permutation_invariant_in_eval_mode():
    model = _build("mean_pool").eval()
    features = torch.randn(2, 3, 4, model.feature_dim)
    permuted = features[:, [2, 0, 1]][:, :, [3, 1, 0, 2]]
    with torch.no_grad():
        torch.testing.assert_close(
            model._aggregate_triplanar_features(features),
            model._aggregate_triplanar_features(permuted),
            rtol=2e-5, atol=2e-6,
        )


def test_no_position_has_no_within_plane_order_leakage():
    model = _build("no_position").eval()
    assert model.position_embedding is None
    assert "position_embedding" not in model.state_dict()
    assert model.plane_embedding is not None
    features = torch.randn(2, 3, 4, model.feature_dim)
    permuted = features.clone()
    permuted[:, 0] = features[:, 0, [3, 1, 0, 2]]
    permuted[:, 1] = features[:, 1, [2, 0, 3, 1]]
    permuted[:, 2] = features[:, 2, [1, 3, 2, 0]]
    with torch.no_grad():
        torch.testing.assert_close(
            model._aggregate_triplanar_features(features),
            model._aggregate_triplanar_features(permuted),
            rtol=2e-5, atol=2e-6,
        )


def test_no_plane_has_no_plane_order_leakage():
    model = _build("no_plane").eval()
    assert model.plane_embedding is None
    assert "plane_embedding" not in model.state_dict()
    assert model.position_embedding is not None
    features = torch.randn(2, 3, 4, model.feature_dim)
    with torch.no_grad():
        torch.testing.assert_close(
            model._aggregate_triplanar_features(features),
            model._aggregate_triplanar_features(features[:, [2, 0, 1]]),
            rtol=2e-5, atol=2e-6,
        )


def test_z_only_encodes_only_fixed_z_and_reports_reduced_instance_budget(monkeypatch):
    model = _build("z_only").eval()
    axes = []
    original = model._encode_plane

    def record_axis(volume, axis):
        axes.append(axis)
        return original(volume, axis)

    monkeypatch.setattr(model, "_encode_plane", record_axis)
    with torch.no_grad():
        assert model(torch.rand(2, 1, 4, 5, 6)).shape == (2, 2)
    assert axes == [4]
    assert model.ablation_config["plane_order"] == ("Z",)
    assert model.ablation_config["num_instances"] == 4
    assert model.plane_embedding.shape == (1, 1, 1, 24)
    assert model.cls_token is not None
    assert len(model.sequence_encoder.layers) == 2


@pytest.mark.parametrize("ablation", ["full", *TRIPLANAR_ABLATION_MODELS.values()])
def test_warmup_and_parameter_groups_cover_remaining_parameters_once(ablation):
    model = _build(ablation)
    assert model.supports_head_warmup()
    model.set_backbone_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    groups = model.get_param_groups(1e-4, 0.05, 0.05)
    ids = [{id(parameter) for parameter in group["params"]} for group in groups]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0] | ids[1] == {id(parameter) for parameter in model.parameters()}
    assert groups[0]["lr"] == pytest.approx(5e-6)
    model.set_backbone_trainable(True)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_full_keeps_legacy_aggregation_and_state_keys_exactly():
    model = _build().eval()
    features = torch.randn(2, 3, 4, model.feature_dim)
    with torch.no_grad():
        tokens = model.projection(features)
        tokens = tokens + model.position_embedding.unsqueeze(1) + model.plane_embedding
        tokens = tokens.flatten(1, 2)
        cls = model.cls_token.expand(features.shape[0], -1, -1)
        expected = model.sequence_norm(model.sequence_encoder(torch.cat((cls, tokens), dim=1))[:, 0])
        torch.testing.assert_close(model._aggregate_triplanar_features(features), expected, rtol=0, atol=0)
    keys = model.state_dict()
    assert {"cls_token", "position_embedding", "plane_embedding"}.issubset(keys)
    assert any(key.startswith("sequence_encoder.layers.") for key in keys)
    assert not any("ablation" in key or "plane_axes" in key for key in keys)
    assert model.variant == "convnext_tiny_triplanar_25d_mil"


@pytest.mark.parametrize("name, ablation", TRIPLANAR_ABLATION_MODELS.items())
def test_named_variants_have_identical_retained_initialization_and_factory_support(name, ablation):
    torch.manual_seed(42)
    full = _build()
    torch.manual_seed(42)
    variant = _build(ablation)
    for key, value in variant.state_dict().items():
        reference = full.state_dict()[key]
        if key == "plane_embedding" and ablation == "z_only":
            reference = reference[:, :1]
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert name in MODEL_CHOICES
    assert build_model(name, 2).ablation == ablation
    registry_model = ModelRegistry.create(name, num_classes=2, pretrained=False)
    assert registry_model.ablation == ablation
    assert name in ModelRegistry.list_models()
    with pytest.raises(ValueError, match="fixes ablation"):
        ModelRegistry.create(name, num_classes=2, pretrained=False, ablation="full")


def test_rejects_unknown_ablation_before_constructing_a_backbone():
    with pytest.raises(ValueError, match="Unsupported TriMIL ablation"):
        _build("not_a_control")
