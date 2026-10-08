from types import SimpleNamespace

import pytest
import torch

from models.model_factory import (
    BENCHMARK_MODELS,
    DB_VCRF_ABLATION_MODELS,
    DB_VCRF_MODEL,
    MODEL_CHOICES,
    build_model,
)
from tester.cli import validate_args as validate_test_args
from trainer.cli import describe_model, model_requires_roi_mask, validate_args as validate_train_args


EXPECTED_MODELS = {
    DB_VCRF_MODEL,
    *DB_VCRF_ABLATION_MODELS,
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "resnext50",
    "resnext101",
    "vit",
    "swin3d",
    "multiscale_gat",
    "r3d",
    "r2plus1d",
    "s3d",
    "convnext_slice_mil",
    "convnext_25d_ordered_mil",
    "convnext_triplanar_25d_mil",
    "convnext_triplanar_25d_mil_center_only",
    "convnext_triplanar_25d_mil_mean_pool",
    "convnext_triplanar_25d_mil_no_position",
    "convnext_triplanar_25d_mil_no_plane",
    "convnext_triplanar_25d_mil_z_only",
    "primus_m",
}


def test_model_choices_only_include_db_vcrf_and_benchmarks():
    assert set(MODEL_CHOICES) == EXPECTED_MODELS
    assert DB_VCRF_ABLATION_MODELS == [
        "dual_backbone_vcrf_single_view",
        "dual_backbone_vcrf_single_view_t0",
        "dual_backbone_vcrf_single_view_td",
        "dual_backbone_vcrf_single_view_th",
        "dual_backbone_vcrf_single_view_tw",
        "dual_backbone_vcrf_t0_oracle_residual_gate",
        "dual_backbone_vcrf_no_depth_flip",
        "dual_backbone_vcrf_audit_depth_only",
        "dual_backbone_vcrf_descriptor_off",
        "dual_backbone_vcrf_feature_only",
        "dual_backbone_vcrf_logit_only",
        "dual_backbone_vcrf_global_static",
        "dual_backbone_vcrf_fullcue_no_aux_viewreg",
        "dual_backbone_vcrf_aux_off",
        "dual_backbone_vcrf_viewreg_off",
        "dual_r3d_vcrf_full",
        "dual_r2plus1d_vcrf_full",
    ]
    assert BENCHMARK_MODELS == [
        "resnet18",
        "resnet34",
        "resnet50",
        "densenet121",
        "densenet161",
        "densenet169",
        "resnext50",
        "resnext101",
        "vit",
        "swin3d",
        "multiscale_gat",
        "r3d",
        "r2plus1d",
        "s3d",
        "convnext_slice_mil",
        "convnext_25d_ordered_mil",
        "convnext_triplanar_25d_mil",
        "primus_m",
    ]


def test_removed_roi_and_mamba_models_are_not_listed():
    removed_models = {
        "pure_3d_mamba",
        "r3d_view_consistency_reliability",
        "r3d_mobile_mamba_adapter",
        "resnet18_mobile_mamba_adapter_stage4",
        "resnet18_roi_mobile_mamba_adapter_stage4",
        "r3d_roi_region_token_mamba_adapter_late_meanmax_anchor_safe",
    }

    assert removed_models.isdisjoint(set(MODEL_CHOICES))


def test_describe_model_covers_remaining_models():
    for model_name in MODEL_CHOICES:
        assert describe_model(model_name)
    assert "Dual R3D-18/R(2+1)D-18" in describe_model(DB_VCRF_MODEL)


def test_remaining_models_do_not_require_roi_masks():
    for model_name in MODEL_CHOICES:
        assert not model_requires_roi_mask(model_name)


def test_train_cli_accepts_remaining_teacher_member_models():
    args = SimpleNamespace(
        model=DB_VCRF_MODEL,
        teacher_model=None,
        teacher_checkpoint=None,
        teacher_members=[("densenet121", "outputs/checkpoints/example.pth", "0.50")],
        use_roi_mask=False,
        hard_negative_mining_csv=None,
        ema_decay=0.0,
        ema_start_epoch=None,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        pairwise_auc_weight=0.0,
        pairwise_auc_margin=0.0,
        focal_penalty_weight=0.0,
        roi_consistency_weight=0.0,
        roi_consistency_jitter_voxels=0,
        roi_consistency_dropout_prob=0.0,
    )

    validate_train_args(args)


def test_train_cli_restricts_primus_pretraining_checkpoint_to_unambiguous_use():
    common = {
        "teacher_model": None,
        "teacher_checkpoint": None,
        "teacher_members": None,
        "use_roi_mask": False,
        "hard_negative_mining_csv": None,
        "ema_decay": 0.0,
        "ema_start_epoch": None,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
        "pairwise_auc_weight": 0.0,
        "pairwise_auc_margin": 0.0,
        "focal_penalty_weight": 0.0,
        "roi_consistency_weight": 0.0,
        "roi_consistency_jitter_voxels": 0,
        "roi_consistency_dropout_prob": 0.0,
    }
    non_primus = SimpleNamespace(
        **common,
        model="r3d",
        primus_pretrained_path="checkpoint.pth",
        init_checkpoint=None,
    )
    with pytest.raises(ValueError, match="only valid"):
        validate_train_args(non_primus)

    ambiguous = SimpleNamespace(
        **common,
        model="primus_m",
        primus_pretrained_path="checkpoint.pth",
        init_checkpoint="fine_tuned.pth",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_train_args(ambiguous)


def test_train_cli_accepts_xy_flip_only_spatial_augmentation():
    validate_train_args(
        SimpleNamespace(model="r3d", spatial_aug_policy="xy_flip_only")
    )


def test_train_cli_rejects_unknown_spatial_augmentation():
    with pytest.raises(ValueError, match="spatial-aug-policy"):
        validate_train_args(
            SimpleNamespace(model="r3d", spatial_aug_policy="depth_only")
        )


def test_train_cli_validates_residual_gate_and_batchnorm_options():
    args = SimpleNamespace(
        model="dual_backbone_vcrf_t0_oracle_residual_gate",
        teacher_model=None,
        teacher_checkpoint=None,
        teacher_members=None,
        use_roi_mask=False,
        hard_negative_mining_csv=None,
        ema_decay=0.0,
        ema_start_epoch=None,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        gate_alpha=1.1,
        gate_temperature=1.0,
        gate_oracle_temperature=1.0,
        gate_loss_weight=0.2,
        freeze_batchnorm=True,
        freeze_batchnorm_affine=True,
        pairwise_auc_weight=0.0,
        pairwise_auc_margin=0.0,
        focal_penalty_weight=0.0,
        roi_consistency_weight=0.0,
        roi_consistency_jitter_voxels=0,
        roi_consistency_dropout_prob=0.0,
        spatial_aug_policy="hw_only",
    )

    with pytest.raises(ValueError, match="gate-alpha"):
        validate_train_args(args)

    args.gate_alpha = 0.5
    args.freeze_batchnorm = False
    with pytest.raises(ValueError, match="freeze-batchnorm-affine"):
        validate_train_args(args)

    args.freeze_batchnorm_affine = False
    args.model = "r3d"
    with pytest.raises(ValueError, match="gate-loss-weight"):
        validate_train_args(args)

    args.gate_loss_weight = 0.0
    args.freeze_source_experts = True
    with pytest.raises(ValueError, match="freeze-source-experts"):
        validate_train_args(args)

    args.model = "dual_backbone_vcrf_t0_oracle_residual_gate"
    args.gate_loss_weight = 0.2
    args.gate_alpha = 0.0
    args.freeze_source_experts = False
    with pytest.raises(ValueError, match="gate-alpha"):
        validate_train_args(args)

    args.gate_loss_weight = 0.0
    args.gate_alpha = 0.5
    args.freeze_source_experts = True
    args.freeze_backbone_epochs = 0
    with pytest.raises(ValueError, match="freeze-backbone-epochs"):
        validate_train_args(args)


def test_train_cli_rejects_removed_teacher_member_models():
    args = SimpleNamespace(
        model="resnet34",
        teacher_model=None,
        teacher_checkpoint=None,
        teacher_members=[
            (
                "r3d_roi_region_token_mamba_adapter_late_meanmax",
                "outputs/checkpoints/example.pth",
                "0.50",
            )
        ],
        use_roi_mask=False,
    )

    with pytest.raises(ValueError, match="Unknown teacher member model"):
        validate_train_args(args)


def test_test_cli_accepts_db_vcrf_without_roi_mask():
    args = SimpleNamespace(
        model=DB_VCRF_MODEL,
        use_roi_mask=False,
        seed=None,
        seeds=None,
        checkpoints=None,
    )

    validate_test_args(args)


def test_build_model_rejects_removed_model_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("resnet18_roi_mobile_mamba_adapter_stage4", num_classes=2)


@pytest.mark.parametrize(
    ("model_class_name", "head_prefix", "expected_weight_key", "expected_bias_key"),
    [
        ("R3DClassifier", "backbone.fc.", "backbone.fc.weight", "backbone.fc.bias"),
        ("R2Plus1DClassifier", "backbone.fc.", "backbone.fc.weight", "backbone.fc.bias"),
        (
            "S3DClassifier",
            "backbone.classifier.",
            "backbone.classifier.1.weight",
            "backbone.classifier.1.bias",
        ),
    ],
)
def test_video_expert_head_warmup_and_lr_groups_preserve_classifier_state_keys(
    model_class_name,
    head_prefix,
    expected_weight_key,
    expected_bias_key,
):
    from models import model as model_module

    model_class = getattr(model_module, model_class_name)
    model = model_class(num_classes=2, in_chans=1, pretrained=False)
    assert model.supports_head_warmup()
    assert expected_weight_key in model.state_dict()
    assert expected_bias_key in model.state_dict()

    model.set_backbone_trainable(False)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(head_prefix)

    groups = model.get_param_groups(
        base_lr=3e-5,
        backbone_lr_scale=0.1,
        weight_decay=1e-2,
    )
    backbone_ids = {id(parameter) for parameter in groups[0]["params"]}
    head_ids = {id(parameter) for parameter in groups[1]["params"]}
    all_ids = {id(parameter) for parameter in model.parameters()}
    assert backbone_ids.isdisjoint(head_ids)
    assert backbone_ids | head_ids == all_ids
    assert groups[0]["lr"] == pytest.approx(3e-6)
    assert groups[1]["lr"] == pytest.approx(3e-5)
    assert groups[0]["weight_decay"] == pytest.approx(1e-2)
    assert groups[1]["weight_decay"] == pytest.approx(1e-2)

    model.set_backbone_trainable(True)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_s3d_uses_single_channel_stem_binary_head_and_adaptive_pool():
    from models.model import S3DClassifier

    model = S3DClassifier(num_classes=2, in_chans=1, pretrained=False)

    assert model.backbone.features[0][0][0].in_channels == 1
    assert model.backbone.classifier[1].out_channels == 2
    assert isinstance(model.backbone.avgpool, torch.nn.AdaptiveAvgPool3d)


def test_convnext_slice_mil_forward_and_training_groups_without_pretrained_download():
    from models.model import ConvNeXtSliceMILClassifier

    model = ConvNeXtSliceMILClassifier(
        num_classes=2,
        in_chans=1,
        pretrained=False,
        num_slices=2,
        attention_dim=32,
        dropout=0.0,
    )
    assert model.backbone.features[0][0].in_channels == 1
    assert model.classifier.out_features == 2
    assert model.supports_head_warmup()
    assert model.input_mean.item() == pytest.approx((0.485 + 0.456 + 0.406) / 3.0)
    assert model.input_std.item() == pytest.approx((0.229 + 0.224 + 0.225) / 3.0)

    # Cached volumes are [B, C, X, Y, Z]. A non-square tensor makes an
    # accidental selection along X observable in both shape and values.
    volume = torch.arange(1 * 1 * 5 * 7 * 3, dtype=torch.float32).reshape(1, 1, 5, 7, 3)
    selected = model._select_slices(volume)
    assert selected.shape == (2, 1, 5, 7)
    assert torch.equal(selected[0], volume[0, :, :, :, 0])
    assert torch.equal(selected[1], volume[0, :, :, :, 2])
    normalized = model._normalize_slices(torch.full((2, 1, 5, 7), 0.5))
    expected_normalized = (0.5 - model.input_mean.item()) / model.input_std.item()
    assert torch.allclose(normalized, torch.full_like(normalized, expected_normalized))

    model.set_backbone_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.attention_v.parameters())
    assert all(parameter.requires_grad for parameter in model.attention_u.parameters())
    assert all(parameter.requires_grad for parameter in model.attention_score.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    groups = model.get_param_groups(base_lr=3e-5, backbone_lr_scale=0.1, weight_decay=1e-2)
    backbone_ids = {id(parameter) for parameter in groups[0]["params"]}
    head_ids = {id(parameter) for parameter in groups[1]["params"]}
    all_ids = {id(parameter) for parameter in model.parameters()}
    assert backbone_ids.isdisjoint(head_ids)
    assert backbone_ids | head_ids == all_ids
    assert groups[0]["lr"] == pytest.approx(3e-6)
    assert groups[1]["lr"] == pytest.approx(3e-5)

    model.eval()
    image = torch.rand(1, 1, 32, 32, 3)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 2)

    model.set_backbone_trainable(True)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_dual_backbone_view_consistency_reliability_forward_components():
    from models.dual_backbone_view_consistency_reliability import (
        DualBackboneViewConsistencyReliabilityFusion,
        create_dual_backbone_view_consistency_reliability,
    )

    model = create_dual_backbone_view_consistency_reliability(
        num_classes=2,
        in_chans=1,
        pretrained=False,
    )
    model.eval()
    image = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        model_output = model(image)
        logits, source_logits = model_output[:2]
        components = model.forward_components(image)

    assert isinstance(model, DualBackboneViewConsistencyReliabilityFusion)
    assert logits.shape == (1, 2)
    assert source_logits.shape == (1, 8, 2)
    assert components["r3d_logits"].shape == (1, 4, 2)
    assert components["r2p1d_logits"].shape == (1, 4, 2)
    assert components["reliability_weights"].shape == (1, 8, 1)
    assert torch.allclose(
        components["reliability_weights"].sum(dim=1),
        torch.ones(1, 1),
        atol=1e-6,
    )


def test_t0_residual_gate_sources_match_initialized_single_expert_logits():
    from models.dual_backbone_view_consistency_reliability import (
        create_dual_backbone_view_consistency_reliability,
    )
    from models.model import R2Plus1DClassifier, R3DClassifier
    from scripts.experiments.build_dual_expert_init_checkpoint import (
        build_dual_expert_init_state_dict,
    )

    torch.manual_seed(17)
    r3d = R3DClassifier(num_classes=2, in_chans=1, pretrained=False).eval()
    r2p1d = R2Plus1DClassifier(num_classes=2, in_chans=1, pretrained=False).eval()
    dual = create_dual_backbone_view_consistency_reliability(
        num_classes=2,
        in_chans=1,
        pretrained=False,
        views=("none",),
        residual_reliability_gate=True,
        aux_loss_weight=0.0,
        view_consistency_loss_weight=0.0,
    ).eval()
    init_state, _ = build_dual_expert_init_state_dict(
        dual,
        r3d.state_dict(),
        r2p1d.state_dict(),
    )
    dual.load_state_dict(init_state, strict=True)
    image = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        r3d_logits = r3d(image)
        r2p1d_logits = r2p1d(image)
        _, source_logits, _ = dual(image)

    assert torch.allclose(source_logits[:, 0], r3d_logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(source_logits[:, 1], r2p1d_logits, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("model_name", "num_sources"),
    [
        ("dual_backbone_vcrf_single_view", 2),
        ("dual_backbone_vcrf_single_view_t0", 2),
        ("dual_backbone_vcrf_single_view_td", 2),
        ("dual_backbone_vcrf_single_view_th", 2),
        ("dual_backbone_vcrf_single_view_tw", 2),
        ("dual_backbone_vcrf_t0_oracle_residual_gate", 2),
        ("dual_backbone_vcrf_no_depth_flip", 6),
        ("dual_backbone_vcrf_audit_depth_only", 6),
        ("dual_backbone_vcrf_descriptor_off", 8),
        ("dual_backbone_vcrf_feature_only", 8),
        ("dual_backbone_vcrf_logit_only", 8),
        ("dual_backbone_vcrf_global_static", 8),
        ("dual_backbone_vcrf_fullcue_no_aux_viewreg", 8),
        ("dual_backbone_vcrf_aux_off", 8),
        ("dual_backbone_vcrf_viewreg_off", 8),
    ],
)
def test_db_vcrf_ablation_variants_forward(model_name, num_sources):
    from models.dual_backbone_view_consistency_reliability import (
        DualBackboneViewConsistencyReliabilityFusion,
        create_dual_backbone_view_consistency_reliability,
    )

    ablation_kwargs = {
        "dual_backbone_vcrf_single_view": {"views": ("none",)},
        "dual_backbone_vcrf_single_view_t0": {"views": ("none",)},
        "dual_backbone_vcrf_single_view_td": {"views": ("d",)},
        "dual_backbone_vcrf_single_view_th": {"views": ("h",)},
        "dual_backbone_vcrf_single_view_tw": {"views": ("w",)},
        "dual_backbone_vcrf_t0_oracle_residual_gate": {
            "views": ("none",),
            "residual_reliability_gate": True,
            "gate_alpha": 0.5,
            "gate_temperature": 2.0,
            "freeze_source_experts": True,
            "aux_loss_weight": 0.0,
            "view_consistency_loss_weight": 0.0,
        },
        "dual_backbone_vcrf_no_depth_flip": {"views": ("none", "h", "w")},
        "dual_backbone_vcrf_audit_depth_only": {
            "views": ("none", "h", "w"),
            "audit_only_views": ("d",),
        },
        "dual_backbone_vcrf_descriptor_off": {"fusion_input_mode": "no_descriptors"},
        "dual_backbone_vcrf_feature_only": {"fusion_input_mode": "feature_only"},
        "dual_backbone_vcrf_logit_only": {"fusion_input_mode": "logit_only"},
        "dual_backbone_vcrf_global_static": {"fusion_input_mode": "global_static"},
        "dual_backbone_vcrf_fullcue_no_aux_viewreg": {
            "aux_loss_weight": 0.0,
            "view_consistency_loss_weight": 0.0,
        },
        "dual_backbone_vcrf_aux_off": {"aux_loss_weight": 0.0},
        "dual_backbone_vcrf_viewreg_off": {"view_consistency_loss_weight": 0.0},
    }[model_name]
    model = create_dual_backbone_view_consistency_reliability(
        num_classes=2,
        in_chans=1,
        pretrained=False,
        **ablation_kwargs,
    )
    assert isinstance(model, DualBackboneViewConsistencyReliabilityFusion)
    model.eval()
    image = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        model_output = model(image)
        logits, source_logits = model_output[:2]
        components = model.forward_components(image)

    assert logits.shape == (1, 2)
    assert source_logits.shape == (1, num_sources, 2)
    assert components["reliability_weights"].shape == (1, num_sources, 1)

    if model_name == "dual_backbone_vcrf_t0_oracle_residual_gate":
        expected = 0.5 / num_sources + 0.5 * torch.softmax(
            components["reliability_logits"] / 2.0,
            dim=1,
        )
        assert model.variant.endswith("oracle_residual_gate")
        assert torch.allclose(components["reliability_weights"], expected)
        model.set_backbone_trainable(False)
        assert not any(parameter.requires_grad for parameter in model.r3d_backbone.parameters())
        assert not any(parameter.requires_grad for parameter in model.r2p1d_backbone.parameters())
        assert not any(parameter.requires_grad for parameter in model.r3d_classifier.parameters())
        assert not any(parameter.requires_grad for parameter in model.r2p1d_classifier.parameters())
        assert all(parameter.requires_grad for parameter in model.reliability_head.parameters())
        frozen_bn_stats = {
            name: buffer.clone()
            for name, buffer in model.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        }
        model.train()
        assert not model.r3d_backbone.training
        assert not model.r2p1d_backbone.training
        assert not model.r3d_classifier.training
        assert not model.r2p1d_classifier.training
        with torch.no_grad():
            frozen_sources_a = model(image)[1]
            frozen_sources_b = model(image)[1]
        assert torch.allclose(frozen_sources_a, frozen_sources_b)
        for name, before in frozen_bn_stats.items():
            assert torch.equal(dict(model.named_buffers())[name], before)
        model.set_backbone_trainable(True)
        assert all(parameter.requires_grad for parameter in model.r3d_classifier.parameters())
        assert model.r3d_backbone.training
        assert model.r2p1d_classifier.training

    if model_name == "dual_backbone_vcrf_audit_depth_only":
        assert model.evaluation_views == ("none", "h", "w", "d")
        assert components["audit_source_logits"].shape == (1, 8, 2)
        assert components["audit_only_source_logits"].shape == (1, 2, 2)
