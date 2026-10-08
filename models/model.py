import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
from monai.networks import nets as monai_nets
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.nets import UNet
import torchvision.models.video as tv_video

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from monai.networks.nets.resnet import get_medicalnet_pretrained_resnet_args
except ImportError:
    get_medicalnet_pretrained_resnet_args = None

__all__ = [
    "ResNet3D",
    "DenseNet3DClassifier",
    "ResNeXt3DClassifier",
    "VisionTransformer3DClassifier",
    "MultiScaleFeatureGATClassifier",
    "SwinClassifier3D",
    "R2Plus1DClassifier",
    "R3DClassifier",
    "S3DClassifier",
    "ConvNeXtSliceMILClassifier",
    "ConvNeXt25DOrderedMILClassifier",
    "ConvNeXtTriPlanar25DMILClassifier",
    "UNetEncoderClassifier",
]

MEDICALNET_RESNET_CONFIGS = {
    "resnet18": {
        "shortcut_type": "A",
        "bias_downsample": True,
        "filename": "resnet_18_23dataset.pth",
    },
    "resnet34": {
        "shortcut_type": "A",
        "bias_downsample": True,
        "filename": "resnet_34_23dataset.pth",
    },
    "resnet50": {
        "shortcut_type": "B",
        "bias_downsample": True,
        "filename": "resnet_50_23dataset.pth",
    },
}


def _get_monai_builder(name: str):
    builder = getattr(monai_nets, name, None)
    if builder is None:
        raise ValueError(f"{name} 未在 monai.networks.nets 中提供，请检查 MONAI 版本或更换可用变体。")
    return builder


def _adapt_conv3d_weight(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    if weight.ndim != 5 or weight.shape[1] == in_chans:
        return weight
    if in_chans == 1:
        return weight.mean(dim=1, keepdim=True)
    repeat = (in_chans + weight.shape[1] - 1) // weight.shape[1]
    weight = weight.repeat(1, repeat, 1, 1, 1)[:, :in_chans]
    return weight / repeat


def _get_resnet_pretrain_kwargs(variant: str, pretrained: bool) -> dict:
    if pretrained and variant in MEDICALNET_RESNET_CONFIGS:
        cfg = MEDICALNET_RESNET_CONFIGS[variant]
        return {
            "bias_downsample": cfg["bias_downsample"],
            "shortcut_type": cfg["shortcut_type"],
        }
    if not pretrained:
        return {
            "bias_downsample": True,
            "shortcut_type": "B",
        }
    if get_medicalnet_pretrained_resnet_args is None:
        # 兼容不提供该辅助函数的 MONAI 版本，交给当前版本的 resnet builder 使用其默认参数。
        return {}
    depth = int(variant.replace("resnet", ""))
    bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(depth)
    return {
        "bias_downsample": bias_downsample,
        "shortcut_type": shortcut_type,
    }


def _get_medicalnet_pretrained_path(variant: str, pretrained_path: str | None = None) -> str:
    if pretrained_path is not None:
        if os.path.isabs(pretrained_path):
            return pretrained_path
        return os.path.join(PROJECT_ROOT, pretrained_path)
    filename = MEDICALNET_RESNET_CONFIGS[variant]["filename"]
    return os.path.join(PROJECT_ROOT, "assets", "pretrained", filename)


def _load_medicalnet_resnet_weights(
    model: nn.Module,
    variant: str,
    in_chans: int,
    pretrained_path: str | None = None,
) -> None:
    weight_path = _get_medicalnet_pretrained_path(variant, pretrained_path)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"MedicalNet 预训练权重不存在：{weight_path}")

    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"无法解析 MedicalNet 权重格式：{weight_path}")

    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k.startswith("fc."):
            continue
        if k == "conv1.weight":
            v = _adapt_conv3d_weight(v, in_chans)
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    total = len(model_state)
    print(f"[ResNet3D] {variant} 预训练权重加载：{loaded}/{total} 个参数匹配。")
    if loaded == 0:
        raise RuntimeError(f"[ResNet3D] 未匹配到任何权重，请检查 {weight_path} 与模型结构是否一致。")
    if incompatible.missing_keys:
        missing = [k for k in incompatible.missing_keys if not k.startswith("fc.")]
        if missing:
            print(f"[ResNet3D] 未加载参数示例：{missing[:5]}")


class ResNet3D(nn.Module):
    """
    3D ResNet 分类封装，默认使用 MedicalNet 的预训练权重（仅在 spatial_dims=3 且 in_chans=1 时可用）。
    variant 可选：resnet18/resnet34/resnet50/resnet101/resnet152。
    """

    def __init__(
        self,
        variant: str = "resnet34",
        num_classes: int = 3,
        in_chans: int = 1,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained
        supported = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
        if variant not in supported:
            raise ValueError(f"不支持的 ResNet 变体：{variant}，可选 {sorted(supported)}。")
        builder = _get_monai_builder(variant)
        base_kwargs = dict(
            spatial_dims=3,
            n_input_channels=in_chans,
            num_classes=num_classes,  # 此处仅为保持接口一致，实际 feed_forward=False 时 fc 被去掉
            feed_forward=False,  # 为了兼容 MedicalNet 预训练，统一去掉原始 fc，外部自建分类头
            pretrained=False,
        )
        base_kwargs.update(_get_resnet_pretrain_kwargs(variant, pretrained))
        base_kwargs.update(kwargs)
        self.model = builder(**base_kwargs)
        self.out_channels = 512 if variant in ("resnet18", "resnet34") else 2048
        self.head = nn.Linear(self.out_channels, num_classes)
        if pretrained:
            if variant not in MEDICALNET_RESNET_CONFIGS:
                raise ValueError(f"{variant} 暂未配置 MedicalNet 预训练权重，目前支持 {sorted(MEDICALNET_RESNET_CONFIGS)}。")
            _load_medicalnet_resnet_weights(self.model, variant, in_chans, pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)  # (B, C_out)
        return self.head(feats)


class DenseNet3DClassifier(nn.Module):
    """MONAI 3D DenseNet benchmark wrapper."""

    _CONFIGS = {
        "densenet121": {
            "builder": "DenseNet121",
            "init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
        },
        "densenet161": {
            "builder": None,
            "init_features": 96,
            "growth_rate": 48,
            "block_config": (6, 12, 36, 24),
        },
        "densenet169": {
            "builder": "DenseNet169",
            "init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 32, 32),
        },
    }

    def __init__(
        self,
        variant: str = "densenet121",
        num_classes: int = 3,
        in_chans: int = 1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if variant not in self._CONFIGS:
            raise ValueError(f"不支持的 DenseNet 变体：{variant}，可选 {sorted(self._CONFIGS)}。")
        cfg = self._CONFIGS[variant]
        kwargs = dict(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=num_classes,
            init_features=cfg["init_features"],
            growth_rate=cfg["growth_rate"],
            block_config=cfg["block_config"],
            dropout_prob=dropout_prob,
        )
        if cfg["builder"] is None:
            self.model = monai_nets.DenseNet(**kwargs)
        else:
            builder = _get_monai_builder(cfg["builder"])
            self.model = builder(pretrained=False, **kwargs)
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _ResNeXtBottleneck3D(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 32,
        base_width: int = 4,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv3d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNeXt3DClassifier(nn.Module):
    """3D ResNeXt benchmark matching the ResNeXt-50/101 comparison rows."""

    _LAYERS = {
        "resnext50": (3, 4, 6, 3),
        "resnext101": (3, 4, 23, 3),
    }

    def __init__(
        self,
        variant: str = "resnext50",
        num_classes: int = 3,
        in_chans: int = 1,
        groups: int = 32,
        width_per_group: int = 4,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if variant not in self._LAYERS:
            raise ValueError(f"不支持的 ResNeXt 变体：{variant}，可选 {sorted(self._LAYERS)}。")
        self.variant = variant
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        layers = self._LAYERS[variant]
        self.layer1 = self._make_layer(64, layers[0], stride=1, groups=groups, width_per_group=width_per_group, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(128, layers[1], stride=2, groups=groups, width_per_group=width_per_group, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(256, layers[2], stride=2, groups=groups, width_per_group=width_per_group, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(512, layers[3], stride=2, groups=groups, width_per_group=width_per_group, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * _ResNeXtBottleneck3D.expansion, num_classes)
        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        groups: int,
        width_per_group: int,
        dropout_prob: float,
    ) -> nn.Sequential:
        layers = [
            _ResNeXtBottleneck3D(
                self.inplanes,
                planes,
                stride=stride,
                groups=groups,
                base_width=width_per_group,
                dropout_prob=dropout_prob,
            )
        ]
        self.inplanes = planes * _ResNeXtBottleneck3D.expansion
        for _ in range(1, blocks):
            layers.append(
                _ResNeXtBottleneck3D(
                    self.inplanes,
                    planes,
                    groups=groups,
                    base_width=width_per_group,
                    dropout_prob=dropout_prob,
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class VisionTransformer3DClassifier(nn.Module):
    """MONAI 3D ViT classification baseline for cached ABUS volumes."""

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        img_size: tuple[int, int, int] = (128, 128, 64),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        hidden_size: int = 384,
        mlp_dim: int = 1536,
        num_layers: int = 8,
        num_heads: int = 6,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.model = monai_nets.ViT(
            in_channels=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=3,
            qkv_bias=True,
        )
        self.variant = "vit3d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits


class _ConvBNAct3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ScaleGraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.mlp(tokens))
        return tokens


class MultiScaleFeatureGATClassifier(nn.Module):
    """Multi-scale CNN feature extractor with graph attention over scale tokens."""

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        hidden_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBNAct3D(in_chans, 32),
            _ConvBNAct3D(32, 32),
        )
        self.stage1 = nn.Sequential(_ConvBNAct3D(32, 64, stride=2), _ConvBNAct3D(64, 64))
        self.stage2 = nn.Sequential(_ConvBNAct3D(64, 128, stride=2), _ConvBNAct3D(128, 128))
        self.stage3 = nn.Sequential(_ConvBNAct3D(128, 256, stride=2), _ConvBNAct3D(256, 256))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projections = nn.ModuleList(
            [
                nn.Linear(32, hidden_dim),
                nn.Linear(64, hidden_dim),
                nn.Linear(128, hidden_dim),
                nn.Linear(256, hidden_dim),
            ]
        )
        self.scale_embedding = nn.Parameter(torch.zeros(1, 4, hidden_dim))
        self.graph_blocks = nn.Sequential(
            _ScaleGraphAttentionBlock(hidden_dim, num_heads=4, dropout=dropout),
            _ScaleGraphAttentionBlock(hidden_dim, num_heads=4, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.variant = "multiscale_feature_gat"
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        x = self.stem(x)
        feats.append(x)
        x = self.stage1(x)
        feats.append(x)
        x = self.stage2(x)
        feats.append(x)
        x = self.stage3(x)
        feats.append(x)

        tokens = [
            projection(self.pool(feat).flatten(1))
            for projection, feat in zip(self.projections, feats, strict=True)
        ]
        tokens = torch.stack(tokens, dim=1) + self.scale_embedding
        tokens = self.graph_blocks(tokens)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)



def _adapt_video_first_conv(model: nn.Module, in_chans: int):
    """将 torchvision.video 预训练权重的首层 3 通道卷积适配为 in_chans 通道。"""
    if in_chans == 3:
        return model
    conv1 = model.stem[0] if hasattr(model, "stem") else getattr(model, "conv1", None)
    if conv1 is None:
        return model
    new_conv = nn.Conv3d(
        in_channels=in_chans,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        if in_chans == 1:
            w = conv1.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(w)
        else:
            # 随机初始化非预训练通道，其余沿用原权重
            new_conv.weight.zero_()
            repeat = torch.ceil(torch.tensor(in_chans / 3)).int().item()
            w = conv1.weight.repeat(1, repeat, 1, 1, 1)[:, :in_chans]
            new_conv.weight.copy_(w / repeat)
        if conv1.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    if hasattr(model, "stem"):
        model.stem[0] = new_conv
    else:
        model.conv1 = new_conv
    return model


class _VideoClassifierTrainingMixin:
    """Expose head warmup and differential-LR groups without re-keying fc."""

    backbone: nn.Module
    classifier_parameter_prefixes: tuple[str, ...] = ("fc.",)

    def _is_classifier_parameter(self, name: str) -> bool:
        return name.startswith(self.classifier_parameter_prefixes)

    def supports_head_warmup(self) -> bool:
        return True

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = bool(trainable) or self._is_classifier_parameter(name)

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float,
        weight_decay: float,
    ) -> list[dict]:
        backbone_parameters = []
        classifier_parameters = []
        for name, parameter in self.backbone.named_parameters():
            if self._is_classifier_parameter(name):
                classifier_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        return [
            {
                "params": backbone_parameters,
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": classifier_parameters,
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
        ]


class R2Plus1DClassifier(_VideoClassifierTrainingMixin, nn.Module):
    """
    torchvision R(2+1)D-18 分类封装，支持 Kinetics400 预训练。
    """

    def __init__(self, num_classes: int = 3, in_chans: int = 1, pretrained: bool = True):
        super().__init__()
        weights = tv_video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = tv_video.r2plus1d_18(weights=weights)
        self.backbone = _adapt_video_first_conv(self.backbone, in_chans)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.variant = "r2plus1d_18"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class R3DClassifier(_VideoClassifierTrainingMixin, nn.Module):
    """
    torchvision R3D-18 分类封装，可作为 I3D 类基线（Kinetics400 预训练）。
    """

    def __init__(self, num_classes: int = 3, in_chans: int = 1, pretrained: bool = True):
        super().__init__()
        weights = tv_video.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = tv_video.r3d_18(weights=weights)
        self.backbone = _adapt_video_first_conv(self.backbone, in_chans)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.variant = "r3d_18"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class S3DClassifier(_VideoClassifierTrainingMixin, nn.Module):
    """torchvision S3D classifier with Kinetics-400 pretraining support."""

    classifier_parameter_prefixes = ("classifier.",)

    def __init__(self, num_classes: int = 3, in_chans: int = 1, pretrained: bool = True):
        super().__init__()
        weights = tv_video.S3D_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = tv_video.s3d(weights=weights)

        first_conv = self.backbone.features[0][0][0]
        if not isinstance(first_conv, nn.Conv3d):
            raise RuntimeError("Unexpected torchvision S3D stem layout; expected features[0][0][0] to be Conv3d.")
        if first_conv.in_channels != in_chans:
            adapted_conv = nn.Conv3d(
                in_channels=in_chans,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=first_conv.bias is not None,
                padding_mode=first_conv.padding_mode,
            )
            with torch.no_grad():
                adapted_conv.weight.copy_(_adapt_conv3d_weight(first_conv.weight, in_chans))
                if first_conv.bias is not None:
                    adapted_conv.bias.copy_(first_conv.bias)
            self.backbone.features[0][0][0] = adapted_conv

        classifier = self.backbone.classifier[1]
        if not isinstance(classifier, nn.Conv3d):
            raise RuntimeError("Unexpected torchvision S3D classifier layout; expected classifier[1] to be Conv3d.")
        self.backbone.classifier[1] = nn.Conv3d(
            classifier.in_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # torchvision S3D uses a fixed (2, 7, 7) pool for video crops. ABUS
        # tensors can have a 64-voxel axis, so use shape-agnostic global pooling.
        self.backbone.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.variant = "s3d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ConvNeXtSliceMILClassifier(nn.Module):
    """ImageNet-pretrained 2D ConvNeXt with gated attention over ABUS slices."""

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        pretrained: bool = True,
        num_slices: int = 64,
        attention_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_slices <= 0:
            raise ValueError("num_slices must be positive.")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive.")

        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.convnext_tiny(weights=weights)
        first_conv = self.backbone.features[0][0]
        if not isinstance(first_conv, nn.Conv2d):
            raise RuntimeError("Unexpected torchvision ConvNeXt stem layout; expected features[0][0] to be Conv2d.")
        if first_conv.in_channels != in_chans:
            adapted_conv = nn.Conv2d(
                in_channels=in_chans,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=first_conv.bias is not None,
                padding_mode=first_conv.padding_mode,
            )
            if pretrained:
                with torch.no_grad():
                    if in_chans == 1:
                        adapted_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                    else:
                        repeat = (in_chans + first_conv.in_channels - 1) // first_conv.in_channels
                        weight = first_conv.weight.repeat(1, repeat, 1, 1)[:, :in_chans]
                        adapted_conv.weight.copy_(weight / repeat)
                    if first_conv.bias is not None:
                        adapted_conv.bias.copy_(first_conv.bias)
            self.backbone.features[0][0] = adapted_conv

        classifier = self.backbone.classifier[-1]
        if not isinstance(classifier, nn.Linear):
            raise RuntimeError("Unexpected torchvision ConvNeXt classifier layout; expected a final Linear layer.")
        self.feature_dim = classifier.in_features
        # Keep ConvNeXt's global pooling and final LayerNorm while exposing its
        # ImageNet representation as one instance embedding per ABUS slice.
        self.backbone.classifier[-1] = nn.Identity()
        self.attention_v = nn.Linear(self.feature_dim, attention_dim)
        self.attention_u = nn.Linear(self.feature_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.num_slices = int(num_slices)
        self.register_buffer(
            "input_mean",
            torch.tensor((0.485 + 0.456 + 0.406) / 3.0).view(1, 1, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor((0.229 + 0.224 + 0.225) / 3.0).view(1, 1, 1, 1),
        )
        self.variant = "convnext_tiny_slice_mil"

    def supports_head_warmup(self) -> bool:
        return True

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)
        for module in (self.attention_v, self.attention_u, self.attention_score, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float,
        weight_decay: float,
    ) -> list[dict]:
        head_parameters = [
            *self.attention_v.parameters(),
            *self.attention_u.parameters(),
            *self.attention_score.parameters(),
            *self.classifier.parameters(),
        ]
        return [
            {
                "params": list(self.backbone.parameters()),
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": head_parameters,
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
        ]

    def _uniform_slice_indices(self, num_available_slices: int, device: torch.device) -> torch.Tensor:
        if num_available_slices <= 0:
            raise ValueError("ABUS slice axis must be positive.")
        return torch.linspace(
            0,
            num_available_slices - 1,
            steps=self.num_slices,
            device=device,
        ).round().long()

    def _select_slices(self, x: torch.Tensor) -> torch.Tensor:
        """Select the last (Z/W) axis and return [B*S, C, X, Y] images."""
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, X, Y, Z] input, got shape {tuple(x.shape)}.")
        batch_size, channels, x_size, y_size, num_available_slices = x.shape
        indices = self._uniform_slice_indices(num_available_slices, x.device)
        slices = x.index_select(4, indices)
        return slices.permute(0, 4, 1, 2, 3).reshape(
            batch_size * self.num_slices,
            channels,
            x_size,
            y_size,
        )

    def _normalize_slices(self, slices: torch.Tensor) -> torch.Tensor:
        return (slices - self.input_mean) / self.input_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slices = self._normalize_slices(self._select_slices(x))
        batch_size = x.shape[0]
        features = self.backbone(slices).reshape(batch_size, self.num_slices, self.feature_dim)
        attention = torch.tanh(self.attention_v(features)) * torch.sigmoid(self.attention_u(features))
        attention = torch.softmax(self.attention_score(attention), dim=1)
        pooled = torch.sum(attention * features, dim=1)
        return self.classifier(self.dropout(pooled))


class ConvNeXt25DOrderedMILClassifier(nn.Module):
    """ConvNeXt-Tiny 2.5D instances with an order-aware Transformer aggregator.

    Volumes use ``[B, 1, X, Y, Z]`` layout. Uniformly spaced center slices
    are converted to three-channel ``(z-1, z, z+1)`` instances, preserving
    the original ImageNet ConvNeXt stem and its three-channel pretraining.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        pretrained: bool = True,
        num_slices: int = 32,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        feedforward_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_chans != 1:
            raise ValueError("ConvNeXt25DOrderedMILClassifier expects in_chans=1.")
        if num_slices <= 0:
            raise ValueError("num_slices must be positive.")
        if embed_dim <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads.")
        if num_layers <= 0 or feedforward_dim <= 0:
            raise ValueError("num_layers and feedforward_dim must be positive.")

        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.convnext_tiny(weights=weights)
        first_conv = self.backbone.features[0][0]
        if not isinstance(first_conv, nn.Conv2d) or first_conv.in_channels != 3:
            raise RuntimeError(
                "Unexpected torchvision ConvNeXt stem layout; expected a three-channel Conv2d."
            )
        classifier = self.backbone.classifier[-1]
        if not isinstance(classifier, nn.Linear):
            raise RuntimeError("Unexpected torchvision ConvNeXt classifier layout; expected a final Linear layer.")
        self.feature_dim = classifier.in_features
        self.backbone.classifier[-1] = nn.Identity()

        self.projection = nn.Linear(self.feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, num_slices + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sequence_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_slices = int(num_slices)
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.variant = "convnext_tiny_25d_ordered_mil"

    def supports_head_warmup(self) -> bool:
        return True

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)
        for module in (
            self.projection,
            self.sequence_encoder,
            self.sequence_norm,
            self.classifier,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.cls_token.requires_grad = True
        self.position_embedding.requires_grad = True

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float,
        weight_decay: float,
    ) -> list[dict]:
        backbone_parameters = list(self.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter for parameter in self.parameters() if id(parameter) not in backbone_ids
        ]
        return [
            {
                "params": backbone_parameters,
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": head_parameters,
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
        ]

    def _uniform_slice_indices(self, num_available_slices: int, device: torch.device) -> torch.Tensor:
        if num_available_slices <= 0:
            raise ValueError("ABUS slice axis must be positive.")
        return torch.linspace(
            0,
            num_available_slices - 1,
            steps=self.num_slices,
            device=device,
        ).round().long()

    def _select_25d_instances(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B*S, 3, X, Y]`` instances ordered as z-1/z/z+1."""
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, X, Y, Z] input, got shape {tuple(x.shape)}.")
        batch_size, _, x_size, y_size, num_available_slices = x.shape
        centers = self._uniform_slice_indices(num_available_slices, x.device)
        neighbors = torch.stack((centers - 1, centers, centers + 1), dim=1)
        neighbors = neighbors.clamp(0, num_available_slices - 1)
        instances = x[:, 0, :, :, neighbors]
        return instances.permute(0, 3, 4, 1, 2).reshape(
            batch_size * self.num_slices,
            3,
            x_size,
            y_size,
        )

    def _normalize_instances(self, instances: torch.Tensor) -> torch.Tensor:
        return (instances - self.input_mean) / self.input_std

    def _aggregate_ordered_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (self.num_slices, self.feature_dim):
            raise ValueError(
                f"Expected [B, {self.num_slices}, {self.feature_dim}] features, "
                f"got shape {tuple(features.shape)}."
            )
        tokens = self.projection(features)
        cls_token = self.cls_token.expand(features.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.position_embedding
        encoded = self.sequence_encoder(tokens)
        return self.sequence_norm(encoded[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        instances = self._normalize_instances(self._select_25d_instances(x))
        features = self.backbone(instances).reshape(
            x.shape[0],
            self.num_slices,
            self.feature_dim,
        )
        return self.classifier(self._aggregate_ordered_features(features))


class ConvNeXtTriPlanar25DMILClassifier(nn.Module):
    """Hierarchical 2.5D MIL over all three orthogonal ABUS planes.

    Each plane contributes ordered three-slice ImageNet-compatible instances.
    A shared ConvNeXt encoder keeps the parameter count controlled, while plane
    embeddings and one cross-plane Transformer let the classifier reconstruct
    complementary volumetric morphology without using a conventional 3D CNN.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        pretrained: bool = True,
        num_slices_per_plane: int = 24,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 3,
        feedforward_dim: int = 1536,
        dropout: float = 0.1,
        ablation: str = "full",
    ):
        super().__init__()
        if in_chans != 1:
            raise ValueError("ConvNeXtTriPlanar25DMILClassifier expects in_chans=1.")
        if num_slices_per_plane <= 0:
            raise ValueError("num_slices_per_plane must be positive.")
        if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads.")
        if num_layers <= 0 or feedforward_dim <= 0:
            raise ValueError("num_layers and feedforward_dim must be positive.")
        supported_ablations = {
            "full", "center_only", "mean_pool", "no_position", "no_plane", "z_only",
        }
        if ablation not in supported_ablations:
            raise ValueError(f"Unsupported TriMIL ablation: {ablation}.")

        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.convnext_tiny(weights=weights)
        classifier = self.backbone.classifier[-1]
        if not isinstance(classifier, nn.Linear):
            raise RuntimeError("Unexpected torchvision ConvNeXt classifier layout.")
        self.feature_dim = classifier.in_features
        self.backbone.classifier[-1] = nn.Identity()

        self.projection = nn.Linear(self.feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, num_slices_per_plane, embed_dim)
        )
        self.plane_embedding = nn.Parameter(torch.empty(1, 3, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sequence_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_slices_per_plane = int(num_slices_per_plane)
        self.register_buffer(
            "input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.plane_embedding, std=0.02)
        # Preserve the full model's original initialization order and state keys.
        # Disabled components are removed only after that initialization, so all
        # retained initial weights match a full model built from the same seed.
        self.ablation = ablation
        self.plane_axes = (4,) if ablation == "z_only" else (4, 3, 2)
        if ablation == "no_position":
            self.position_embedding = None
        elif ablation == "no_plane":
            self.plane_embedding = None
        elif ablation == "mean_pool":
            self.cls_token = None
            self.sequence_encoder = nn.Identity()
        elif ablation == "z_only":
            self.plane_embedding = nn.Parameter(
                self.plane_embedding[:, :1].detach().clone()
            )
        self.variant = "convnext_tiny_triplanar_25d_mil"
        if ablation != "full":
            self.variant += f"_{ablation}"
        # Plain metadata is intentionally not added to state_dict, preserving
        # strict reload of existing full checkpoints. The named CLI model choice
        # fixes this configuration when loading each ablation checkpoint.
        self.ablation_config = {
            "ablation": ablation,
            "plane_axes": self.plane_axes,
            "plane_order": ("Z",) if ablation == "z_only" else ("Z", "Y", "X"),
            "num_slices_per_plane": self.num_slices_per_plane,
            "num_instances": len(self.plane_axes) * self.num_slices_per_plane,
            "neighbor_offsets": (0, 0, 0) if ablation == "center_only" else (-1, 0, 1),
            "aggregation": "mean" if ablation == "mean_pool" else "cls_transformer",
            "position_embedding": self.position_embedding is not None,
            "plane_embedding": self.plane_embedding is not None,
            "mean_pool_embedding_effect": (
                "instance embeddings average to a learned constant offset"
                if ablation == "mean_pool" else None
            ),
        }

    def supports_head_warmup(self) -> bool:
        return True

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                parameter.requires_grad = True

    def get_param_groups(self, base_lr: float, backbone_lr_scale: float, weight_decay: float):
        backbone_parameters = list(self.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter for parameter in self.parameters() if id(parameter) not in backbone_ids
        ]
        return [
            {"params": backbone_parameters, "lr": base_lr * backbone_lr_scale,
             "weight_decay": weight_decay},
            {"params": head_parameters, "lr": base_lr, "weight_decay": weight_decay},
        ]

    def _uniform_slice_indices(self, size: int, device: torch.device) -> torch.Tensor:
        if size <= 0:
            raise ValueError("ABUS plane axis must be positive.")
        return torch.linspace(
            0, size - 1, steps=self.num_slices_per_plane, device=device
        ).round().long()

    def _select_plane_instances(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, X, Y, Z], got {tuple(x.shape)}.")
        if axis not in (2, 3, 4):
            raise ValueError("axis must be one of X=2, Y=3, Z=4.")
        centers = self._uniform_slice_indices(x.shape[axis], x.device)
        if self.ablation == "center_only":
            neighbors = centers.unsqueeze(1).expand(-1, 3)
        else:
            neighbors = torch.stack((centers - 1, centers, centers + 1), dim=1)
        neighbors = neighbors.clamp(0, x.shape[axis] - 1)
        selected = x.index_select(axis, neighbors.flatten())
        remaining = [dimension for dimension in (2, 3, 4) if dimension != axis]
        selected = selected.permute(0, axis, 1, *remaining)
        return selected.reshape(
            x.shape[0] * self.num_slices_per_plane, 3,
            x.shape[remaining[0]], x.shape[remaining[1]],
        )

    def _encode_plane(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        instances = self._select_plane_instances(x, axis)
        instances = (instances - self.input_mean) / self.input_std
        return self.backbone(instances).reshape(
            x.shape[0], self.num_slices_per_plane, self.feature_dim
        )

    def _aggregate_triplanar_features(self, features: torch.Tensor) -> torch.Tensor:
        expected = (len(self.plane_axes), self.num_slices_per_plane, self.feature_dim)
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(f"Expected [B, {expected}], got {tuple(features.shape)}.")
        tokens = self.projection(features)
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding.unsqueeze(1)
        if self.plane_embedding is not None:
            tokens = tokens + self.plane_embedding
        tokens = tokens.flatten(1, 2)
        if self.ablation == "mean_pool":
            return self.sequence_norm(tokens.mean(dim=1))
        cls_token = self.cls_token.expand(features.shape[0], -1, -1)
        encoded = self.sequence_encoder(torch.cat((cls_token, tokens), dim=1))
        return self.sequence_norm(encoded[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            [self._encode_plane(x, axis) for axis in self.plane_axes], dim=1
        )
        return self.classifier(self._aggregate_triplanar_features(features))


class UNetEncoderClassifier(nn.Module):
    """
    基于 MONAI 3D UNet encoder 的分类封装：将 UNet 输出特征全局池化后接线性层。
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        channels=(8,16,32,64,128),
        strides=(2, 2, 2, 2),
        norm="instance",
    ):
        super().__init__()
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=channels[-1],
            channels=channels,
            strides=strides,
            num_res_units=2,
            norm=norm,

        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(channels[-1], num_classes)
        self.variant = "unet_encoder"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.avgpool(feats).flatten(1)
        return self.head(feats)


class SwinClassifier3D(nn.Module):
    """
    基于 MONAI SwinTransformer 编码器的简单 3D 分类头：取最后一层特征做全局平均池化 + 全连接。
    预训练权重需自行加载（MONAI 不附带分类预训练）。
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        patch_size=(4, 4, 4),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained: bool = False,
        pretrained_path: str | None = None,
        pretrained_url: str | None = None,
    ):
        super().__init__()
        self.backbone = SwinTransformer(
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # MONAI 的 SwinTransformer 在默认 downsample="merging" 下最后一层仍会执行 PatchMerging，
        # 因此输出通道数是 num_features 的 2 倍（例如 1536 而不是 768）。这里按最后一层是否存在 downsample 自适应确定维度。
        layer_lists = [self.backbone.layers1, self.backbone.layers2, self.backbone.layers3, self.backbone.layers4]
        last_layer = layer_lists[self.backbone.num_layers - 1][0]
        out_channels = self.backbone.num_features * (2 if getattr(last_layer, "downsample", None) is not None else 1)
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(out_channels)
        self.head = nn.Linear(out_channels, num_classes)
        if pretrained:
            self._load_pretrained(pretrained_path, pretrained_url)

    def _load_pretrained(self, pretrained_path: str | None, pretrained_url: str | None) -> None:
        if pretrained_path is None and pretrained_url is None:
            raise ValueError("未提供预训练权重路径或 URL。")

        if pretrained_path is None:
            filename = os.path.basename(pretrained_url)
            pretrained_path = os.path.join("assets", "pretrained", filename)

        if not os.path.exists(pretrained_path):
            if pretrained_url is None:
                raise FileNotFoundError(f"预训练权重不存在：{pretrained_path}")
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            try:
                from monai.apps import download_url
                download_url(pretrained_url, pretrained_path)
            except Exception as exc:
                raise RuntimeError(
                    f"下载预训练权重失败，请手动下载到 {pretrained_path}"
                ) from exc

        ckpt = torch.load(pretrained_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        backbone_state = self.backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            if k.startswith("encoder."):
                k = k[len("encoder."):]
            if k.startswith("swinViT."):
                k = k[len("swinViT."):]
            # SSL 权重里层级形如 layers1.0.0.blocks.*，需去掉多余的 ".0."
            if k.startswith("layers"):
                k = k.replace(".0.0.", ".0.")
            if k in backbone_state and backbone_state[k].shape == v.shape:
                filtered[k] = v

        incompatible = self.backbone.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = len(backbone_state)
        print(f"[SwinClassifier3D] 预训练权重加载：{loaded}/{total} 个参数匹配。")
        if loaded == 0:
            print("[SwinClassifier3D] 警告：未匹配到任何权重，请检查配置是否一致。")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]  # 取最高层输出
        features = self.avgpool(features).flatten(1)
        features = self.norm(features)
        return self.head(features)
