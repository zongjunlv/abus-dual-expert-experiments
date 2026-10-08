import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.batch import batch_to_device, forward_model


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        use_amp: bool = False,
        accum_steps: int = 1,
        teacher_model=None,
        kd_alpha: float = 0.0,
        kd_temperature: float = 4.0,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        roi_consistency_weight: float = 0.0,
        roi_consistency_jitter_voxels: int = 0,
        roi_consistency_dropout_prob: float = 0.0,
        gate_loss_weight: float = 0.0,
        gate_alpha: float = 0.5,
        gate_temperature: float = 1.0,
        gate_oracle_temperature: float = 1.0,
        freeze_batchnorm: bool = False,
        freeze_batchnorm_affine: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.accum_steps = max(1, int(accum_steps))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
            except TypeError:
                self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.teacher_model = teacher_model
        self.kd_alpha = max(0.0, min(1.0, float(kd_alpha)))
        self.kd_temperature = max(1e-6, float(kd_temperature))
        self.mixup_alpha = max(0.0, float(mixup_alpha))
        self.mixup_prob = max(0.0, min(1.0, float(mixup_prob)))
        self.roi_consistency_weight = max(0.0, float(roi_consistency_weight))
        self.roi_consistency_jitter_voxels = max(0, int(roi_consistency_jitter_voxels))
        self.roi_consistency_dropout_prob = max(
            0.0,
            min(1.0, float(roi_consistency_dropout_prob)),
        )
        self.gate_loss_weight = max(0.0, float(gate_loss_weight))
        bare_model = self.model.module if hasattr(self.model, "module") else self.model
        gate_config = getattr(bare_model, "gate_hyperparameters", None)
        if callable(gate_config) and bool(getattr(bare_model, "residual_reliability_gate", False)):
            actual_gate_alpha, actual_gate_temperature = gate_config()
            self.gate_alpha = float(actual_gate_alpha)
            self.gate_temperature = float(actual_gate_temperature)
        else:
            self.gate_alpha = max(0.0, min(1.0, float(gate_alpha)))
            self.gate_temperature = max(1e-6, float(gate_temperature))
        self.gate_oracle_temperature = max(1e-6, float(gate_oracle_temperature))
        self.freeze_batchnorm = bool(freeze_batchnorm)
        self.freeze_batchnorm_affine = bool(freeze_batchnorm_affine)
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def _autocast(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            except TypeError:
                return torch.amp.autocast(enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    @staticmethod
    def _extract_logits(model_output):
        if isinstance(model_output, tuple):
            return model_output[0], model_output[1] if len(model_output) > 1 else None
        return model_output, None

    @staticmethod
    def _extract_training_outputs(model_output):
        if isinstance(model_output, tuple):
            logits = model_output[0]
            source_logits = model_output[1] if len(model_output) > 1 else None
            gate_logits = model_output[2] if len(model_output) > 2 else None
            return logits, source_logits, gate_logits
        return model_output, None, None

    @staticmethod
    def configure_batchnorm(model, freeze_affine: bool = False) -> None:
        bare_model = model.module if hasattr(model, "module") else model
        for module in bare_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                if freeze_affine and module.affine:
                    if module.weight is not None:
                        module.weight.requires_grad_(False)
                    if module.bias is not None:
                        module.bias.requires_grad_(False)

    def _freeze_batchnorm_modules(self) -> None:
        if not self.freeze_batchnorm:
            return
        self.configure_batchnorm(self.model, self.freeze_batchnorm_affine)

    def _gate_supervision_loss(
        self,
        gate_logits,
        source_logits,
        label_a,
        label_b=None,
        lam: float = 1.0,
    ):
        """Match gate scores to detached per-source label evidence.

        The oracle distribution never backpropagates into source classifiers;
        only the predicted gate distribution receives gradients from this loss.
        """
        if not torch.is_tensor(gate_logits) or not torch.is_tensor(source_logits):
            raise ValueError("Gate supervision requires source and gate logits.")
        if source_logits.ndim != 3:
            raise ValueError("source_logits must have shape [batch, source, class].")
        if gate_logits.ndim != 3 or gate_logits.shape[-1] != 1:
            raise ValueError("gate_logits must have shape [batch, source, 1].")

        batch_size, num_sources, num_classes = source_logits.shape
        if gate_logits.shape[:2] != (batch_size, num_sources):
            raise ValueError(
                "gate_logits batch/source dimensions must match source_logits; "
                f"got gate={tuple(gate_logits.shape)} source={tuple(source_logits.shape)}."
            )
        with torch.autocast(device_type=source_logits.device.type, enabled=False):
            detached_sources = source_logits.detach().float().reshape(
                batch_size * num_sources,
                num_classes,
            )
            repeated_a = label_a.repeat_interleave(num_sources)
            source_ce = F.cross_entropy(detached_sources, repeated_a, reduction="none").view(
                batch_size,
                num_sources,
            )
            if label_b is not None:
                repeated_b = label_b.repeat_interleave(num_sources)
                source_ce_b = F.cross_entropy(
                    detached_sources,
                    repeated_b,
                    reduction="none",
                ).view(batch_size, num_sources)
                source_ce = lam * source_ce + (1.0 - lam) * source_ce_b

            oracle_raw = F.softmax(
                -source_ce / self.gate_oracle_temperature,
                dim=1,
            ).detach()
            oracle_weights = (
                (1.0 - self.gate_alpha) / num_sources
                + self.gate_alpha * oracle_raw
            )
            learned_weights = F.softmax(
                gate_logits.float().squeeze(-1) / self.gate_temperature,
                dim=1,
            )
            predicted_weights = (
                (1.0 - self.gate_alpha) / num_sources
                + self.gate_alpha * learned_weights
            )
            gate_loss = F.kl_div(
                predicted_weights.clamp_min(1e-7).log(),
                oracle_weights,
                reduction="batchmean",
            )
        return gate_loss, oracle_weights

    def _distillation_loss(self, student_logits, teacher_logits):
        temperature = self.kd_temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    def _sample_mixup_params(self, image):
        if self.mixup_alpha <= 0.0 or self.mixup_prob <= 0.0 or image.size(0) < 2:
            return None
        if torch.rand((), device=image.device).item() > self.mixup_prob:
            return None
        beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = float(beta.sample().to(image.device).item())
        permutation = torch.randperm(image.size(0), device=image.device)
        identity = torch.arange(image.size(0), device=image.device)
        if torch.equal(permutation, identity):
            permutation = torch.roll(permutation, shifts=1)
        return lam, permutation

    def _apply_mixup(self, image, label, mask):
        params = self._sample_mixup_params(image)
        if params is None:
            return image, label, None, 1.0, mask
        lam, permutation = params
        mixed_image = lam * image + (1.0 - lam) * image[permutation]
        mixed_mask = None
        if mask is not None:
            mixed_mask = lam * mask + (1.0 - lam) * mask[permutation]
        return mixed_image, label, label[permutation], lam, mixed_mask

    def _supervised_loss(self, logits, label_a, label_b=None, lam: float = 1.0):
        if label_b is None:
            return self.criterion(logits, label_a)
        return lam * self.criterion(logits, label_a) + (1.0 - lam) * self.criterion(logits, label_b)

    def _auxiliary_supervised_loss(self, aux_logits, logits, label_a, label_b=None, lam: float = 1.0):
        if not torch.is_tensor(aux_logits):
            return None
        if aux_logits.ndim == 3:
            batch_size, num_views, num_classes = aux_logits.shape
            flat_logits = aux_logits.reshape(batch_size * num_views, num_classes)
            aux_label_a = label_a.repeat_interleave(num_views)
            aux_label_b = label_b.repeat_interleave(num_views) if label_b is not None else None
            return self._supervised_loss(flat_logits, aux_label_a, aux_label_b, lam)
        return self._supervised_loss(aux_logits, label_a, label_b, lam)

    @staticmethod
    def _view_consistency_loss(logits, aux_logits):
        if not torch.is_tensor(aux_logits) or aux_logits.ndim != 3:
            return logits.new_tensor(0.0)
        batch_size, num_views, num_classes = aux_logits.shape
        flat_view_logits = aux_logits.reshape(batch_size * num_views, num_classes)
        repeated_logits = logits.detach().unsqueeze(1).expand(-1, num_views, -1)
        target_probs = F.softmax(repeated_logits.reshape(batch_size * num_views, num_classes), dim=1)
        return F.kl_div(
            F.log_softmax(flat_view_logits, dim=1),
            target_probs,
            reduction="batchmean",
        )

    def _shift_mask_tensor(self, mask, shifts):
        shifted = torch.zeros_like(mask)
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

    def _make_consistency_mask(self, mask):
        if (
            mask is None
            or self.roi_consistency_weight <= 0.0
            or (
                self.roi_consistency_jitter_voxels <= 0
                and self.roi_consistency_dropout_prob <= 0.0
            )
        ):
            return None

        consistency_mask = mask.detach().clone()
        if self.roi_consistency_dropout_prob > 0.0:
            keep = torch.rand(
                (consistency_mask.size(0), 1, 1, 1, 1),
                device=consistency_mask.device,
                dtype=consistency_mask.dtype,
            ) >= self.roi_consistency_dropout_prob
            consistency_mask = consistency_mask * keep

        if self.roi_consistency_jitter_voxels > 0:
            max_shift = self.roi_consistency_jitter_voxels
            shifted_masks = []
            for sample_mask in consistency_mask:
                shifts = [
                    int(torch.randint(-max_shift, max_shift + 1, (1,), device=mask.device).item())
                    for _ in range(3)
                ]
                shifted_masks.append(self._shift_mask_tensor(sample_mask, shifts))
            consistency_mask = torch.stack(shifted_masks, dim=0)

        return (consistency_mask > 0).to(dtype=mask.dtype)

    @staticmethod
    def _consistency_loss(logits, consistency_logits):
        student_log_probs = F.log_softmax(logits, dim=1)
        consistency_log_probs = F.log_softmax(consistency_logits, dim=1)
        student_probs = student_log_probs.detach().exp()
        consistency_probs = consistency_log_probs.detach().exp()
        return 0.5 * (
            F.kl_div(student_log_probs, consistency_probs, reduction="batchmean")
            + F.kl_div(consistency_log_probs, student_probs, reduction="batchmean")
        )

    def train(self, dataloader):
        self.model.train()
        self._freeze_batchnorm_modules()
        if self.teacher_model is not None:
            self.teacher_model.eval()
        stats = {
            'total_loss' : 0.0,
            'gate_loss': 0.0,
            'processed_samples' : 0
        }

        pbar = tqdm(
            dataloader,
            desc='Train',
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='blue',
            disable=not self.is_main_process,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            moved = batch_to_device(batch, self.device)
            img = moved["image"]
            label = moved["label"]
            mask = moved["mask"]
            img, label_a, label_b, mixup_lam, mask = self._apply_mixup(img, label, mask)

            with self._autocast():
                gate_loss = img.new_tensor(0.0)
                logits, aux_logits, gate_logits = self._extract_training_outputs(
                    forward_model(self.model, image=img, mask=mask)
                )

                supervised_loss = self._supervised_loss(logits, label_a, label_b, mixup_lam)
                total_loss = supervised_loss
                if self.teacher_model is not None and self.kd_alpha > 0.0:
                    with torch.no_grad():
                        teacher_logits, _ = self._extract_logits(
                            forward_model(self.teacher_model, image=img, mask=mask)
                        )
                    kd_loss = self._distillation_loss(logits, teacher_logits)
                    total_loss = (1.0 - self.kd_alpha) * supervised_loss + self.kd_alpha * kd_loss
                if aux_logits is not None:
                    bare_model = self.model.module if hasattr(self.model, "module") else self.model
                    aux_loss_weight = float(getattr(bare_model, "aux_loss_weight", 0.0))
                    if aux_loss_weight > 0:
                        aux_supervised_loss = self._auxiliary_supervised_loss(
                            aux_logits,
                            logits,
                            label_a,
                            label_b,
                            mixup_lam,
                        )
                        if aux_supervised_loss is not None:
                            total_loss = total_loss + aux_loss_weight * aux_supervised_loss
                    view_consistency_weight = float(
                        getattr(bare_model, "view_consistency_loss_weight", 0.0)
                    )
                    if view_consistency_weight > 0:
                        total_loss = total_loss + view_consistency_weight * self._view_consistency_loss(
                            logits,
                            aux_logits,
                        )
                if self.gate_loss_weight > 0.0:
                    gate_loss, _ = self._gate_supervision_loss(
                        gate_logits,
                        aux_logits,
                        label_a,
                        label_b,
                        mixup_lam,
                    )
                    total_loss = total_loss + self.gate_loss_weight * gate_loss
                consistency_mask = self._make_consistency_mask(mask)
                if consistency_mask is not None:
                    consistency_logits, _ = self._extract_logits(
                        forward_model(self.model, image=img, mask=consistency_mask)
                    )
                    total_loss = total_loss + self.roi_consistency_weight * self._consistency_loss(
                        logits,
                        consistency_logits,
                    )

            loss_for_backward = total_loss / self.accum_steps
            self.scaler.scale(loss_for_backward).backward()

            if step % self.accum_steps == 0 or step == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = img.size(0)
            stats['total_loss'] += total_loss.item() * batch_size
            stats['gate_loss'] += gate_loss.item() * batch_size
            stats['processed_samples'] += batch_size

            postfix = {
                'loss': f"{stats['total_loss']/stats['processed_samples']:.4f}",
                'lr': f"{max(group['lr'] for group in self.optimizer.param_groups):.2e}"
            }

            pbar.set_postfix(postfix)

        self.last_train_stats = {
            "total_loss": stats["total_loss"] / stats["processed_samples"],
            "gate_loss": stats["gate_loss"] / stats["processed_samples"],
        }
        return stats['total_loss'] / stats['processed_samples']

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        processed_sample = 0

        pbar = tqdm(
            dataloader,
            desc='val',
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='green',
            disable=not self.is_main_process,
        )

        with torch.inference_mode():
            for batch in pbar:
                moved = batch_to_device(batch, self.device)
                img = moved["image"]
                label = moved["label"]
                mask = moved["mask"]

                logits, _ = self._extract_logits(forward_model(self.model, image=img, mask=mask))
                loss = self.criterion(logits, label)

                batch_size = img.size(0)
                total_loss += loss.item() * batch_size
                processed_sample += batch_size

                pbar.set_postfix({
                        'loss':f'{total_loss/processed_sample:.4f}'
                    }
                )

        return total_loss / processed_sample
