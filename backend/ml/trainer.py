"""
MODNet training loop adapted for MPS / CPU / CUDA.

Implements:
  - Supervised training (requires image, trimap, alpha GT)
  - SOC Adaptation (self-supervised, no labels needed)
  - Real-time metrics via asyncio.Queue
  - Checkpointing every N epochs
"""

import asyncio
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from scipy.ndimage import grey_dilation, grey_erosion

from ml.modnet import MODNet, GaussianBlurLayer
from ml.dataset import MattingDataset
from ml.metrics import compute_sad, compute_mse, compute_gradient_error


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    dataset_dir: str = ""
    checkpoint_dir: str = "./checkpoints"
    pretrained_ckpt: str | None = None
    device: str = "mps"

    # Training params
    stage: str = "supervised"  # "supervised" or "soc"
    epochs: int = 40
    lr: float = 0.01
    batch_size: int = 4
    img_size: int = 512
    num_workers: int = 0  # 0 for MPS compatibility

    # Loss weights
    semantic_loss_weight: float = 10.0
    detail_loss_weight: float = 10.0
    matte_loss_weight: float = 1.0

    # SOC params
    soc_lr: float = 0.00001
    soc_epochs: int = 10

    # Checkpointing
    save_every: int = 5
    val_split: float = 0.1

    # Split
    train_split: float = 0.8
    val_split_ratio: float = 0.1
    test_split: float = 0.1

    # Backgrounds for synthetic composition
    backgrounds_dir: str | None = None

    # Run name
    run_name: str = "run_001"


@dataclass
class TrainingState:
    """Mutable training state."""

    status: str = "idle"  # idle, running, finished, error, stopped
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    semantic_loss: float = 0.0
    detail_loss: float = 0.0
    matte_loss: float = 0.0
    val_loss: float = 0.0
    lr: float = 0.0
    samples_processed: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    error_message: str = ""
    best_val_loss: float = float("inf")
    checkpoints: list[str] = field(default_factory=list)


# Global state
training_state = TrainingState()
training_event_queue: asyncio.Queue | None = None
_stop_flag = False


def get_training_state() -> TrainingState:
    return training_state


def stop_training():
    global _stop_flag
    _stop_flag = True


def _send_event(event: dict):
    """Send metric event to SSE queue (non-blocking)."""
    if training_event_queue is not None:
        try:
            training_event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # skip if queue is full


# ── Loss functions ───────────────────────────────────────
def supervised_loss(
    img: torch.Tensor,
    pred_semantic: torch.Tensor,
    pred_detail: torch.Tensor,
    pred_matte: torch.Tensor,
    gt_matte: torch.Tensor,
    trimap: torch.Tensor,
    blurer: GaussianBlurLayer,
    weights: tuple[float, float, float] = (10.0, 10.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three supervised losses for MODNet.

    Returns: (semantic_loss, detail_loss, matte_loss)
    """
    # ── Semantic loss: MSE on blurred downsampled GT ──
    gt_semantic = F.interpolate(
        gt_matte, scale_factor=1 / 16, mode="bilinear", align_corners=False
    )
    gt_semantic = blurer(gt_semantic)
    semantic_loss = F.mse_loss(pred_semantic, gt_semantic) * weights[0]

    # ── Detail loss: L1 only at transition region of trimap ──
    trimap_hr = F.interpolate(trimap, size=pred_detail.shape[2:], mode="nearest")
    transition_mask = (trimap_hr == 0.5).float()
    if transition_mask.sum() > 0:
        detail_loss = (
            F.l1_loss(pred_detail * transition_mask, gt_matte * transition_mask)
            * weights[1]
        )
    else:
        detail_loss = torch.tensor(0.0, device=img.device)

    # ── Matte loss: L1 + compositional L1 (weighted heavier at boundaries) ──
    # Boundary weighting: 4x weight at transition regions
    boundary_weight = torch.ones_like(gt_matte)
    trimap_full = F.interpolate(trimap, size=gt_matte.shape[2:], mode="nearest")
    boundary_weight[trimap_full == 0.5] = 4.0

    l1_loss = (torch.abs(pred_matte - gt_matte) * boundary_weight).mean()

    # Compositional loss: compare composited image
    pred_comp = pred_matte * img
    gt_comp = gt_matte * img
    comp_loss = (torch.abs(pred_comp - gt_comp) * boundary_weight).mean()

    matte_loss = (l1_loss + comp_loss) * weights[2]

    return semantic_loss, detail_loss, matte_loss


# ── Supervised training ──────────────────────────────────
def run_supervised_training(
    config: TrainingConfig, event_queue: asyncio.Queue | None = None
):
    """
    Run supervised fine-tuning of MODNet.

    This function runs synchronously (blocking). Call from asyncio.to_thread().
    """
    global _stop_flag, training_state, training_event_queue
    _stop_flag = False
    training_event_queue = event_queue

    device = _resolve_device(config.device)
    training_state = TrainingState(status="running", total_epochs=config.epochs)

    try:
        # ── Model ──
        model = _load_model(config, device)
        model.train()

        blurer = GaussianBlurLayer(1, 3).to(device)

        # ── Data ──
        dataset = MattingDataset(
            root=config.dataset_dir,
            split="train",
            img_size=config.img_size,
            augment=True,
            backgrounds_dir=config.backgrounds_dir,
        )

        if len(dataset) == 0:
            raise ValueError(f"No samples found in {config.dataset_dir}")

        val_size = max(1, int(len(dataset) * config.val_split_ratio))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False,
        )

        # ── Optimizer ──
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(config.epochs * 0.25)),
            gamma=0.1,
        )

        ckpt_dir = Path(config.checkpoint_dir) / config.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(1, config.epochs + 1):
            if _stop_flag:
                training_state.status = "stopped"
                _send_event({"type": "stopped", "epoch": epoch})
                break

            model.train()
            epoch_losses = {"semantic": 0, "detail": 0, "matte": 0, "total": 0}
            n_batches = 0

            for batch_idx, (imgs, trimaps, alphas) in enumerate(train_loader):
                if _stop_flag:
                    break

                imgs = imgs.to(device)
                trimaps = trimaps.to(device)
                alphas = alphas.to(device)

                optimizer.zero_grad()

                pred_sem, pred_det, pred_mat = model(imgs, inference=False)

                sem_loss, det_loss, mat_loss = supervised_loss(
                    imgs,
                    pred_sem,
                    pred_det,
                    pred_mat,
                    alphas,
                    trimaps,
                    blurer,
                    weights=(
                        config.semantic_loss_weight,
                        config.detail_loss_weight,
                        config.matte_loss_weight,
                    ),
                )
                loss = sem_loss + det_loss + mat_loss

                loss.backward()
                optimizer.step()

                epoch_losses["semantic"] += sem_loss.item()
                epoch_losses["detail"] += det_loss.item()
                epoch_losses["matte"] += mat_loss.item()
                epoch_losses["total"] += loss.item()
                n_batches += 1

            scheduler.step()

            # Averages
            if n_batches > 0:
                for k in epoch_losses:
                    epoch_losses[k] /= n_batches

            # Validation
            val_loss = _validate(model, val_loader, blurer, device)

            elapsed = time.time() - start_time
            eta = (elapsed / epoch) * (config.epochs - epoch) if epoch > 0 else 0

            # Update state
            training_state.current_epoch = epoch
            training_state.current_loss = epoch_losses["total"]
            training_state.semantic_loss = epoch_losses["semantic"]
            training_state.detail_loss = epoch_losses["detail"]
            training_state.matte_loss = epoch_losses["matte"]
            training_state.val_loss = val_loss
            training_state.lr = scheduler.get_last_lr()[0]
            training_state.samples_processed = epoch * train_size
            training_state.elapsed_seconds = elapsed
            training_state.eta_seconds = eta

            # Checkpoint
            if (
                epoch % config.save_every == 0
                or val_loss < training_state.best_val_loss
            ):
                ckpt_path = ckpt_dir / f"modnet_epoch{epoch:03d}_val{val_loss:.4f}.ckpt"
                torch.save(model.state_dict(), ckpt_path)
                training_state.checkpoints.append(str(ckpt_path))
                if val_loss < training_state.best_val_loss:
                    training_state.best_val_loss = val_loss
                    best_path = ckpt_dir / "best.ckpt"
                    torch.save(model.state_dict(), best_path)

            # Send SSE event
            _send_event(
                {
                    "type": "epoch_end",
                    "epoch": epoch,
                    "total_epochs": config.epochs,
                    "semantic_loss": epoch_losses["semantic"],
                    "detail_loss": epoch_losses["detail"],
                    "matte_loss": epoch_losses["matte"],
                    "total_loss": epoch_losses["total"],
                    "val_loss": val_loss,
                    "lr": training_state.lr,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "samples_processed": training_state.samples_processed,
                }
            )

            print(
                f"[epoch {epoch}/{config.epochs}] "
                f"loss={epoch_losses['total']:.4f} val={val_loss:.4f} "
                f"lr={training_state.lr:.6f}"
            )

        if not _stop_flag:
            training_state.status = "finished"
            import math

            best = training_state.best_val_loss
            _send_event(
                {
                    "type": "finished",
                    "best_val_loss": None if math.isinf(best) else best,
                }
            )

    except Exception as e:
        training_state.status = "error"
        training_state.error_message = str(e)
        _send_event({"type": "error", "message": str(e)})
        raise


# ── SOC Adaptation ───────────────────────────────────────
def run_soc_adaptation(
    config: TrainingConfig, event_queue: asyncio.Queue | None = None
):
    """
    Self-supervised SOC adaptation (no labels needed).
    Uses consistency between sub-branches as supervision.
    """
    global _stop_flag, training_state, training_event_queue
    _stop_flag = False
    training_event_queue = event_queue

    device = _resolve_device(config.device)
    epochs = config.soc_epochs
    training_state = TrainingState(status="running", total_epochs=epochs)

    try:
        model = _load_model(config, device)
        backup_model = copy.deepcopy(model)
        backup_model.eval()
        for p in backup_model.parameters():
            p.requires_grad = False

        model.freeze_norm()
        model.train()

        dataset = MattingDataset(
            root=config.dataset_dir,
            split="train",
            img_size=config.img_size,
            augment=False,
        )

        loader = DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=config.num_workers
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.soc_lr, betas=(0.9, 0.99)
        )

        ckpt_dir = Path(config.checkpoint_dir) / config.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if _stop_flag:
                training_state.status = "stopped"
                break

            total_loss = 0
            n = 0
            for imgs, _, _ in loader:
                if _stop_flag:
                    break

                imgs = imgs.to(device)
                optimizer.zero_grad()

                pred_sem, pred_det, pred_mat = model(imgs, inference=False)

                # Backup predictions (frozen)
                with torch.no_grad():
                    _, b_det, b_mat = backup_model(imgs, inference=False)

                # SOC semantic loss: consistency between semantic and matte
                sem_target = F.interpolate(
                    pred_mat.detach(),
                    scale_factor=1 / 16,
                    mode="bilinear",
                    align_corners=False,
                )
                soc_sem_loss = F.mse_loss(pred_sem, sem_target) * 100.0

                # SOC detail loss: consistency at boundaries using backup
                # Generate boundary mask from predicted matte
                mat_np = pred_mat.detach().squeeze().cpu().numpy()
                boundary = _compute_boundary_mask(mat_np)
                boundary_t = (
                    torch.from_numpy(boundary)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                boundary_t = F.interpolate(
                    boundary_t, size=pred_det.shape[2:], mode="nearest"
                )
                if boundary_t.sum() > 0:
                    soc_det_loss = (
                        F.l1_loss(pred_det * boundary_t, b_det * boundary_t) * 1.0
                    )
                else:
                    soc_det_loss = torch.tensor(0.0, device=device)

                loss = soc_sem_loss + soc_det_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n += 1

            avg_loss = total_loss / max(n, 1)
            elapsed = time.time() - start_time
            eta = (elapsed / epoch) * (epochs - epoch)

            training_state.current_epoch = epoch
            training_state.current_loss = avg_loss
            training_state.elapsed_seconds = elapsed
            training_state.eta_seconds = eta

            _send_event(
                {
                    "type": "epoch_end",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "total_loss": avg_loss,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                }
            )

            if epoch % config.save_every == 0 or epoch == epochs:
                ckpt_path = ckpt_dir / f"modnet_soc_epoch{epoch:03d}.ckpt"
                torch.save(model.state_dict(), ckpt_path)
                training_state.checkpoints.append(str(ckpt_path))

            print(f"[SOC epoch {epoch}/{epochs}] loss={avg_loss:.4f}")

        if not _stop_flag:
            training_state.status = "finished"

    except Exception as e:
        training_state.status = "error"
        training_state.error_message = str(e)
        raise


# ── Helpers ──────────────────────────────────────────────
def _resolve_device(device_str: str) -> torch.device:
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(config: TrainingConfig, device: torch.device) -> MODNet:
    if config.pretrained_ckpt and Path(config.pretrained_ckpt).exists():
        model = MODNet.from_pretrained(
            config.pretrained_ckpt, device=str(device), backbone_pretrained=False
        )
    else:
        model = MODNet(backbone_pretrained=True)
        model.to(device)
    return model


def _validate(
    model: MODNet,
    val_loader: DataLoader,
    blurer: GaussianBlurLayer,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for imgs, trimaps, alphas in val_loader:
            imgs = imgs.to(device)
            trimaps = trimaps.to(device)
            alphas = alphas.to(device)

            pred_sem, pred_det, pred_mat = model(imgs, inference=False)
            sem_l, det_l, mat_l = supervised_loss(
                imgs, pred_sem, pred_det, pred_mat, alphas, trimaps, blurer
            )
            total_loss += (sem_l + det_l + mat_l).item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def _compute_boundary_mask(alpha: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Compute boundary region using morphological operations."""
    dilated = grey_dilation(alpha, size=(kernel_size, kernel_size))
    eroded = grey_erosion(alpha, size=(kernel_size, kernel_size))
    boundary = (dilated - eroded) > 0.05
    return boundary.astype(np.float32)
