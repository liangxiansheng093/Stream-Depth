from typing import Tuple

import torch
import torch.nn.functional as F

from models import pose_vec_to_matrix


def _meshgrid(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).float().unsqueeze(0).repeat(batch, 1, 1, 1)
    return pix


def projective_warp(
    frame_tp1: torch.Tensor,
    depth_t: torch.Tensor,
    pose_t_to_tp1: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    b, _, h, w = frame_tp1.shape
    pix = _meshgrid(b, h, w, frame_tp1.device).reshape(b, 3, -1)

    inv_k = torch.inverse(intrinsics)
    cam_points = inv_k @ pix
    cam_points = cam_points * depth_t.reshape(b, 1, -1)

    pose = pose_vec_to_matrix(pose_t_to_tp1)
    cam_points_h = torch.cat([cam_points, torch.ones(b, 1, h * w, device=frame_tp1.device)], dim=1)
    cam_points_next = pose @ cam_points_h
    cam_points_next = cam_points_next[:, :3, :]

    pix_next = intrinsics @ cam_points_next
    x = pix_next[:, 0, :] / (pix_next[:, 2, :] + 1e-8)
    y = pix_next[:, 1, :] / (pix_next[:, 2, :] + 1e-8)

    x_norm = 2.0 * (x / (w - 1.0)) - 1.0
    y_norm = 2.0 * (y / (h - 1.0)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(b, h, w, 2)

    return F.grid_sample(frame_tp1, grid, mode="bilinear", padding_mode="border", align_corners=True)


def depth_supervision_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, has_depth: torch.Tensor) -> torch.Tensor:
    valid = (gt_depth > 1e-6).float() * has_depth[:, None, None, None]
    denom = valid.sum().clamp_min(1.0)
    return (torch.abs(pred_depth - gt_depth) * valid).sum() / denom


def photometric_loss(frame_t: torch.Tensor, reconstructed_t: torch.Tensor) -> torch.Tensor:
    l1 = torch.abs(frame_t - reconstructed_t).mean()
    return l1


def smoothness_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    img_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    return (depth_dx * torch.exp(-img_dx)).mean() + (depth_dy * torch.exp(-img_dy)).mean()


def total_loss(
    frame_t: torch.Tensor,
    frame_tp1: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    has_depth: torch.Tensor,
    pred_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    lambda_photo: float,
    lambda_smooth: float,
    lambda_depth: float,
) -> Tuple[torch.Tensor, dict]:
    reconstructed = projective_warp(frame_tp1, pred_depth, pred_pose, intrinsics)

    loss_d = depth_supervision_loss(pred_depth, gt_depth, has_depth)
    loss_p = photometric_loss(frame_t, reconstructed)
    loss_s = smoothness_loss(pred_depth, frame_t)

    loss = lambda_depth * loss_d + lambda_photo * loss_p + lambda_smooth * loss_s
    metrics = {
        "loss": loss.item(),
        "loss_depth": loss_d.item(),
        "loss_photo": loss_p.item(),
        "loss_smooth": loss_s.item(),
    }
    return loss, metrics
