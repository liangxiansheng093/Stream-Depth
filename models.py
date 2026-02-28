from collections import deque
from typing import Deque, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT encoder returning multi-scale token features."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.feature_hooks = [2, 5, 8, 11]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, _, h, w = x.shape
        tokens = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.backbone.pos_drop(tokens + self.backbone.pos_embed)

        features = []
        for i, blk in enumerate(self.backbone.blocks):
            tokens = blk(tokens)
            if i in self.feature_hooks:
                feat = tokens[:, 1:, :]
                ph = h // self.patch_size
                pw = w // self.patch_size
                feat = feat.transpose(1, 2).reshape(b, self.embed_dim, ph, pw)
                features.append(feat)
        return features


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DPTHead(nn.Module):
    """DPT-like depth head fusing transformer features."""

    def __init__(self, in_dim: int, out_channels: Tuple[int, int, int, int] = (256, 128, 64, 32)) -> None:
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(in_dim, out_channels[0], 1),
            nn.Conv2d(in_dim, out_channels[1], 1),
            nn.Conv2d(in_dim, out_channels[2], 1),
            nn.Conv2d(in_dim, out_channels[3], 1),
        ])
        self.fuse4 = ConvBlock(out_channels[3], out_channels[3])
        self.fuse3 = ConvBlock(out_channels[2] + out_channels[3], out_channels[2])
        self.fuse2 = ConvBlock(out_channels[1] + out_channels[2], out_channels[1])
        self.fuse1 = ConvBlock(out_channels[0] + out_channels[1], out_channels[0])

        self.depth_pred = nn.Sequential(
            ConvBlock(out_channels[0], 64),
            nn.Conv2d(64, 1, 1),
            nn.Softplus(),
        )

    def forward(self, features: List[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = [proj(f) for proj, f in zip(self.proj, features)]

        x4 = self.fuse4(f4)
        x3 = self.fuse3(torch.cat([f3, F.interpolate(x4, size=f3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        x2 = self.fuse2(torch.cat([f2, F.interpolate(x3, size=f2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        x1 = self.fuse1(torch.cat([f1, F.interpolate(x2, size=f1.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        depth = self.depth_pred(x1)
        depth = F.interpolate(depth, size=out_hw, mode="bilinear", align_corners=False)
        return depth + 1e-3


class PoseNetHead(nn.Module):
    """PoseNet head that regresses SE(3) between two frames."""

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 6)

    def forward(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([frame_t, frame_tp1], dim=1)
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return 0.01 * self.fc(x)


class InterFrameInteractionAttention(nn.Module):
    """Cross-attention block that fuses current features with historical memory."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        memory_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if memory_size < 1:
            raise ValueError("memory_size must be >= 1")

        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self._memory: Deque[torch.Tensor] = deque(maxlen=memory_size)

    def reset_memory(self) -> None:
        self._memory.clear()

    def _tokens_from_feature(self, feat: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        b, c, h, w = feat.shape
        if c != self.embed_dim:
            raise ValueError(f"Expected channel dimension {self.embed_dim}, got {c}")
        return feat.flatten(2).transpose(1, 2), h, w

    def _build_memory_tokens(
        self,
        current_tokens: torch.Tensor,
        external_memory: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if external_memory is not None:
            return external_memory
        if len(self._memory) == 0:
            return None

        batch = current_tokens.shape[0]
        if self._memory[0].shape[0] != batch:
            self.reset_memory()
            return None
        return torch.cat(list(self._memory), dim=1)

    def forward(self, current_feat: torch.Tensor, memory_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            current_feat: [B, C, H, W] current frame features.
            memory_tokens: Optional external history [B, T, C]. If None, use internal memory queue.
        """
        cur_tokens, h, w = self._tokens_from_feature(current_feat)
        mem_tokens = self._build_memory_tokens(cur_tokens, external_memory=memory_tokens)

        if mem_tokens is None:
            fused_tokens = cur_tokens
        else:
            q = self.norm_q(cur_tokens)
            kv = self.norm_kv(mem_tokens)
            attn_tokens, _ = self.attn(q, kv, kv, need_weights=False)
            fused_tokens = cur_tokens + attn_tokens

        fused_tokens = fused_tokens + self.ffn(self.norm_out(fused_tokens))

        # Store detached history to avoid backprop through long temporal chains.
        self._memory.append(fused_tokens.detach())

        return fused_tokens.transpose(1, 2).reshape(current_feat.shape[0], self.embed_dim, h, w)


class VideoDepthModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        pretrained_encoder: bool = False,
        enable_temporal_attention: bool = False,
        temporal_memory_size: int = 4,
        temporal_num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = DINOv2Encoder(encoder_name, pretrained_encoder)
        self.temporal_attention = (
            InterFrameInteractionAttention(
                embed_dim=self.encoder.embed_dim,
                memory_size=temporal_memory_size,
                num_heads=temporal_num_heads,
            )
            if enable_temporal_attention
            else None
        )
        self.depth_head = DPTHead(self.encoder.embed_dim)
        self.pose_head = PoseNetHead()

    def reset_temporal_memory(self) -> None:
        if self.temporal_attention is not None:
            self.temporal_attention.reset_memory()

    def forward(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats_t = self.encoder(frame_t)
        if self.temporal_attention is not None:
            feats_t[-1] = self.temporal_attention(feats_t[-1])
        depth_t = self.depth_head(feats_t, out_hw=frame_t.shape[-2:])
        pose_t_to_tp1 = self.pose_head(frame_t, frame_tp1)
        return depth_t, pose_t_to_tp1


def euler_angles_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = euler[:, 0], euler[:, 1], euler[:, 2]

    cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)

    rot_x = torch.stack(
        [
            torch.ones_like(cx),
            torch.zeros_like(cx),
            torch.zeros_like(cx),
            torch.zeros_like(cx),
            cx,
            -sx,
            torch.zeros_like(cx),
            sx,
            cx,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    rot_y = torch.stack(
        [
            cy,
            torch.zeros_like(cy),
            sy,
            torch.zeros_like(cy),
            torch.ones_like(cy),
            torch.zeros_like(cy),
            -sy,
            torch.zeros_like(cy),
            cy,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    rot_z = torch.stack(
        [
            cz,
            -sz,
            torch.zeros_like(cz),
            sz,
            cz,
            torch.zeros_like(cz),
            torch.zeros_like(cz),
            torch.zeros_like(cz),
            torch.ones_like(cz),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    return rot_z @ rot_y @ rot_x


def pose_vec_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    rot = euler_angles_to_matrix(pose[:, :3])
    trans = pose[:, 3:].unsqueeze(-1)
    mat = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(pose.shape[0], 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3:] = trans
    return mat
