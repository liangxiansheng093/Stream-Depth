from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDepthDataset(Dataset):
    """
    Expected layout:
    root/
      seq_000/
        frames/000000.png
        frames/000001.png
        depths/000000.npy (optional, HxW depth in meters)
        intrinsics.txt (optional 3x3)
    """

    def __init__(self, root: str, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.samples: List[Dict] = []
        self._scan()
        if not self.samples:
            raise RuntimeError(f"No frame pairs found under {root}")

    def _scan(self) -> None:
        for seq in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            frame_dir = seq / "frames"
            if not frame_dir.exists():
                continue

            frames = sorted(frame_dir.glob("*.png"))
            if len(frames) < 2:
                continue

            intrinsics = self._load_intrinsics(seq / "intrinsics.txt")
            depth_dir = seq / "depths"

            for i in range(len(frames) - 1):
                depth_file = depth_dir / f"{frames[i].stem}.npy"
                self.samples.append(
                    {
                        "frame_t": frames[i],
                        "frame_tp1": frames[i + 1],
                        "depth_t": depth_file if depth_file.exists() else None,
                        "intrinsics": intrinsics,
                    }
                )

    def _load_intrinsics(self, intr_path: Path) -> np.ndarray:
        if intr_path.exists():
            mat = np.loadtxt(str(intr_path), dtype=np.float32).reshape(3, 3)
            return mat
        fx = fy = float(self.image_size[1])
        cx = self.image_size[1] / 2.0
        cy = self.image_size[0] / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_img(self, path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_depth(self, depth_path: Optional[Path]) -> Optional[torch.Tensor]:
        if depth_path is None:
            return None
        d = np.load(str(depth_path)).astype(np.float32)
        d = cv2.resize(d, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(d).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        depth = self._load_depth(sample["depth_t"])
        has_depth = torch.tensor(1.0 if depth is not None else 0.0, dtype=torch.float32)
        if depth is None:
            depth = torch.zeros(1, *self.image_size, dtype=torch.float32)

        return {
            "frame_t": self._load_img(sample["frame_t"]),
            "frame_tp1": self._load_img(sample["frame_tp1"]),
            "depth_t": depth,
            "has_depth": has_depth,
            "intrinsics": torch.from_numpy(sample["intrinsics"]),
        }
