import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDepthDataset
from losses import total_loss
from models import VideoDepthModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Video stream depth training with DINOv2 + DPT + PoseNet")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--encoder", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-h", type=int, default=224)
    parser.add_argument("--img-w", type=int, default=224)
    parser.add_argument("--lambda-photo", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoDepthDataset(args.data_root, image_size=(args.img_h, args.img_w))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = VideoDepthModel(args.encoder, pretrained_encoder=args.pretrained_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            frame_t = batch["frame_t"].to(device)
            frame_tp1 = batch["frame_tp1"].to(device)
            gt_depth = batch["depth_t"].to(device)
            has_depth = batch["has_depth"].to(device)
            intrinsics = batch["intrinsics"].to(device)

            pred_depth, pred_pose = model(frame_t, frame_tp1)

            loss, metrics = total_loss(
                frame_t,
                frame_tp1,
                pred_depth,
                gt_depth,
                has_depth,
                pred_pose,
                intrinsics,
                args.lambda_photo,
                args.lambda_smooth,
                args.lambda_depth,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch + 1:03d}.pth")

    print(f"Training done. Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
