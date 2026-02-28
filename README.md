# Stream-Depth

一个完整可执行的视频流深度估计训练工程，模型结构为：

- **DINOV2 Encoder**（ViT 主干）
- **DPT Head**（深度估计）
- **PoseNet Head**（相邻帧位姿映射）
- **InterFrameInteractionAttention**（可选帧间记忆交叉注意力）

训练采用混合损失：

1. 深度监督损失（有 GT 时）
2. 基于相邻帧重投影的 photometric loss
3. 深度平滑损失

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 数据目录格式

```text
data_root/
  seq_000/
    frames/
      000000.png
      000001.png
      ...
    depths/               # 可选；没有也可以训练（依赖光度重投影）
      000000.npy
      000001.npy
      ...
    intrinsics.txt        # 可选，3x3 相机内参；不存在则自动构造默认内参
  seq_001/
    ...
```

- `depths/*.npy` 是深度图（单位米，`H x W`）
- 每个样本由 `(frame_t, frame_t+1)` 组成

## 3. 开始训练

```bash
python train.py \
  --data-root ./data_root \
  --save-dir ./checkpoints \
  --encoder vit_small_patch14_dinov2.lvd142m \
  --batch-size 4 \
  --epochs 20 \
  --lr 1e-4
```

如果你希望加载预训练 DINOV2 权重（会尝试联网下载）：

```bash
python train.py --data-root ./data_root --pretrained-encoder
```

## 4. 核心文件

- `models.py`：DINOV2 encoder + DPT depth head + PoseNet
- `models.py`：DINOV2 encoder + DPT depth head + PoseNet + 帧间交互注意力模块
- `dataset.py`：视频帧对数据集读取
- `losses.py`：重投影、监督、平滑损失
- `train.py`：完整训练循环与 checkpoint 保存

## 5. 说明

- 该实现注重**可执行与可扩展**，可作为后续加入多尺度损失、自动遮挡掩码、SSIM photometric loss、时序窗口训练的基础。
- 若训练分辨率变化较大，建议在 `dataset.py` 中按实际相机模型同步缩放内参。
