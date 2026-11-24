# HVI-CIDNet 配对低光增强实验

从仓库根目录使用 `python -m experiments.<方案> train` 训练，使用同一模块的 `eval` 评测。每个方案文件都定义了自己的改动。公共训练器在 `experiments/runner.py`。五组均保留原 CIDNet 解码连线，包括原实现中输出被覆盖的 I_LCA5。

| 模块 | 实际总参数 | 额外参数 | 训练差异 |
|---|---:|---:|---|
| baseline | 1,975,569 | 0 | 同训练器下的原始 CIDNet |
| a | 1,975,619 | 50 | HV_LCA2 输入的 3→8→2 局部门控。同曝光、独立附加噪声的两份输入各用同一 GT 监督，并对输出 HV 加一致性损失 |
| b | 1,975,569 | 0 | 在原有损失上叠加按低光输入强度软分组的 HV 误差 |
| c | 1,977,931 | 2,362 | HV_LCA2 后增加两个 72→8→72 残差专家，用输入图像局部均值、最大强度、色度半径均值和纹理强度路由 |
| d | 1,977,873 | 2,304 | HV_LCA5 后接两个 rank-8 的 1×1 适配器，分别接收重建和 VGG 感知损失的梯度 |

A 的门控初始为恒等混合，C、D 的残差投影初始为零，均可从原版权重开始。A 的 `--noise-std` 在 [0,1] sRGB 张量上定义，初值 0.015。一批裁剪产生两份训练观测，显存紧张时可调整所有对照组的 batch 大小。B 默认用合成的同曝光噪声对估计当前批次各亮度组的 HV 扰动方差，做有界归一化和最差组加权。它测量对附加噪声的敏感度，真实相机噪声标定需要另外采集数据。C 在输入坐标上按 32 像素窗口汇总并插值融合权重。D 只投影细节适配器输出矩阵的感知梯度，使其与保真适配器输出矩阵当前的列空间正交。两个目标都训练主干。

## 训练与评测

使用项目原有的 PyTorch、torchvision、lpips 等依赖。云实例可以运行以下命令选择已配置的解释器和预训练特征权重缓存。

```bash
cd /workspace/HVI-CIDNet
source /workspace/.hvi-venv/bin/activate
export TORCH_HOME=/workspace/.cache/hvi-torch
```

准备 LOL-v2 Real 时，`--train-dir` 和 `--val-dir` 均指向一组配对图片的父目录。例如，下面两张编号相同的训练图分别作为输入和监督目标，同一对图片的宽高须一致。

```text
datasets/LOLv2/Real_captured/
├── Train/
│   ├── Low/low00690.png
│   └── Normal/normal00690.png
└── Test/
    ├── Low/low00001.png
    └── Normal/normal00001.png
```

默认裁剪为 256×256。如果训练图像的短边小于 256，请将所有对照组的 `--crop` 调为不超过最短边的正整数，且须为 8 的倍数。单组训练与评测命令见[仓库首页](../Readme.md)。

从头训练的示例依次运行同一协议下的五组。分别启动单组时，在循环列表中只保留对应字母。也可以给每组加相同的 `--init /path/to/original_cidnet.pth` 进行微调。`--init` 接受原仓库的纯 state_dict。

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in baseline a b c d; do
  python -m "experiments.${variant}" train \
    --train-dir "$DATA/Train" --out "./results/real/${variant}" \
    --epochs 1000 --batch-size 8 --crop 256 --lr 0.0001 --seed 42 || break
done
```

训练集、验证集和测试集的每个目录包含 `Low/Normal` 或 `low/high`。按文件标识配对，同名或 `low00690.png` 与 `normal00690.png` 均可。改变 `DATA` 即可测试 Synthetic，使用单独的输出目录。LOL-v1 可分别指定 `our485` 和 `eval15`。

每个 epoch 原子写入 `latest.pt` 并记录 `metrics.jsonl`。如有从训练数据留出的独立验证集，可添加 `--val-dir /path/to/Val --eval-every 10` 记录验证指标并保存最佳 PSNR 的 `best.pt`。中断后用原训练参数、相同目录和 `--resume` 恢复，去掉 `--init`。保持原定总 epoch 数以延续学习率曲线。

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in baseline a b c d; do
  python -m "experiments.${variant}" eval \
    --checkpoint "./results/real/${variant}/latest.pt" \
    --val-dir "$DATA/Test" --lpips || break
done
```

评测逐图计算 RGB 8-bit PSNR 和 SSIM 再取平均，`--lpips` 额外计算 AlexNet LPIPS。使用固定输出增益 `--alpha 1`，关闭 GT-Mean。图像会补齐到 8 的倍数，评测前裁回原尺寸。若对照原仓库中采用 0.8 增益的 Real 配方，可另列所有方案在 `--alpha 0.8` 下的结果。五组主比较使用同一权重初始化方式、固定增益、训练集和指标实现。

## 训练设置与单项对照

四个方案的实现分别位于 [a.py](a.py)、[b.py](b.py)、[c.py](c.py) 和 [d.py](d.py)。主干把编码器的 `HV_LCA2` 调用封装为 `encode_hv_level2`，供 A 和 C 改写，把解码器的 `HV_LCA5` 调用封装为 `decode_hv_level2`，供 D 改写。B 直接使用原版模型，只在训练时计算分组损失。[主干实现](../net/CIDNet.py)

共享配方使用 Adam、余弦退火、反传后梯度裁剪 0.01，以及原仓库的 RGB/HVI 双域 L1、SSIM、边缘和 VGG19 特征损失。该训练器独立于原版 train.py，学习率曲线采用无预热的余弦退火。首次运行感知损失或 LPIPS 需要可用的预训练权重。以新训练器的 baseline 作为归因对照。

- A 可用 `--consistency-weight 0` 对照额外的一致性监督。两组保持相同的 `--noise-std`、训练步数及 batch 大小。
- B 可用 `--band-mode uniform` 检查各组等权结果。默认 `calibrated` 使用扰动方差和动态权重。两组可固定 `--bands 5 --band-weight 0.1`。
- C 只有局部专家与路由器的新增参数。比较 baseline 和 C，检查同时有高光和阴影的图像。
- D 可用 `--disable-projection` 在相同双适配器与损失分工下关闭梯度投影。`--perceptual-weight` 默认 0.01，应为正值。

文中的参数量对应默认模型。示例超参数是实验起点，指标以真实数据训练与评测结果为准。
