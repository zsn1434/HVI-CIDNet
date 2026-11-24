# 使用 CVPR 2026 模块训练 HVI

这五组实验在配对低光增强上分别测试超分、复原、事件辅助增强和去噪模块。基线与各组共享 RGB/HVI 重建目标、数据加载和评测。R 额外训练信息密度估计器，V 增加同曝光去噪辅助监督。[论文拆解与原始数据](independent-papers-2026.md)

| 入口 | 接入模块和位置 | 默认训练参数总量 | 推理新增参数 |
|---|---|---:|---:|
| [r](r.py) | IAFMNet 的 IDE 与 ARM。在四分之一分辨率用 I 特征估计高斯尺度，由尺度图调节同尺度 HV 编码跳连 | 1,985,441 | 9,872 |
| [s](s.py) | C2SSM 的 CCSM。半分辨率 I 解码特征按余弦相似度匹配到 2×2 个中心，扫描中心后把结果回写像素 | 1,989,152 | 13,583 |
| [t](t.py) | UCAN 的 16 通道 LKSA。在半分辨率 I 解码特征上融合大核深度卷积和局部瓶颈 | 1,977,674 | 2,105 |
| [u](u.py) | EIC-LIE 的 IAEF 采样机制。I 生成横纵核，HV 生成偏移和邻域权重，修正四分之一分辨率的 HV 解码特征 | 1,993,570 | 18,001 |
| [v](v.py) | TM-BSN 作者的盲点教师生成同曝光去噪 RGB。用 I 分支的第一层特征预测该目标，导出时仅保留 CIDNet | 1,975,680 | 0 |

R 从论文公式复现量化噪声下的高斯区间似然和尺度调制，并以正值下限约束 GDN 和高斯尺度。S 使用与选择性扫描对应的纯 PyTorch 递推，沿四个中心迭代。U 用标准双线性采样和边界复制，对每个位置的 25 个抽样点联合归一化权重，并将横纵偏移限制在特征图的 ±1 像素。这些实现沿用项目的 PyTorch 环境。

## 准备数据和环境

在仓库根目录启用安装了项目依赖的 Python 环境。云环境可用 `source /workspace/.hvi-venv/bin/activate`，并设置 `export TORCH_HOME=/workspace/.cache/hvi-torch` 读取预训练特征权重。首次训练感知损失和使用 `--lpips` 时需要对应的 VGG19、AlexNet 权重。[安装与完整实验协议](README.md#训练与评测)

以 LOL-v2 Real 为例，将 `DATA` 改为你的数据路径。每组需要同编号、同尺寸的低光和正常光图片。Val 从训练数据预先划出，Test 保留给最后评测。如果不划 Val，省略训练命令的 `--val-dir`，最后使用 `latest.pt`。

```text
Real_captured/
├── Train/
│   ├── Low/
│   └── Normal/
├── Val/
│   ├── Low/
│   └── Normal/
└── Test/
    ├── Low/
    └── Normal/
```

## 训练基线和 R、S、T、U

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in baseline r s t u; do
  python -m "experiments.${variant}" train \
    --train-dir "$DATA/Train" --val-dir "$DATA/Val" \
    --out "results/real/${variant}" --epochs 1000 --batch-size 8 \
    --crop 256 --lr 0.0001 --seed 42 --eval-every 10 || break
done
```

有同协议的基线时可以从循环中移除 `baseline`。如果统一微调，可给所有组加 `--init /path/to/original_cidnet.pth`。它接受原版 CIDNet 的纯 state_dict。设备默认优先选择 CUDA，可用 `--device cuda:0` 指定。裁剪边长为 8 的倍数且不能超过训练图像短边。若降低批量大小或裁剪尺寸，用同一配置训练基线和待比较方案。

## 准备教师目标并训练 V

在 HVI 仓库根目录运行下面命令。最后的 `--teacher-dir` 对应前一步的输出，准备时只需给出训练集的低光图片。TM-BSN 按带上下文的方块处理长宽不等的照片，然后拼回原尺寸。显存紧张时可减小 `--tile`，保留默认上下文边宽 `--halo 64`。

```bash
TEACHER=/tmp/hvi-tmbsn-teacher
git clone https://github.com/parkjun210/TM-BSN.git "$TEACHER"
git -C "$TEACHER" checkout efde5a84e3ffa70258d4a1f460c581e2cc94d4ee

python -m experiments.v prepare \
  --teacher-source "$TEACHER/model/TMBSN.py" \
  --teacher-checkpoint "$TEACHER/ckpt/TMBSN_SIDD.pth" \
  --low-dir "$DATA/Train/Low" --out "$DATA/Train/Teacher_TMB_SIDD"

python -m experiments.v train \
  --train-dir "$DATA/Train" --val-dir "$DATA/Val" \
  --teacher-dir "$DATA/Train/Teacher_TMB_SIDD" --teacher-weight 0.1 \
  --out results/real/v --epochs 1000 --batch-size 8 \
  --crop 256 --lr 0.0001 --seed 42 --eval-every 10
```

也可把已有的同曝光去噪图放在一个单独目录，并让文件编号与低光图片逐一对应。运行时三张图共享裁剪和翻转。验证和测试只读取 Low 与 Normal。教师来自 SIDD，可先放大查看几张生成的暗部图，确认小字和细纹理仍可辨认。[教师网络与权重](https://github.com/parkjun210/TM-BSN/tree/efde5a84e3ffa70258d4a1f460c581e2cc94d4ee)

## 消融、评测与续训

按上面的训练命令启动新组，并为每个消融指定独立的 `--out`。R、S、T、U 的模式随 checkpoint 记录，评测时按训练模式运行。

| 组 | 选项 | 检验目标 |
|---|---|---|
| R | `--prior constant` 或 `--prior gradient` | 将学习尺度图换成常数或原始亮度梯度。单独检验信息密度图的贡献 |
| S | `--cluster-only` | 保留聚类和回写，检验中心扫描的额外贡献 |
| T | `--local-only` | 保留局部 LKSA 支路，检验长程卷积的贡献 |
| U | `--fixed-grid` | 保留动态权重和滤波核，将采样偏移固定为零 |
| V | `--identity-teacher` 代替 `--teacher-dir` | 用原始低光图作辅助目标，检验去噪目标的贡献 |

训练使用 Val 时，`best.pt` 对应验证 PSNR 最佳的轮次。训练中断后运行原命令并加 `--resume`，去掉此前可能使用的 `--init`，保持数据路径、总轮数、消融选项和输出目录一致。最终评测 RGB PSNR、SSIM 和可选 LPIPS。

```bash
for variant in baseline r s t u v; do
  python -m "experiments.${variant}" eval \
    --val-dir "$DATA/Test" --checkpoint "results/real/${variant}/best.pt" \
    --lpips --alpha 1 || break
done

python -m experiments.v export \
  --checkpoint results/real/v/best.pt --out results/real/v/cidnet.pth
```

V 导出的 `cidnet.pth` 是纯 CIDNet 权重，可由原版 CIDNet 加载。评测循环读取的是实验 checkpoint `best.pt`。
