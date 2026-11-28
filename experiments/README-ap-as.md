# AP–AS 多指标融合实验

四个入口均使用同一个 CIDNet 主干和公共训练器。参数量按实际模型统计，CIDNet 有 1,975,569 个参数。四个方案都可在 LOLv1、LOLv2-Real 和 LOLv2-Syn 上独立训练。

| 入口 | 改动 | 总参数 | 相对 CIDNet |
|---|---|---:|---:|
| [ap](ap.py) | 从 W 出发，共享一份 M 字典。保留 M 的 72 通道直接修正和 Y 的区域路由，同时沿用 W 的亮度门 | 2,033,469 | +2.93% |
| [aq](aq.py) | 保留 R 的 HV 编码调制和尺度辅助损失。在 I 解码器沿用 Z 的平滑与细节更新，用 16 通道尺度图预测 8 组细节门 | 2,059,939 | +4.27% |
| [ar](ar.py) | 保留 T、O 的候选更新，在半分辨率 I 解码器学习三个平滑窗口的混合权重，将 T 的平滑项和 O 的细节项写回 | 2,034,349 | +2.98% |
| [as_fusion](as_fusion.py) | 联合 AP、AQ。先对 AP 的 HV 跳连结果应用 R 调制，再在 I 解码器使用 W 更新和 AQ 的分频修正 | 2,117,839 | +7.20% |

AQ 和联合版的训练参数包含 R 的高斯均值头，共 2,592 个参数。该头用于辅助损失，推理前向只读取尺度头。

## 开始训练

在仓库根目录使用已安装依赖的 Python 环境。云环境可以运行 `source /workspace/.hvi-venv/bin/activate`，并设 `export TORCH_HOME=/workspace/.cache/hvi-torch`。训练默认使用 VGG 感知损失，计算 LPIPS 需要 AlexNet 权重。

把下列数据路径换成实际目录。每个划分使用配对的 Low/Normal 或 low/high 子目录。从原训练集固定留出 Val，Test 用于最后评测。

```bash
REAL=./datasets/LOLv2/Real_captured
SYN=./datasets/LOLv2/Synthetic
V1=./datasets/LOLdataset

train_fusion() {
  local variant="$1" data="$2" protocol="$3" out="$4"
  shift 4
  python -m "experiments.$variant" train \
    --train-dir "$data/Train" --val-dir "$data/Val" --eval-protocol "$protocol" \
    --out "$out" --epochs 1500 --batch-size 8 --crop 256 --lr 0.0001 \
    --seed 42 --eval-every 10 "$@"
}

train_fusion ap        "$REAL" lolv2-real results/real/ap
train_fusion aq        "$REAL" lolv2-real results/real/aq
train_fusion ar        "$REAL" lolv2-real results/real/ar
train_fusion as_fusion "$REAL" lolv2-real results/real/as_fusion

train_fusion ap "$SYN" plain results/syn/ap --epochs 500
train_fusion aq "$SYN" plain results/syn/aq --epochs 500
train_fusion ap "$V1" lolv1 results/v1/ap --gt-mean
```

每条命令独立运行。先比较 AP、AQ 及各自对照，再验证联合版。将版本名替换成 w、m、y、r、z、o、t 可运行对应对照，其中各旧版所需特殊选项见其指南。对照使用相同训练步数、选模频率与数据划分。可为各组追加同一份原版 CIDNet state_dict，通过 `--init /path/to/cidnet.pth` 初始化。用相同入口、目录与参数加 `--resume` 续训，同时省略 `--init`。

## 消融

| 入口 | 对照设置 | 检验内容 |
|---|---|---|
| AP | `--no-direct --no-scene-router` | 只保留 W 的计算路径 |
| AP | `--no-scene-router` | W 加 M 直接调制 |
| AP | `--no-direct` | W 加 Y 的场景路由 |
| AQ | `--detail-mode independent` | R 与 Z 独立工作 |
| AQ | `--detail-mode mean` | 解码门只看到逐通道空间均值，R 的编码路径继续用完整尺度图 |
| AR | `--split-mode sum` | T、O 的更新直接相加 |
| AR | `--split-mode swapped` | 交换 T、O 的平滑与细节分工 |
| 联合版 | AP 和 AQ 的开关均可用 | 检查各机制在联合训练中的贡献 |

在训练命令末尾加开关，为每组指定新的输出目录。设置随 checkpoint 保存，评测自动恢复。关闭分支的对照仍保存该分支的参数张量。

## 最终评测

```bash
python -m experiments.ap eval --checkpoint results/real/ap/best.pt \
  --val-dir "$REAL/Test" --eval-protocol lolv2-real --lpips
python -m experiments.aq eval --checkpoint results/syn/aq/best.pt \
  --val-dir "$SYN/Test" --eval-protocol plain --lpips
python -m experiments.ap eval --checkpoint results/v1/ap/best.pt \
  --val-dir "$V1/eval15" --eval-protocol lolv1 --gt-mean --lpips
```

Real 使用 gated2、α=0.8、关闭 GT-Mean。Syn 使用 plain。跨版本对比时，在 Val 按同一规则选一个 checkpoint，并用它报告完整的 PSNR、SSIM 和 LPIPS。此处的参数表为已测模型体量，完整 LOL 指标待训练。

## 实现来源

AP 复用 M 的 InterLight 字典、C/Y 的区域路由和 W 的亮度门。AQ 复用 R 的 IAFMNet 尺度估计与 Z 的频带分解。AR 复用 O 的 D²R²OSR 融合及 T 的 UCAN 大核分支。新增的是这些模块之间的连接和控制。 [已有方案与来源](README.md) · [R、T 的论文与实现说明](README-r-v.md) · [W、Y、Z 实现说明](README-w-z.md)
