# AE–AI 训练、评测和消融

七个入口分别从原版 CIDNet 出发。建议先在 LOLv2-Real 跑 AF 加宽版，在 LOLv2-Syn 跑 AE 标准版。下表为默认配置实例化后的模型参数量，原版有 1,975,569 个参数。

| 入口 | 修改内容和接入位置 | 新增参数 | 总参数 |
|---|---|---:|---:|
| [ae](ae.py) | HV 编码器前两级下采样。固定低通后按每组六通道对四个采样位置加权，用滤波前后之差学习细节补偿。两项分别由零初始化系数控制 | 187,204 | 2,162,773 |
| [ae_wide](ae_wide.py) | AE 内部宽度由 48/96 调至 64/128 | 295,684 | 2,271,253 |
| [af](af.py) | PHVIT 前联合输入 HVI 和预测残差，在半分辨率用两个稠密块校正，再以一个全分辨率稠密块细调。两级输出层零初始化 | 135,510 | 2,111,079 |
| [af_wide](af_wide.py) | AF 半分辨率头增至 48 通道、增长通道 24，全分辨率头仍为 16 通道 | 285,526 | 2,261,095 |
| [ag](ag.py) | 两分支同时从各自特征生成 Q/V，从对方生成 K。1/8 尺度沿整行整列检索，1/4 解码层沿最长 16 个位置的行列片段检索 | 138,244 | 2,113,813 |
| [ah](ah.py) | 部分训练批次复用编码结果，在 1/4 尺度互补遮挡两分支的局部特征。完整输出使用原目标，受扰输出在遮挡区域计算 RGB L1，推理走正常路径 | 0 | 1,975,569 |
| [ai](ai.py) | 在半分辨率分别读取 I/HV 梯度，第一层混合连续逻辑关系，第二层结合局部上下文修正 HV 主路径 | 5,856 | 1,981,425 |

这些是针对 HVI 的机制迁移。AE 借鉴三维相位聚合，AF 借鉴超分中的残差校正，AG 借鉴屏下相机复原的行列检索，AH 借鉴多模态干预训练，AI 借鉴复原中的可微逻辑门。[CROWn](https://github.com/IMOP-lab/CROWn) · [FiDeSR](https://github.com/Ar0Kim/FiDeSR) · [UCMNet](https://github.com/kdhRick2222/UCMNet) · [ISFuse](https://arxiv.org/abs/2603.23272) · [LogicIR](https://github.com/jimmy9704/LogicIR)

## 开始训练

在仓库根目录、装有项目依赖的 Python 环境下运行。默认感知损失需要 VGG19 权重，最终 LPIPS 评测需要 lpips 和 AlexNet 权重。本项目云环境可用以下设置。

```bash
source /workspace/.hvi-venv/bin/activate
export TORCH_HOME=/workspace/.cache/hvi-torch
```

将下面的路径改为自己的数据目录。每个数据集使用 Train、Val、Test 三个互不重叠的配对划分，每个划分内含 Low/Normal 或 low/high。若只有官方 Train、Test，可从 Train 固定留出 Val。下面是 Bash 函数，定义一次后任选实验运行。

```bash
REAL=./datasets/LOLv2/Real_captured
SYN=./datasets/LOLv2/Synthetic
train_hvi() {
  local version="$1" data="$2" protocol="$3" out="$4"
  shift 4
  python -m "experiments.$version" train \
    --train-dir "$data/Train" --val-dir "$data/Val" --out "$out" \
    --eval-protocol "$protocol" --epochs 1000 --batch-size 8 --crop 256 \
    --lr 0.0001 --seed 42 --eval-every 10 "$@"
}

train_hvi af_wide "$REAL" lolv2-real results/real/af_wide
train_hvi ae      "$SYN"  plain      results/syn/ae
train_hvi ae_wide "$SYN"  plain      results/syn/ae_wide
train_hvi af      "$REAL" lolv2-real results/real/af
train_hvi ag      "$REAL" lolv2-real results/real/ag
train_hvi ah      "$REAL" lolv2-real results/real/ah
train_hvi ai      "$SYN"  plain      results/syn/ai
```

从原版权重启动可追加 `--init /path/to/cidnet.pth`（纯 state_dict）。续训时使用相同入口、参数和目录，追加 `--resume`，省略 `--init`。训练保存 latest.pt 和按验证 PSNR 选出的 best.pt。显存紧张可减小 batch size，对照组保持相同设置。

## 消融实验

```bash
train_hvi ae      "$SYN"  plain      results/syn/ae_phase_only --no-detail
train_hvi af_wide "$REAL" lolv2-real results/real/af_residual_only --residual-only
train_hvi ag      "$REAL" lolv2-real results/real/ag_self --self-only
train_hvi ah      "$REAL" lolv2-real results/real/ah_dropout --mask-mode dropout
train_hvi ai      "$SYN"  plain      results/syn/ai_bilinear --bilinear
```

AE 关闭细节补偿，AF 将两级头看到的输入 HVI 置零，AG 改用本分支生成 K。AH 对照为两分支独立的随机位置遮挡，默认每批触发概率 0.5，遮挡区辅助损失权重 0.1，可分别用 `--intervention-prob` 和 `--mask-weight` 修改。AI 对照使用自由双线性系数，每组读取四项。AE 和 AI 的对照保留完整参数张量，其中部分参数不参与计算。原版可用同一个函数将版本换为 baseline，另选输出目录。

## 最终评测

使用训练该权重的同一模块名。评测从 checkpoint 恢复消融配置，评测协议在命令行指定。

```bash
python -m experiments.af_wide eval --checkpoint results/real/af_wide/best.pt \
  --val-dir "$REAL/Test" --eval-protocol lolv2-real --lpips
python -m experiments.ae eval --checkpoint results/syn/ae/best.pt \
  --val-dir "$SYN/Test" --eval-protocol plain --lpips
```

Real 使用 gated2、内部 α=0.8，Syn 使用 plain。LOLv1 的官方对照使用 `--eval-protocol lolv1 --gt-mean --lpips`。同数据集固定划分、训练预算和选模频率，并记录 PSNR、SSIM、LPIPS 和选中轮次。完整 LOL 结果待实测。
