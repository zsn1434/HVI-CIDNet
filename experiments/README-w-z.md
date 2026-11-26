# 训练 HVI 的四个融合方案 W–Z

这四组实验来自 C、P、R、M 在 LOL 上的实测差异。Real 优先跑 W，对照 P 的 24.10 dB；Synthetic 优先跑 X，对照 C 的 26.31 dB。Y 检查逐图曝光信息能否帮助区域路由，Z 用两个系数检验 P 的收益来自哪一段空间频率。以下参数量按默认模型实例统计，PSNR 参照值来自已有单方案实验。W–Z 的正式指标需训练后填写。

| 版本 | 改动位置和训练目标 | 模型总参数 | 相对 CIDNet 新增 |
|---|---|---:|---:|
| [W](w.py) | C 路由器共享 8 通道区域特征，同时控制 HV 专家和 P 的亮度更新 | 1,978,028 | 2,459 |
| [X](x.py) | R 从 I 特征估计尺度，作为 C 路由器的第五个区域输入；训练时加高斯编码损失 | 1,984,275 | 8,706 |
| [Y](y.py) | M 的 32 维字典向量经线性层调节 C 的 8 通道路由隐藏层 | 2,028,619 | 53,050 |
| [Z](z.py) | 将 P 的跨分支更新拆成平滑和细节两项，各学一个标量 | 1,975,571 | 2 |

基线为 1,975,569 个参数。表中计入整份 checkpoint 的模型权重。X 的均值估计分支有 2,592 个参数，只在训练时计算辅助损失。默认学习尺度模式推理用到的模型参数按扣除这条分支计为 1,981,683。

## 准备数据

在仓库根目录运行。已配置的云环境可以先执行下面两行，其他机器使用装有本仓库依赖的 Python 环境。训练默认使用 VGG19 感知损失，评测加上 `--lpips` 时还需要 AlexNet 权重。

```bash
source /workspace/.hvi-venv/bin/activate
export TORCH_HOME=/workspace/.cache/hvi-torch
```

每个配对目录内放 `Low/` 和 `Normal/`，也支持 `low/` 和 `high/`。示例假定 Real 和 Synthetic 均有 `Train/`、`Val/`、`Test/`。请从原训练集划出固定 Val，并移出训练样本；LOLv1 同理，可把拆出的两个目录命名为 `Train/`、`Val/`，保留官方 `eval15/` 作测试。执行下列 Bash 代码设置数据路径和训练公共参数。

```bash
REAL=./datasets/LOLv2/Real_captured
SYN=./datasets/LOLv2/Synthetic
V1=./datasets/LOLdataset
common=(--epochs 1000 --batch-size 8 --crop 256 --lr 0.0001 --seed 42 --eval-every 10)

train_variant() {
  local variant="$1" data="$2" out="$3"
  shift 3
  python -m "experiments.$variant" train \
    --train-dir "$data/Train" --val-dir "$data/Val" --out "$out" \
    "${common[@]}" "$@"
}
```

## 开始训练

以下每一行独立启动一个实验。若复用你已有的训练预算，把 `common` 中的轮数、裁剪、批量和学习率统一改成之前的配置。

```bash
train_variant w "$REAL" results/real/w-region --eval-protocol lolv2-real
train_variant x "$SYN"  results/syn/x-learned --eval-protocol plain
train_variant y "$REAL" results/real/y-own    --eval-protocol lolv2-real
train_variant z "$REAL" results/real/z-split  --eval-protocol lolv2-real
```

在 LOLv1 上验证 Y，可运行 `train_variant y "$V1" results/v1/y-own --eval-protocol lolv1 --gt-mean`。独立从头训练是默认行为。统一微调时，各组加同一个 `--init /path/to/cidnet.pth`，传入原版 CIDNet 的纯 state_dict。中断后重跑原命令，去掉 `--init` 并加上 `--resume`，沿用原数据、总轮数、评测协议、消融模式和输出路径。程序从 `latest.pt` 恢复；`best.pt` 按验证 PSNR 保存，`metrics.jsonl` 记录逐轮指标。

## 每组如何做消融

**W · 区域指导亮度解码。** 在 `HV_LCA2` 的跳连上沿用 C 的双专家。从同一张区域网格取 8 个隐藏通道，插值到解码器分辨率，与注意力前 I 特征和跨分支更新 ΔI 的通道平均绝对值拼接。10→8→1 门控头输出逐位置系数，乘到 ΔI 再送入 `ID_block2`。末层零初始化。三个模式都保留 C 的专家；`decoder` 将门控头的区域输入置零，`scalar` 仅启用标量偏置。重点比较 `region` 和 `decoder`。[C](c.py) · [P](p.py)

```bash
train_variant w "$REAL" results/real/w-decoder --eval-protocol lolv2-real --gate-mode decoder
train_variant w "$REAL" results/real/w-scalar  --eval-protocol lolv2-real --gate-mode scalar
```

**X · 尺度指导专家选择。** 在四分之一分辨率预测正尺度 σ。对 log σ 取通道均值，再按 8×8 窗口池化，与 C 在原图坐标的 32×32 区域对齐。各 X 组都使用权重 0.0001 的高斯辅助损失，以便隔离路由器实际收到的信号。`gradient` 输入亮度梯度，`zero` 将第五通道置零，`shuffled` 在每张图内打乱尺度网格位置。训练时随机打乱，评测时固定排列；观察打乱效应的裁剪应覆盖多个区域。原 C 用于单独对照辅助训练与新增通道的合计影响。[R 的高斯估计](r.py) · [IAFMNet 论文与移植说明](independent-papers-2026.md)

```bash
for prior in gradient zero shuffled; do
  train_variant x "$SYN" "results/syn/x-$prior" --eval-protocol plain --prior "$prior" || break
done
```

**Y · 全局字典指导局部路由。** 从输入 HVI 检索 16 个字典原子，得到 32 维图像描述。32→16 投影为路由隐藏层生成逐通道缩放和偏置，影响每块区域两个专家的混合权重。训练用本图描述，验证时可在同一权重上比较三种来源。`mean` 使用所给验证集的平均描述，`other` 按已排序的配对列表循环取下一张图，至少需要两张。两种替换仅用于验证集诊断。正式测试用默认 `own`。[M 的字典](m.py) · [InterLight](https://arxiv.org/abs/2605.19982)

```bash
for source in own mean other; do
  python -m experiments.y eval --val-dir "$REAL/Val" \
    --checkpoint results/real/y-own/best.pt --eval-protocol lolv2-real \
    --descriptor "$source" --lpips || break
done
```

**Z · 分频亮度更新。** 从 P 的 ΔI 取固定 5×5 均值滤波结果作为低频，边界用复制填充；高频取 ΔI 与低频之差。默认两个标量从零学习。`low`、`high` 各启用一路。`tied` 调用 P 的单系数公式，可以直接作为同配置的 P 对照。[P 的亮度连线](p.py)

```bash
for band in tied low high; do
  train_variant z "$REAL" "results/real/z-$band" --eval-protocol lolv2-real --band-mode "$band" || break
done
```

## 按现有 LOL 口径测试

训练时指定的协议用于验证选模。单独运行 `eval` 时再次传入目标协议。W、X、Z 的模型消融配置会从 checkpoint 自动恢复。评测输出逐图平均的 RGB PSNR、SSIM 和可选 LPIPS，协议开关也可用于同训练器的 baseline、C、P、R、M。

| 数据集 | 参数 | 计算方法 |
|---|---|---|
| LOLv1 | `--eval-protocol lolv1 --gt-mean --lpips` | HVI 逆变换开启 gated，饱和度系数 1.3；按每张 GT 灰度均值校正 8 位预测，再计算指标。去掉 `--gt-mean` 可观察实际曝光误差 |
| LOLv2-Real | `--eval-protocol lolv2-real --lpips` | 开启 gated2，内部 α 固定为 0.8 |
| LOLv2-Syn | `--eval-protocol plain --lpips` | 默认逆变换、增益 1 |

```bash
python -m experiments.w eval --val-dir "$REAL/Test" --checkpoint results/real/w-region/best.pt --eval-protocol lolv2-real --lpips
python -m experiments.x eval --val-dir "$SYN/Test"  --checkpoint results/syn/x-learned/best.pt --eval-protocol plain --lpips
python -m experiments.y eval --val-dir "$V1/eval15" --checkpoint results/v1/y-own/best.pt --eval-protocol lolv1 --gt-mean --lpips
python -m experiments.z eval --val-dir "$REAL/Test" --checkpoint results/real/z-split/best.pt --eval-protocol lolv2-real --lpips
```

评测先将非 8 倍数的尺寸反射填充，输出裁回原尺寸。GT-Mean 遇到预测灰度均值为零时保留原预测。正式比较请统一划分、选模频率、推理设置和 seed，并记录最佳轮次。[官方推理](../eval.py) · [官方指标](../measure.py)
