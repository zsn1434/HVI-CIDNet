# HVI-CIDNet 配对低光增强实验

从仓库根目录使用 `python -m experiments.<方案> train` 训练，使用同一模块的 `eval` 评测。每个方案文件都定义了自己的改动。公共训练器在 `experiments/runner.py`。基线沿用原 CIDNet 的解码连线。P 单独检验将 I_LCA5 输出接回主路径的作用。

AE–AI 的七个入口和参数表见[训练与消融指南](README-ae-ai.md)。

AA–AD 的训练入口、来源论文、模型参数、消融和 AD 权重导出见[独立方案指南](README-aa-ad.md)。

W–Z 接续 C、P、R、M 的实验结果，提供四个独立训练入口及对应的验证协议。[W–Z 的改动、参数量、训练和消融](README-w-z.md)

| 模块 | 实际总参数 | 额外参数 | 训练差异 |
|---|---:|---:|---|
| baseline | 1,975,569 | 0 | 同训练器下的原始 CIDNet |
| a | 1,975,619 | 50 | HV_LCA2 输入的 3→8→2 局部门控。同曝光、独立附加噪声的两份输入各用同一 GT 监督，并对输出 HV 加一致性损失 |
| b | 1,975,569 | 0 | 在原有损失上叠加按低光输入强度软分组的 HV 误差 |
| c | 1,977,931 | 2,362 | HV_LCA2 后增加两个 72→8→72 残差专家，用输入图像局部均值、最大强度、色度半径均值和纹理强度路由 |
| d | 1,977,873 | 2,304 | HV_LCA5 后接两个 rank-8 的 1×1 适配器，分别接收重建和 VGG 感知损失的梯度 |
| [f](f.py) | 1,975,890 | 321 | 对输出施加局部曝光曲线。根据输入和预测亮度估计共享 RGB 增益，调整暗部和中间调 |
| [g](g.py) | 1,982,481 | 6,912 | 在解码器 HV_LCA5 后做二维 Haar 分解，让四个子带分别学习色度特征的残差，再重建到空间域 |
| [h](h.py) | 1,975,569 | 0 | 在基础损失上增加局部 RGB 色度比例监督，以参考图亮度降低近黑区域的权重 |
| [i](i.py) | 1,975,569 | 0 | 按每张训练图的局部 RGB 均方误差选困难区域，对误差最高的一部分额外监督 |
| [j](j.py) | 1,999,501 | 23,932 | TPCNet 的 CG_MSA_V，以 I 特征引导 HV 解码 |
| [k](k.py) | 1,998,465 | 22,896 | LAformer 的秩增强全局线性注意力 |
| [l](l.py) | 1,985,506 | 9,937 | MArFE 的傅里叶重参数化线性层与正弦激活 |
| [m](m.py) | 2,030,482 | 54,913 | InterLight 退化字典按图像调制编码器 HV 跳连 |
| [n](n.py) | 1,982,102 | 6,533 | DRWKV 可学习 Scharr 算子增强半分辨率亮度特征 |
| [o](o.py) | 2,000,996 | 25,427 | D²R²OSR 空间与通道双门控，将 HV 信息注入同尺度 I 解码特征 |
| [p](p.py) | 1,975,570 | 1 | 将已有 I_LCA5 输出经可学习系数接回亮度解码器，验证原版连线 |
| [q](q.py) | 1,975,785 | 216 | Multinex 六种解析亮度图补充 I 分支的输入 |

A 的门控初始为恒等混合，C、D 的残差投影初始为零，均可从原版权重开始。A 的 `--noise-std` 在 [0,1] sRGB 张量上定义，初值 0.015。一批裁剪产生两份训练观测，显存紧张时可调整所有对照组的 batch 大小。B 默认用合成的同曝光噪声对估计当前批次各亮度组的 HV 扰动方差，做有界归一化和最差组加权。它测量对附加噪声的敏感度，真实相机噪声标定需要另外采集数据。C 在输入坐标上按 32 像素窗口汇总并插值融合权重。D 只投影细节适配器输出矩阵的感知梯度，使其与保真适配器输出矩阵当前的列空间正交。两个目标都训练主干。

R–V 已可独立训练，分别来自 IAFMNet、C2SSM、UCAN、EIC-LIE 和 TM-BSN。[启动训练与消融](README-r-v.md) · [CVPR 2026 独立主干论文选读](independent-papers-2026.md)

## 亮度分支与退化先验实验

此前 J、K、L 均在同一级 HV 解码特征上增加计算。本轮把问题拆为输入亮度如何表示、编码器如何适应不同场景，以及亮度解码器如何利用边缘和色度。M、N、O、Q 对应四篇 2026 年论文的独立模块。P 是来自本仓库代码检查的对照实验。

| 方案 | 作者方法与本仓库实现 | 最有辨别力的对照与观察 |
|---|---|---|
| **M · InterLight，IJCAI 2026** | 作者源码已经在 CIDNet 上使用 RIN、PAB 和特征记忆。M 保留 RIN 的五层卷积、全局池化、softmax 字典检索与 GELU，输入改用 HVI；16 个 32 维原子生成图像级描述，再用 PAB 的通道缩放与偏置调制四分之一分辨率的 HV 跳连。未引入作者的空间提示、记忆更新或数据扩增。 | 适合检验不同照明场景是否需要不同的色彩恢复映射。若有提升，可在验证时把每图字典系数替换成训练集平均系数，观察逐图条件的贡献。训练使用局部裁剪，整图推理时字典系数可能变化。[论文](https://arxiv.org/abs/2605.19982) · [RIN 原码](https://github.com/House-yuyu/InterLight/blob/590f1e7e9e9d74b163302f40a0a6edf4201d9627/src/net/rin.py) · [PAB 原码](https://github.com/House-yuyu/InterLight/blob/590f1e7e9e9d74b163302f40a0a6edf4201d9627/src/net/pab.py) |
| **N · DRWKV，WACV 2026** | 从 CFEM 提取 Scharr 分支，保留两个可学习方向卷积的 Scharr 初始化、绝对值梯度和、BatchNorm 以及卷积融合。输入取半分辨率的 I 解码特征，经零初始化系数作残差相加。 | 针对灯牌文字、门框等明暗边界断裂。比较 SSIM，并同时查看平坦暗部的噪点是否变粗。要隔离方向先验，可在相同结构上仅改成随机初始化。[论文](https://openaccess.thecvf.com/content/WACV2026/papers/Bai_DRWKV_Focusing_on_Object_Edges_for_Low-Light_Image_Enhancement_WACV_2026_paper.pdf) · [Scharr 原码](https://github.com/Bai-Xuecheng/DRWKV/blob/88cb54313692b279d6f1204b6353d38e36242a4f/Block.py#L373) |
| **O · D²R²OSR，ECCV 2026** | 全景超分模型用 AdaptiveFusion 合并全景与透视特征。空间门由两路特征共同预测，通道门由主路全局池化预测。O 保留两道门及乘法注入，主路改为 I，引导路改为 HV，两路均为半分辨率的 36 通道。 | 与 J 有实质差别。J 用 I 引导色度，O 用色度帮助亮度定位边界。若 O 提升，再将其中一道门固定为 1，可区分空间选择与通道选择的作用。原文全景指标采用 Y 通道、裁边与球面加权，HVI 对照沿用 LOL 的 RGB 协议。[论文](https://eccv.ecva.net/virtual/2026/poster/5315) · [AdaptiveFusion 原码](https://github.com/nichenxingmeng/D2R2OSR/blob/02cac413e9c3f599b9c01579e7cab5bb25d4cd0c/d2r2osr/archs/d2r2osr_arch.py#L245) |
| **P · CIDNet 原始连线对照** | 原版计算 I_LCA5 后，ID_block2 仍取融合前的 I。P 让它接收 `I + gain × (I_LCA5(I, HV) − I)`。初始 gain 为 0，设为 1 时等于完全接入该层。梯度检查确认原版 I_LCA5 参数没有来自输出的梯度，P 在非零 gain 时可收到梯度。 | 与同初始化的 baseline 比较即可。它回答的是这一级已有交互能否提高指标。原始参数量已经计入 I_LCA5；预训练基线中这组参数可能没有通过重建任务得到训练。[CIDNet 解码器](../net/CIDNet.py) |
| **Q · Multinex，CVPR 2026** | 保留作者的六种解析亮度量：RGB 均值、Rec.709、最大通道、中间亮度、YCgCo 的 Y 和 RGB 的 L2 范数。零初始化的 1×1 投影将它们加到 I 分支第一层。 | HVI 的 I 已经包含最大通道。Q 测试其余亮度表示能否帮助区分同峰值、不同颜色的区域。可在同一 216 参数投影下重复输入六份最大通道值，对照额外输入信息的作用。六张图在全分辨率计算，显存开销随图像面积增长。[论文](https://arxiv.org/abs/2604.10359) · [原始定义](https://github.com/albrateanu/multinex/blob/e15790a963935569fff1ba6a8b0d33a791765749/basicsr/models/archs/Multinex_arch.py#L151) |

**原文指标的含义。** Multinex 作者发布的表中，完整 Multinex 为 0.0446 M 参数，在 LOL-v2 Real 和 Synthetic 上分别为 23.04 和 25.04 dB；同表 CIDNet 为 24.11 和 25.71 dB。这支持将六种测量作为补充输入做小规模消融。Q 的最终分数需要用本仓库同协议的基线实测。[作者结果表](https://github.com/albrateanu/multinex/blob/e15790a963935569fff1ba6a8b0d33a791765749/assets/LOL_table.png)

**另两篇值得跟踪的论文。** CVPR 2026 Highlight《Missing No More》使用共享字典与稀疏系数，在红外模态缺失时估计跨模态信息。若 HVI 的色度仅在极暗区丢失，可进一步检验用亮度结构条件化色度字典；原方法有真实红外训练信号和多阶段训练，迁移时需要重新定义监督。IJCV 2026 的 P-PatchDiff 提供渐进调整块大小与步长的方案；当小裁剪训练导致整图曝光不一致时，可另开训练配方对照。它的扩散采样过程需要额外推理预算。以上两项是研究备选。[Missing No More](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Missing_No_More_Dictionary-Guided_Cross-Modal_Image_Fusion_under_Missing_Infrared_CVPR_2026_paper.html) · [P-PatchDiff](https://doi.org/10.1007/s11263-026-02995-w)

优先比较 P、M 与同协议基线；若现有输出主要损失纹理，则把 N 提前。五组均可单独运行。以下命令假定已经准备好 LOL-v2 Real 数据以及本项目环境。选模使用从训练集划出的独立验证集，固定初始化、划分、增益和训练预算；第一轮胜出的方案可用多个种子复核。

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in p m n o q; do
  python -m "experiments.${variant}" train \
    --train-dir "$DATA/Train" --out "./results/real/${variant}" \
    --epochs 1000 --batch-size 8 --crop 256 --lr 0.0001 --seed 42 || break
done
```

上例按固定轮数训练，最终评测使用 `latest.pt`。若要按验证集选择 `best.pt`，给每组训练都加 `--val-dir "$DATA/Val" --eval-every 10`，再用独立 Test 做最终评测。以下命令以 M 为例。

```bash
python -m experiments.m eval --val-dir "$DATA/Test" --checkpoint ./results/real/m/latest.pt --lpips
```

## 论文模块移植

J、K、L 均接在 `HV_LCA5` 后，使用同一套 RGB/HVI 配对损失。每次只启用一个新分支。复用现有基线时请保持数据划分、初始权重、训练轮数和评测增益一致。

| 方案与论文 | 保留的原模块运算 | 适配到 HVI 的改动 |
|---|---|---|
| **J · TPCNet，ECCV 2026，低光增强** | `CG_MSA_V` 对两路特征分别作对角线配对下采样，交叉构造 Q/K，融合两路 V，做通道注意力，经 PixelShuffle 恢复分辨率，并加输入投影 | 两个输入分别取 HV 解码特征和同尺度 I 特征，各从 72 压到 24 通道，使用四头注意力，再投回 72 通道。输出投影以零初始化。适合检验亮度引导对暗部色彩恢复的作用。[论文](https://eccv.ecva.net/virtual/2026/poster/4351) · [作者源码 CG_MSA_V](https://github.com/2020shijingyi/TPCNet/blob/123b09fbc46dc5be8bce296f13f0efdec9d5589d/arch/TPCNet_arch.py#L280) |
| **K · LAformer，ECCV 2026，图像复原** | 全局 `LinearAttention` 使用 ELU+1 核、按空间位置归一化的线性注意力和 V 分支的 5×5 深度卷积，两路相加后投影 | 在同一 HV 特征上用 72 通道、四个头运行，分母下限取 1e-6，输出投影以零初始化。适合检验远距离上下文对局部增强是否有帮助。[论文](https://eccv.ecva.net/virtual/2026/poster/5618) · [作者源码 LinearAttention](https://github.com/shallowdream204/LAformer/blob/b4a3dc705874631fa180105b38f3478226d9933f/basicsr/models/archs/laformerv2_arch.py#L92) |
| **L · MArFE，ECCV 2026，多对比度 MRI 超分辨率** | 取 Q 分支中的“点卷积 → 正弦 → `Fourier_reparam_linear`”，按固定余弦基与可学习系数的乘积生成逐像素线性层权重 | 使用 72 通道，将原配置的 128 低频、128 高频和 32 相位缩为 8、8 和 4，相同的 0.05 基函数缩放。加零初始化的残差系数。这里的频率基沿**特征通道轴**展开。其参数量含 9,937 个新增可训练参数，另有 4,608 个固定基值随权重保存。[论文](https://eccv.ecva.net/virtual/2026/poster/3657) · [作者源码 Fourier_reparam_linear](https://github.com/zhiwen-shi/MArFE/blob/db507eea2ed9666b97d8c492a391cca964ec9496/model/dualref.py#L36) |

**先试 J 或 K。** J 直接回答 I 分支提供的信息是否改善 HV 恢复，K 检验单路 HV 特征的空间上下文。L 可用于探索固定频率基对通道映射的作用。若 L 相对基线有提升，可进一步用同位置普通线性层对照，区分额外残差分支和傅里叶参数化的贡献。

以下命令从仓库根目录运行这三组；数据和环境准备见下文。已有同协议基线可直接复用；如需微调，在三组和基线的训练命令中均加入同一 `--init` 权重。

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in j k l; do
  python -m "experiments.${variant}" train \
    --train-dir "$DATA/Train" --out "./results/real/${variant}" \
    --epochs 1000 --batch-size 8 --crop 256 --lr 0.0001 --seed 42 || break
done
```

三种算子已在相同尺寸、相同权重的随机 FP32 输入上分别与作者实现对照，检查了前向输出和输入梯度。三个完整实验均在 CPU 合成配对图上完成两轮训练和一次独立评测；真实 LOL 排名有待实测。

**前两批方案的出处。** A 的局部门控与独立噪声一致性是本仓库的设计。B 对应 BBRD 的亮度分组与风险加权思路，改用 HV 误差、等距软分组和合成扰动估计。C 借鉴 RegionFuse 的区域专家思想，采用四种局部统计量进行软路由。D 借鉴 FreqOrtho-SR 的正交专家思想，将投影用于两个适配器的指定梯度。F–I 的设计动机与论文链接见下一节。这些实验的指标需分别在 HVI 上测量。[BBRD](https://eccv.ecva.net/virtual/2026/poster/5112) · [RegionFuse](https://github.com/search?q=RegionFuse+Region-Adaptive+Pixel+Distribution&type=repositories) · [FreqOrtho-SR](https://eccv.ecva.net/virtual/2026/poster/4909)

## 第二批实验怎么选

| 观察到的基线误差 | 优先试 | 实现与对照 |
|---|---|---|
| 暗部或中间调整体偏暗，高光已经接近参考图 | F | 根据 16×16 图块的输入和预测亮度估计平滑系数。三个通道共享增益，对纯黑和峰值为 1 的像素固定端点。曲线头以零初始化，可从同一原版权重与基线对比。 |
| 暗部边缘有彩色纹理，或恢复平滑区域时损失了边缘 | G | 对解码器 HV 特征的一个低频和三个方向细节子带分别做 72→12→72 通道变换。末层以零初始化。以基线作对照，同时检查 SSIM 和暗部细节。 |
| 曝光接近参考图，但中性物体仍明显偏色 | H | 比较 4×4 区域的平均 RGB 经通道和归一化后的比例。`--chroma-weight 0.1` 控制该项，设为 0 可作对照。 |
| 局部大误差集中在少数区域，拉低 PSNR | I | 每张裁剪按 8×8 块计算 RGB MSE，默认给误差最高的 25% 额外权重。对比 `--hard-fraction 0.25` 和 `--hard-fraction 1` 可检验挑选困难块的作用，固定 `--hard-weight 0.5`。若出现错配，优先检查被选中区域的参考图。 |

**2026 年论文与设计取舍。** F 借鉴 HDR 重建用增益图控制色调映射的思路，这里用单幅低光输入学习有界曲线。G 借鉴跨任务小波建模，将子带处理放进现有 HV 解码器。两组都只用 LOL 的成对图像训练。[HDR 增益图，ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4299) · [SFKD 小波知识蒸馏，ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/5232)

H 参考相机颜色校正中显式建模色度的方向，这里用配对参考图监督局部 RGB 比例。I 参考三维显微分割中的困难图块训练思路，这里用增强输出的 RGB MSE 排序。表中参数量和训练命令对应本仓库的适配实现。[相机色度校正，ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/3428) · [显微分割困难图块，ECCV 2026](https://eccv.ecva.net/virtual/2026/poster/4987)

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

从头训练的示例依次运行同一协议下的全部方案及基线。如果已经完成首批实验，训练和评测的循环列表都只填 `f g h i` 即可。分别启动单组时，只保留对应字母。也可以给每组加相同的 `--init /path/to/original_cidnet.pth` 进行微调。`--init` 接受原仓库的纯 state_dict。

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in baseline a b c d f g h i j k l m n o p q; do
  python -m "experiments.${variant}" train \
    --train-dir "$DATA/Train" --out "./results/real/${variant}" \
    --epochs 1000 --batch-size 8 --crop 256 --lr 0.0001 --seed 42 || break
done
```

训练集、验证集和测试集的每个目录包含 `Low/Normal` 或 `low/high`。按文件标识配对，同名或 `low00690.png` 与 `normal00690.png` 均可。改变 `DATA` 即可测试 Synthetic，使用单独的输出目录。LOL-v1 可分别指定 `our485` 和 `eval15`。

每个 epoch 原子写入 `latest.pt` 并记录 `metrics.jsonl`。如有从训练数据留出的独立验证集，可添加 `--val-dir /path/to/Val --eval-every 10` 记录验证指标并保存最佳 PSNR 的 `best.pt`。中断后用原训练参数、相同目录和 `--resume` 恢复，去掉 `--init`。保持原定总 epoch 数以延续学习率曲线。

```bash
DATA=./datasets/LOLv2/Real_captured
for variant in baseline a b c d f g h i j k l m n o p q; do
  python -m "experiments.${variant}" eval \
    --checkpoint "./results/real/${variant}/latest.pt" \
    --val-dir "$DATA/Test" --lpips || break
done
```

评测逐图计算 RGB 8-bit PSNR 和 SSIM 再取平均，`--lpips` 额外计算 AlexNet LPIPS。使用固定输出增益 `--alpha 1`，关闭 GT-Mean。图像会补齐到 8 的倍数，评测前裁回原尺寸。若对照原仓库中采用 0.8 增益的 Real 配方，可另列所有方案在 `--alpha 0.8` 下的结果。各组主比较使用同一权重初始化方式、固定增益、训练集和指标实现。

## 训练设置与单项对照

首批四个方案的实现分别位于 [a.py](a.py)、[b.py](b.py)、[c.py](c.py) 和 [d.py](d.py)。主干把编码器的 `HV_LCA2` 调用封装为 `encode_hv_level2`，供 A 和 C 改写，把解码器的 `HV_LCA5` 调用封装为 `decode_hv_level2`，供 D 改写。B 直接使用原版模型，只在训练时计算分组损失。[主干实现](../net/CIDNet.py)

共享配方使用 Adam、余弦退火、反传后梯度裁剪 0.01，以及原仓库的 RGB/HVI 双域 L1、SSIM、边缘和 VGG19 特征损失。该训练器独立于原版 train.py，学习率曲线采用无预热的余弦退火。首次运行感知损失或 LPIPS 需要可用的预训练权重。以新训练器的 baseline 作为归因对照。

- A 可用 `--consistency-weight 0` 对照额外的一致性监督。两组保持相同的 `--noise-std`、训练步数及 batch 大小。
- B 可用 `--band-mode uniform` 检查各组等权结果。默认 `calibrated` 使用扰动方差和动态权重。两组可固定 `--bands 5 --band-weight 0.1`。
- C 只有局部专家与路由器的新增参数。比较 baseline 和 C，检查同时有高光和阴影的图像。
- D 可用 `--disable-projection` 在相同双适配器与损失分工下关闭梯度投影。`--perceptual-weight` 默认 0.01，应为正值。

文中的参数量对应默认模型。示例超参数是实验起点，指标以真实数据训练与评测结果为准。
