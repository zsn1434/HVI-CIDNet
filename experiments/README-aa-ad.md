# 训练 HVI 的四个独立方案 AA–AD

AA、AB、AC、AD 分别从原版 CIDNet 出发。各组使用同一训练器，支持从原版权重初始化、按验证 PSNR 选模和断点续训。下表是默认模型实例的参数总量。完整 LOL 实验的 PSNR、SSIM、LPIPS 待测。

| 入口 | 修改位置与目的 | 训练参数 | 推理模型参数 | 优先验证 |
|---|---|---:|---:|---|
| [aa](aa.py) | 整图缩略图预测正常曝光分布，按裁剪位置指导半分辨率 I 解码 | 1,978,443 | 1,978,443 | LOLv2-Real |
| [ab](ab.py) | 在 HVD_block2 的 2× 上采样处，用 I 引导有方向的高斯核传播 HV 特征 | 1,976,839 | 1,976,839 | LOLv2-Syn |
| [ac](ac.py) | 在 HVI 逆变换前，用空间位置和预测亮度查询局部 HV 仿射变换 | 1,979,657 | 1,979,657 | LOLv2-Real |
| [ad](ad.py) | 从低光图的互斥像素构造下一尺度 RGB 预测任务，训练共享 HV 浅层 | 1,976,013 | 1,975,569（导出后） | LOLv2-Real |

## 启动训练

在仓库根目录运行。已配置的云环境使用 `source /workspace/.hvi-venv/bin/activate` 和 `export TORCH_HOME=/workspace/.cache/hvi-torch`。其他机器请使用装有项目依赖的环境。默认训练感知损失需要 VGG19 权重，评测 LPIPS 需要 AlexNet 权重。

每个数据集使用固定且互不重叠的 Train、Val、Test。若只有官方 Train 和 Test，先从 Train 留出 Val。每个划分内部包含 `Low/` 与 `Normal/`，也支持 `low/` 与 `high/`。低光和正常曝光图应同尺寸且文件编号对应，例如 `Train/Low/Low0001.png` 与 `Train/Normal/Normal0001.png`。LOLv1 可以通过命令行分别指定配对目录。

在 Bash 中先定义数据路径和训练函数，再选任意一行训练。下例给 Real 验证启用 gated2、α=0.8，给 Syn 验证使用 plain。模型默认在可用的 CUDA 设备上训练。

```bash
REAL=./datasets/LOLv2/Real_captured
SYN=./datasets/LOLv2/Synthetic
common=(--epochs 1000 --batch-size 8 --crop 256 --lr 0.0001 --seed 42 --eval-every 10)

train_hvi() {
  local variant="$1" data="$2" protocol="$3" out="$4"
  shift 4
  python -m "experiments.$variant" train \
    --train-dir "$data/Train" --val-dir "$data/Val" --eval-protocol "$protocol" \
    --out "$out" "${common[@]}" "$@"
}

train_hvi aa "$REAL" lolv2-real results/real/aa-aligned
train_hvi ab "$SYN"  plain      results/syn/ab-offset
train_hvi ac "$REAL" lolv2-real results/real/ac-full
train_hvi ad "$REAL" lolv2-real results/real/ad-heldout
```

`--crop` 为 8 的倍数，且不超过训练图短边。若比较基线，将首个参数换成 `baseline`，单独指定输出目录。统一微调时给各组传同一份 `--init /path/to/original_cidnet.pth`（原版纯 state_dict）。若 AB 在大图上显存紧张，可降低 batch size，并用同一训练配置重跑对照组。

## 修改内容与消融

### AA · 整图曝光监督

作者 P-PatchDiff 用整张正常曝光图的 RGB 均值图与最大值图监督预测器。AA 取这一监督方式。数据加载器同步翻转配对整图并换算裁剪框；先在 GT 原分辨率计算两种统计量，再缩至 64×64。3→16→16→2 预测头读取低光缩略图，按裁剪框采样两张曝光图。无偏置、零初始化的 2→36 投影将它们注入 ID_block2 后。辅助 L1 默认权重 0.1，可用 `--exposure-weight` 调整。整图推理只从输入低光 RGB 计算缩略图。[IJCV 2026 论文](https://doi.org/10.1007/s11263-026-02995-w) · [作者训练路径](https://github.com/RuoyuGuo/P-PatchDiff/blob/9c90499/PPatchDiff/pydiff/archs/pgcddpm_arch.py#L479)

```bash
train_hvi aa "$REAL" lolv2-real results/real/aa-crop    --context-mode crop
train_hvi aa "$REAL" lolv2-real results/real/aa-shifted --context-mode shifted
```

默认 `aligned` 使用整图和正确位置。`crop` 只从当前裁剪生成缩略图和辅助 GT；整图评测时，当前输入就是整图。`shifted` 保留整图监督，把注入的预测图沿宽、高各循环平移 32 个缩略图像素，训练和评测使用相同错位规则。训练图尺寸大于裁剪尺寸时，`aligned` 与 `crop` 才能比较上下文范围。

### AB · 高斯色度上采样

GaussM2ASR 用高斯几何描述引导超分。AB 只改 HVD_block2 的插值步骤。72 通道亮度特征与同尺度的平滑亮度梯度驱动 73→16→5 几何头，预测中心的双轴偏移、两条轴长和方向角。每个源位置在输出网格的 4×4 邻域写入 HV 特征，并按每个目标像素实际收到的权重归一化。轴长限定为 0.5–2 个输出像素，中心偏移最多半个输出像素。零初始化系数控制相对原双线性的更新。实现使用 PyTorch fold，可在 CPU 或 GPU 运行。[CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Yan_Adaptive_Anisotropic_Gaussian_Splatting_for_Multi-contrast_MRI_Arbitrary-Scale_Super-Resolution_with_CVPR_2026_paper.pdf) · [作者几何预测器](https://github.com/Qiuhai-CV/GaussM2ASR/blob/2f0acd5/basicsr/archs/aggp_arch.py)

```bash
for kernel in bilinear circular ellipse; do
  train_hvi ab "$SYN" plain "results/syn/ab-$kernel" --kernel-mode "$kernel" || break
done
```

`bilinear` 走原上采样；`circular` 学半径且固定中心；`ellipse` 学两条轴长与方向且固定中心；默认 `offset` 进一步学习中心偏移。比较圆核与椭圆核时可同时看细彩线和近似等亮度的颜色交界，检查 I 梯度弱时是否串色。

### AC · 按亮度查询的色度网格

Hist2Style 用双边网格做摄影色彩变换。AC 从四分之一分辨率的 72 通道 HV 解码特征生成网格，空间最多 16×16，亮度方向 8 层。每格预测 2×2 矩阵残差、两项偏置和一个正权重。在原分辨率按位置与预测 I 三线性插值，对加权系数除以插值权重，再变换二维 HV。初始为恒等变换，I 直接传给 HVI 逆变换。网络只读取低光输入。[CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Galor_Hist2Style_Histogram-Guided_Stylization_with_Bilateral_Grids_CVPR_2026_paper.pdf) · [作者网格查询](https://github.com/dgalor/hist2style/blob/a3ef9ba/demo/model.py#L51)

```bash
train_hvi ac "$REAL" lolv2-real results/real/ac-fixed --guide fixed
train_hvi ac "$REAL" lolv2-real results/real/ac-diag  --diagonal
```

`--guide fixed` 始终查询亮度轴中点，保留位置差异。`--diagonal` 关闭 H、V 之间的两个交叉项，仍学习缩放与偏置。两选项可组合。整图指标之外，可查看中性物体偏色和高饱和区域色相误差。

### AD · 同曝光下一尺度辅助任务

NSP 从真实含噪照片的互斥像素构造跨尺度监督。AD 在每个 5×5 原始低光 RGB 块中随机留出连续 2×2 目标，从其余 21 个位置均匀抽一个输入像素，再分别拼成小输入图和双倍尺寸的目标图。整批三个通道共享各自像素的坐标。小输入图经过共享 HVIT 和 HVE_block0，由 36→12 投影与 PixelShuffle(2) 预测 RGB。每侧不满 5 像素的尾块只在辅助任务中裁掉。主任务继续使用完整训练裁剪与正常曝光 GT。辅助 L1 默认权重 0.1。[CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Shan_Next-Scale_Prediction_A_Self-Supervised_Approach_for_Real-World_Image_Denoising_CVPR_2026_paper.pdf) · [作者采样代码](https://github.com/XLearning-SCU/2026-CVPR-NSP/blob/e95b6df/train_nsp_dbsn.py#L141)

```bash
train_hvi ad "$REAL" lolv2-real results/real/ad-visible --aux-target visible
train_hvi ad "$REAL" lolv2-real results/real/ad-noaux   --nsp-weight 0
```

默认 `heldout` 预测留出的 2×2 RGB。`visible` 将可见输入重复为 2×2 目标，使用相同的预测头和辅助计算量。`--nsp-weight 0` 仅训练主任务。

## 评测、续训与导出

按相同口径对照已有单方案结果。各变体结构的消融模式随 checkpoint 保存。单独运行 eval 时显式指定评测协议和 LPIPS。

```bash
python -m experiments.aa eval --val-dir "$REAL/Test" --checkpoint results/real/aa-aligned/best.pt --eval-protocol lolv2-real --lpips
python -m experiments.ab eval --val-dir "$SYN/Test"  --checkpoint results/syn/ab-offset/best.pt  --eval-protocol plain --lpips
python -m experiments.ac eval --val-dir "$REAL/Test" --checkpoint results/real/ac-full/best.pt    --eval-protocol lolv2-real --lpips
python -m experiments.ad eval --val-dir "$REAL/Test" --checkpoint results/real/ad-heldout/best.pt --eval-protocol lolv2-real --lpips
```

LOLv1 对应 `--eval-protocol lolv1 --gt-mean --lpips`。额外去掉 `--gt-mean` 可评估模型自身的曝光预测。验证集选模时也传入计划使用的协议。如果训练省略 `--val-dir`，请评测 `latest.pt`。训练日志在各自的 `metrics.jsonl`；`best.pt` 按验证 PSNR 选取，`latest.pt` 供中断续训。

```bash
train_hvi aa "$REAL" lolv2-real results/real/aa-aligned --resume

python -m experiments.ad export --checkpoint results/real/ad-heldout/best.pt \
  --out results/real/ad-heldout/cidnet.pth
```

续训沿用原命令和所有训练选项，去掉可能使用过的 `--init` 并加 `--resume`。AD 导出的纯 state_dict 可以直接由原版 CIDNet 严格加载。仓库实验入口的 eval 读取 `best.pt` 或 `latest.pt`。运行数值回归检查可使用 `python -m unittest discover -s tests -p 'test_aa_ad.py' -v`。
