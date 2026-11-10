> 说明：这是内部  复现协议的中文原始记录（historical, for provenance only）。
> 英文 `docs/PROTOCOL.md`、`docs/ASSUMPTIONS.md`、`docs/METRICS.md`、最终
> `stage5_full` 结果及 `results/summary/` 审计为准；文中部分历史变体描述
> 仅作过程记录。**请勿用本文件核对论文设置或结果；以英文文档为准。**
>
> This is the original internal protocol, kept for provenance only.  The
> English `PROTOCOL.md`, `ASSUMPTIONS.md`, `METRICS.md`, the released
> `stage5_full` results and `results/summary/` audit are authoritative.
> Some passages describe historical variants; do not cite them as paper
> settings.

# ComGEN  复现实验协议

本协议依据 ComGEN 论文（ICRA 2025），将论文明确要求与补充实现假设分开记录。目标是复现伪异常生成、掩码生成和检测完整流程，并客观比较 Table II 的结果；不承诺达到 91.2%。实际运行参数、数据清单、检查点及机器可读 manifest 是具体实验的依据。

## 指标与实验状态

论文摘要和 Table II Mean 报告 91.2%。每类将真实测试集的 logical_anomalies 和 structural_anomalies 一并视为异常，与 good 比较。Table II 列出的五类 AUROC 为 breakfast_box 87.2%、juice_bottle 99.1%、pushpins 92.9%、screw_bag 79.8%、splicing_connectors 95.2%；这五个已四舍五入的数字算术平均为 **90.84%**，与表中 Mean 91.2% 不一致。按测试图数量加权也约为 90.81%，无法从该表还原 91.2%。因此复现报告需并列给出逐类 AUROC、实际五类算术平均、论文报告值 91.2% 和差距；不应称 91.2% 为这些逐类数值的算术平均，也不应为匹配 91.2% 而改变评估方法。仅生成图像、完成数量审计或得到单类结果，均不代表复现此指标。

历史 v4/v5 产物、旧嵌入上的新检测器实验、短轮数 smoke test 只作为基线或运行验证。它们不能替代完成本协议 MCL、生成和检测的最终  结果。最终报告须注明生成检查点训练轮数、每类实际样本数、检测器选择规则及五类宏平均；未完成的类别不能填入论文数值计算平均。

## 论文明确设置

| 模块 | 明确要求 | 论文位置 |
| --- | --- | --- |
| MCL | 同时学习多个组件，每个新词嵌入 768 参数，只训练名词对应嵌入，冻结文本编码器其余参数与扩散网络 | III-B、Eq.4 |
| 提示词 | 用形容词、连接词、介词组织组件；使用源于 CLIP 的模板引入多样性 | III-B |
| 训练数据与时长 | 每类别使用正常图的三分之一，6100 epochs，恒定学习率 1e-5，batch size 16 | IV-A Implementation Details |
| 增强 | 随机裁剪和旋转 | IV-A |
| PM | 删除相关 token 模拟缺失语义 | III-B、Fig.2-I |
| 生成 | DDIM，去噪步数 T=50，每类别生成 1000 张 | IV-A |
| RNA | 正常参考库，多尺度特征比较，寻找最近正常参考 | III-C、Eq.5 |
| 特征 | ResNet18 block 1/2/3，维度 64×64×64、128×32×32、256×16×16 | IV-A |
| RM | 平均所有 16×16 cross-attention 层和 heads，t=0 组件图，以大小为 3 的高斯核平滑 | III-C、Eq.7 |
| 正常参考重建 | 确定性 DDIM 正向编码和反向重建 | III-C、Eq.8–9 |
| 残差掩码 | 对组件注意力绝对差求和后上采样 | III-C、Eq.10 |
| CSDA | 通道均值差异图按 Eq.11 逐尺度相乘增强；U-Net 交换尺度信息；下采样深度可分离卷积，上采样双线性插值接 1×1 卷积 | III-D |
| 检测损失 | Smooth L1 + 5×Focal + Dice | Eq.12、IV-A |

### 6100 epochs 与 textual inversion 的 6100 steps

Textual Inversion 作者发布的 [Stable Diffusion 配置](https://github.com/rinongal/textual_inversion/blob/main/configs/stable-diffusion/v1-finetune.yaml) 中，trainer 设置为 `max_steps: 6100`。本次查阅的[原始配置](https://raw.githubusercontent.com/rinongal/textual_inversion/main/configs/stable-diffusion/v1-finetune.yaml)可用于核对。该设置与 ComGEN IV-A 明文的 **6100 epochs** 不同；不能据此悄悄将 ComGEN 改成 6100 次更新。

本复现以 ComGEN 明文为准执行 6100 epochs。当前每轮遍历所选三分之一正常图，使用 batch 16、`drop_last=True`，更新总数为 `6100 × floor(所选正常图数 / 16)`。取样用固定随机种子，三分之一数量按 round 取整；取整和丢弃不足一个 batch 的尾部是补充假设。日志和 manifest 必须同时记录 epochs、steps_per_epoch 和累计 updates。提前检查点只能标为中间结果。

## 论文未完全公开时采用的设置

以下是可复核的实现选择，并非论文明确给出的全部参数。

- 基础模型 SD1.5；生成分辨率 512×512；组件初始化词、六种模板，以及早餐盒之外四类的组件划分和提示词。早餐盒语句参考 Fig.2；其他类别不宣称为作者原始提示词。
- MCL 使用 Adam，无 weight decay；可训练文本向量和 VAE 为 fp32，冻结 UNet 为 bf16。增强为 resize 544、反射 padding 32、旋转 ±5°、center crop 544、random crop 512；增强的具体幅度为假设。
- PM 使用原提示词以及逐个删除一个组件 token 的变体，删除时保留周围词。变体比例与是否删除容器均记录在生成协议；论文没有公开完整变体分配表。
- 默认生成 2000 候选，再按提示词组固定配额选择，共保留 1000 对；CFG 默认 7.5。具体候选量、配额和 CFG 均非论文公开数值。
- Eq.6 将 SSIM 之和称为距离，但通常 SSIM 越大越相似，与文中的距离上限解释存在歧义。当前采用 `sum(1-SSIM)`，保留较小距离，并用组内排序阈值满足配额。SSIM 窗口 7、高斯 sigma 1.5、按参考通道范围计算常数，均为补充选择。不得将其标为作者公布的数值阈值。
- RNA 使用三尺度完整空间特征 L2 范数之和，为每张样本选择一张共同正常参考；特征输入统一双线性 resize 256×256，使用 ImageNet 归一化。
- 正常参考执行 50 步 DDIM inversion 和 50 步 reconstruction，两者 guidance=1。scheduler alpha、steps_offset、timestep spacing 随已记录配置确定；记录重建误差，不假定数值上能无损复原原图。
- RM 对正常重建 latent 与生成 latent 均用**完整正常 prompt 再执行一次 t=0 查询**，使组件通道按同一 token 身份对齐。论文没有明确说明删除 token 后如何对齐所有组件，以及 A0 是否再次查询，因此 common full prompt 属于明确记录的歧义处理。
- Eq.10 原始残差保存为 `raw_residual`，仅供审计。训练软目标由每图残差 min-max 映射至 [0,1]，保存为 mask PNG。这是明确假设，**不是 Eq.10 明文操作**。训练器读取 PNG 是当前设计；release 版已移除未使用的 `mask_raw` 分支。若改用浮点训练目标，应独立记录目标含义及数值范围。
- 检测器默认 Adam、100 epochs、学习率 1e-4、batch 16；具体 U-Net 通道数、Focal gamma=2/无类别 alpha 权重、连续软目标、正常样本加入比例、正常训练图排除自身最近参考、图像分数取 sigmoid 图最大值，均为实现选择。
- 1000 对合成数据中按最近参考分组留约 10% 作验证，结合官方 validation/good，以验证损失选择检查点。因此当前训练不直接使用全部 1000 对合成样本；这是防止模型选择泄漏的验证设计，需要在最终报告注明。

## 数据隔离与质量检查

生成和 MCL 仅使用 train/good。测试图、测试标签、真实异常掩码不得用于组件词学习、伪异常筛选、阈值选择、调参或检查点选择。检测器选择使用合成验证与官方 validation/good。协议冻结后才执行显式测试评估；已看过测试结果的后续修改必须注明探索性质，不能再描述为完全未观察测试集的模型选择。

数量、尺寸、文件完整性通过不等于生成质量通过。正式训练前检查固定种子、固定位置的生成样例及正常重建，检查组件是否仍可辨认、PM 是否改变目标组件、残差是否主要对应异常，而不是正常位置差异或重建失真。保留原始残差幅度、mask 覆盖率和重建 PSNR 分布。逐图 min-max 可能放大小残差；固定配额可能保留较差样本；完整 prompt 的 attention 可能受生成后的重新解释影响。这些是需观察的质量风险，不是未经验证即可认定的代码错误。

质量检查、消融或算法修订的依据应来自训练/验证数据与论文公式。不能以“哪个版本测试 AUROC 更接近 91.2”作为选择规则。

## 追溯与缓存建议

每次运行保存参数、随机种子、训练文件清单、类别提示词、嵌入检查点 hash、训练 updates、软件版本、scheduler 配置、基础模型与 ResNet 权重标识/hash。源码追溯应覆盖三个顶层  脚本及实际导入的 vendor 配置、utils、attention processor、SSIM 实现；仅 hash 顶层文件不足以发现依赖变化。

建议使用完整协议 hash 作为缓存身份的一部分。模型、嵌入、组件配置、mask 规则或 scheduler 改变后，应使用新输出目录或经明确验证重建受影响缓存，避免把旧 candidates、normal_cache 和新 mask 混在一个结果中。此段为待落实的追溯建议，不声称所有建议已由代码自动强制执行。
