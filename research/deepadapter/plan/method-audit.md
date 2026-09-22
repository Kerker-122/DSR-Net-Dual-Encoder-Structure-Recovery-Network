# 方法审查与证据边界

## 后续澄清（优先于下方早期疑问）

用户已确认 SAM2 与 DINOv2 两路均输出四级特征、逐尺度融合。用户采用的是 NightAdapter 第 4.2 节 Illumination-Insensitive Band Adaptation 的结构，而非完整 NightAdapter 全部频带分支。原文图 4 区分光照不敏感和敏感频带适配；前者用可学习 token 与低频表示建立相似度，并映射 token 信息进行特征适配。原文的“低”指频率，不是网络深度或空间分辨率。附图所示 Insensitive Tokens 等结构与这一来源对应；具体算子仍不可把图中所有 C 当作原文运算。

贡献定位随之更新：应明确借用该低频适配结构，并论证它在 SAM2–DINOv2 多尺度融合与 PSD 框架中的作用；不能仅凭没有采用原文全部分支就声称低频适配器内部全新。后续无需反复询问已知的模块来源。

## NightAdapter 核查

Qi Bi 等，NightAdapter: Learning a Frequency Adapter for Generalizable Night-time Scene Segmentation，CVPR 2025，23838–23849。

官方来源：
- https://openaccess.thecvf.com/content/CVPR2025/html/Bi_NightAdapter_Learning_a_Frequency_Adapter_for_Generalizable_Night-time_Scene_Segmentation_CVPR_2025_paper.html
- https://openaccess.thecvf.com/content/CVPR2025/papers/Bi_NightAdapter_Learning_a_Frequency_Adapter_for_Generalizable_Night-time_Scene_Segmentation_CVPR_2025_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bi_NightAdapter_Learning_a_CVPR_2025_supplemental.pdf

官方搜索可读取的内容确认：主实验使用 DINOv2 编码器及 Mask2Former 解码器，表中为 ViT-L；DST 用于 VFM 特征；区分光照敏感和不敏感频带。补充材料明确频带性质在夜间图像上研究，其他不利环境可能不同。因此水下适用性属于需要验证的迁移假设。网页直接打开部分失败，但官方搜索片段可读取；未获取可验证的具体 checkpoint 文件名及下载链接。

## 架构含义

空间分辨率 H×W、网络层级、特征通道 C 和频率分量是四个不同概念。低层通常更靠近输入，低分辨率通常更粗；低频并不等于低层。用户图同时标注 Low-Level Branch、Low-Res Input、Low-Frequency Adapter，需要统一。

图中两个编码器都输出四级特征，并非明确的“低层仅辅助编码器、高层仅 SAM2”结构。辅助 encoder 若沿用普通 ViT，不能直接将四个抽取层画为原生不同空间分辨率，需明确金字塔构建方式。

## 创新判定

现有图未清楚展示 DeepAdapter 超出 NightAdapter 的运算差别。改名、替换数据集或放入第二分支不足以单独证明新 adapter 机制。可以研究尺度选择与结构保留的互补设计，但这些目前是候选假设，不是已实现事实。

PSD/ICCS 是作者已有工作。新论文可沿用，但需要将新编码器、decoder 增量和已有组件区分；自有且未收录不自动免除相关工作的适当说明。

四数据集领先是用户报告的性能结果，不是第三个独立算法创新点。补齐对比表、协议和模型版本后再决定摘要与贡献中的准确措辞。

## PSD/ICCS 已知风险

旧代码的 ICCS 位于 no_grad 中，field_head 输出仅流入该路径；若无其他训练路径，形状头得不到来自该输出的梯度。新损失实现未提供，不能假定此问题已经修复，也不能据旧片段否认用户的新实验。

原图概率权重在 ReLU 外，旧代码在梯度项内；三阶段细化图与仅最终阶段细化代码有差别。任意二维形状场不自动保证星形约束、拓扑保持或迭代收敛。

## 附图修订清单（本轮未改图）

- Skin Feature 改为 Skip Feature。
- Low-Level、Low-Res、Low-Frequency 分别命名，避免混用。
- DST 若实际用于 encoder 中间特征，应移动至相应层位置，不能画成原始图像变换后直接送入 encoder。
- 标出输入尺寸、输出形状、频率变换轴与冻结范围。
- 明确四组融合分别接收哪两路张量，现图灰色输入的来源不清楚。
- 概率、logits、形状场与特征张量采用不同符号。
- 示例掩码与形状场如为示意，应标注；未经预测文件核对不能作为真实定性实验结果。
