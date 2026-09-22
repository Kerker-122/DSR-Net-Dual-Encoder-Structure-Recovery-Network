# 论文大纲与论证安排

更新：用户已确认写作，指定 Introduction、Related Work、Method、Experiments、Conclusion 五部分，覆盖下方旧的章节划分；最终文件和分工以 chapter-architecture.md 为准。原 Introduction 中相关工作迁入独立 Related Work；数据设置、结果与讨论并入 Experiments；保留原有论证要点。双分支均提供四级特征、逐尺度融合已确认。

## 中心论证

低照度下弱结构证据不稳定，加上触手和鱼鳍等结构所占像素较少，可能造成漏检、断裂和边界偏移。研究应检验：光照鲁棒辅助表征与 SAM2 表征是否互补，PSD 是否能进一步把这种互补转化为结构质量提升。不要把“深海”“低亮度”“低对比”当作同一数据标签。

## 摘要（正文定稿后撰写）

组织顺序：具体失效现象 → 双分支互补假设 → DeepAdapter 的实际新增机制 → PSD 与损失的作用 → 四数据集可核查结果。未收到实验表前不补写提升数值。

## 1 Introduction

- 从海洋动物细长结构分割需求切入，限定研究对象为低照度或低对比样本；深海作为应用动机，除非数据支持，不把全部实验表述为深海验证。
- 将挑战拆为外观可辨性和结构完整性；避免未经实验认定光照是唯一原因。
- 将 MAS、基础模型适配、NightAdapter 相关工作融入研究缺口，后续补齐 MAS 原文核查。
- 解释为何引入辅助表征，以及为什么预测端细化仍有必要。
- 明确新机制、已有 PSD/ICCS 的沿用/修改和实验证据三者的贡献边界。

## 2 Methodology

### 2.1 Problem definition and architecture

定义输入、二值标注、输出及尺度索引；明确输入分辨率与每层张量形状，说明两路并行全尺度融合还是按层分工。

### 2.2 SAM2 branch with adapters

说明 Hiera 型号、预训练来源、冻结范围、adapter 插入位置和四级输出。不要把未确认的适配器默认写为 Mona。

### 2.3 DeepAdapter

交代 NightAdapter 来源；明确辅助 backbone、预训练权重版本、特征变换位置、DST 作用维度、频带划分、token 交互及逆变换位置。单列与直接迁移 NightAdapter 的差别。若使用下采样输入，说明细节保留责任由哪一路承担。

### 2.4 Cross-branch feature alignment and fusion

定义对应层的空间对齐、通道映射和融合输出。区分 backbone 原生层级与重采样构造的金字塔；融合方式以代码为准，不能仅凭图中 C 符号假定完整运算。

### 2.5 Progressive SAG-CCS decoder

沿附图顺序解释注意力门、上采样、skip 拼接、双卷积、分割头和形状头；说明 ICCS 每阶段使用还是仅最终阶段使用。交代本篇相对上一篇 PSD 的实际变化。

### 2.6 Training objective

按真实实现写出分割损失、形状监督与各尺度权重；提供形状场目标生成方式和梯度路径。不能把常见 BCE/IoU/边界项的组合自动称为全新损失。

## 3 Datasets and Experimental Setting

四数据集名称、规模、来源、划分、标注类型；训练/测试重叠排查；低照度子集的独立定义；预训练、输入尺寸、优化器、学习率、训练预算、随机种子、指标实现与推理处理。

## 4 Experimental Results and Analysis

### 4.1 Comparison with state-of-the-art methods

主表按四数据集列真实指标，交代比较方法版本与复现设置。不要将部分指标领先改写为所有指标最优。

### 4.2 Low-light and thin-structure evaluation

将整体性能与目标问题分开验证。报告预先定义的低照度分组、边界质量和细结构召回；如采用骨架指标，说明算法、阈值与适用范围。亮度分组不能作为海深标签。

### 4.3 Component and interaction ablations

固定解码器比较 SAM2-UNeXT、直接迁移 NightAdapter、DeepAdapter；固定编码器比较原解码器与 PSD；以 2×2 编码器/解码器组合检查互补性。使用相同训练预算并报告参数量与计算量。

### 4.4 Scale and frequency analysis

验证低/高层适配位置、输入分辨率、频带保留方式与融合位置。特别检查低频适配是否伴随细结构损失。

### 4.5 Qualitative comparison and efficiency

展示触手断裂、鱼鳍漏检、背景误检及失败案例；报告推理速度、峰值显存和可训练/总参数。

## 5 Discussion

解释何种场景受益、何种场景失败；区分实验相关性与机制因果证据；讨论深海证据不足、极暗区域信息不可恢复和双编码器开销。

## Conclusion

仅总结最终实现与实验证实的结论，避免宣称恢复不可观测结构或普遍解决深海分割。

## 图表计划

- 图 1：真实细结构失效与动机，统一裁剪和尺度。
- 图 2：澄清尺度后的总网络图。
- 图 3：DeepAdapter 细节，突出相对 NightAdapter 的实际增量。
- 图 4：PSD/ICCS 细节，与实现逐步对应。
- 表 1：四数据集主结果；表 2：2×2 互补消融；表 3：频带/尺度/损失消融；表 4：低照度与细结构分组及开销。
