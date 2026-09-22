# DeepAdapter 论文规划（待确认）

日期：2026-09-19。阶段：S4 首版草稿写作。用户在澄清架构后明确要求一次完成五部分，视为本轮结构与写作授权，无需再逐章请求许可。

确认结构：Introduction、Related Work、Method、Experiments、Conclusion。中文工作初稿，英文标题，Markdown 输出。精确章节文件见 chapter-architecture.md。

- 任务：分析 MAS 方案、确定大纲，再分章节形成初稿。
- 学科：计算机视觉、海洋动物分割。按研究型投稿论文规划，具体期刊/会议未定。
- 暂定题目：DeepAdapter：面向低照度海洋动物分割的互补表征适配与渐进形状细化。
- 英文候选：DeepAdapter: Complementary Representation Adaptation and Progressive Shape Refinement for Low-Light Marine Animal Segmentation。
- 当前输出：中文方案与 Markdown 大纲；英文投稿稿语言和模板后续确定。项目内现有图中公式 tex 不是投稿模板。
- 用户报告：四个数据集优于 SOTA；数据集名称、表格、指标和评测协议尚未提供。本记录不否定该结果，但不能据此补写数字、显著性或逐指标排名。
- 既有研究：PSD/ICCS 为作者上一篇工作，详见 ../../PRIOR_WORK_ICCS_CONTEXT.md。第二篇需要区分继承与增量。
- 本轮不修改模型，不训练，不安装依赖，只在本项目内保存研究文件。

## 已确认的架构与来源（后续用户澄清）

- 两路分别为 SAM2 encoder 与 DINOv2 encoder，两路均提供四级特征并逐尺度融合，不是按浅/深层互斥分配编码器。
- 用户确认采用 NightAdapter 中低频、光照不敏感频带适配部分的结构，参照已提供网络图中的 Illumination-Insensitive Adapter。不要再把来源当作完全未知，也不要要求用户从零重新描述原模块。
- 原文第 4.2 节与图 4 对应此模块；术语应为低频而非低尺度。用户图中的 tokens、映射与残差路径在概念上与该模块相对应，但图中 C 被标为拼接，不代表原文全部矩阵运算；实现级对应尚需代码。
- DeepAdapter 暂定位为引入该适配结构的 MAS 辅助分支/设计名称；采用已有子模块不等于提出该子模块的新内部运算。是否还有额外模块改动，以用户后续实现为准。
- 用户已有四数据集实验，后续提供；当前先理清思路。

## 已消费资料

用户最新描述、附图 codex-clipboard-23cb97b7-69a7-4d13-87a6-e62097b8d73a.png、既有 ICCS 归档、NightAdapter 官方论文搜索可访问内容和补充材料。
