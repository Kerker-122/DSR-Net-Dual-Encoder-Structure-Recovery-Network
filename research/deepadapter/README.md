# DeepAdapter 初稿工作区

本稿以已确认的 SAM2–DINOv2 四尺度融合结构、借用 NightAdapter 光照不敏感低频适配器以及已有 PSD/ICCS 为基础。中文正文配英文章节标题，供作者讨论与逐轮完善；不是可直接投稿的最终稿。

最新阅读入口：manuscript_zh_v2.md（依据三篇用户 PDF 完善）；manuscript_zh_v1.md 保留旧版；chapters/ 为当前五章，versions/ 保存版本快照；refs/ 保存文献与证据映射；plan/ 保存技术边界、实验待补信息和审查记录。

正文 〔待补：...〕 是缺失的真实配置或结果，不是虚构数据。方法公式中的抽象模块表示数据流，不代表已核实的完整实现。原有 PSD/ICCS 的使用不被当作本篇新提出的全部机制。

v2 已补齐 NightAdapter 低频 token 适配、UNeXT 融合、SAG/ICCS 与 HSAL 的论文级公式，并参照 SPMNet 重排实验章节。当前仍缺：HSAL 中 C 函数和真实形状场构造、当前训练梯度路径及配置、四数据集结果与实际消融。未把参考论文的数值或超参数自动当作本研究实测内容。

公式核对：plan/review/source-formula-map-v2.md。原 PDF 只读，提取文本与必要的公式页预览仅保存在本项目 sources/ 中，不上传匿名稿。
