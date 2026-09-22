# DeepAdapter：面向低照度海洋动物分割的互补表征适配与渐进形状细化

DeepAdapter: Complementary Representation Adaptation and Progressive Shape Refinement for Low-Light Marine Animal Segmentation

> 中文工作初稿 v2：依据用户提供的 SPMNet、NightAdapter 与 UNeXT 全文完善。公式来源与实现差异见独立核对表；正文的〔待补〕与〔待核〕仅保留当前模型配置和真实实验仍未确认的内容。SPMNet 的实验数值未转用为本研究结果。

# 1 Introduction

海洋动物分割旨在从水下图像中提取动物的像素级区域，为形态记录和后续视觉分析提供目标掩码。与仅定位动物主体相比，准确保留触手、鱼鳍等局部结构对掩码的形态完整性提出了更高要求。在低照度或局部对比不足的图像中，主体仍可能具有可辨轮廓，而细长附属结构因所占像素少、边界响应弱而容易被遗漏。面向海洋场景的基础模型适配研究已经探索多层级提示与相互监督等途径[1]；本文进一步关注目标区域判断与细部恢复之间的联系，将弱结构信息的利用作为网络设计的出发点。

视觉基础模型为这一任务提供了可迁移表征。SAM 2 面向图像和视频中的可提示分割，DINOv2 则通过自监督预训练学习通用视觉特征[2,3]。SAM2-UNeXT 已将二者组合，通过双分辨率输入和密集融合扩展下游分割的表征来源[4]。这一框架为海洋动物分割提供了基础，但第二条编码分支如何针对水下任务进行调整，仍是具体应用中需要回答的问题。两个 backbone 的预训练目标不同，并不自动保证它们在低照度样本上形成有效互补；融合效果还取决于辅助表示包含什么信息，以及这些信息如何进入解码过程。

频域适配为调整辅助表示提供了研究思路。NightAdapter 利用光照敏感与不敏感频带的差异进行特征适配，其中低频子结构通过 token 交互调整表征[5]。本文采用这一已有子结构构建 DeepAdapter 分支，以研究其对海洋动物分割的作用。该选择来源于夜间场景的研究启发，而不是假设水下图像与夜间图像具有完全一致的分布。水下场景中的低照度还可能与背景相似、局部遮挡等因素共同影响预测，因此需要通过任务实验考察适配表示是否对动物区域判断有帮助。

针对细长结构的分割，仅在编码阶段引入频域信息仍不足以界定完整方案。低频表征并不等同于精细边缘，解码阶段需要结合不同尺度的空间信息，逐步形成与目标形态对应的预测。为此，本文将适配后的 DINOv2 表示与 SAM2 特征逐尺度融合，并采用前期 SPMNet 中的 Progressive SAG-CCS Decoder（PSD）[8]。其中，Spatial Attention Gate（SAG）调节解码响应，Iterative CCS Refinement（ICCS）利用形状场与当前概率进行迭代细化。编码适配和预测细化分别作用于表示形成与掩码生成，使两者能够在同一网络中接受分解评价。

本文构建的网络包含 SAM2 适配分支、DINOv2 低频适配分支和四尺度融合接口。四级特征描述的是融合与解码所使用的空间组织，并不要求 DINOv2 具有与 Hiera 相同的原生层级结构。沿用 UNeXT 的映射方式，辅助输出在各尺度上完成通道与尺寸对齐，再与 SAM2 表示融合；PSD 据此逐级恢复预测。训练目标采用 SPMNet 的 Hybrid Shape-Aware Loss（HSAL），将区域 BCE 与细化预测的梯度方向一致性结合[8]。区域学习与结构约束由此对应到明确的监督对象，而不是将所有改善笼统归于“特征增强”。

本研究的贡献定位在上述机制面向 MAS 的组合与验证：将已有低频适配子结构引入 SAM2–DINOv2 框架，使辅助分支在融合前获得任务适配；将四尺度融合与 PSD、HSAL 连接，考察编码调整和形状细化是否共同支持弱结构预测。实验以四个数据集为评价范围，结合定量、定性和消融分析组织证据〔待补：I1，数据集名称及最终结果摘要〕。SAM2–DINOv2 的配对、NightAdapter 的子模块及 PSD/HSAL 均明确保留其来源关系，本研究的具体收益由当前网络的真实实验确定。


# 2 Related Work

## 2.1 Marine animal segmentation

海洋动物分割需要在复杂背景中识别动物主体，并区分与背景相近的局部边缘。对于触手和鳍部等结构，区域定位与轮廓刻画并不总是同步改善：主体区域可以被正确识别，细长附属结构仍可能出现遗漏。面向这一任务，基础模型适配提供了利用大规模预训练表征的途径。Dual SAM 将多层级提示与相互监督引入海洋动物分割，说明任务相关信息可以通过提示和模型协作参与预测[1]。本文同样关注基础模型向海洋场景的迁移，但采用自动生成掩码的双编码器流程，研究辅助表征在特征融合之前的适配作用。

频率信息也已被引入海洋动物分割。HFP-SAM 使用频率引导适配器、频率感知点选择和 Full-View Mamba，将频域先验、提示生成与上下文建模结合[6]。这一工作表明，频率建模在 MAS 中已有明确的方法基础，因此本文的研究定位不在于首次使用频率信息。本文关注的是低频适配如何作用于 DINOv2 辅助分支，以及适配后的信息如何与 SAM2 的多尺度表示共同服务于形状细化。两种研究分别从提示及上下文组织、双分支特征组织出发，具体效果需要在一致的评测协议下比较。

## 2.2 Foundation models and dual-encoder adaptation

SAM 2 为图像和视频中的可提示分割提供统一的基础模型，DINOv2 则通过自监督预训练学习可用于不同视觉任务的特征[2,3]。预训练目标的差异提供了多源表征融合的动机，但并不能预先决定一个编码器仅负责语义、另一个仅负责纹理。对下游分割而言，特征是否互补取决于其提取位置、适配方式以及融合后接受的监督。因此，双编码器设计不仅需要说明使用了哪些 backbone，还需要交代不同来源的表示如何在空间和通道上对应。

与本文最直接相关的是 SAM2-UNeXT。该框架在 SAM2 基础上引入辅助 DINOv2 编码器，通过双分辨率策略与密集融合扩展表征能力[4]。本文沿用这一双基础模型组合，将研究重点转向辅助分支的低频适配和预测端的形状细化。与将第二个编码器仅作为附加特征来源的配置相比，本文在其进入多尺度融合之前加入指定的适配结构。这里的差别是辅助特征的任务适配路径及其与解码的配合，而不是 SAM2 与 DINOv2 的组合本身。

## 2.3 Frequency-aware feature adaptation

NightAdapter 将特征划分为光照敏感与不敏感频带，并分别适配[5]。两类分支都包含 token 关联、映射与残差更新，敏感频带还引入随机化处理；本文采用其中的低频、不敏感子结构。其核心运算予以保留，研究重点是该子结构在水下双编码器框架中的用途，而非将相同运算重新定义为全新的频域模块。

频率适配的目标还需要与结构细化区分。HFS-SAM2 面向伪装目标检测研究高频特征补充[7]；它与本文的低频适配具有不同的直接关注点。低频、网络浅层和低分辨率属于不同概念，不能从低频适配直接推出细长边缘已被增强。本文保留双路多尺度融合和独立的预测细化环节，考察区域判断所需的适配信息能否与形状恢复共同发挥作用。该定位不排斥高频信息的价值，也不预设某一频带在所有水下条件下均有相同性质。

## 2.4 Progressive shape refinement

区域分割与结构细化分别作用于掩码的不同方面。区域预测描述像素属于前景的倾向，形状引导则为局部预测更新引入额外的空间关系。作者前期 SPMNet 提出了 Progressive SAG-CCS Decoder（PSD），通过 SAG 调节空间响应，利用 ICCS 根据 logits 与二维形状场迭代细化概率；其 HSAL 联合区域 BCE 与预测梯度方向一致性[8]。本文沿用这部分解码和监督设计，将研究对象扩展到与低频适配辅助分支的配合。

本文将辅助分支的特征适配与已有的预测细化连接起来，形成编码与解码之间的研究联系。低频适配改变进入融合的表示，PSD/ICCS 则利用融合结果形成并调整预测。两个环节的目标可以互补，但在没有独立对照之前，不能将总体性能变化归因于某个组件。因此，评价不仅包括所使用的模块，还需要在固定其他条件时分别考察适配分支、解码细化及其联合配置。


# 3 Method

## 3.1 Overall architecture

给定海洋动物图像 $I\in\mathbb R^{3\times H\times W}$，本文预测前景概率图 $P\in[0,1]^{H\times W}$，训练标注为二值掩码 $Y\in\{0,1\}^{H\times W}$。网络由 SAM2 编码分支、带 DeepAdapter 的 DINOv2 编码分支、逐尺度融合层与 Progressive SAG-CCS Decoder（PSD）组成。两条分支在融合接口上均提供四级特征，PSD 从粗到细重建预测，并利用 Iterative CCS Refinement（ICCS）进行形状引导的细化。

按照 SAM2-UNeXT 的双分辨率设计[4]，将同一图像分别调整为 $I_S=\mathcal R_{H_S,W_S}(I)$ 和 $I_N=\mathcal R_{H_N,W_N}(I)$，其中辅助输入采用较低分辨率，以控制 DINOv2 分支的计算量。两路实际尺寸及预训练 checkpoint 为〔待补：M1，当前模型配置〕。输入分辨率与频率分量属于不同概念：低分辨率描述图像尺寸，DeepAdapter 中的低频则描述变换后的特征分量。

整体特征流写为

$$
\{F_S^i\}_{i=1}^{4}=\mathcal E_S(I_S),\qquad
Z_N=\mathcal E_N(I_N;\theta_{\mathrm{DA}}),\qquad
F_N^i=\mathcal P_i(Z_N),\quad i=1,\ldots,4.\tag{1}
$$

$\mathcal P_i$ 将辅助表示组织为与第 $i$ 级 SAM2 特征对应的尺度，$\theta_{\mathrm{DA}}$ 表示低频适配参数。本文按 UNeXT 的密集融合路径展开式（1）：同一张辅助输出经四组映射得到四级融合输入。因此，四级输出不要求 DINOv2 具有四个原生降采样阶段。若当前 DeepAdapter 实现改为抽取不同中间层，则仅需据实调整 $Z_N$ 的层来源及 $\mathcal P_i$，不能将两种提取方式混写〔待核：M2，辅助输出层来源〕。

## 3.2 SAM2 encoder with lightweight adapters

主分支采用 SAM2 的 Hiera 编码器[2]。参照 SAM2-UNeXT，在冻结的 Hiera block 之前引入轻量残差适配器[4]。令 $X_\ell$ 为输入 token 表示，$g(\cdot)$ 为 GELU，适配器及对应 block 的计算为

$$
\mathcal A_\ell(X_\ell)
=g\!\left(g(X_\ell W_{\ell,d}+b_{\ell,d})W_{\ell,u}+b_{\ell,u}\right),\qquad
X_{\ell+1}=\mathcal B_\ell\!\left(X_\ell+\mathcal A_\ell(X_\ell)\right).\tag{2}
$$

$W_{\ell,d}$ 和 $W_{\ell,u}$ 分别完成通道压缩与恢复，$\mathcal B_\ell$ 为预训练 Hiera block。原 UNeXT 使用 32 通道瓶颈；本研究的对应配置需与实际模型一致。式（2）保留了参考结构中的两次非线性映射，不将其与 SPMNet 中采用的 MonaAdapter 混为同一个具体模块。

SAM2 的四级特征 $F_S^i\in\mathbb R^{C_i\times H_i\times W_i}$ 为融合提供层级空间表示。主分支承担通用表征向 MAS 的适配，辅助分支则通过特定频带的调整提供另一条任务适配路径。两路的角色来自其设计和学习过程，而非预设所有局部信息仅存在于某一个 backbone。

## 3.3 Low-frequency adaptation in DeepAdapter

DeepAdapter 采用 NightAdapter 的光照不敏感频带适配结构[5]，将其用于 DINOv2 编码分支。该结构的 token 关联、映射和残差融合来自原方法；本文关注其在海洋动物分割中的迁移及与 SAM2 多尺度特征的结合。这里保留借鉴关系，不将已有运算重新表述为全新的频域算子。

令某一适配位置的 DINOv2 特征为 $Z_\ell\in\mathbb R^{C\times N}$，$N$ 为 patch token 数量。用 $D_C$ 和 $D_N$ 表示相应维度的离散正弦变换矩阵，则频域表示为

$$
V_\ell=D_C Z_\ell D_N^{\mathsf T}.\tag{3}
$$

式（3）将 NightAdapter 的特征变换写成矩阵形式。它作用于通道与 token 组织的表示，不能等同于对原始 RGB 图像直接进行二维空间变换。具体 DST 类型、归一化与频带排列应与代码保持一致。选取低频部分并转换为 patch 优先排列，记为 $B_\ell^L\in\mathbb R^{N\times d_L}$；其中 $d_L$ 是参与适配的特征宽度。NightAdapter 的示例在其指定排列下选用 $C=1024$ 的低频组，实际截取索引不能脱离该排列直接套用于其他实现。

引入可学习 token $T_\ell^L\in\mathbb R^{M\times d_L}$，通过归一化相似度建立 patch 与 token 的联系：

$$
A_\ell^L=\operatorname{Softmax}_{M}\!\left(\frac{B_\ell^L(T_\ell^L)^{\mathsf T}}{\sqrt{d_L}}\right),\qquad
A_\ell^L\in\mathbb R^{N\times M}.\tag{4}
$$

token 经可学习映射后，根据相似度投影回对应 patch：

$$
U_\ell^L=A_\ell^L\!\left(T_\ell^L W_{\ell,1}+b_{\ell,1}\right),\qquad
U_\ell^L\in\mathbb R^{N\times d_L}.\tag{5}
$$

将投影信息与原低频表示结合，得到适配结果：

$$
\widetilde B_\ell^L=B_\ell^L+(U_\ell^L+B_\ell^L)W_{\ell,2}+b_{\ell,2}.\tag{6}
$$

式（4）—（6）对应 NightAdapter 式（4）—（6）的关联、映射和残差融合流程。这里将参与运算的维度统一记为 $d_L$，从而明确 token 与所选频带的维度关系；这一记号整理不是新增模块。若实现先将低频映射回原通道空间，则取 $d_L=C$，并保留对应投影及缩放设置。偏置沿 patch/token 维广播，两个映射的输出与 $B_\ell^L$ 保持同宽。当前实现中的 token 数量、交互维度、缩放系数与插入层为〔待补：M3，DeepAdapter 适配配置〕。

适配结果回填至频域表示后，通过逆变换返回编码特征。令 $\operatorname{Merge}_L$ 按原索引组织低频与其余分量，则

$$
\widetilde V_\ell=\operatorname{Merge}_L\!\left((\widetilde B_\ell^L)^{\mathsf T},V_\ell^{\mathrm{rest}}\right),\qquad
\widetilde Z_\ell=D_C^{-1}\widetilde V_\ell(D_N^{\mathsf T})^{-1}.\tag{7}
$$

式（7）按“仅适配低频、其余频带旁路保留”的工作版本表达，以保持完整的逆变换接口。这一路径不包含完整 NightAdapter 的敏感频带随机化，但也不意味着把其他频带删除。实际 DeepAdapter 若采用其他剩余频带处理方式，需相应替换式（7）〔待核：M4，剩余频带与逆变换路径〕。

低频适配为辅助分支提供光照稳健表征的设计动机，细长结构恢复则依赖于适配表示、主分支信息和解码细化的共同作用。本文不将低频等同于高分辨率边缘，也不将夜间图像中的频带性质直接视为所有水下场景的固定规律。

## 3.4 Multi-scale dense feature fusion

融合层采用 UNeXT 的通道投影、尺寸对齐、拼接与压缩流程[4]。对辅助输出 $Z_N$，各尺度分别使用独立的 $1\times1$ 卷积调整通道，再将其调整至 $F_S^i$ 的空间尺寸：

$$
F_N^i=\operatorname{Resize}_{H_i,W_i}\!\left(\operatorname{Conv}_{1\times1}^{N,i}(Z_N)\right),\qquad
F_O^i=\operatorname{Conv}_{1\times1}^{O,i}\!\left(\operatorname{Concat}_{c}[F_S^i,F_N^i]\right).\tag{8}
$$

$\operatorname{Concat}_{c}$ 表示沿通道维拼接，融合后的 $F_O^i$ 具有统一的解码通道宽度 $C_f$。在参考 UNeXT 的 large 配置中，DINOv2 输出为 1024 通道，Hiera 四级通道为 $(144,288,576,1152)$，拼接后压缩至 128 通道。本稿将这些数值作为参考配置，实际运行值需由当前模型确认〔待补：M5，融合通道配置〕。

该融合方式将适配后的辅助表示传播至各解码尺度，而不只作用于最深层特征。本文使用其作为两个编码器与 PSD 之间的接口；其作用需要通过保持解码器一致的对照分析，不能将沿用的密集融合操作单独宣称为新的融合机制。

## 3.5 Progressive SAG-CCS decoding

PSD 沿用 SPMNet 的空间门控与形状细化框架[8]。设 $D^4=F_O^4$，在解码层级 $i=3,2,1$，SAG 根据输入特征生成单通道空间权重：

$$
M_i=\sigma\!\left(\operatorname{Conv}_{7\times7}\left(\operatorname{Conv}_{1\times1}(D^{i+1})\right)\right),\qquad
\widehat D^{i+1}=D^{i+1}\odot M_i.\tag{9}
$$

式（9）对应 SPMNet 式（11）—（12）。空间权重沿通道广播，用于调节解码特征的局部响应。经上采样后，与当前尺度融合特征拼接并进行双卷积：

$$
D^i=\operatorname{DoubleConv}_i\!\left(\operatorname{Concat}_{c}\left[\operatorname{Up}_{H_i,W_i}(\widehat D^{i+1}),F_O^i\right]\right).\tag{10}
$$

该过程将上一阶段的解码信息与当前尺度跳跃信息结合。每个启用预测的阶段通过分割头与形状头生成

$$
O_i=\mathcal H_{\mathrm{seg},i}(D^i),\qquad
S_i=\mathcal H_{\mathrm{shape},i}(D^i)=(S_i^x,S_i^y).\tag{11}
$$

$O_i$ 表示初始 logits，$S_i$ 为二维结构引导场。后续 ICCS 使用二者细化概率预测。本文按所提供总图描述三个渐进阶段；各阶段是否同时启用形状头、细化和监督，需要与新代码核对〔待核：M6，多阶段预测设置〕。SPMNet 原有的 MonaAdapter、CFEM 和 RFB 不因借用 PSD 而自动成为本篇双编码器的组成部分。

## 3.6 Iterative CCS refinement

ICCS 依据 SPMNet 的公式与结构图进行表述[8]。对于尺度 $i$，初始概率和二维通量为

$$
P_i^{(0)}=\sigma(O_i),\qquad Q_i^{(0)}=0.\tag{12}
$$

每次迭代计算当前概率的梯度，并根据预测的模糊程度调节更新：

$$
V_i^{(t)}=\nabla P_i^{(t-1)},\qquad
U_i^{(t)}=4P_i^{(t-1)}(1-P_i^{(t-1)}),\qquad
W_i^{(t)}=1+\beta U_i^{(t)}.\tag{13}
$$

形状场与概率梯度共同更新通量：

$$
Q_i^{(t)}=\operatorname{ReLU}\!\left(Q_i^{(t-1)}-\tau(S_i\odot V_i^{(t)})\right)\odot W_i^{(t)}.\tag{14}
$$

通量与形状场结合后通过散度形成演化项，并重建概率：

$$
E_i^{(t)}=\nabla\!\cdot\!\left(Q_i^{(t)}\odot S_i\right),\qquad
P_i^{(t)}=\sigma\!\left(\frac{O_i-E_i^{(t)}}{\delta}\right),\qquad
P_i^r=P_i^{(T)}.\tag{15}
$$

$\tau$ 为通量更新步长，$\beta$ 为概率权重系数，$\delta$ 为概率重建缩放参数，$T$ 为迭代次数。式（14）按 SPMNet 论文将 $W_i^{(t)}$ 置于 ReLU 之外；这与早期代码中将权重乘入梯度项的版本不同。此处采用用户指定的论文版本，最终实现需据此统一。最细尺度概率图经插值恢复到输入尺寸，得到最终预测 $P$。

该迭代使预测更新依赖于形状场和当前概率分布，而非仅使用一次像素分类。Sigmoid 保证概率取值范围，但不由此推导星形约束、拓扑保持或收敛保证。固定迭代次数的具体选择应由本研究的参数分析确定，不能直接把 SPMNet 选择 $T=10$ 的结果视为新网络的最优结论。

## 3.7 Hybrid Shape-Aware Loss

本文沿用 SPMNet 的 Hybrid Shape-Aware Loss（HSAL），联合监督区域预测与梯度方向一致性[8]。为区分尺度与像素，以下以 $i$ 表示预测尺度，$j$ 表示像素位置，$n_i=H_iW_i$ 表示相应像素总数。单尺度目标为

$$
\mathcal L_{\mathrm{HSAL}}^i=\mathcal L_{\mathrm{BCE}}^i+\lambda\mathcal L_{\mathrm{shape}}^i.\tag{16}
$$

区域项直接作用于细化概率 $p_{i,j}^r$ 与对应标注 $y_{i,j}$：

$$
\mathcal L_{\mathrm{BCE}}^i=-\frac{1}{n_i}\sum_{j=1}^{n_i}\left[y_{i,j}\log p_{i,j}^r+(1-y_{i,j})\log(1-p_{i,j}^r)\right].\tag{17}
$$

形状项通过 Sobel 算子 $K_x,K_y$ 提取预测梯度，并在每个像素处对二维向量归一化：

$$
G_i=\operatorname{Concat}_{c}\left[K_x*P_i^r,K_y*P_i^r\right],\qquad
v_{i,j}=\frac{G_i(j)}{\|G_i(j)\|_2+\epsilon}.\tag{18}
$$

设 $s_{\mathrm{gt},i,j}$ 为由对应标注预计算的真实形状场，按照 SPMNet 式（17）得到

$$
\mathcal L_{\mathrm{shape}}^i=\frac{1}{n_i}\sum_{j=1}^{n_i}\mathcal C\!\left(-v_{i,j}^{\mathsf T}s_{\mathrm{gt},i,j}\right).\tag{19}
$$

$\mathcal C$ 为方向一致性惩罚函数，$\epsilon$ 用于避免零梯度处的除零问题。SPMNet 所给公式尚未展开 $\mathcal C$ 的具体形式以及真实形状场的生成规则，二者需补充对应实现〔待补：M7，方向函数与真实形状场构造〕。因此不能自行将该损失替换成梯度 L1、余弦距离、ReLU 惩罚或预测形状头与真实场之间的直接回归。式（19）的监督对象是细化概率的梯度方向。

若多个阶段参与训练，以 $\mathcal I$ 表示被监督尺度，可组织为

$$
\mathcal L_{\mathrm{total}}=\sum_{i\in\mathcal I}\alpha_i\mathcal L_{\mathrm{HSAL}}^i.\tag{20}
$$

式（20）是对多阶段接口的训练组织方式；实际 $\mathcal I$、$\alpha_i$、$\lambda$ 及辅助输出是否使用其他损失为〔待补：M8，当前监督配置〕。式（16）—（19）作用于细化概率，训练时需要保留相应可微路径或提供明确的其他监督机制。早期代码中的无梯度 ICCS 调用不能直接支持此描述，故反向传播路径仍需与当前训练代码对应〔待核：M9，ICCS 与形状头训练路径〕。


# 4 Experiments

## 4.1 Implementation Details

本文采用预训练 SAM2 和 DINOv2 构建双编码器网络，在冻结 backbone 参数的条件下训练适配器、融合层和解码部分。两路输入分别按照主分支和辅助分支的分辨率设置进行预处理，以保持对应图像内容的一致性。模型型号、checkpoint、输入尺寸及参数冻结范围为〔待补：E1，当前网络配置〕。尽管实验组织参照 SPMNet[8]，双编码器引入后的输入与训练配置仍以本研究实际运行记录为准。

优化器、初始学习率、权重衰减、批量大小、训练轮数、学习率调度和数据增强设置为〔待补：E2，训练配置〕。软件版本、GPU 型号和随机种子为〔待补：E3，运行环境〕。ICCS 的迭代次数及 HSAL 权重应与方法部分对应，并记录辅助输出的监督方式。SPMNet 中的单分支训练参数不直接作为本网络的已用参数，避免不同训练条件被误认为相同实验协议。

## 4.2 Datasets

本文在四个海洋动物分割数据集上评价所提出方法。数据集名称、样本规模、划分及标注类型见表 1〔待补：E4，四数据集信息〕。各数据集的描述围绕目标尺度、背景复杂度、光照变化和动物形态展开，以说明整体评价覆盖的条件。SPMNet 使用的 MAS3K、RMAS、UFO120 和 RUWI 可作为组织数据集小节的参考；本篇是否沿用这四个数据集及相同划分，需以实际实验记录为准。

若在不同数据集上独立训练，应分别列出各自训练与测试设置；若使用统一训练集跨数据集测试，则需明确训练来源与测试域，并避免将跨域结果和域内结果混在同一比较中。图像亮度不能替代采集水深标签，因此低照度样本分析与深海场景验证分别表述。对于带有相邻视频帧或重复图像的数据，还需说明划分时如何避免训练与测试内容重叠。

## 4.3 Evaluation Metrics

参照 SPMNet 的评价组织，本稿以 $m\mathrm{IoU}$、$S_\alpha$、$F_\beta^\omega$、$mE_\phi$ 和 MAE 为五指标报告模板[8]。$m\mathrm{IoU}$ 衡量区域重叠，$S_\alpha$ 从结构层面评价预测与标注的一致性，$F_\beta^\omega$ 表征加权精确率与召回表现，$mE_\phi$ 衡量增强对齐表现，MAE 反映概率预测与标注之间的平均绝对差异。前四项取值越高越好，MAE 越低越好。实际使用指标、脚本版本与平均方式为〔待核：E5，评价实现〕。

这些指标共同描述整体分割质量，但不能单独证明细长部位的连通性得到改善。针对论文关注的触手和鳍部，定性比较应提供统一的局部放大区域，并结合实际完成的边界或细结构评价解释结果。如果尚未计算此类专门指标，则将结论限定为整体分割表现和可观察到的局部差异，不额外主张经过量化验证的拓扑保持能力。

## 4.4 Comparative Experiments

### 4.4.1 Quantitative Comparisons

比较方法按 CNN、Transformer 和基础模型适配三类组织，以呈现不同表征与解码策略的表现。SAM2-UNeXT 作为最直接的双编码器参照，用于判断低频适配、PSD 与 HSAL 在原有框架上的综合影响[4]；SPMNet 用于比较新的辅助分支与已有形状解码体系的组合[8]。最终方法清单与结果来源为〔待补：E6，对比方法、复现或引用说明〕，不直接照搬 SPMNet 的方法列表或既有数值。

四数据集的定量结果分别列于表 2 和表 3〔待补：E7，本研究真实结果〕。作者已报告所提出方法在四个数据集上优于所比较方法，具体优势应按各项指标的实际排名陈述。对于越高越好的指标，提升量按本方法与对照方法之差计算；对于 MAE，则报告误差下降量。百分比与百分点采用一致口径，避免将绝对差与相对变化混写。

主结果的分析需要同时考虑区域得分与结构相关表现。若某数据集主要体现为区域重叠改善，应将结论限定为该表现；只有局部预测与相应指标共同支持时，再讨论触手遗漏、鳍部边界等问题的缓解。若存在指标持平或下降，也应保留该结果，分析是否与前景扩张、背景误检或过度平滑有关。各数据集的具体结论为〔待补：E8，与真实表格和预测一致的比较分析〕。

### 4.4.2 Qualitative Comparisons

定性比较沿用 SPMNet 的原图、真实标注及不同方法预测并列形式，以触手、鱼鳍、弱边界和背景干扰区域为主要观察对象。相同样本采用一致的裁剪范围与显示尺寸，避免不同放大倍数改变对细节的判断。结果图及样本编号为〔待补：E9，真实预测图〕，网络结构图中的示意掩码不作为实验输出使用。

分析中分别观察动物主体是否完整、细长部位是否连续、前景与背景交界是否偏移，以及是否产生额外背景响应。更平滑的轮廓不一定更接近标注，因此局部讨论应同时保留真实形状中的细小分支。除优势样本外，还需呈现极暗区域、遮挡或高背景相似度下的失败情况，以说明编码适配与形状细化的适用范围。

## 4.5 Ablation Experiments

消融沿用 SPMNet 的组件分析、特征响应、迭代细化与损失比较结构，并增加与本篇核心变化对应的 DeepAdapter 对照。基线采用 SAM2–DINOv2 编码框架、原始融合接口及常规解码器；不将 SPMNet 的单编码器、MonaAdapter、CFEM 和 RFB 配置直接命名为本篇相同基线。消融数据集、训练预算和已完成的配置为〔待补：E10，消融协议〕。以下小节规定结果应回答的问题，未提供数值的分析不表示相应结果已核验。

### 4.5.1 Effectiveness of DeepAdapter and Its Cooperation with PSD

DeepAdapter 的独立作用通过固定融合层、解码器和损失，对比未适配与低频适配的 DINOv2 分支进行分析。该对照将辅助分支适配与解码器变化分开，避免同时增加多个模块后无法定位收益。若已有完整 NightAdapter 对照，则将其与本篇选用的低频子结构分别列出，以评价选择该子结构后的性能与开销差别。对应结果为〔待补：E11，辅助分支适配对照〕。

编码适配与 PSD 的配合可通过二因素设计分析：基线、仅采用 DeepAdapter、仅采用 PSD，以及二者联合。四种配置应维持相同的数据划分与训练预算，并控制损失设置；若 HSAL 仅适用于某些配置，需另列这一条件，避免将目标函数变化混入解码器效应。联合模型优于基线能够支持整体收益，但要主张互补或额外增益，还需结合单组件结果进行判断。相关数值为〔待补：E12，编码与解码组合表〕。

### 4.5.2 Effectiveness of PSD Components

参照 SPMNet 对 SAG 和 ICCS 的分析，在固定编码配置后比较常规解码、仅 SAG、仅 ICCS 以及完整 PSD。各配置使用一致的训练监督规则，并说明 ICCS 的加入是否同时改变形状头与反向传播路径。该实验分别考察空间响应调节和概率细化的作用，不将继承模块在旧网络上的有效性自动视为其在当前双编码器中的有效性。结果为〔待补：E13，SAG 与 ICCS 消融表〕。

特征响应可视化与组件得分结合呈现，比较引入 PSD 前后相同尺度的目标响应与背景激活。热图应采用一致的归一化方式，并交代取自融合特征、解码特征还是预测 logits。响应更集中可以作为观察现象，但不能单凭热图证明边界精度或因果机制。需要结合掩码结果检查是否同时减少背景误检并保留细小附属结构。

### 4.5.3 Visualization of ICCS Iterative Refinement

迭代分析比较同一输入、相同参数和初始预测下不同 $T$ 对细化结果的影响。SPMNet 展示的 $T\in\{1,5,10,15\}$ 可作为本节的参考设置；本研究实际取值及对应输出为〔待补：E14，迭代配置、预测与运行时间〕。使用统一局部放大区域，有助于观察初始遗漏是否减少、边界是否更贴近标注，以及迭代增加后是否出现结构收缩或过度平滑。

迭代次数的确定应同时考虑分割质量与运行开销。若若干次数的结果接近，应据实说明性能差异，而不是依据单张图宣称某一次数普遍最优。SPMNet 选择 $T=10$ 的结论属于该网络及其分析条件，本研究的默认值需要由当前实验支持。定性演化也不替代对迭代收敛性的数学证明。

### 4.5.4 Effectiveness of HSAL

损失分析参照 SPMNet，将常规 IoU–BCE 目标与 HSAL 进行比较，保持网络结构、训练预算和评价协议一致。由于两个目标的项数及量纲不同，训练曲线可分别除以各自初始损失以便观察相对变化，但不能据归一化曲线更低直接推断最终分割更好。损失配方、权重、曲线及对应验证指标为〔待补：E15，HSAL 对照结果〕。

HSAL 的作用还可通过保持 BCE 项不变、加入方向一致性项的对照进一步分析，以减少同时替换 IoU 与增加形状项带来的混杂。如果当前尚未完成该补充对照，应将其保留为后续分析，而不写成已有发现。判断形状监督是否有效时，需要共同检查区域指标、局部预测及形状项的梯度路径；训练曲线下降本身并不证明形状头获得了预期监督。

本章的解释边界由真实结果决定。四数据集的整体优势与针对低照度细长结构的机制证据分别呈现，既不将所有水下样本称为深海样本，也不推断模型能够恢复完全不可观测的动物部位。双编码器的资源消耗可在真实运行记录齐备后补充报告，为性能改进提供相应的计算代价背景。


# 5 Conclusion

本文面向低照度海洋动物图像中的弱边界与细长结构遗漏，构建结合辅助表征适配、多尺度融合和渐进形状细化的分割框架。在 SAM2–DINOv2 双编码器基础上，DeepAdapter 将 NightAdapter 的低频适配子结构用于辅助分支，并通过四尺度映射将适配表示与 SAM2 特征融合。解码端采用 SPMNet 的 PSD，在空间响应调节后利用形状场细化概率预测；HSAL 则通过区域监督与梯度方向一致性组织训练目标。

该设计将辅助表示的任务适配和预测侧的结构恢复联系起来，同时保留各部分的可分解评价接口。四个数据集构成实验范围，作者报告的方法比较结果将在完整主表中呈现〔待补：C1，各数据集的实际领先指标与结论〕。低照度鲁棒性、细长部位改善及组件间配合的具体程度，应分别由相应预测与对照结果支持。极暗区域中的信息缺失、其他水下退化条件及双编码器计算开销仍是需要进一步分析的边界。


# References

[1] Zhang et al. Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual SAM. CVPR, 2024. [Official paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.pdf).

[2] Ravi, N., et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024. [Paper](https://arxiv.org/abs/2408.00714).

[3] Oquab, M., et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193, 2023. [Paper](https://arxiv.org/abs/2304.07193).

[4] Xiong, X., et al. SAM2-UNeXT: An Improved High-Resolution Baseline for Adapting Foundation Models to Downstream Segmentation Tasks. arXiv:2508.03566, 2025. Technical report. [Paper](https://arxiv.org/abs/2508.03566).

[5] Bi, Q., Yi, J., Huang, H., Zheng, H., Zhan, H., Huang, Y., Li, Y., Wu, X., and Zheng, Y. NightAdapter: Learning a Frequency Adapter for Generalizable Night-time Scene Segmentation. CVPR, 2025, pp. 23838–23849. [Official paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Bi_NightAdapter_Learning_a_Frequency_Adapter_for_Generalizable_Night-time_Scene_Segmentation_CVPR_2025_paper.pdf).

[6] Zhang, P., et al. HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation. arXiv:2603.12708, 2026. Author record states acceptance by IEEE TIP 2026; publisher volume and pages not verified. [Paper](https://arxiv.org/abs/2603.12708).

[7] HFS-SAM2: Segment Anything Model 2 with High-Frequency Feature Supplementation for Camouflaged Object Detection. IEEE Signal Processing Letters, 2025, as identified by the authors' repository. 〔待补：出版记录中的作者、卷页与 DOI〕. [Author repository](https://github.com/WZH0120/HFS-SAM2).

[8] SPMNet: Shape-Prior Guided Progressive Decoding for Camouflaged Marine Animal Segmentation. 作者提供的匿名稿，公开发表状态未确认。本文仅据该稿标识沿用机制；正式投稿时的作者及引用形式待补。[Provided manuscript](<C:/Users/12269/Desktop/SPNet/SPMNet Anonymous.pdf>).

