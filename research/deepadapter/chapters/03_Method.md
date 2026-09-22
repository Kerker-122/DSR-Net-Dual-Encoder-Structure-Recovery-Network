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
