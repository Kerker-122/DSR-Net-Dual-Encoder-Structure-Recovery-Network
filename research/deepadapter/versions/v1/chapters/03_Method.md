# 3 Method

## 3.1 Overview

给定输入图像 $I\in\mathbb{R}^{3\times H\times W}$，海洋动物分割的目标是生成与图像对应的前景概率图 $P\in[0,1]^{H\times W}$。训练阶段使用二值标注 $Y\in\{0,1\}^{H\times W}$，其中前景表示目标动物区域。本文构建由双编码器、逐尺度融合和渐进形状解码组成的网络。SAM2 编码器分支和 DINOv2 编码器分支分别为四级融合提供特征；DeepAdapter 指引入低频适配结构的辅助分支设计。该命名不改变所采用适配结构与 NightAdapter 的来源关系[5]。

将两路特征记为 $\{F_S^i\}_{i=1}^{4}$ 和 $\{F_N^i\}_{i=1}^{4}$，尺度索引由细到粗排列。对应尺度融合得到 $\{F_O^i\}_{i=1}^{4}$，随后由 PSD 从粗尺度向细尺度逐级解码。整体计算可以写为

$$
\{F_S^i\}=\mathcal E_S(I_S),\qquad
\{F_N^i\}=\mathcal E_N(I_N;\theta_A),\qquad
P=\mathcal D_{\mathrm{PSD}}\big(\{\mathcal F_i(F_S^i,F_N^i)\}_{i=1}^{4}\big).
$$

其中，$I_S$ 和 $I_N$ 表示送入两路编码器的图像，$\theta_A$ 表示辅助分支中的适配参数，$\mathcal F_i$ 为尺度对齐与融合操作。两路是否使用不同输入分辨率、具体尺寸及归一化策略为〔待补：M1，两路输入预处理〕。本框架并非将浅层交给一个编码器、深层交给另一个编码器，而是在四个尺度上均保留两路信息参与融合。

## 3.2 SAM2 encoding branch

主分支使用 SAM2 的 Hiera 编码器，并配备可训练适配器以适应海洋动物分割。SAM2 提供预训练视觉表示，适配器使该表示能够在下游监督下进行任务相关调整[2]。依照当前网络图，预训练 backbone 保持冻结，适配器参与训练；实际型号、冻结范围和适配器插入位置为〔待补：M2，SAM2 checkpoint 与 adapter 配置〕。四级输出作为后续融合与跳跃连接的特征来源。

以 $h_S^\ell$ 表示第 $\ell$ 个位置的表示，主分支的适配过程可抽象记为 $h_S^{\ell+1}=\mathcal B_S^\ell(h_S^\ell;\theta_S^\ell,\theta_{AS}^\ell)$，其中 $\theta_S^\ell$ 为预训练参数，$\theta_{AS}^\ell$ 为适配参数。该记号只描述冻结与可训练部分的关系，不预设适配器位于 block 之前或之后。与直接修改最终预测不同，该分支的训练调整作用于特征形成过程，并通过后续融合影响多个解码尺度。

## 3.3 DINOv2 branch with DeepAdapter

辅助分支采用 DINOv2 编码器[3]，其预训练表示通过选定的低频适配结构进行任务调整。本文沿用 NightAdapter 的光照不敏感频带适配机制[5]，将其作为 DeepAdapter 分支的组成部分。频域运算作用于编码特征，采用 token 关联与信息映射完成选定频带的适配。具体计算以该借鉴结构为基础，不将相同的内部运算另行主张为原创。

设编码位置 $\ell$ 的表示为 $Z_\ell$，以抽象算子表达数据流：

$$
V_\ell=\mathcal T_{\mathrm{DST}}(Z_\ell),\qquad
\widetilde V_\ell^{L}=\mathcal A_{\mathrm{II}}(V_\ell^{L};T_\ell),\qquad
\widetilde Z_\ell=\mathcal T_{\mathrm{IDST}}\!\left(\mathcal R(\widetilde V_\ell^{L},V_\ell^{\mathrm{rest}})\right).
$$

$L$ 标识所选低频部分，$T_\ell$ 表示可学习 token，$\mathcal R$ 表示适配结果与其余分量的组织。$\mathcal R$ 的实际定义、变换维度、频带索引、插入层及逆变换位置为〔待补：M3，DeepAdapter 实现〕。该式是适配流程的抽象表示，不据此认定其余频带被删除或保持不变。辅助 backbone 的型号和预训练权重为〔待补：M4，DINOv2 checkpoint〕。

辅助分支为四级融合提供 $F_N^i$，但四级融合接口并不意味着 DINOv2 原生具有与 Hiera 相同的空间金字塔。本文将层选择、token 到空间特征的转换及必要的尺度重采样统一记为 $F_N^i=\mathcal P_i(\{\widetilde Z_\ell\})$，其具体定义为〔待补：M5，层索引、空间尺寸及金字塔构建〕。这一表示既保留用户确定的四级输出设计，也避免将网络深度误写为空间分辨率。

低频适配的目标是为水下目标判断提供经过任务调整的补充表示，而不是单独恢复所有细长边缘。触手和鳍部的最终预测还依赖其他特征来源与解码路径。因此，本文保留完整的双分支融合接口，以便在同一预测框架内利用两路信息；是否形成有效互补，需要在固定解码器的条件下与未适配辅助分支进行比较。

## 3.4 Multi-scale feature alignment and fusion

在尺度 $i$，来自两路编码器的特征可能具有不同通道数和空间尺寸。以 $\mathcal P_S^i$、$\mathcal P_N^i$ 表示各自的通道映射，$\mathcal R_i$ 表示空间对齐，则融合可抽象写为

$$
F_O^i=\Phi_i\!\left(\operatorname{Concat}\left[\mathcal P_S^i(F_S^i),\mathcal R_i\big(\mathcal P_N^i(F_N^i)\big)\right]\right).
$$

$\operatorname{Concat}$ 对应网络图中的两路拼接，$\Phi_i$ 表示融合后的特征变换。其核大小、通道维度、归一化及激活设置为〔待补：M6，融合层定义〕。SAM2–DINOv2 双编码器及特征融合已有 SAM2-UNeXT 作为基础[4]；本研究将低频适配后的辅助表示送入这一类接口，并使融合结果服务于 PSD。融合层本身是否存在新增机制，应以实际实现为准。

## 3.5 Progressive SAG-CCS decoder

PSD 从最粗尺度融合表示开始，在逐级恢复空间分辨率的过程中引入对应尺度的跳跃特征。按照网络图，令 $D^4=F_O^4$，第 $i$ 级解码特征可表示为

$$
D^i=\mathcal C_i\!\left(\operatorname{Concat}\left[\operatorname{Up}_i\big(\mathcal G_i(D^{i+1})\big),F_O^i\right]\right),\quad i=3,2,1.
$$

其中，$\mathcal G_i$ 为 Spatial Attention Gate，$\operatorname{Up}_i$ 将输入恢复到相应跳跃特征的尺寸，$\mathcal C_i$ 为图中的 Double Conv。门控作用于上采样之前的输入特征，随后通过跳跃连接补充对应尺度的信息。具体门控公式与卷积设置为〔待补：M7，PSD block 实现及相对前期版本的改动〕。

解码特征经分割头和形状头分别得到 logits $O_i$ 与二维形状场 $S_i=(S_i^x,S_i^y)$。前者用于产生初始前景概率，后者参与 ICCS 的局部更新。PSD 与 ICCS 沿用作者前期工作的设计，本文在新的双编码器特征接口上使用该解码路径。当前总图包含多阶段预测，而前期提供代码仅在最终阶段执行 ICCS；本篇实际执行尺度为〔待补：M8，预测头与细化的启用尺度〕，不能由示意图推定所有阶段均参与相同监督。

## 3.6 Shape-guided iterative refinement

依据作者此前提供的有效代码路径，ICCS 以 $P^{(0)}=\sigma(O)$ 和 $Q_x^{(0)}=Q_y^{(0)}=0$ 初始化。省略尺度下标，在第 $t$ 次更新中，概率相关权重与通量计算为

$$
U^{(t)}=4P^{(t-1)}(1-P^{(t-1)}),\qquad W^{(t)}=1+\beta U^{(t)},
$$

$$
Q_d^{(t)}=\operatorname{ReLU}\!\left(Q_d^{(t-1)}-\tau_q S_d\odot\nabla_dP^{(t-1)}\odot W^{(t)}\right),\quad d\in\{x,y\},
$$

$$
D^{(t)}=\operatorname{div}(Q_x^{(t)}\odot S_x,Q_y^{(t)}\odot S_y),\qquad
P^{(t)}=\sigma\!\left(\frac{O-D^{(t)}}{\xi}\right).
$$

这里 $\odot$ 为逐元素乘法，梯度与散度由离散卷积计算，$\tau_q$、$\beta$、$\xi$ 分别控制通量更新、概率权重与输出缩放。权重在概率接近 0.5 时较大，用于调节当前不确定区域的更新强度，但不将其解释为能够识别伪装成因的专用不确定性估计。原始 logits 在迭代中保持固定，概率和通量逐步更新。

上述式子忠实于已有代码中权重位于梯度项内的形式；本篇最终公式和超参数仍需由当前实现确认〔待补：M9，ICCS 版本、迭代次数与算子设置〕。输出经 Sigmoid 保持在概率范围内，但这不等价于证明迭代收敛、星形约束或拓扑保持。细化后的概率图经尺寸恢复得到输入分辨率的预测，二值化规则为〔待补：M10，推理阈值及后处理〕。

## 3.7 Learning objective

训练目标需要同时说明前景区域学习、形状场学习和多尺度监督之间的关系。以 $\mathcal I$ 表示实际接受监督的尺度集合，损失组织暂以通用记号表示为

$$
\mathcal L=\sum_{i\in\mathcal I}\lambda_i\left[\mathcal L_{\mathrm{seg}}(\widehat P_i,Y_i)+\mu_i\mathcal L_{\mathrm{shape}}(S_i,Y_i)\right].
$$

该式仅表达区域项、形状项及尺度权重的组织，不代表作者尚未提供的创新损失的精确定义。$\widehat P_i$ 究竟为初始概率或细化概率、$Y_i$ 的生成方式、形状场监督目标和全部权重为〔待补：M11，实际损失公式与实现〕。损失应与输出接口及梯度路径共同说明，避免仅根据模块名称认定形状头已经获得有效监督。

预训练 backbone、适配器、融合层和解码器的参数更新范围以实际训练配置为准。前期代码将 ICCS 调用放在无梯度上下文中，因而不能直接据该代码把本篇描述为通过最终细化概率端到端训练形状场。当前模型是否采用可微细化、独立形状监督或其他训练路径为〔待补：M12，形状头与细化模块的反向传播路径〕。明确这一点后，损失设计与最终预测之间才能形成可复现的对应关系。
