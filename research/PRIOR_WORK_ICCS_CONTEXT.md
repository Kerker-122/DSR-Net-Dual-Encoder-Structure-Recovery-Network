# MAS 第二篇论文研究上下文：用户已有 SPNet / TSAMNet、PSD、ICCS

记录日期：2026-09-18。来源：用户在本任务中直接提供的代码、两张方法论截图、ICCS.png、主网络图.tif。这是用户自述的上一篇论文核心工作，尚未收录；确切投稿、接受、公开状态未知。不要因为检索不到而否认其存在，也不要把以下已有设计当成第二篇新提出的贡献。具体引用/披露方式应结合后续投稿状态与要求判断，不能由“自己提出且未收录”直接推出一概无需引用。

## 当前研究要求

- 第二篇以 SAM2-UNeXT 为基础，优先方法创新，当前不修改模型、不启动实现或训练。
- 用户明确关注：模糊边界、细长部位分割错误。
- 用户对 NightAdapter（CVPR 2025）的编码器频率适配感兴趣，考虑小波或其他机制改造。
- 上一篇 decoder 可复用。新 encoder 应改善特征形成与细节保留；已有 PSD/ICCS 负责预测侧形状引导细化。两者是否互补需实验验证。
- 小波替换本身不自动构成充分创新；不得保证模块数量等于发表价值。
- 用户要求保存本次材料，后续不要反复要求其提供同一内容。缺失的损失、模块定义等仍应明确为未知。

## 文件来源与已查看内容

- 总网络图：`C:\Users\12269\Desktop\SPNet\图片\主网络图.tif`，5463×3522；已读取并转为缩略 PNG 查看。
- 方法截图 1：`C:\Users\12269\AppData\Local\Temp\codex-clipboard-458991c1-a8b1-4137-8bfb-8890e9dec865.png`。
- 方法截图 2：`C:\Users\12269\AppData\Local\Temp\codex-clipboard-b955cf52-e7c1-419a-ac4f-37980ad7cfa2.png`。
- ICCS 图：`C:\Users\12269\Desktop\SPNet\图片\ICCS.png`。
- 总图查看副本：`C:\Users\12269\.codex\visualizations\2026\09\18\01a0b397-da54-75a1-9103-663fdf1f4881\spnet_network_preview.png`。
- 临时图片路径可能失效，下面的文字与代码记录用于保留核心信息；不是完整论文归档。

## 总网络图所表达的设计

1. 冻结 SAM2 Hiera，四阶段配 MonaAdapter，输出 1/4、1/8、1/16、1/32 特征 C1…C4。
2. Neck：每级 CFEM（Context-aware Feature Enhancement Module）+ RFB（Receptive Field Block）。
3. Progressive SAG-CCS Decoder（PSD），逐级由高层语义恢复低层细节，图中三个阶段均产生 refined prediction。
4. PSD Block：Spatial Attention Gate（图示 7×7 depthwise conv + sigmoid）→ Upsample → 与 skip feature 拼接 → Double Conv → Seg Head 与 Shape Head → ICCS。
5. 图中 Hybrid Shape-Aware Loss（每尺度）：区域 BCE + λ × Shape Loss（Gradient Consistency），涉及 GT Mask 与 GT Shape Field。具体 loss 公式、实现、形状场 GT 生成方式未提供。
6. 总图中出现 DA-CCS iteration 标注；用户正文与模块图主要称 ICCS。不要自行认定它们是不同已实现模块。

## 用户方法论（截图）

标题：3.3.1 Iterative CCS Refinement。

目的：缓解伪装海洋动物的边界模糊与结构不完整，为低对比区域、触手与鳍等细长结构提供几何引导。

CCS 先验在文稿中与 star-shape constraint 关联：存在中心 c，使对边界点 x 的线段满足 `(1−λ)c+λx ∈ Ω, λ∈[0,1]`。

Shape Head 从解码特征预测二维场 `S_i=(S_i^x,S_i^y)`。Seg Head 输出 logits O_i。初始概率 `P_i^(0)=sigmoid(O_i)`，初始通量 Q=0。每轮计算概率梯度，再用形状场、步长和概率相关权重更新通量；计算 `D=div(Q ⊙ S)`，以 `P=sigmoid((O−D)/δ)` 重建概率。固定 T 次后输出 refined prediction。

ICCS 图：`U=4P(1−P)`；`W=1+βU`；图示更新为 `Q^(t)=ReLU(Q^(t−1)−τ(S⊙V^(t)))⊙W^(t)`。

### 按提供代码确认的实际行为

- 类名 CCSSigmoid；T=10、tau_q=1、xi=1。
- kx=[[1,0,−1],[2,0,−2],[1,0,−1]]，ky=kx 转置；零填充卷积计算梯度，负核计算 divergence。
- 每轮 uncertainty=4u(1−u)，cam_weight=1+2×uncertainty（即 β=2，权重范围 [1,3]）。这只是概率接近 0.5 的指标，不是经校准或专门鉴别伪装原因的不确定性。
- 实际通量：`qx=ReLU(qx−tau_q*s_x*gx*cam_weight)`，qy 同理。
- 实际 divergence：`div(qx*s_x,qy*s_y)`。
- 每轮 `u=sigmoid((o−div_qs)/xi)`，原始 logits o 固定，迭代 u 和 q。
- TSAMNet 中只有最终解码阶段调用一次 ICCS，调用被 `torch.no_grad()` 包围。两个 side 输出是普通 side logits。
- 返回 `(o, out_ccs, out1, out2)`；不返回 s。

## 待核对项（不要静默修复或混同图文与实现）

1. **形状场梯度路径**：按所示 forward，field_head 唯一输出 s 仅用于 no_grad 内的 ICCS，且未返回 s。常规基于返回值的 loss 无法训练 field_head；out_ccs 自身也不能作为可反传监督路径。完整训练脚本或其他路径未知，不能断言完整实验一定如此。
2. **权重位置不同**：图文为 ReLU(q−τS∇u)×W；代码为 ReLU(q−τS∇u×W)。从 q 非零开始通常不等价。
3. **多尺度差异**：总图表示多尺度 ICCS 与 shape loss；所示代码只有末级 ICCS。Up、TwinAttn、MonaAdaper 定义及 loss 均未提供，不能填补推断。
4. **star-shape 的实现证据**：任意二维 field_head + 当前迭代式不自动保证星形约束。中心构造、场监督、约束推导尚未提供，先记录为作者设计动机，不能写成已证明硬约束。
5. Sigmoid 保证输出范围，不自动证明迭代收敛、稳定性或拓扑连接保证。

## 第二篇方法设计边界

- 已有工作侧重 logits/probability 层面的形状场引导迭代修正。
- 新工作宜侧重 encoder 阶段的弱结构证据保留、局部空间—频率适配，与已有预测细化区别开。
- 暂定 NightAdapter-inspired structure-preserving frequency adaptation 仅是草案，不是用户已确定的最终结构。
- 不要默认第二篇必须复用原全部 Mona/CFEM/RFB/PSD 组件，也不要默认必须替换 DINOv2。
- 实验需分离新 encoder、旧 decoder 和二者联合效果，同时比较直接迁移 NightAdapter、仅换小波、加入任务机制。

## 用户代码归档

下面整理粘贴时被 Markdown 损坏的缩进与 __init__ 写法，保留有效计算路径。没有把此代码写入模型文件或执行。原注释中的无 uncertainty 迭代是早期备选，未启用。

```python
class CCSSigmoid(nn.Module):
    def __init__(self, T=10, tau_q=1.0, xi=1.0):
        super(CCSSigmoid, self).__init__()
        self.T = T
        self.tau_q = tau_q
        self.xi = xi
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                          dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = kx.transpose(2, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def gradient(self, u):
        gx = F.conv2d(u, self.kx, padding=1)
        gy = F.conv2d(u, self.ky, padding=1)
        return gx, gy

    def divergence(self, qx, qy):
        dx = F.conv2d(qx, -self.kx, padding=1)
        dy = F.conv2d(qy, -self.ky, padding=1)
        return dx + dy

    def forward(self, o, s):
        u = torch.sigmoid(o)
        qx = torch.zeros_like(o)
        qy = torch.zeros_like(o)
        for _ in range(self.T):
            uncertainty = 4.0 * u * (1.0 - u)
            cam_weight = 1.0 + 2.0 * uncertainty
            gx, gy = self.gradient(u)
            qx = F.relu(qx - self.tau_q * s[:, 0:1] * gx * cam_weight)
            qy = F.relu(qy - self.tau_q * s[:, 1:2] * gy * cam_weight)
            div_qs = self.divergence(qx * s[:, 0:1], qy * s[:, 1:2])
            u = torch.sigmoid((o - div_qs) / self.xi)
        return u


class TSAMNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(TSAMNet, self).__init__()
        model_cfg = 'sam2_hiera_l.yaml'
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(MonaAdaper(block))
        self.encoder.blocks = nn.Sequential(*blocks)
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(128, 64)
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.gam1 = TwinAttn(144)
        self.gam2 = TwinAttn(288)
        self.gam3 = TwinAttn(576)
        self.gam4 = TwinAttn(1152)
        self.field_head = nn.Conv2d(64, 2, kernel_size=1)
        self.ccs_sigmoid = CCSSigmoid(T=10)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.gam1(x1), self.gam2(x2), self.gam3(x3), self.gam4(x4)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        o = self.head(x)
        s = self.field_head(x)
        with torch.no_grad():
            out_ccs = self.ccs_sigmoid(o, s)
        out_ccs = F.interpolate(out_ccs, scale_factor=4, mode='bilinear')
        o = F.interpolate(o, scale_factor=4, mode='bilinear')
        return o, out_ccs, out1, out2
```
