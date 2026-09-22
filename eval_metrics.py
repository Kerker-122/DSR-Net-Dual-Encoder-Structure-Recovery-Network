import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timm
from timm.models.layers import trunc_normal_

# 尝试导入 SAM2，如果没有安装则提示
try:
    from sam2.build_sam import build_sam2
except ImportError:
    print("【错误】未检测到 SAM2 库。")
    print("请先安装 SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
    exit()

try:
    from thop import profile
except ImportError:
    print("【错误】未检测到 thop 库。请安装: pip install thop")
    exit()


# ==============================================================================
# 1. 模型定义区域 (直接包含你提供的代码)
# ==============================================================================

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.prompt_learn.apply(_init_weights)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            # 处理尺寸不匹配时的 padding
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class SAM2UNeXT(nn.Module):
    def __init__(self, checkpoint_path=None, dinov2_path=None) -> None:
        super(SAM2UNeXT, self).__init__()

        print("正在加载 SAM2 (Hiera-Large)...")
        # ===== SAM2 Encoder =====
        # 注意: 这里的 model_cfg 必须在你的 SAM2 安装路径或当前路径下能被找到
        model_cfg = "sam2_hiera_l.yaml"

        try:
            if checkpoint_path:
                model = build_sam2(model_cfg, checkpoint_path)
            else:
                # 如果没有 checkpoint，这里可能会报 warning，但不影响跑结构测试
                print("Warning: 未提供 SAM2 checkpoint，将使用随机初始化用于测试...")
                # 这里为了跑通代码，我们尝试不传 checkpoint，看 build_sam2 是否允许
                # 实际上 build_sam2 通常强制要求 checkpoint，如果报错请下载 sam2_hiera_large.pt
                model = build_sam2(model_cfg, checkpoint=None)
        except Exception as e:
            print(f"SAM2 加载失败: {e}")
            print("提示: 你可能需要手动下载 'sam2_hiera_large.pt' 并传入路径")
            raise e

        # 删除不需要的部分以节省显存
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.sam = model.image_encoder.trunk
        for param in self.sam.parameters():
            param.requires_grad = False

        # 插入 Adapter
        blocks = []
        for block in self.sam.blocks:
            blocks.append(Adapter(block))
        self.sam.blocks = nn.Sequential(*blocks)

        print("正在加载 DINOv2 (ViT-Large)...")
        # ===== DINOv2 Encoder =====
        # 为了测试方便，如果下载失败，我们可以临时把 pretrained=False (虽然这没有意义，但能跑通测试脚本)
        try:
            dino_kwargs = dict(
                features_only=True,
                img_size=(448, 448),
                pretrained=True
            )
            if dinov2_path:
                dino_kwargs['pretrained_cfg_overlay'] = dict(file=dinov2_path)

            self.dino = timm.create_model('vit_large_patch14_dinov2', **dino_kwargs)
        except Exception as e:
            print(f"DINOv2 下载或加载失败: {e}")
            print("尝试使用未预训练版本进行测试...")
            self.dino = timm.create_model('vit_large_patch14_dinov2', features_only=True, img_size=(448, 448),
                                          pretrained=False)

        for param in self.dino.parameters():
            param.requires_grad = False

        self.align1 = nn.Conv2d(1024, 144, 1)
        self.align2 = nn.Conv2d(1024, 288, 1)
        self.align3 = nn.Conv2d(1024, 576, 1)
        self.align4 = nn.Conv2d(1024, 1152, 1)

        self.reduce1 = nn.Conv2d(144 + 144, 128, 1)
        self.reduce2 = nn.Conv2d(288 + 288, 128, 1)
        self.reduce3 = nn.Conv2d(576 + 576, 128, 1)
        self.reduce4 = nn.Conv2d(1152 + 1152, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        # SAM2 branch
        x1_s, x2_s, x3_s, x4_s = self.sam(x)

        # DINOv2 branch (resize to 448 internally)
        x_d = self.dino(F.interpolate(x, size=(448, 448), mode='bilinear'))[-1]

        x1_d = F.interpolate(self.align1(x_d), size=x1_s.shape[-2:], mode='bilinear')
        x2_d = F.interpolate(self.align2(x_d), size=x2_s.shape[-2:], mode='bilinear')
        x3_d = F.interpolate(self.align3(x_d), size=x3_s.shape[-2:], mode='bilinear')
        x4_d = F.interpolate(self.align4(x_d), size=x4_s.shape[-2:], mode='bilinear')

        x1 = torch.cat([x1_s, x1_d], dim=1)
        x2 = torch.cat([x2_s, x2_d], dim=1)
        x3 = torch.cat([x3_s, x3_d], dim=1)
        x4 = torch.cat([x4_s, x4_d], dim=1)

        x1, x2, x3, x4 = self.reduce1(x1), self.reduce2(x2), self.reduce3(x3), self.reduce4(x4)
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        return out


# ==============================================================================
# 2. 评测逻辑
# ==============================================================================

def cal_params_flops(model, input_size, device):
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
    model.eval()

    print(f"正在计算 FLOPs (输入: {input_size}x{input_size})...")
    # SAM2UNeXT 只需要一个输入 x
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return params, flops


def cal_fps(model, input_size, device, iterations=50):
    model.eval()
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)

    print("正在进行 FPS 预热...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    print(f"开始 FPS 测试 (循环 {iterations} 次)...")

    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / 1000.0
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        curr_time = time.time() - start_time

    if curr_time == 0: return 0
    fps = iterations / curr_time
    return fps


def main():
    # 检测 GPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {DEVICE}")
    if DEVICE.type == 'cpu':
        print("警告: 使用 CPU 跑 SAM2 + DINOv2 会非常慢！")

    # 配置输入尺寸 (SAM 通常是 1024)
    INPUT_SIZE = 512

    # 实例化模型
    # 注意: 如果你有 sam2_hiera_large.pt，请填入 checkpoint_path="..."
    print("-" * 30)
    try:
        model = SAM2UNeXT(checkpoint_path=None, dinov2_path=None).to(DEVICE)
    except Exception as e:
        print(f"\n【模型初始化失败】\n原因: {e}")
        print("\n解决建议:")
        print("1. 确保安装了 SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
        print("2. 确保 'sam2_hiera_l.yaml' 配置文件存在")
        print("3. 如果需要 checkpoint，请下载并修改 main 函数中的 checkpoint_path")
        return

    # 测试 FLOPs / Params
    print("-" * 30)
    try:
        params, flops = cal_params_flops(model, INPUT_SIZE, DEVICE)
        print(f"Params: {params / 1e6:.2f} M")
        print(f"FLOPs:  {flops / 1e9:.2f} G")
    except Exception as e:
        print(f"FLOPs 计算失败: {e}")

    # 测试 FPS
    print("-" * 30)
    try:
        # 这个模型非常大，FPS 测试次数可以设少一点
        fps = cal_fps(model, INPUT_SIZE, DEVICE, iterations=30)
        print(f"FPS: {fps:.2f}")
    except Exception as e:
        print(f"FPS 计算失败: {e}")
    print("=" * 30)


if __name__ == "__main__":
    main()