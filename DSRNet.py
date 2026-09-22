import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from sam2.build_sam import build_sam2
from timm.layers import trunc_normal_


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


def orthogonal_dst(x):
    """Orthogonal DST-I along the feature dimension; also its own inverse."""
    n = x.shape[-1]
    padded = torch.cat((torch.zeros_like(x[..., :1]), x,
                        torch.zeros_like(x[..., :1]), -x.flip(-1)), dim=-1)
    spectrum = torch.fft.rfft(padded.float(), dim=-1)
    result = -spectrum.imag[..., 1:n + 1] * (0.5 * math.sqrt(2.0 / (n + 1)))
    return result.to(x.dtype)


class IlluminationInsensitiveBand(nn.Module):
    """Token similarity and feature mapping for one selected frequency band."""
    def __init__(self, width, num_tokens=8):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, width) * 0.02)
        self.token_projection = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.feature_mapping = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
        self.projection = nn.Linear(2 * width, width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, band):
        similarity = torch.softmax(
            torch.matmul(band, self.tokens.t()) / math.sqrt(band.shape[-1]), dim=-1)
        token_features = torch.matmul(similarity, self.token_projection(self.tokens))
        mapped = self.feature_mapping(torch.cat((band, token_features, band), dim=-1))
        return band + self.projection(torch.cat((mapped, band), dim=-1))


class FrequencyAdapterBlock(nn.Module):
    """Adapt DINOv2 patch tokens in the frequency domain before a frozen block."""
    def __init__(self, block, channels=1024, band_ranges=((768, 896), (896, 1024))):
        super().__init__()
        self.block = block
        for parameter in self.block.parameters():
            parameter.requires_grad = False
        self.band_ranges = band_ranges
        self.bands = nn.ModuleList([
            IlluminationInsensitiveBand(end - start) for start, end in band_ranges])
        if any(start < 0 or end > channels or start >= end for start, end in band_ranges):
            raise ValueError('Frequency bands must lie within the DINOv2 feature dimension')

    def forward(self, x):
        # The class token has no spatial frequency interpretation.
        class_token, patches = x[:, :1], x[:, 1:]
        # Reverse the DST ordering so the selected upper indices are low frequencies.
        spectrum = orthogonal_dst(patches).flip(-1)
        pieces = []
        previous = 0
        for (start, end), adapter in zip(self.band_ranges, self.bands):
            pieces.append(spectrum[..., previous:start])
            pieces.append(adapter(spectrum[..., start:end]))
            previous = end
        pieces.append(spectrum[..., previous:])
        adapted = orthogonal_dst(torch.cat(pieces, dim=-1).flip(-1))
        return self.block(torch.cat((class_token, adapted), dim=1))
    

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
    
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFBModified(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(BasicConv2d(in_channels, out_channels, 1))])
        for size, dilation in ((3, 3), (5, 5), (7, 7)):
            self.branches.append(nn.Sequential(
                BasicConv2d(in_channels, out_channels, 1),
                BasicConv2d(out_channels, out_channels, (1, size), padding=(0, size // 2)),
                BasicConv2d(out_channels, out_channels, (size, 1), padding=(size // 2, 0)),
                BasicConv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation),
            ))
        self.fuse = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.residual = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return F.relu(self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
                      + self.residual(x), inplace=True)


class SAGAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.Conv2d(channels // 8, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class TwinAttn(nn.Module):
    def __init__(self, channels, rate=4):
        super().__init__()
        hidden = channels // rate
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, hidden, 7, padding=3),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 7, padding=3),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        channel_weight = self.channel_attention(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).sigmoid()
        x = x * channel_weight
        return x * self.spatial_attention(x).sigmoid()


class CCSSigmoid(nn.Module):
    def __init__(self, T=10, tau_q=1.0, xi=1.0):
        super().__init__()
        self.T, self.tau_q, self.xi = T, tau_q, xi
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)[None, None]
        self.register_buffer('kx', kx)
        self.register_buffer('ky', kx.transpose(2, 3).contiguous())

    def forward(self, logits, field):
        u = torch.sigmoid(logits)
        qx, qy = torch.zeros_like(logits), torch.zeros_like(logits)
        kx, ky = self.kx.to(dtype=logits.dtype), self.ky.to(dtype=logits.dtype)
        for _ in range(self.T):
            uncertainty = 4.0 * u * (1.0 - u)
            weight = 1.0 + 2.0 * uncertainty
            gx = F.conv2d(u, kx, padding=1)
            gy = F.conv2d(u, ky, padding=1)
            qx = F.relu(qx - self.tau_q * field[:, 0:1] * gx * weight)
            qy = F.relu(qy - self.tau_q * field[:, 1:2] * gy * weight)
            div = F.conv2d(qx * field[:, 0:1], -kx, padding=1)
            div += F.conv2d(qy * field[:, 1:2], -ky, padding=1)
            u = torch.sigmoid((logits - div) / self.xi)
        return u


class Up(nn.Module):
    """Upsample the decoder feature, fuse a skip feature, then refine it."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sag = SAGAttention(in_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.conv(self.sag(torch.cat([x2, x1], dim=1)))

    
class DSRNet(nn.Module):
    def __init__(self, checkpoint_path=None, dinov2_path=None) -> None:
        super(SAM2UNeXT, self).__init__()

        # ===== SAM2 Encoder =====    
        model_cfg = "sam2_hiera_l.yaml"
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
        self.sam = model.image_encoder.trunk
        for param in self.sam.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.sam.blocks:
            blocks.append(
                Adapter(block)
            )
        self.sam.blocks = nn.Sequential(
            *blocks
        )

        # ===== DINOv2 Encoder =====
        if dinov2_path:
            self.dino = timm.create_model('vit_large_patch14_dinov2',
                                        img_size=(448, 448),
                                        pretrained=True,
                                        pretrained_cfg_overlay=dict(file=dinov2_path))
        else:
            self.dino = timm.create_model('vit_large_patch14_dinov2',
                                        img_size=(448, 448))
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.blocks = nn.Sequential(*[
            FrequencyAdapterBlock(block) for block in self.dino.blocks])

        self.align1 = nn.Conv2d(1024, 144, 1)
        self.align2 = nn.Conv2d(1024, 288, 1)
        self.align3 = nn.Conv2d(1024, 576, 1)
        self.align4 = nn.Conv2d(1024, 1152, 1)

        self.reduce1 = nn.Conv2d(144+144, 128, 1)
        self.reduce2 = nn.Conv2d(288+288, 128, 1)
        self.reduce3 = nn.Conv2d(576+576, 128, 1)
        self.reduce4 = nn.Conv2d(1152+1152, 128, 1)

        self.attentions = nn.ModuleList([TwinAttn(128) for _ in range(4)])
        self.rfbs = nn.ModuleList([RFBModified(128, 64) for _ in range(4)])
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.head = nn.Conv2d(64, 1, 1)
        self.field_head = nn.Conv2d(64, 2, 1)
        self.ccs_sigmoid = CCSSigmoid(T=10)
        

    def forward(self, x, return_ccs=False):
        x1_s, x2_s, x3_s, x4_s = self.sam(x)
        dino_input = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
        dino_tokens = self.dino.forward_features(dino_input)
        patch_tokens = dino_tokens[:, self.dino.num_prefix_tokens:]
        x_d = patch_tokens.transpose(1, 2).reshape(x.shape[0], 1024, 32, 32)

        x1_d = F.interpolate(self.align1(x_d), size=x1_s.shape[-2:], mode='bilinear')
        x2_d = F.interpolate(self.align2(x_d), size=x2_s.shape[-2:], mode='bilinear')
        x3_d = F.interpolate(self.align3(x_d), size=x3_s.shape[-2:], mode='bilinear')
        x4_d = F.interpolate(self.align4(x_d), size=x4_s.shape[-2:], mode='bilinear')

        x1, x2, x3, x4 = torch.cat([x1_s,x1_d], dim=1), torch.cat([x2_s,x2_d], dim=1), torch.cat([x3_s,x3_d], dim=1), torch.cat([x4_s,x4_d], dim=1)
        features = [reduce(feature) for reduce, feature in zip(
            (self.reduce1, self.reduce2, self.reduce3, self.reduce4), (x1, x2, x3, x4))]
        x1, x2, x3, x4 = [rfb(attn(feature)) for feature, attn, rfb in
                          zip(features, self.attentions, self.rfbs)]
        decoded = self.up1(x4, x3)
        decoded = self.up2(decoded, x2)
        decoded = self.up3(decoded, x1)
        logits = self.head(decoded)
        out = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=True)
        if not return_ccs:
            return out
        field = self.field_head(decoded)
        with torch.no_grad():
            refined = self.ccs_sigmoid(logits, field)
        refined = F.interpolate(refined, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out, refined
