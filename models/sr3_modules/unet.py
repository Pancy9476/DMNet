import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)



class AdaptiveMixPool2d(nn.Module):
    def __init__(self, output_size=1, alpha=0.5):
        super(AdaptiveMixPool2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        self.alpha = alpha

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        return self.alpha * avg_pooled + (1 - self.alpha) * max_pooled
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mix_pool = AdaptiveMixPool2d(1,0.5)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgooled_x = self.avg_pool(x)
        max_pooled_x = self.max_pool(x)
        mix_pooled_x = self.mix_pool(x)
        # out = self.shared_MLP(avgooled_x) + self.shared_MLP(max_pooled_x)
        out = self.shared_MLP(max_pooled_x) + self.shared_MLP(mix_pooled_x)
        # out = self.shared_MLP(avgooled_x) + self.shared_MLP(mix_pooled_x)
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgooled_x = torch.mean(x, dim=1, keepdim=True)
        max_pooled_x, _ = torch.max(x, dim=1, keepdim=True)
        mix_pooled_x = 0.5 * avgooled_x + 0.5 * max_pooled_x
        concatenated = torch.cat([max_pooled_x, mix_pooled_x], dim=1)
        out = self.conv2d(concatenated)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# building block modules
class upSkip(nn.Module):
    def __init__(self):
        super(upSkip, self).__init__()

        # 128 -> 64
        self.to2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )

        # 256 -> 64
        self.to4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )

        # 512 -> 64
        self.to8 = nn.Sequential(
            nn.ConvTranspose2d(512, 64, kernel_size=4, stride=4, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )

        # 1024 -> 64
        self.to161 = nn.Sequential(
            nn.ConvTranspose2d(1024, 64, kernel_size=8, stride=8, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )

        # 1024 -> 64
        self.to162 = nn.Sequential(
            nn.ConvTranspose2d(1024, 64, kernel_size=16, stride=16, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )
        # 960 -> 64
        self.to15 = nn.Sequential(
            nn.ConvTranspose2d(960, 64, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )

    def forward(self, x):
        x0=self.to2(x[0])
        x1=self.to2(x[1])
        x2=self.to2(x[2])
        x3=self.to4(x[3])
        x4=self.to4(x[4])
        x5=self.to4(x[5])
        x6=self.to8(x[6])
        x7=self.to8(x[7])
        x8=self.to8(x[8])
        x9=self.to161(x[9])
        x10=self.to161(x[10])
        x11=self.to161(x[11])
        x12=self.to162(x[12])
        x13=self.to162(x[13])
        x14=self.to162(x[14])


        cbam = CBAM(in_channels=64).to(device)
        x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14= cbam(x0),cbam(x1),cbam(x2),cbam(x3),cbam(x4),cbam(x5),cbam(x6),cbam(x7),cbam(x8),cbam(x9),cbam(x10),cbam(x11),cbam(x12),cbam(x13),cbam(x14)


        return (x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14)/75
class downSkip(nn.Module):
    def __init__(self):
        super(downSkip, self).__init__()
        # 64 - > 128
        self.to2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.GELU()
        )
        # 64 - > 256
        self.to4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.GELU()
        )
        # 64 - > 512
        self.to8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.GELU()
        )
        # 64 - > 1024
        self.to16 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16),
            nn.Conv2d(64, 1024, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.GELU()
        )

    def forward(self, x):
        return [self.to2(x), self.to4(x), self.to8(x), self.to16(x)]

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

def Reverse(lst):
    return [ele for ele in reversed(lst)]


class UNet(nn.Module):
    def __init__(
        self,
        image_size=128,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        self.init_conv = nn.Conv2d(in_channels=in_channel, out_channels=inner_channel, kernel_size=3, padding=1)
        downs = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)


        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)    # 要改返回

    def forward(self, x, time, feat_need=False):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        x = self.init_conv(x)

        feats = [x]
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        
        if feat_need:
            fe = feats.copy()

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        if feat_need:
            fd = []
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
                if feat_need:
                    fd.append(x)

            else:
                x = layer(x)

        # Final Diffusion layer
        x = self.final_conv(x)
        r_fd = Reverse(fd)
        unet_res = upSkip().to(device)(r_fd)
        if feat_need:
            return fe, unet_res
        else:
            return x
