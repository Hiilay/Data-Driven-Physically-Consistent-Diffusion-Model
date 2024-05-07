import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

class DRRN(nn.Module):
    def __init__(self):
        super(DRRN,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.input = nn.Conv3d(in_channels=1, out_channels = 32, kernel_size=3,stride=1, padding=1)
        self.conv1 = nn.Conv3d(32, 32, kernel_size=3, stride=1,padding=1)
        self.donw1 = nn.Conv3d(32, 128, kernel_size=3, stride=2,padding=1)
        self.conv2 = nn.Conv3d(128, 128, kernel_size=3, stride=1,padding=1)
        self.donw2 = nn.Conv3d(128, 256, kernel_size=3, stride=2,padding=1)
        self.conv3 = nn.Conv3d(256, 256, kernel_size=3, stride=1,padding=1)
        self.donw3 = nn.Conv3d(256, 512, kernel_size=3, stride=2,padding=1)
        self.conv4 = nn.Conv3d(512, 512, kernel_size=3, stride=1,padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self,x):
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(2):
            out = self.conv1(self.conv1(self.relu(out)))
            out = torch.add(out,inputs)
        out = self.donw1(self.relu(out))
        # print(out.shape)
        middle = out
        for _ in range(2):
            out = self.conv2(self.conv2(self.relu(out)))
            out = torch.add(out,middle)
        out = self.donw2(self.relu(out))
        # print(out.shape)
        middle = out
        for _ in range(2):
            out = self.conv3(self.conv3(self.relu(out)))
            out = torch.add(out,middle)
        out = self.donw3(self.relu(out))
        # print(out.shape)
        out = self.conv4(out)
        return out
    
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
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
                batch, -1, 1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            # nn.ReLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
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
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w,d = x.shape
        ha = self.block1(x)
        ha = self.noise_func(ha, time_emb)
        ha = self.block2(ha)
        return ha + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width,depth)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwd, bncyxd -> bnhwyxd", query, key
        ).contiguous() / math.sqrt(channel)
        #attn = attn.view(batch, n_head, height, width,depth)



        attn = torch.softmax(attn, -1)

        #attn = attn.view(batch, n_head, height, width,depth, height, width,depth)

        out = torch.einsum("bnhwyxd, bncyxd -> bnchwd", attn, value).contiguous()

        out = self.out(out.view(batch, channel, height, width,depth))

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


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=(4),
        res_blocks=3,
        dropout=0,
        drrn=None,
        with_noise_level_emb=True,
        image_size=128,
    ):
        super().__init__()
        self.drrn = DRRN()
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
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
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

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time,x_lr):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        #print(x.shape)
        #print(t.shape)
        # print(lr.shape)
        features = self.drrn(x_lr)
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                #print(x.shape)
            else:
                x = layer(x)
                #print(x.shape)
            feats.append(x)
        
        feats[-1] = torch.add(feats[-1], features)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                #print(x.shape)
            else:
                x = layer(x)
                #print(x.shape)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
                #print(x.shape)
            else:
                x = layer(x)
                #print(x.shape)

        return self.final_conv(x)
    

if __name__ == '__main__':
    batch_size = 1
    device = torch.device("cuda:1")
    model = UNet(in_channel=2, out_channel=1, inner_channel=64,norm_groups=32,channel_mults=[1,2,4,8,8],
                 attn_res=[64],res_blocks=2,dropout=0,image_size=256).to(device)
    
    x1 = torch.randn(batch_size, 1, 64, 64, 128).to(device)
    x2 = torch.randn(batch_size, 1, 64, 64, 128).to(device)
    x3 = torch.randn(batch_size, 1, 32, 32, 64).to(device)
    t = torch.randint(1000, (batch_size, )).to(device)
    #x, t=t, noise=model(torch.cat([x1, x2], dim=1), t)
    x_x = torch.cat([x1, x2],dim=1) 
    # 切分输入数据成多个小批次
    # num_batches = 8 # 你可以根据需求调整小批次数量
    # batched_x = torch.split(x_x, x_x.shape[4] // num_batches, dim=4)

    #     # 计算每个小批次的输出
    # output_list = []
    # for batch_x in batched_x:
    #     batch_y = model(batch_x, t).to(device)
    #     output_list.append(batch_y)
    output_list=model(x_x, t,x3).to(device)
    # 合并每个小批次的输出

    #print(y.shape)
    #print(y.shape)
