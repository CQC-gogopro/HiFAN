import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat
from typing import Optional, Callable, Any
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from mmcv.ops import DeformConv2dPack as DCN

try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe(x, normed_mask, kernel_size, group=1, up=1):
            """
            carafe:
                features: Tensor,     特征     b c h w
                masks: Tensor,        卷积核    b k*k h w
                kernel_size: int,     卷积核大小  k
                group_size: int,      条件组数量
                scale_factor: int
            """
            b, c, h, w = x.shape
            _, m_c, m_h, m_w = normed_mask.shape
            print('x', x.shape)
            print('normed_mask', normed_mask.shape)
            # assert m_c == kernel_size ** 2 * up ** 2
            assert m_h == up * h
            assert m_w == up * w
            pad = kernel_size // 2
            # print(pad)
            pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
            # print(pad_x.shape)
            unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
            # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
            unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
            unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
            # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
            unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
            normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
            res = unfold_x * normed_mask
            # test
            # res[:, :, 0] = 1
            # res[:, :, 1] = 2
            # res[:, :, 2] = 3
            # res[:, :, 3] = 4
            res = res.sum(dim=2).reshape(b, c, m_h, m_w)
            # res = F.pixel_shuffle(res, up)
            # print(res.shape)
            # print(res)
            return res

    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        

INTERPOLATE_MODE = 'bilinear'

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


class InitialBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(InitialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.SyncBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConvBlock, self).__init__()
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels, 
            in_channels,  # Depthwise convolution uses the same number of input and output channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # groups = in_channels for depthwise convolution
            bias=False  # Depthwise conv typically does not require bias
        )
        
        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels, 
            out_channels,  # Pointwise convolution changes the number of channels
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=False  # Pointwise conv typically does not require bias
        )
        
        # SiLU activation function
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.depthwise_conv(x)  # Apply depthwise convolution
        x = self.pointwise_conv(x)  # Apply pointwise convolution
        x = self.silu(x)  # Apply SiLU activation
        return x

class DimPooling(nn.Module):
    def __init__(self, 
                 dim, 
                 r, #2的指数倍
                 channel_last = True
                 ):
        super().__init__()
        self.dim = dim
        self.num = int(math.log2(r))
        self.r = r
        self.channel_last = channel_last
            
    def forward(self, x):
        if self.channel_last:
            b, h, w, c =  x.shape
            x = rearrange(x, " b h w c -> b (h w) c")
        else:
            b, c, h, w =  x.shape
            x = rearrange(x, " b c h w -> b (h w) c")
            
        for i in range(self.num):
            c = c//2
            x = (F.adaptive_avg_pool1d(x, output_size=(int(c))) + F.adaptive_max_pool1d(x, output_size=(int(c))))/2
            
        if self.channel_last:
            x = rearrange(x, " b (h w) c  -> b h w c", h=h, w=w)
        else:
            x = rearrange(x, " b (h w) c -> b c h w", h=h, w=w)
        
        return x


class CEIM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CEIM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)

        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def channel_attention(self, feat_A, feat_B):
        # Cross Attention for Channel Attention
        Q = self.avg_pool(feat_A)  # Query from modality A
        K = self.avg_pool(feat_B)  # Key from modality B
        V = K  # Value from modality B

        # Compute attention weights
        attn = F.softmax((Q @ K.transpose(-2, -1)) / (self.channels ** 0.5), dim=-1)
        cross_feat = attn @ V

        # Channel Attention
        avg_out = self.fc2(F.relu(self.fc1(cross_feat)))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(feat_A))))
        channel_attn = torch.sigmoid(avg_out + max_out)
        return channel_attn

    def spatial_attention(self, feat_A, feat_B):
        # Cross Attention for Spatial Attention
        Q_avg = torch.mean(feat_A, dim=1, keepdim=True)  # Query from modality A
        K_avg = torch.mean(feat_B, dim=1, keepdim=True)  # Key from modality B
        V_avg = K_avg  # Value from modality B

        # Compute attention weights
        attn = F.softmax((Q_avg @ K_avg.transpose(-2, -1)) / (self.channels ** 0.5), dim=-1)
        cross_feat = attn @ V_avg

        # Spatial Attention
        avg_out = cross_feat
        max_out, _ = torch.max(cross_feat, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return spatial_attn

    def forward(self, feat_A, feat_B):
        # Channel Attention
        channel_attn = self.channel_attention(feat_A, feat_B)
        feat_A = feat_A * channel_attn

        # Spatial Attention
        spatial_attn = self.spatial_attention(feat_A, feat_B)
        feat_A = feat_A * spatial_attn

        return feat_A


class SCTM(nn.Module):
    def __init__(
        self,
        p,
        tasks,
        stage,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.tasks = tasks
        self.norm_share = norm_layer(hidden_dim*len(tasks))
        self.selective = nn.ModuleDict({t: nn.ModuleDict({task: CEIM(hidden_dim,16) for task in self.tasks}) for t in self.tasks})
                
        self.fusion_conv = nn.ModuleDict({t: nn.Sequential(nn.Conv2d(hidden_dim*len(tasks), hidden_dim, 1),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)) for t in self.tasks})

        self.norm = nn.ModuleDict()
        self.op = nn.ModuleDict()
        self.fusion_norm = nn.ModuleDict()
        for tname in self.tasks:
            self.norm[tname] = norm_layer(hidden_dim)
            self.op[tname] = TaskFusionBlock(p, dim=hidden_dim)
            self.fusion_norm[tname] = norm_layer(hidden_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: dict):
        b, h, w, c = input[list(input.keys())[0]].shape
        
        _input = {t: input[t].permute(0,3,1,2) for t in self.tasks}
        #* 选择性任务特征融合
        fusion_selective = {}
        for task in self.tasks:
            fusion = []
            
            for t in self.tasks:
                task_selective = self.selective[task][t](_input[t],_input[task]) + _input[t]  # 用task信息，来选择t的信息，加入跳连
                fusion.append(task_selective)
                
            fusion_feature = self.fusion_conv[task](torch.cat(fusion,dim=1))
            fusion_selective[task] = self.fusion_norm[task](fusion_feature.permute(0,2,3,1))
            
        out = {}
        for t in self.tasks:
            x_t = input[t]
            x = x_t + self.drop_path(self.op[t](self.norm[t](x_t), fusion_selective[t]))    
            out[t] = x
        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by heads"
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # 定义线性变换层用于查询（Q）、键（K）和值（V）
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # 最终的全连接层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, queries, keys, values, mask=None):
        N, seq_length_q, embed_size = queries.shape
        N, seq_length_k, embed_size = keys.shape
        N, seq_length_v, embed_size = values.shape
        
        # 将嵌入维度分割成多个头
        values = values.reshape(N, seq_length_v, self.heads, self.head_dim)
        keys = keys.reshape(N, seq_length_k, self.heads, self.head_dim)
        queries = queries.reshape(N, seq_length_q, self.heads, self.head_dim)
        
        # 通过线性层
        values = self.values(values)  # (N, seq_length, heads, head_dim)
        keys = self.keys(keys)        # (N, seq_length, heads, head_dim)
        queries = self.queries(queries)  # (N, seq_length, heads, head_dim)
        
        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, seq_length, seq_length)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # 归一化分数
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, seq_length, seq_length)
        
        # 加权和值
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length_q, self.heads * self.head_dim)
        
        # 通过最终的全连接层
        out = self.fc_out(out)  # (N, seq_length, embed_size)
        
        return out


class GlobalPerceptionBlock(nn.Module):
    def __init__(self, p, dim, stage):
        super().__init__()
        self.p = p
        self.dim = dim
        self.stage = stage
        if 'attn_dim_r' in p:
            self.attn_dim_r = p['attn_dim_r']
        else:
            self.attn_dim_r = 2
        
        if "attn_hp_r" in p:
            self.attn_hp_r = p['attn_hp_r']
            assert self.attn_hp_r>=1 and type(self.attn_hp_r) == int, "config中attn_hp_r必须是大于等于1的整数"
        else:
            self.attn_hp_r = 2
            
        self.attn_dim = self.dim // self.attn_dim_r
        
        self.dim_down = nn.Conv2d(dim, self.attn_dim,1)
        
        self.hp = HP(p,self.attn_dim,stage,self.attn_hp_r)
        
        # 用平均池化和最大池化之和替代原来的 patchify 操作
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        
        self.attn = SelfAttention(self.attn_dim, 8)
        
        self.dim_up = nn.Conv2d(self.attn_dim,dim,1)
        self.norm2 = nn.GroupNorm(16, dim, eps=1e-6)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.dim_down(x)
        
        # 采用平均池化和最大池化的结果相加
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_patches = x_avg + x_max
        
        hp_enhance = self.hp(x)
        hp_enhance = F.interpolate(hp_enhance, size=(h // 4, w // 4), mode=INTERPOLATE_MODE)
        
        q = rearrange(x, "b c h w -> b (h w) c")
        k = rearrange(x_patches + hp_enhance, "b c h w -> b (h w) c")
        v = k
        x_sa = self.attn(q, k, v)
        x_sa = rearrange(x_sa, "b (h w) c -> b c h w", h=h, w=w)
        
        out = self.dim_up(x_sa)
        out = F.interpolate(out, size=(h, w), mode=INTERPOLATE_MODE)
        out = self.norm2(out)
        
        return out


class HPD(nn.Module):
    def __init__(self, p, dim, stage):
        super().__init__()
        self.linear = nn.Conv2d(dim,dim,1,1,0)
        self.norm_layer_1 = nn.GroupNorm(16, dim, eps=1e-6)
        self.dwconv_act = DWConvBlock(dim,dim,3,1,1)
        self.norm_layer_2 = nn.GroupNorm(16, dim, eps=1e-6)
        self.GP = GlobalPerceptionBlock(p, dim, stage)
        
    def forward(self, x):
        x = self.norm_layer_1(F.silu(self.linear(x.permute(0,3,1,2))))
        res = x
        x_1 = self.norm_layer_2(self.dwconv_act(x))
        x_2 = self.GP(x)
        out = (res + x_1 + x_2).permute(0,2,3,1)
        
        return out


class TaskFusionBlock(nn.Module):
    def __init__(self, p, dim):
        super().__init__()
        self.norm_layer_1 = nn.GroupNorm(16, dim, eps=1e-6)
        self.gate_prj = nn.Conv2d(dim,dim,1,1,0)
        self.dwconv_act = DWConvBlock(dim,dim,3,1,1)
        self.norm_layer_2 = nn.GroupNorm(16, dim, eps=1e-6)
        self.linear = nn.Conv2d(dim,dim,1,1,0)
        
    def forward(self, x_st, x_ct):
        x_st = x_st.permute(0,3,1,2)
        res = x_st
        x_ct = x_ct.permute(0,3,1,2)
        G = F.sigmoid(self.gate_prj(x_st))
        x_st = self.dwconv_act(self.norm_layer_1(x_st))
        out = (G*x_st + (1-G)*x_ct)
        out = self.linear(self.norm_layer_2(out))
        out = (out + res).permute(0,2,3,1)
        
        return out
    
    
class LP(nn.Module):
    def __init__(self, p, dim, stage, r, size_r=1):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.r = r
            
        if "feauture_sizes" in p:
            feauture_sizes = p['feauture_sizes']
        else:
            feauture_sizes = [[16,16],[32,32],[64,64],[128,128]]
        self.feauture_size = feauture_sizes[stage+1]
        
        self.register_buffer('hamming_win', torch.FloatTensor(hamming2D(int(self.feauture_size[0]*size_r), int(self.feauture_size[1]*size_r)))[None, None,])
        
        dim_hidden = dim//r
        # self.proj_in = nn.Conv2d(dim,dim_hidden,1)
        self.proj_in = DimPooling(dim, r, channel_last=False)
        self.dwconv_w_act = DWConvBlock(dim_hidden*2,dim_hidden*2)
        self.proj_out = nn.Conv2d(dim_hidden,dim,1)


    def forward(self, input):
        b, c, h, w = input.shape
        input = self.proj_in(input)

        y = torch.fft.fft2(input.float())
        y = y * self.hamming_win
        y_imag = y.imag
        y_real = y.real
        y = torch.cat([y_real, y_imag], dim=1)
        y = self.dwconv_w_act(y)            
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.ifft2(y, s=(h, w)).float()
        
        y = self.proj_out(y)
        return y
    

class LPBlock(nn.Module):
    def __init__(self, p, dim, stage):
        super().__init__()
        self.dim = dim
        self.stage = stage
        if "lp_r" in p:
            self.lp_r = p['lp_r']
            assert self.lp_r>=1 and type(self.lp_r) == int, "config中lp_r必须是大于等于1的整数"
        else:
            self.lp_r = 4
        
        # 定义网络
        self.lp = LP(p, dim, stage, self.lp_r, 0.5)
        self.norm = nn.GroupNorm(16, dim, eps=1e-6)
        
    def forward(self, x):
        b, c, h, w = x.shape
        res = x
        
        y = self.lp(x)
        out = self.norm(y + res)
        return out


class HP(nn.Module):
    def __init__(self, p, dim, stage, r, size_r=1):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.r = r
            
        if "feauture_sizes" in p:
            feauture_sizes = p['feauture_sizes']
        else:
            feauture_sizes = [[16,16],[32,32],[64,64],[128,128]]
        self.feauture_size = feauture_sizes[stage+1]
        
        self.register_buffer('hamming_win', 1 - torch.FloatTensor(hamming2D(int(self.feauture_size[0]*size_r), int(self.feauture_size[1]*size_r)))[None, None,])
        
        # 定义网络
        
        dim_hidden = dim//r
        
        # self.proj_in = nn.Conv2d(dim,dim_hidden,1)
        self.proj_in = DimPooling(dim, r, channel_last=False)
        self.dwconv_w_act = DWConvBlock(dim_hidden*2,dim_hidden*2)
        self.proj_out = nn.Conv2d(dim_hidden,dim,1)


    def forward(self, input):
        b, c, h, w = input.shape
        input = self.proj_in(input)

        y = torch.fft.fft2(input.float())
        y = y * self.hamming_win
        y_imag = y.imag
        y_real = y.real
        y = torch.cat([y_real, y_imag], dim=1)
        y = self.dwconv_w_act(y)            
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.ifft2(y, s=(h, w)).float()
        
        y = self.proj_out(y)
        return y
    

class HPBlock(nn.Module):
    def __init__(self, p, dim, stage):
        super().__init__()
        self.dim = dim
        self.stage = stage
        if "hp_r" in p:
            self.hp_r = p['hp_r']
            assert self.hp_r>=1 and type(self.hp_r) == int, "config中hp_r必须是大于等于1的整数"
        else:
            self.hp_r = 4
        
        # 定义网络
        self.hp = HP(p, dim, stage, self.hp_r, 1)
        self.norm = nn.GroupNorm(16, dim, eps=1e-6)
        
    def forward(self, x):
        b, c, h, w = x.shape
        res = x
        
        y = self.hp(x)
        out = self.norm(y + res)
        return out
    

class FAFM(nn.Module):
    def __init__(self, p, current_channel, skip_channel, tasks, stage):
        super().__init__()
        self.stage = stage
        self.tasks = tasks
        self.current_channel = current_channel
        self.skip_channel = skip_channel
        self.low_ks = 3
        self.high_ks = 3
        
        self.register_buffer('hamming_low_conv_win', torch.FloatTensor(hamming2D(self.low_ks , self.low_ks))[None, None,])
        self.register_buffer('hamming_high_conv_win', torch.FloatTensor(hamming2D(self.high_ks , self.high_ks))[None, None,])
        
        if "compress_r" in p:
            self.compress_r = p['compress_r']
            assert self.compress_r>=1 and type(self.compress_r) == int, "config中compress_r必须是大于等于1的整数"
        else:
            self.compress_r = 1
            
        if "deform_groups_SE_hp" in p:
            self.deform_groups_SE_hp = p['deform_groups_SE_hp']
            assert self.deform_groups_SE_hp>=1 and type(self.deform_groups_SE_hp) == int, "config中deform_groups_SE_hp必须是大于等于1的整数"
        else:
            self.deform_groups_SE_hp = 1
            
        if "deform_groups_SE_lp" in p:
            self.deform_groups_SE_lp = p['deform_groups_SE_lp']
            assert self.deform_groups_SE_lp>=1 and type(self.deform_groups_SE_lp) == int, "config中deform_groups_SE_lp必须是大于等于1的整数"
        else:
            self.deform_groups_SE_lp = 1
        
        self.compress_dim = skip_channel//self.compress_r

        self.lp_enhance = nn.ModuleDict()
        self.hp_enhance = nn.ModuleDict()
        self.compress_highpath = nn.ModuleDict()
        self.compress_lowpath = nn.ModuleDict()
        self.low_encoder = nn.ModuleDict()
        self.high_encoder = nn.ModuleDict()
        self.dim_down = nn.ModuleDict()
        
        for t in self.tasks:
            self.lp_enhance[t] = LPBlock(p, skip_channel, stage)
            self.hp_enhance[t] = HPBlock(p, skip_channel, stage)
            self.dim_down[t] = nn.Conv2d(current_channel, skip_channel, 1)
            self.low_encoder[t] = nn.Conv2d(self.compress_dim, 3*self.low_ks ** 2, 3, 1, 1)
            self.high_encoder[t] = nn.Conv2d(self.compress_dim, 3*self.high_ks ** 2, 3, 1, 1)
            self.compress_lowpath[t] = nn.Conv2d(skip_channel, self.compress_dim, 1)
            self.compress_highpath[t] = nn.Conv2d(skip_channel, self.compress_dim, 1)
            
    def forward(self, x_dict, skip):
        b, h, w, c = x_dict[list(x_dict.keys())[0]].shape
        x_dict = {t: self.dim_down[t](x_dict[t]) for t in self.tasks}
        x_low_enhanced_1 = {t: F.interpolate(self.lp_enhance[t](x_dict[t]), scale_factor=2, mode=INTERPOLATE_MODE) for t in self.tasks}
        x_high_enhanced_1 = {t: self.hp_enhance[t](skip) for t in self.tasks}
        
        x_low_enhanced_1_dimdown = {t: self.compress_lowpath[t](x_low_enhanced_1[t]) for t in self.tasks}
        x_high_enhanced_1_dimdown = {t: self.compress_highpath[t](x_high_enhanced_1[t]) for t in self.tasks}
        
        x_fusion = {t: x_low_enhanced_1_dimdown[t] + x_high_enhanced_1_dimdown[t] for t in self.tasks}
        x_low_enhanced_2 = {}
        x_high_enhanced_2 = {}
        for t in self.tasks:
            kernal_feature = x_fusion[t]
            
            # low path
            low_conv_feature = self.low_encoder[t](kernal_feature)
            offset = low_conv_feature[:, :2*self.low_ks*self.low_ks, :, :]  # 形状: (b, k^2, h, w)
            low_conv_weights = low_conv_feature[:, 2*self.low_ks*self.low_ks:, :, :]  # 形状: (b, k^2, h, w)
            low_conv_weights = self.kernel_normalizer(low_conv_weights, self.low_ks, hamming=self.hamming_low_conv_win) 
            low_feature = F.interpolate(x_dict[t], scale_factor=2, mode=INTERPOLATE_MODE)
            x_low_enhanced_2[t] = low_feature + self.carafe_deformable(low_feature, offset, low_conv_weights, self.low_ks, padding=0)

            # high path
            high_conv_feature = self.high_encoder[t](kernal_feature)
            offset = high_conv_feature[:, :2*self.high_ks*self.high_ks, :, :]  # 形状: (b, k^2, h, w)
            high_conv_weights = high_conv_feature[:, 2*self.high_ks*self.high_ks:, :, :]  # 形状: (b, k^2, h, w)
            high_conv_weights = self.kernel_normalizer(high_conv_weights, self.high_ks, hamming=self.hamming_high_conv_win) 
            x_high_enhanced_2[t] = skip + skip - self.carafe_deformable(low_feature, offset, high_conv_weights, self.high_ks, padding=0)
        
        return x_low_enhanced_1, x_high_enhanced_1, x_low_enhanced_2, x_high_enhanced_2
    
    def carafe_deformable(self, features, offset, conv_weights, k, padding=0):
        """
        在神经网络中通过卷积提取特征图，并基于偏移量和卷积权重进行高效采样和卷积操作。

        :param features: Tensor of shape (b, c, h, w), 特征图
        :param offset: Tensor of shape (b, 2*k^2, h, w), 每个像素的偏移量
        :param conv_weights: Tensor of shape (b, k^2, h, w), 每个像素对应的卷积权重
        :param k: 卷积核大小（例如，k=3 表示 3x3 卷积核）
        :param padding: 卷积的填充大小（默认为 0）
        :return: Tensor of shape (b, c, h, w), 卷积后的输出特征图
        """
        b, c, h, w = features.shape
        _, k2, h_offset, w_offset = offset.shape
        
        assert k2 == 2 * k * k, f"Offset 的通道数应为 2*k^2, 但得到 {k2}"
        assert h_offset == h and w_offset == w, "Offset 的空间维度应与特征图一致"
        
        offset_x = offset[:, :k*k, :, :]  # 形状: (b, k^2, h, w)
        offset_y = offset[:, k*k:, :, :]  # 形状: (b, k^2, h, w)
        base_x = torch.arange(w, device=features.device).view(1, 1, 1, w).expand(b, k*k, h, w)
        base_y = torch.arange(h, device=features.device).view(1, 1, h, 1).expand(b, k*k, h, w)
        
        grid_x = base_x + offset_x  # 形状: (b, k^2, h, w)
        grid_y = base_y + offset_y  # 形状: (b, k^2, h, w)
        grid_x = 2.0 * grid_x / (w - 1) - 1.0
        grid_y = 2.0 * grid_y / (h - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # 形状: (b, k^2, h, w, 2)
        grid = grid.view(b * k * k, h, w, 2)        # 形状: (b*k^2, h, w, 2)

        features_expanded = features.unsqueeze(1).repeat(1, k*k, 1, 1, 1)  # 形状: (b, k^2, c, h, w)
        features_expanded = features_expanded.view(b * k * k, c, h, w)      # 形状: (b*k^2, c, h, w)
        sampled_features = F.grid_sample(features_expanded, grid, align_corners=True, padding_mode="reflection")  # 形状: (b*k^2, c, h, w)
        sampled_features = sampled_features.view(b, k * k, c, h, w)  # 形状: (b, k^2, c, h, w)
        
        conv_weights = conv_weights.unsqueeze(2)  # 形状: (b, k^2, 1, h, w)
        weighted_features = sampled_features * conv_weights  # 形状: (b, k^2, c, h, w)
        output = weighted_features.sum(dim=1)  # 形状: (b, c, h, w)
        
        return output
    
    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2)) # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)  # 卷积核softmax归一化
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming                           # 用hamming窗作为基本的卷积基础，mask是学习的卷积核参数变化
        mask /= mask.sum(dim=(-1, -2), keepdims=True)   # 卷积核线性归一化
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask


class HiFAN(nn.Module):
    def __init__(self, p, backbone, d_state=16, dt_rank="auto", ssm_ratio=2, mlp_ratio=0):
        super().__init__()
        self.p = p
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.feature_channel = backbone.num_features

        each_stage_depth = 3
        stage_num = len(self.feature_channel) - 1

        dpr = [x.item() for x in torch.linspace(0.2, 0, stage_num*each_stage_depth)]

        self.SE = nn.ModuleDict()
        
        self.concat_layers = nn.ModuleDict()
        self.block_stm = nn.ModuleDict()
        self.final_project = nn.ModuleDict()
        self.final_expand = nn.ModuleDict()
        self.SE_norm_low = nn.ModuleDict()
        self.SE_norm_high = nn.ModuleDict()
        
        for stage in range(len(self.feature_channel) - 1):
            current_channel = self.feature_channel[::-1][stage]
            skip_channel = self.feature_channel[::-1][stage+1]
            self.SE[f'{stage}'] = FAFM(p, current_channel, skip_channel, self.tasks, stage)
        
        for t in self.tasks:
            for stage in range(len(self.feature_channel) - 1):
                current_channel = self.feature_channel[::-1][stage]
                skip_channel = self.feature_channel[::-1][stage+1]
                
                self.SE_norm_low[f'{t}_{stage}'] = nn.GroupNorm(16, skip_channel, eps=1e-6)
                self.SE_norm_high[f'{t}_{stage}'] = nn.GroupNorm(16, skip_channel, eps=1e-6)
                self.concat_layers[f'{t}_{stage}'] = nn.Conv2d(2*skip_channel, skip_channel, 1)

                stm_layer = [HPD(p, dim=skip_channel, stage=stage) for i in range(2)]
                self.block_stm[f'{t}_{stage}'] = nn.Sequential(*stm_layer)

            self.final_expand[t] = nn.Sequential(
                nn.Conv2d(self.feature_channel[0], 96, 3, padding=1), 
                nn.SyncBatchNorm(96), 
                nn.ReLU(True)
            )
            trunc_normal_(self.final_expand[t][0].weight, std=0.02)

            self.final_project[t] = nn.Conv2d(96, p.TASKS.NUM_OUTPUT[t], 1)

        self.block_ctm = nn.ModuleDict()
        for stage in range(len(self.feature_channel) - 1):
            skip_channel = self.feature_channel[::-1][stage+1]

            ctm_layer = [SCTM(p,
                            tasks=self.tasks,
                            stage=stage,
                            hidden_dim=skip_channel,
                            drop_path=dpr[each_stage_depth*(stage)+2],
                            norm_layer=nn.LayerNorm,
                            ssm_ratio=ssm_ratio,
                            d_state=d_state,
                            mlp_ratio=mlp_ratio,
                            dt_rank=dt_rank)]
            self.block_ctm[f'{stage}'] = nn.Sequential(*ctm_layer)
            
        self.intermediate_head = nn.ModuleDict()
        self.preliminary_decoder = nn.ModuleDict()
        channels = self.feature_channel[::-1]
        for t in p.TASKS.NAMES:
            self.preliminary_decoder[t] = InitialBlock(channels[0], channels[0])
            if p['intermediate_supervision']:
                self.intermediate_head[t] = nn.Conv2d(channels[0], p.TASKS.NUM_OUTPUT[t], 1)

    def _forward_expand(self, x_dict, selected_fea: list, stage: int) -> dict:
        aux_out = -1
        if stage == 0:
            x_dict = {t: self.preliminary_decoder[t](selected_fea[-1]) for t in self.tasks}
            if self.p['intermediate_supervision']:
                aux_out = {t: self.intermediate_head[t](x_dict[t]) for t in self.tasks}
            
        skip = selected_fea[::-1][stage+1]
        out = {}
        x_low_enhanced_1, x_high_enhanced_1, x_low_enhanced_2, x_high_enhanced_2 = self.SE[f'{stage}'](x_dict, skip)
        x_low_enhanced = {t: self.SE_norm_low[f'{t}_{stage}'](x_low_enhanced_1[t] + x_low_enhanced_2[t]) for t in self.tasks}
        x_high_enhanced = {t: self.SE_norm_high[f'{t}_{stage}'](x_high_enhanced_1[t] + x_high_enhanced_2[t]) for t in self.tasks}
        for t in self.tasks:
            x = torch.cat((x_low_enhanced[t], x_high_enhanced[t]), dim=1)
            x = self.concat_layers[f'{t}_{stage}'](x)
            out[t] = x.permute(0,2,3,1)
        return out, aux_out # out: B,H,W,C

    def _forward_block_stm(self, x_dict: dict, stage: int) -> dict:
        out = {}
        for t in self.tasks:
            out[t] = self.block_stm[f'{t}_{stage}'](x_dict[t])
        return out

    def _forward_block_ctm(self, x_dict: dict, stage: int) -> dict:
        return self.block_ctm[f'{stage}'](x_dict)


    def forward(self, x):
        img_size = x.size()[-2:]

        # Backbone 
        selected_fea = self.backbone(x)

        x_dict = None
        aux_outs = []
        for stage in range(len(self.feature_channel) - 1):
            x_dict, aux_out = self._forward_expand(x_dict, selected_fea, stage)
            aux_outs.append(aux_out)
            x_dict = self._forward_block_ctm(x_dict, stage)
            x_dict = self._forward_block_stm(x_dict, stage)
            x_dict = {t: xx.permute(0,3,1,2) for t, xx in x_dict.items()}

        out = {}
        for t in self.tasks:
            z = self.final_expand[t](x_dict[t])
            z = self.final_project[t](z)
            out[t] = F.interpolate(z, img_size, mode=INTERPOLATE_MODE)
        
        if self.p['intermediate_supervision']:
            aux_out = aux_outs[0]
            for t in self.tasks:
                aux_out[t] = F.interpolate(aux_out[t], img_size, mode=INTERPOLATE_MODE)
            out['inter_preds'] = aux_out
        
        return out
