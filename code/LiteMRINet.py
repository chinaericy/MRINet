from __future__ import print_function, division

import torch.utils.data

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    # print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")



class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)

        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class UnetDecoder(nn.Module):
    def __init__(self, depths=[32, 32, 32, 32, 32]):
        super().__init__()

        self.ConvTranspose4_3 = up_conv(depths[4], depths[3])
        self.Conv4_3 = conv_block(depths[3] * 2, depths[3])

        self.ConvTranspose4_2 = up_conv(depths[3], depths[2])
        self.Conv4_2 = conv_block(depths[2] * 2, depths[2])

        self.ConvTranspose4_1 = up_conv(depths[2], depths[1])
        self.Conv4_1 = conv_block(depths[1] * 2, depths[1])

        self.ConvTranspose4_0 = up_conv(depths[1], depths[0])
        self.Conv4_0 = conv_block(depths[0], depths[0])

        self.Out = nn.Conv2d(depths[0], 2, kernel_size=1, stride=1, padding=0)

    def forward(self, levels):
        # level4的解码
        level4 = self.ConvTranspose4_3(levels[4])
        level4 = torch.cat([level4, levels[3]], dim=1)
        level4 = self.Conv4_3(level4)

        level4 = self.ConvTranspose4_2(level4)
        level4 = torch.cat([level4, levels[2]], dim=1)
        level4 = self.Conv4_2(level4)

        level4 = self.ConvTranspose4_1(level4)
        level4 = torch.cat([level4, levels[1]], dim=1)
        level4 = self.Conv4_1(level4)

        level4 = self.ConvTranspose4_0(level4)
        # level4 = torch.cat([level4, levels[0]], dim=1)
        level4 = self.Conv4_0(level4)

        out = self.Out(level4)

        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#,B_,num_heads,N,C // self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        torch.cuda.empty_cache()

        attn = (q @ k.transpose(-2, -1))#q为B_,num_heads,N,C // self.num_heads，K为#,B_,num_heads,C // self.num_heads,N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # def extra_repr(self) -> str:
    #     return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    #
    # def flops(self, N):
    #     # calculate flops for 1 window with token length of N
    #     flops = 0
    #     # qkv = self.qkv(x)
    #     flops += N * self.dim * 3 * self.dim
    #     # attn = (q @ k.transpose(-2, -1))
    #     flops += self.num_heads * N * (self.dim // self.num_heads) * N
    #     #  x = (attn @ v)
    #     flops += self.num_heads * N * N * (self.dim // self.num_heads)
    #     # x = self.proj(x)
    #     flops += N * self.dim * self.dim
    #     return flops
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ViTSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,input_resolution, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        '''self.attn=BuildFormer.geoseg.models.BuildFormer.LWMSA(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias)
        '''
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):

        B, C,H,W = x.shape
        x = x.permute(0, 2, 3, 1)  # b,h,w,c

        shortcut = x
        x = self.norm1(x)


        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2) #B,C,H,W

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


#LiteMRINet
class LiteMRINet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.local = nn.Sequential(
            conv_block(in_ch, filters[0]),
            conv_block(filters[0], filters[0]),
            conv_block(filters[0], filters[0]),
            conv_block(filters[0], filters[0]),
            conv_block(filters[0], filters[0]),
            # conv_block(filters[0], filters[0]),
            # conv_block(filters[0], filters[0]),
            # conv_block(filters[0], filters[0]),
            # conv_block(filters[0], filters[0]),
            # conv_block(filters[0], filters[0])
        )

        # self.Conv1 = conv_block(in_ch, 64)


        # self.Conv2 = nn.Sequential(
        #     conv_block(filters[0], filters[1]),
        #     conv_block(filters[1], filters[1])
        #                            )
        # self.Conv3 = nn.Sequential(
        #     conv_block(filters[1], filters[2]),
        #     conv_block(filters[2], filters[2])
        #                            )
        # self.Conv4 = nn.Sequential(
        #     conv_block(filters[2], filters[3]),
        #     conv_block(filters[3], filters[3])
        #                            )
        # self.Conv5 = nn.Sequential(
        #     conv_block(filters[3], filters[4]),
        #     conv_block(filters[4], filters[4])
        #                            )




        # self.Trans1=nn.Sequential(
        #     ViTSwinTransformerBlock([256,256],dim=64,num_heads=1,window_size=8,shift_size=0),
        #     ViTSwinTransformerBlock([256, 256], dim=64, num_heads=1, window_size=8, shift_size=4),
        #     )
        # self.Trans2 = nn.Sequential(
        #     ViTSwinTransformerBlock([128, 128], dim=64, num_heads=2, window_size=8, shift_size=0),
        #     ViTSwinTransformerBlock([128, 128], dim=64, num_heads=2, window_size=8, shift_size=4),
        # )
        # self.Trans3 = nn.Sequential(
        #     ViTSwinTransformerBlock([64, 64], dim=64, num_heads=4, window_size=8, shift_size=0),
        #     ViTSwinTransformerBlock([64, 64], dim=64, num_heads=4, window_size=8, shift_size=4),
        # )
        # self.merge4=VitPatchMerging(filters[0])
        self.Trans4 = nn.Sequential(
            ViTSwinTransformerBlock([32, 32], dim=filters[0], num_heads=8, window_size=32, shift_size=0),
            ViTSwinTransformerBlock([32, 32], dim=filters[0], num_heads=8, window_size=32, shift_size=0),
            ViTSwinTransformerBlock([32, 32], dim=filters[0], num_heads=8, window_size=32, shift_size=0),
            ViTSwinTransformerBlock([32, 32], dim=filters[0], num_heads=8, window_size=32, shift_size=0),
            ViTSwinTransformerBlock([32, 32], dim=filters[0], num_heads=8, window_size=32, shift_size=0),
            ViTSwinTransformerBlock([32, 32], dim=filters[0], num_heads=8, window_size=32, shift_size=0),
        )
        # self.merge5 = VitPatchMerging(filters[1])
        self.Trans5 = nn.Sequential(
            ViTSwinTransformerBlock([16, 16], dim=filters[0], num_heads=8, window_size=16, shift_size=0),
            ViTSwinTransformerBlock([16, 16], dim=filters[0], num_heads=8, window_size=16, shift_size=0),
            ViTSwinTransformerBlock([16, 16], dim=filters[0], num_heads=8, window_size=16, shift_size=0),
            ViTSwinTransformerBlock([16, 16], dim=filters[0], num_heads=8, window_size=16, shift_size=0),

        )

        # self.Trans = nn.Sequential(
        #     ViTSwinTransformerBlock([16, 16], dim=64, num_heads=16, window_size=16, shift_size=0),
        #     ViTSwinTransformerBlock([16, 16], dim=64, num_heads=16, window_size=16, shift_size=4),
        #
        #     ViTSwinTransformerBlock([16, 16], dim=64, num_heads=16, window_size=16, shift_size=0),
        #     ViTSwinTransformerBlock([16, 16], dim=64, num_heads=16, window_size=16, shift_size=4),
        # )


        # self.mutiDecoder = mutiDecoder(filters)
        # self.UnetDecoder = UnetDecoder(filters)
        # self.Conv64_2 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        #
        # self.Conv128_64=nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0)
        # self.Conv256_64=nn.Conv2d(256,64,kernel_size=1,stride=1,padding=0)
        # self.Conv512_64=nn.Conv2d(512,64,kernel_size=1,stride=1,padding=0)
        # self.Conv1024_64=nn.Conv2d(1024,64,kernel_size=1,stride=1,padding=0)
        # #
        # self.ConvOut1 = nn.Conv2d(64, 16 , kernel_size=1, stride=1, padding=0)
        # self.ConvOut2 = nn.Conv2d(16 , 2, kernel_size=1, stride=1, padding=0)
        self.UnetDecoder=UnetDecoder([filters[0],filters[0],filters[0],filters[0],filters[0]])
        # self.MutiDecoder=mutiDecoder([64,64,64,64,64])
    def forward(self, input):

        e1=input

        e2=F.interpolate(input,size=(128,128))
        e2= self.local(e2)

        e3=F.interpolate(input,size=(64,64))
        e3= self.local(e3)

        # e4=F.interpolate(input,size=(32,32))
        # e4= self.local(e4)
        # #
        # e5=F.interpolate(input,size=(16,16))
        # e5= self.local(e5)


        # local = self.Conv(input)
        #

        e4=self.Maxpool1(e3)
        # e4=self.merge4(e3)
        e4=self.Trans4(e4)


        e5=self.Maxpool4(e4)
        # e5 = self.merge5(e4)
        e5=self.Trans5(e5)


        #
        # g1 = self.Conv64_2(g1)
        #
        # out=F.interpolate(g1,size=(256,256))
        #
        #
        # # out=torch.cat([local,g1],dim=1)
        #
        #
        # # out=self.ConvOut1(out)
        # # out=self.ConvOut2(out)


        out = self.UnetDecoder([e1, e2, e3, e4, e5])

        return out




