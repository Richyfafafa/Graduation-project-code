import numpy as np

import torch
import torch.nn as nn
#import better_cubemap
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _cubemapencoder as _backend
except ImportError:
    from .backend import _backend

_interp_to_id = {
    'nearest': 0,
    'linear': 1
}

class _cubemap_encode(torch.autograd.Function):
    """
    自定义 autograd Function，直接调用 CUDA 扩展完成 cubemap 采样。

    约定张量形状:
    - inputs: [B, 3]，每行为一个方向向量 (x, y, z)
    - embeddings: [6, C, L, L]，六个面、每面 C 通道、分辨率 LxL
    - fail_value: [C]，当输入方向为 (0,0,0) 时的回退输出
    - outputs(内部): [C, B]，最终在模块层会转置为 [B, C]
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    #@custom_fwd
    def forward(ctx, inputs, embeddings, fail_value, interpolation, enable_seamless):
        # CUDA 侧要求 contiguous，避免隐式拷贝和 stride 带来的额外开销
        embeddings = embeddings.contiguous()
        inputs = inputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        # CUDA kernel 输出采用 [C, B] 的内存布局
        outputs = torch.empty([C,B],dtype=embeddings.dtype,device=embeddings.device)

        # 调用 C++/CUDA 扩展：
        # interpolation: 0=nearest, 1=bilinear
        # enable_seamless: 0=普通采样, 1=跨面无缝采样
        _backend.cubemap_encode_forward(
            inputs, embeddings, fail_value, outputs, interpolation, enable_seamless,
            B, C, L
        )

        # backward 需要的上下文
        params = torch.tensor([interpolation,enable_seamless], dtype=torch.int64)
        ctx.save_for_backward(inputs, embeddings, params)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        # 取回 forward 保存的变量
        inputs, embeddings, params = ctx.saved_variables
        grad_outputs = grad_outputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        # 对 embeddings、inputs、fail_value 分别求梯度
        grad_embeddings = torch.zeros_like(embeddings)
        grad_inputs = torch.empty_like(inputs)
        grad_fail =  torch.zeros([C], dtype=embeddings.dtype, device=embeddings.device)

        _backend.cubemap_encode_backward(
            grad_outputs, inputs, embeddings,
            grad_embeddings, grad_inputs, grad_fail,
            params[0].item(), params[1].item(),
            B, C, L
        )
        return grad_inputs, grad_embeddings, grad_fail, None, None

cubemap_encode = _cubemap_encode.apply

'''
class _mipcubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,xxx):
        pass # TODO
    @staticmethod
    @custom_bwd
    def backward(ctx,xxx):
        pass

mipcubemap_encode = _mipcubemap_encode.apply
'''

class CubemapEncoder(nn.Module):
    """
    单层可训练 Cubemap 编码器。

    作用:
    - 将方向向量映射到可训练的 cubemap 纹理上，输出 C 维特征。
    - 常用于环境光/反射/透射颜色查询。

    参数:
    - output_dim: 每个方向输出的通道数 C（例如 RGB 时为 3）
    - resolution: 每个 cubemap 面的分辨率 L
    - interpolation: 'nearest' 或 'linear'
    """
    def __init__(self, output_dim = 6, resolution = 256, interpolation='linear'):
        super().__init__()

        self.input_dim = 3
        self.resolution = resolution
        self.output_dim = output_dim

        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        # 开启跨面无缝采样，降低 cubemap 六个面拼接边界处的接缝
        self.seamless = 1

        self.params = nn.ParameterDict({
            # 六个面: [+x, -x, +y, -y, +z, -z]
            # 初始化在较大范围，后续通常配合外部 sigmoid 使用
            'Cubemap_texture': nn.Parameter(torch.rand(6, self.output_dim, resolution, resolution)*10-5), 
            # 输入方向为零向量时的回退值
            'Cubemap_failv': nn.Parameter(torch.zeros(self.output_dim))
        })
        self.n_elems = 6 * self.output_dim * resolution * resolution + self.output_dim

    def __repr__(self):
        return f"CubemapEncoder: input_dim={self.input_dim} output_dim={self.output_dim} resolution={self.resolution} -> {self.n_elems} interpolation={self.interpolation} seamless={self.seamless}"

    def forward(self, inputs):
        # cubemap_encode 返回 [C, B]，这里转为 [B, C] 便于上层网络使用
        outputs = cubemap_encode(inputs, self.params['Cubemap_texture'], self.params['Cubemap_failv'], self.interp_id, self.seamless)
        return outputs.permute(1,0) # CxN -> NxC


class MipCubemapEncoder(nn.Module):
    """
    多分辨率（mip-like）Cubemap 编码器。

    思路:
    - 维护多个分辨率层级的 cubemap 参数。
    - 每层独立采样后，按 concat 或 sum 方式融合。

    典型用途:
    - 提升表达能力（低频+高频联合）
    - 在不同尺度上表达方向相关外观
    """
    def __init__(self, num_levels=4, level_dim=6, per_level_scale=4, base_resolution=4,interpolation='linear',concat=True):
        super().__init__()
        # default 4x 16x 64x 256x

        self.input_dim = 3
        self.num_levels = num_levels
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.concat = concat
        self.output_dim = num_levels * level_dim if concat else level_dim

        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        self.seamless = 1

        params_list = []
        L = float(base_resolution)
        n_elems = 0
        for ii in range(num_levels):
            iL = int(np.ceil(L))
            # 每层均为 [6, level_dim, iL, iL]
            param = nn.Parameter(torch.empty(6,self.level_dim,iL,iL))
            n_elems += 6 * self.level_dim * iL * iL
            params_list.append(param)
            L = L * per_level_scale
        #self.params_list = nn.ModuleList(*params_list)
        self.params_list = nn.ParameterList(params_list)

        self.fail_value = nn.Parameter(torch.zeros(self.level_dim))
        n_elems += self.level_dim
        self.n_elems = n_elems # total parameters

        self.reset_parameters()

    def reset_parameters(self):
        # 小范围初始化，多层叠加时更稳定
        std = 1e-4
        for ii in range(self.num_levels):
            self.params_list[ii].data.uniform_(-std,std)

    def __repr__(self):
        return f"MipCubemapEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} -> {self.n_elems} per_level_scale={self.per_level_scale:.4f} interpolation={self.interpolation} seamless={self.seamless}"

    def forward(self, inputs):
        outputs = []
        for ii in range(self.num_levels):
            # 每层输出形状: [level_dim, B]
            x = cubemap_encode(inputs, self.params_list[ii], self.fail_value, self.interp_id, self.seamless)
            outputs.append(x)
        if self.concat:
            # 按通道拼接: [num_levels * level_dim, B]
            outputs = torch.cat(outputs,dim=0)
        else:
            # 逐层相加: [level_dim, B]
            outputs = sum(outputs)
        return outputs.permute(1,0) # CxN -> NxC





