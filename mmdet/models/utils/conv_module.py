import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import constant_init, kaiming_init, normal_init

from .conv_ws import ConvWS2d
from .norm import build_norm_layer

conv_cfg = {
    'Conv': nn.Conv2d,
    'ConvWS': ConvWS2d,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(conv_cfg,
                                     in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x


class NonLocalModule(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 sub_sample=False):
        super(NonLocalModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.sub_sample = sub_sample
        self.inter_channels = inter_channels

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build value conv, no norm after conv, so with bias
        self.value_conv = build_conv_layer(conv_cfg,
                                           in_channels,
                                           inter_channels,
                                           kernel_size=1,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=True)
        # # export the attributes of self.conv to a higher level for convenience
        # self.in_channels = self.conv.in_channels
        # self.out_channels = self.conv.out_channels
        # self.kernel_size = self.conv.kernel_size
        # self.stride = self.conv.stride
        # self.padding = self.conv.padding
        # self.dilation = self.conv.dilation
        # self.transposed = self.conv.transposed
        # self.output_padding = self.conv.output_padding
        # self.groups = self.conv.groups

        # build query conv
        self.query_conv = build_conv_layer(conv_cfg,
                                           in_channels,
                                           inter_channels,
                                           kernel_size=1,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=True)

        # build key conv
        self.key_conv = build_conv_layer(conv_cfg,
                                         in_channels,
                                         inter_channels,
                                         kernel_size=1,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True)

        # build the conv after aggeregate non-local features
        self.W = build_conv_layer(conv_cfg,
                                  inter_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias)

        # build the normalization layer after aggeregate non-local features
        if self.with_norm:
            norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        # whether downsample the key and value
        if self.sub_sample:
            self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))

        # Use msra init by default for the first three convs anh const_init for the fourth
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):

        normal_init(self.query_conv, std=0.01)
        normal_init(self.key_conv, std=0.01)
        normal_init(self.value_conv, std=0.01)
        normal_init(self.W, std=0.01)
        if self.with_norm:
            constant_init(self.norm, val=0, bias=0)

    def forward(self, x, activate=True, norm=True):
        b, c, h, w = x.shape
        value_x = self.value_conv(x).view(b, c, -1)
        value_x = value_x.permute(0, 2, 1)
        query_x = self.query_conv(x).view(b, c, -1)
        query_x = query_x.permute(0, 2, 1)
        key_x = self.key_conv(x).view(b, c, -1)

        f = torch.matmul(query_x, key_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, value_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=2):
        '''
        state_dim: 输入通道数（特征维度）
        out_state_dim: 输出通道数
        '''
        super(CircConv, self).__init__()
        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim,
                            out_state_dim,
                            kernel_size=self.n_adj * 2 + 1)

    def forward(self, input, adj):
        input = torch.cat(
            [input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


_conv_factory = {'grid': CircConv}


class SnakeBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type='grid', n_adj=1):
        super(SnakeBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj=None):
        '''
            x:(num_center,66,40) 
            adj:(40,4)
        '''
        x = self.conv(x, adj)
        x = self.relu(x)
        x = self.norm(x)
        return x


class SnakeProBlock(nn.Module):
    def __init__(self,
                 state_dim,
                 out_state_dim,
                 feature_dim=128,
                 res_layer_num=4,
                 conv_type='grid',
                 n_adj=1):
        super(SnakeProBlock, self).__init__()

        # self.head = BasicBlock(feature_dim, state_dim,
        #                        conv_type)  # (29,128,40) 圆形卷积

        # compress the feature
        self.compress_conv = nn.Conv1d(state_dim, feature_dim, 1)
        self.res_layer_num = res_layer_num
        # dilation = [1, 1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = SnakeBlock(feature_dim, feature_dim)
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(feature_dim * (self.res_layer_num + 1),
                                fusion_state_dim, 1)  # 卷积核为1
        self.fuse_conv = nn.Conv1d(
            feature_dim * (self.res_layer_num + 1) + fusion_state_dim,
            out_state_dim, 1)
        # self.prediction = nn.Sequential(
        #     nn.Conv1d(
        #         feature_dim * (self.res_layer_num + 1) + fusion_state_dim, 256,
        #         1), nn.ReLU(inplace=True), nn.Conv1d(256, 64, 1),
        #     nn.ReLU(inplace=True), nn.Conv1d(64, 2, 1))

    def forward(self, x, adj=None):
        '''
        x: (num_center,66,num_points) (num_center,66,40)
        adj: (num, 4) 每个点的正负2的下标
        '''
        states = []
        x = self.compress_conv(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x, adj) + x  # 跳跃链接
            states.append(x)

        state = torch.cat(states, dim=1)  # (num_center,128*8,40)
        global_state = torch.max(
            self.fusion(state), dim=2,
            keepdim=True)[0]  # (num_center,256,1) 每个特征维度上40个点的最大值
        global_state = global_state.expand(global_state.size(0),
                                           global_state.size(1),
                                           state.size(2))  # 复制40次全局特征
        # 将全局特征和每个点的局部特征concat (num_center,128*8+256,40)
        state = torch.cat([global_state, state], dim=1)
        x = self.fuse_conv(state)  # (num_center, 2, 40)

        return x


class FuseIntermediateBlock(nn.Module):
    def __init__(self, len_state, fusion_out_state_dim, feature_dim=256):
        '''
        len_state: 包含backbone和mask head中间特征的feat map个数
        fusion_out_state_dim: 全局特征的维度
        '''
        super(FuseIntermediateBlock, self).__init__()
        self.fuse_intermediate = nn.Conv2d(36 * feature_dim * len_state,
                                           36 * fusion_out_state_dim,
                                           3,
                                           1,
                                           1,
                                           groups=36)
        self.conv1 = nn.Conv2d(
            36 * (feature_dim * len_state + fusion_out_state_dim),
            36 * 512,
            3,
            1,
            1,
            groups=36)
        self.conv2 = nn.Conv2d(36 * 512, 36 * 256, 3, 1, 1, groups=36)
        # self.init_weight()

    def init_wieght(self):
        normal_init(self.fuse_intermediate, std=0.01)
        normal_init(self.conv1, std=0.01)
        normal_init(self.conv2, std=0.01)

    def forward(self, states):
        init_feat = torch.cat(states, axis=2)  # (N,36,5C,H,W)
        n, _, _, h, w = init_feat.shape
        global_feat = init_feat.contiguous().view(n, -1, h, w)  # (N,180C,H,W)
        global_feat = self.fuse_intermediate(global_feat)  # (N,36*512,H,W)
        global_feat = global_feat.contiguous().view(n, 36, -1, h, w)
        global_feat = torch.max(global_feat, dim=1,
                                keepdim=True)[0].expand(-1, 36, -1, -1,
                                                        -1)  # (N,36,512,H,W)
        refine_feat = torch.cat([init_feat, global_feat],
                                axis=2)  # (N,36,5C+512,H,W)
        refine_feat = refine_feat.contiguous().view(n, -1, h, w)
        refine_feat = self.conv1(refine_feat)
        refine_feat = self.conv2(refine_feat)
        refine_feat = refine_feat.contiguous().view(n, 36, -1, h, w)
        return refine_feat