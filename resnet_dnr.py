import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ...registry import BACKBONES
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

class bottleneck_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(bottleneck_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 4),
                      nn.ReLU(inplace=True),
                      nn.Linear(input_size // 4, 4 * hidden_size))
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.1):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                ih.append(bottleneck_cell(input_size, hidden_size))
                hh.append(bottleneck_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * torch.sigmoid(ncx)
            cy.append(ncx)
            hy.append(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)  # number of layer * batch * hidden
        return hy, cy

class _BatchCLNorm(_BatchNorm):  #cross-layer dynamic norm
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchCLNorm, self).__init__(num_features, eps, momentum, affine)

    def forward(self, input, ht, ct):
        self._check_input_dim(input)

        out = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        return out * ht + ct

class CL_DNR(_BatchCLNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


class _BatchCTNorm(_BatchNorm):  #cross-temporal dynamic norm
    def __init__(self, num_features, n_segment = 8, num_groups = 4, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchCTNorm, self).__init__(num_features, eps, momentum, affine)
        self.lstm = LSTMCell(num_features,num_features,1) 
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.num_features = num_features
        self.num_groups = num_groups
        self.n_segment = n_segment

    def forward(self, input):
        self._check_input_dim(input)
        NT,C,H,W = input.size()		# input 4D tensor
        n_batch = NT // self.n_segment
        input = input.view(n_batch,self.n_segment,C,H,W)  # N,T,C,H,W
        input = input.permute(0,2,1,3,4)		# N,C,T,H,W
        t_groupsize = int(self.n_segment / self.num_groups)
        nGroups = self.num_groups
        N = n_batch		
        out_temp = []
        out = torch.zeros(n_batch,C,t_groupsize,H,W)
        for i in range(nGroups):
            if i == 0:
                ht = torch.zeros(1, N, C).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, N, C).cuda()

            t_idx = torch.tensor(range(i*t_groupsize,(i+1)*t_groupsize))
            seq = self.avgpool(input[:,:,t_idx,:,:])    # NxCx1x1x1
            seq = seq.view(N, C)    # NxC

            ht, ct = self.lstm(seq, (ht, ct))
            w = ht[-1].view(ht.size(1), ht.size(2), 1, 1, 1)
            b = ct[-1].view(ct.size(1), ct.size(2), 1, 1, 1)

            out = F.batch_norm(
             input[:,:,t_idx,:,:], self.running_mean, self.running_var, None, None,
             self.training, self.momentum, self.eps)
            out_temp.append(out * w + b)   # T-len List; each NxCx(t_groupsize)xHxW
            out = torch.cat(out_temp,2)  
            out = out.permute(0,2,1,3,4) # N,T,C,H,W
        return out.reshape(NT,C,H,W) # NT,C,H,W


class CT_DNR(_BatchCTNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4 :
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.bn1 = CL_DNR(planes)	#CL
        self.bn2 = CT_DNR(planes, n_segment = 8, num_groups = 4) #CT
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)	#CL
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)	#CT
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class wrapper(nn.Module):
    def __init__(self, ModuleList, block_idx, n_segment):
        super(wrapper, self).__init__()
        self.ModuleList = ModuleList
        self.lstm = LSTMCell(64*2**(block_idx-1), 64*2**(block_idx-1), 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.block_idx = block_idx
        self.n_segment = n_segment

    def forward(self, x):	#cross-layer norm relay
        for idx, layer in enumerate(self.ModuleList):
            residual = x
            out = layer.conv1(x)

            nt,C,H,W = out.size()
            n_batch = nt//self.n_segment			
            seq = self.avgpool(out.view(n_batch,self.n_segment,C,H,W).permute(0,2,1,3,4))	#avgpool([n,c,t,h,w])==>nxc
            seq = seq.view(seq.size(0), seq.size(1))
            if idx == 0:
                ht = torch.zeros(1, seq.size(0), seq.size(1)).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
            ht, ct = self.lstm(seq, (ht, ct))
            w = ht[-1].view(ht.size(1), ht.size(2), 1, 1, 1)	# add an extra dimension
            b = ct[-1].view(ct.size(1), ct.size(2), 1, 1, 1)

            out = layer.bn1(out.view(n_batch,self.n_segment,C,H,W).permute(0,2,1,3,4), w, b) #bn1([n,c,t,h,w])
            out = out.permute(0,2,1,3,4).reshape(nt,C,H,W) #nt,c,h,w

            out = layer.relu(out)
            out = layer.conv2(out)

            out = layer.bn2(out)
            out = layer.relu(out)
            out = layer.conv3(out)
            out = layer.bn3(out)
            if layer.downsample is not None:
                residual = layer.downsample(x)
            out += residual
            x = layer.relu(out)
        return x

def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet_DNR(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        partial_bn (bool): Whether to freeze weight and bias of **all but the first** BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False):
        super(ResNet_DNR, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = wrapper(make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp),i+1,8)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_DNR, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
