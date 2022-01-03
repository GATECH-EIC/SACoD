from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)

def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


def calculate_qparams_dws(x, num_bits):
    with torch.no_grad():
        min_values = x.min(-1)[0].min(-1)[0].min(0)[0].view(1, -1, 1, 1)
        max_values = x.max(-1)[0].max(-1)[0].max(0)[0].view(1, -1, 1, 1)

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        scale = qparams.range / (qmax - qmin)

        mask = (scale == 0).float()
        scale += mask

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize

        output = output * (1 - mask)

        return output


    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = Quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def Quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, num_bits=8, num_bits_weight=8, dws=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits

        self.dws = dws

        self.momentum = 0.1

        if self.dws:
            shape_measure = (1, in_channels, 1, 1)
        else:
            shape_measure = (1, 1, 1, 1)

        self.register_buffer('running_zero_point', torch.zeros(*shape_measure).cuda())
        self.register_buffer('running_range', torch.zeros(*shape_measure).cuda())


    def forward(self, input, num_bits):
        if num_bits > 0:
            if self.training:
                if self.dws:
                    qparams = calculate_qparams_dws(input, num_bits=num_bits)
                else:
                    qparams = calculate_qparams(input, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point.cuda() * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range.cuda() * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits)

            qinput = Quantize(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False)

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(self.weight, qparams=weight_qparams)

            if self.bias is not None:
                qbias = Quantize(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None
            
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                                  self.padding, self.dilation, self.groups)

        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision

        self.momentum = 0.1

        shape_measure = (1,)
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure).cuda())
        self.register_buffer('running_range', torch.zeros(*shape_measure).cuda())


    def forward(self, input, num_bits):
        if num_bits > 0:
            if self.training:
                qparams = calculate_qparams(
                        input, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point.cuda() * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range.cuda() * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits)

            qinput = Quantize(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False)

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(self.weight, qparams=weight_qparams)

            if self.bias is not None:
                qbias = Quantize(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None

            output = F.linear(qinput, qweight, qbias)

        else:
            output = F.linear(input, self.weight, self.bias)

        return output


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = Quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
