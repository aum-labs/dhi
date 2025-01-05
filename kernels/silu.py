import triton
import torch
import triton.language as tl

@triton.jit
def silu_fwd_kernel(x):
    return x * tl.sigmoid(x)

@triton.jit
def silu_bwd_kernel(x, grad_x):
    return grad_x * (1 + tl.sigmoid(x) * (1 - x))

class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = silu_fwd_kernel(x)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = silu_bwd_kernel(x, grad_output)
        return grad_x