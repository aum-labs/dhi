import triton
import torch
import triton.language as tl
from typing import Tuple

MAX_FUSED_SIZE: int = 65536

def calculate_settings(n: int) -> Tuple[int, int]:
    BLOCK_SIZE: int = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "
                         f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

def rope1d(x, head_dim):
    theta = 10000 ** (-2 * torch.arange(0, head_dim//2, dtype=torch.float32) / head_dim)
    theta = theta.to(x.device)
    m = torch.arange(x.size(1), dtype=torch.float32).to(x.device)
    freqs = torch.outer(m, theta)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos = torch.cos(freqs)[None, :, None, :]  # [1, seq, 1, head_dim]
    sin = torch.sin(freqs)[None, :, None, :]
    r1 = x * cos
    r2 = torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    return r1 + r2 * sin

def get_cos_sin(head_dim, seq_len):
    theta = 10000 ** (-2 * torch.arange(0, head_dim//2, dtype=torch.float32) / head_dim)
    m = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(m, theta)
    cos = torch.cos(freqs)[None, :, None, :]  # [1, seq, 1, head_dim/2]
    sin = torch.sin(freqs)[None, :, None, :]
    return cos.to('cuda'), sin.to('cuda')

@triton.jit
def rope1d_fwd_kernel(
    inp_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,  
    inp_stride_batch,
    inp_stride_seq,
    inp_stride_head,
    inp_stride_dim, 
    cos_stride_seq,
    cos_stride_dim, 
    head_dim,
    batch_size,
    seq_len,
    n_heads,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (n_heads * seq_len)
    head_idx = (pid // seq_len) % n_heads
    seq_idx = pid % seq_len

    inp_offset = (batch_idx * inp_stride_batch + 
                 head_idx * inp_stride_head + 
                 seq_idx * inp_stride_seq)
    cos_offset = seq_idx * cos_stride_seq

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim // 2  

    x1 = tl.load(inp_ptr + inp_offset + cols * inp_stride_dim, mask=mask)
    x2 = tl.load(inp_ptr + inp_offset + (cols + head_dim // 2) * inp_stride_dim, mask=mask)
    
    cos1 = tl.load(cos_ptr + cos_offset + cols * cos_stride_dim, mask=mask)
    sin1 = tl.load(sin_ptr + cos_offset + cols * cos_stride_dim, mask=mask)
    
    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos1 + x1 * sin1

    tl.store(out_ptr + inp_offset + cols * inp_stride_dim, out1, mask=mask)
    tl.store(out_ptr + inp_offset + (cols + head_dim // 2) * inp_stride_dim, out2, mask=mask)

@triton.jit
def rope1d_bwd_kernel(
    grad_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    grad_stride_batch,
    grad_stride_seq,
    grad_stride_head,
    grad_stride_dim,
    cos_stride_seq,
    cos_stride_dim,
    head_dim,
    batch_size,
    seq_len,
    n_heads,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (n_heads * seq_len)
    head_idx = (pid // seq_len) % n_heads
    seq_idx = pid % seq_len

    grad_offset = (batch_idx * grad_stride_batch + 
                  head_idx * grad_stride_head + 
                  seq_idx * grad_stride_seq)
    cos_offset = seq_idx * cos_stride_seq

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim // 2  

    grad1 = tl.load(grad_ptr + grad_offset + cols * grad_stride_dim, mask=mask)
    grad2 = tl.load(grad_ptr + grad_offset + (cols + head_dim // 2) * grad_stride_dim, mask=mask)
    
    cos1 = tl.load(cos_ptr + cos_offset + cols * cos_stride_dim, mask=mask)
    sin1 = tl.load(sin_ptr + cos_offset + cols * cos_stride_dim, mask=mask)
    
    # forward was: out1 = x1 * cos - x2 * sin
    #                 out2 = x2 * cos + x1 * sin
    # backward is: dx1 = grad1 * cos + grad2 * sin
    #                 dx2 = grad2 * cos - grad1 * sin
    dx1 = grad1 * cos1 + grad2 * sin1
    dx2 = grad2 * cos1 - grad1 * sin1

    tl.store(out_ptr + grad_offset + cols * grad_stride_dim, dx1, mask=mask)
    tl.store(out_ptr + grad_offset + (cols + head_dim // 2) * grad_stride_dim, dx2, mask=mask)

class RoPE1D_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, head_dim):
        BLOCK_SIZE, num_warps = calculate_settings(head_dim)
        B, T, N, H = x.shape  # B: batch_size, T: seq_len, N: n_heads, H: head_dim
        
        output = torch.empty_like(x)
        
        rope1d_fwd_kernel[(B * N * T,)](
            x,
            cos,
            sin,
            output,
            x.stride(0),  # inp_stride_batch
            x.stride(1),  # inp_stride_seq
            x.stride(2),  # inp_stride_head
            x.stride(3),  # inp_stride_dim
            cos.stride(1),  # cos_stride_seq
            cos.stride(3),  # cos_stride_dim
            H,
            B,
            T,
            N,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.save_for_backward(cos, sin)
        ctx.head_dim = head_dim
        ctx.block_size = BLOCK_SIZE
        ctx.num_warps = num_warps
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        head_dim = ctx.head_dim
        BLOCK_SIZE = ctx.block_size
        num_warps = ctx.num_warps
        
        B, T, N, H = grad_output.shape
        
        grad_input = torch.empty_like(grad_output)
        
        rope1d_bwd_kernel[(B * N * T,)](
            grad_output,
            cos,
            sin,
            grad_input,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            cos.stride(1),
            cos.stride(3),
            H,
            B,
            T,
            N,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        return grad_input, None, None, None

    
