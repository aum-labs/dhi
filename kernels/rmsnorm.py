import triton
import triton.language as tl
import torch
import torch.nn as nn
from typing import Tuple
import math

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

def calculate_flops(batch_size: int, seq_len: int, hidden_dim: int) -> int:
    flops_per_seq = (
        hidden_dim +          # Square each element
        (hidden_dim - 1) +    # Sum reduction
        1 +                   # Division by hidden_dim
        1 +                   # Square root
        (2 * hidden_dim)      # Final division and scale multiplication
    )
    return flops_per_seq * batch_size * seq_len

@triton.jit
def _rmsn_fwd_kernel(
    inp_ptr, 
    out_ptr, 
    scale_ptr,
    inp_row_stride,
    out_row_stride,
    g_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * inp_row_stride + cols
    mask = cols < inp_row_stride
    
    x = tl.load(inp_ptr + offset, mask=mask).to(tl.float16)
    
    x_f32 = x.to(tl.float32)
    sum_sqr_x = tl.sum(x_f32 * x_f32, axis=0)
    
    rms = tl.sqrt(sum_sqr_x / inp_row_stride).to(tl.float16)
    scale = tl.load(scale_ptr + cols, mask=mask).to(tl.float16)
    
    out = (x / rms) * scale
    
    tl.store(out_ptr + offset, out, mask=mask)

@triton.jit
def _rmsn_bwd_kernel(
    grad_out_ptr,
    inp_ptr,
    scale_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    inp_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * inp_row_stride + cols
    mask = cols < inp_row_stride
    
    grad_out = tl.load(grad_out_ptr + offset, mask=mask).to(tl.float32)
    x = tl.load(inp_ptr + offset, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / inp_row_stride
    inv_std = 1.0 / tl.sqrt(variance)
    x_norm = x * inv_std
    
    dx_norm = grad_out * scale
    
    # Compute variance gradient
    # First term: dx_norm * x * -1/(2 * variance^(3/2))
    variance_grad = dx_norm * x * (-0.5 / (variance * tl.sqrt(variance)))
    
    # Sum for the second term of input gradient
    sum_dx_norm_x = tl.sum(dx_norm * x_norm, axis=0)
    
    # Compute final input gradient
    # dx = (1/sqrt(variance)) * (dx_norm - mean(dx_norm * x_norm) * x_norm)
    grad_input = inv_std * (dx_norm - (sum_dx_norm_x / inp_row_stride) * x_norm)
    
    tl.store(grad_input_ptr + offset, grad_input.to(tl.float16), mask=mask)
    
    if pid == 0:
        grad_scale = tl.sum(grad_out * x_norm, axis=0).to(tl.float16)
        tl.atomic_add(grad_scale_ptr + cols, grad_scale, mask=mask)

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=None):
        dims = x.shape
        cols = dims[-1]
        x = x.view(-1, cols)
        
        # Create or use provided scale
        if scale is None:
            scale = torch.ones(cols, dtype=torch.float16, device='cuda')
        
        # Initialize output
        out = torch.zeros_like(x, dtype=torch.float16)
        
        block_size, num_warps = calculate_settings(cols)
        rows = x.shape[0]
        
        _rmsn_fwd_kernel[(rows,)](
            x,
            out,
            scale,
            cols,
            cols,
            cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(x, scale)
        ctx.dims = dims
        ctx.cols = cols
        ctx.rows = rows
        
        return out.view(dims)

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        dims = ctx.dims
        cols = ctx.cols
        rows = ctx.rows
        
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.float16:
            grad_output = grad_output.half()
        grad_output = grad_output.view(-1, cols)
        
        grad_input = torch.zeros_like(x)
        grad_scale = torch.zeros_like(scale)
        
        block_size, num_warps = calculate_settings(cols)
        
        _rmsn_bwd_kernel[(rows,)](
            grad_output,
            x,
            scale,
            grad_input,
            grad_scale,
            cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        return grad_input.view(dims), grad_scale if scale.requires_grad else None

def test_backward():
    x = torch.randn(32, 128, 512, dtype=torch.float16, device='cuda', requires_grad=True)
    
    out = RMSNorm.apply(x)
    
    grad_output = torch.randn_like(out)
    
    out.backward(grad_output)
    
    print("Gradient shape:", x.grad.shape)
    print("Gradient contains NaN:", torch.isnan(x.grad).any())
    print("Gradient contains Inf:", torch.isinf(x.grad).any())
def rmsnorm(x: torch.Tensor):
    if x.dtype != torch.float16:
        x = x.half()
    
    scale = torch.ones(x.shape[-1], dtype=torch.float16, device='cuda')
    
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True)).half()
    return (x / rms) * scale

def get_gpu_memory_info():
    t = torch.cuda.get_device_properties(0)
    memory_used = torch.cuda.memory_allocated() / (1024**2)  
    memory_total = t.total_memory / (1024**2) 
    return memory_used, memory_total

def get_gpu_info():
    t = torch.cuda.get_device_properties(0)
    return {
        'name': t.name,
        'compute_capability': f"{t.major}.{t.minor}",
    }

def benchmark():
    batch_size = 512
    seq_len = 1024
    hidden_dim = 512
    num_iterations = 100
    warmup_iterations = 10
    
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
    
    total_flops = calculate_flops(batch_size, seq_len, hidden_dim)
    
    gpu_info = get_gpu_info()
    print("\nGPU Information:")
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    torch.cuda.synchronize()
    for _ in range(warmup_iterations):
        torch_rms = rmsnorm(x)
        triton_rms = RMSNorm.apply(x)
    torch.cuda.synchronize()
    
    import time
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        torch_rms = rmsnorm(x)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / num_iterations
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        triton_rms = RMSNorm.apply(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iterations
    
    torch_flops = total_flops / torch_time
    triton_flops = total_flops / triton_time
    
    memory_used, memory_total = get_gpu_memory_info()
    
    print("\nPerformance Metrics:")
    print(f"Input shape: {x.shape}")
    print(f"FLOPs per iteration: {total_flops:,}")
    print(f"\nPyTorch implementation:")
    print(f"Time: {torch_time*1000:.3f} ms")
    print(f"FLOP/s: {torch_flops/1e9:.2f} GFLOP/s")
    print(f"\nTriton implementation:")
    print(f"Time: {triton_time*1000:.3f} ms")
    print(f"FLOP/s: {triton_flops/1e9:.2f} GFLOP/s")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    
    print("\nMemory Usage:")
    print(f"Used: {memory_used:.2f} MB")
    print(f"Total: {memory_total:.2f} MB")
    print(f"Utilization: {(memory_used/memory_total)*100:.2f}%")
    
    print(f"Implementations match: {torch.testing.assert_close(torch_rms, triton_rms)}")


class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        
    def forward(self, x):
        if x.dtype != torch.float16:
            x = x.half()
            
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 / variance.sqrt()
        
        return (x_norm.half() * self.scale)

def compare_gradients(batch_size=32, seq_len=128, hidden_size=512, atol=1e-2, rtol=1e-2):

    print("\nGradient Verification Test")
    print("-" * 50)
    
    x = torch.randn(batch_size, seq_len, hidden_size, 
                    dtype=torch.float16, 
                    device='cuda', 
                    requires_grad=True)
    grad_output = torch.randn_like(x)
    
    x_torch = x.clone().detach().requires_grad_(True)
    
    triton_out = RMSNorm.apply(x)
    triton_out.backward(grad_output)
    triton_grad = x.grad.clone()
    
    torch_model = TorchRMSNorm(hidden_size).cuda()
    torch_out = torch_model(x_torch)
    torch_out.backward(grad_output)
    torch_grad = x_torch.grad.clone()
    
    output_match = torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol)
    grad_match = torch.allclose(triton_grad, torch_grad, atol=atol, rtol=rtol)
    
    print(f"Test Configuration:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Sequence Length: {seq_len}")
    print(f"- Hidden Size: {hidden_size}")
    print(f"- Tolerance: atol={atol}, rtol={rtol}")
    print("\nResults:")
    print(f"- Forward outputs match: {output_match}")
    print(f"- Backward gradients match: {grad_match}")
    
    if not output_match or not grad_match:
        print("\nDetailed Error Analysis:")
        forward_diff = (triton_out - torch_out).abs()
        print(f"Forward max difference: {forward_diff.max().item():.6f}")
        print(f"Forward mean difference: {forward_diff.mean().item():.6f}")
        
        backward_diff = (triton_grad - torch_grad).abs()
        print(f"Backward max difference: {backward_diff.max().item():.6f}")
        print(f"Backward mean difference: {backward_diff.mean().item():.6f}")

    
    return output_match and grad_match

def run_gradient_tests():

    print("Running Gradient Test Suite")
    print("=" * 50)
    
    test_configs = [
        (32, 128, 512),    
        (1, 128, 512),    
        (32, 1, 512),      
        (32, 128, 128),    
        (64, 256, 1024),   
    ]
    
    all_passed = True
    for batch, seq, hidden in test_configs:
        print(f"\nTesting configuration: batch={batch}, seq_len={seq}, hidden={hidden}")
        passed = compare_gradients(batch, seq, hidden)
        all_passed &= passed
        
    print("\nFinal Results:")
    print(f"All tests {'passed' if all_passed else 'failed'}")
    
    return all_passed

if __name__ == "__main__":
    run_gradient_tests()

# if __name__ == "__main__":
#     benchmark()
# Running Gradient Test Suite
# ==================================================

# Testing configuration: batch=32, seq_len=128, hidden=512

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 32
# - Sequence Length: 128
# - Hidden Size: 512
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=1, seq_len=128, hidden=512

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 1
# - Sequence Length: 128
# - Hidden Size: 512
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=32, seq_len=1, hidden=512

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 32
# - Sequence Length: 1
# - Hidden Size: 512
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=32, seq_len=128, hidden=128

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 32
# - Sequence Length: 128
# - Hidden Size: 128
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=64, seq_len=256, hidden=1024

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 64
# - Sequence Length: 256
# - Hidden Size: 1024
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Final Results:
# All tests passed