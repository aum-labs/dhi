import triton
import triton.language as tl
import torch
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

# b * c * h * w -> so dim is channels or d_model
# b * (hw) * c lets say x coordinate is corresponding to w and y coordinate is corresponding to h
# original rope2d paper proposed two methods to rotate the coordinates
# ø = option + o
# 1. r(n, 2t) = cos(p(x)*ø) + i * sin(p(x)*ø), r(n, 2t+1) = cos(p(y)*ø) + i * sin(p(y)*ø) --> axial frequency
# 2. r(n, 2t) = exp(i * (ø(x) * p(x) + ø(y) * p(y))) , where ø(x) and ø(y) are learnable params

def rope2d(x, dim, width, n_heads):
    b, hw, c = x.shape
    head_dim = c // n_heads
    h = hw // width
    w = width
    
    dim_half = head_dim // 2
    
    theta = 100 ** (-torch.arange(0, dim_half//2, dtype=torch.float32) / (dim_half//2))
    theta = theta.to(x.device)
    
    h_pos = torch.arange(h, dtype=torch.float32).to(x.device)
    w_pos = torch.arange(w, dtype=torch.float32).to(x.device)
    
    freqs_h = torch.outer(h_pos, theta)  
    freqs_w = torch.outer(w_pos, theta)  

    freqs_h = torch.cat((freqs_h,freqs_h), dim = -1)
    freqs_w = torch.cat((freqs_w,freqs_w), dim = -1)
    
    x = x.view(b, n_heads, h, w, head_dim)
    
    x_h = x[..., :dim_half]  
    x_w = x[..., dim_half:]  
    
    cos_h = torch.cos(freqs_h)[None, None, :, None, :]  
    sin_h = torch.sin(freqs_h)[None, None, :, None, :]
    r1_h = x_h * cos_h
    r2_h = torch.cat((-x_h[..., dim_half//2:], x_h[..., :dim_half//2]), dim=-1) * sin_h
    x_h_rotated = r1_h + r2_h
    
    cos_w = torch.cos(freqs_w)[None, None, None, :, :]  
    sin_w = torch.sin(freqs_w)[None, None, None, :, :]
    r1_w = x_w * cos_w
    r2_w = torch.cat((-x_w[..., dim_half//2:], x_w[..., :dim_half//2]), dim=-1) * sin_w
    x_w_rotated = r1_w + r2_w
    
    x_out = torch.cat([x_h_rotated, x_w_rotated], dim=-1)
    
    return x_out.view(b, h*w, c)

def get_cis_mat_2d(head_dim, hw, width, device=None):
    h = hw // width
    w = width
    dim_half = head_dim // 2
    
    theta = 100 ** (-torch.arange(0, dim_half//2, dtype=torch.float32, device=device) / (dim_half//2))
    h_pos = torch.arange(h, dtype=torch.float32, device=device)
    w_pos = torch.arange(w, dtype=torch.float32, device=device)
    
    freqs_h = torch.outer(h_pos, theta)  
    freqs_w = torch.outer(w_pos, theta) 
    
    cos_h = torch.cos(freqs_h) # h * head_dim/2
    sin_h = torch.sin(freqs_h)
    
    cos_w = torch.cos(freqs_w) # w * head_dim/2
    sin_w = torch.sin(freqs_w)

@triton.jit
def _rope2d_fwd_kernel(
    inp_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    out_ptr,
    inp_stride_batch,
    inp_stride_hw,
    inp_stride_head,
    cos_stride_hw,
    cos_stride_dim,
    head_dim,
    batch_size,
    height, 
    width,
    n_heads,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    n_hw = tl.program_id(1)
    hw = height * width
    n = n_hw // hw
    h_w = n_hw % hw

    # height_coordinate hc = y, width_coordinate wc = x 
    # say h_w = 0 1
    #           2 3
    # so for point 2, y = 1, x = 0
    y = h_w // width
    x = h_w % width
    dim_fourth = head_dim // 4

    inp_offset = (b * inp_stride_batch + n * inp_stride_head + h_w * inp_stride_hw)
    h_offset = (y * height)
    w_offset = (x * width)
    cols = tl.arange(0, BLOCK_SIZE)

    mask = cols < dim_fourth
    inp1 = tl.load(inp_ptr + inp_offset + cols * 1, mask=mask)
    inp2 = tl.load(inp_ptr + inp_offset + (cols + dim_fourth)*1, mask=mask)
    inp3 = tl.load(inp_ptr + inp_offset + (cols + 2 * dim_fourth)*1, mask=mask)
    inp4 = tl.load(inp_ptr + inp_offset + (cols + 3 * dim_fourth)*1, mask=mask)

    cos_h = tl.load(cos_h_ptr + h_offset + cols * cos_stride_dim, mask=mask)
    sin_h = tl.load(sin_h_ptr + h_offset + cols * cos_stride_dim, mask=mask)
    cos_w = tl.load(cos_w_ptr + w_offset + cols * cos_stride_dim, mask=mask)
    sin_w = tl.load(sin_w_ptr + w_offset + cols * cos_stride_dim, mask=mask)

    # Height dimension rotation
    r1_h = inp1 * cos_h
    r2_h = inp2 * sin_h
    out1h = r1_h - r2_h
    out2h = inp1 * sin_h + inp2 * cos_h

    # Width dimension rotation
    r1_w = inp3 * cos_w
    r2_w = inp4 * sin_w
    out1w = r1_w - r2_w
    out2w = inp3 * sin_w + inp4 * cos_w

    # Store results
    tl.store(out_ptr + inp_offset + cols * 1, out1h, mask=mask)
    tl.store(out_ptr + inp_offset + (cols + dim_fourth)*1, out2h, mask=mask)
    tl.store(out_ptr + inp_offset + (cols + 2 * dim_fourth)*1, out1w, mask=mask)
    tl.store(out_ptr + inp_offset + (cols + 3 * dim_fourth)*1, out2w, mask=mask)


@triton.jit
def _rope2d_bwd_kernel(
    grad_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    out_ptr,
    grad_stride_batch,
    grad_stride_hw,
    grad_stride_head,
    cos_stride_hw,
    cos_stride_dim,
    head_dim,
    batch_size,
    height, 
    width,
    n_heads,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    n_hw = tl.program_id(1)
    hw = height * width
    n = n_hw // hw
    h_w = n_hw % hw

    y = h_w // width
    x = h_w % width
    dim_fourth = head_dim // 4

    grad_offset = (b * grad_stride_batch + n * grad_stride_head + h_w * grad_stride_hw)
    h_offset = (y * height)
    w_offset = (x * width)
    cols = tl.arange(0, BLOCK_SIZE)

    mask = cols < dim_fourth
    grad1h = tl.load(grad_ptr + grad_offset + cols * 1, mask = mask)
    grad2h = tl.load(grad_ptr + grad_offset + (cols + dim_fourth)*1, mask = mask)
    grad3w = tl.load(grad_ptr + grad_offset + (cols + 2 * dim_fourth)*1, mask = mask)
    grad4w = tl.load(grad_ptr + grad_offset + (cols + 3 * dim_fourth)*1, mask = mask)

    cos_h = tl.load(cos_h_ptr + h_offset + cols * cos_stride_dim, mask = mask)
    sin_h = tl.load(sin_h_ptr + h_offset + cols * cos_stride_dim, mask = mask)

    cos_w = tl.load(cos_w_ptr + w_offset + cols * cos_stride_dim, mask = mask)
    sin_w = tl.load(sin_w_ptr + w_offset + cols * cos_stride_dim, mask = mask)

    # For height dimension:
    # Forward: out1h = inp1 * cos_h - inp2 * sin_h
    #         out2h = inp2 * cos_h + inp1 * sin_h
    # Backward derivation: 'do' is option + d
    # ðL/ðinp1 = ðL/ðout1h * ðout1h/ðinp1 + ðL/ðout2h * ðout2h/ðinp1
    #          = grad1h * cos_h + grad2h * sin_h
    # ðL/ðinp2 = ðL/ðout1h * ðout1h/ðinp2 + ðL/ðout2h * ðout2h/ðinp2
    #          = -grad1h * sin_h + grad2h * cos_h
    out1h = grad1h * cos_h + grad2h * sin_h
    out2h = -grad1h * sin_h + grad2h * cos_h

    # For width dimension:
    # Forward: out1w = inp3 * cos_w - inp4 * sin_w
    #         out2w = inp4 * cos_w + inp3 * sin_w
    # Backward derivation follows same pattern as height
    out1w = grad3w * cos_w + grad4w * sin_w
    out2w = -grad3w * sin_w + grad4w * cos_w

    tl.store(out_ptr + grad_offset + cols * 1, out1h, mask = mask)
    tl.store(out_ptr + grad_offset + (cols + dim_fourth)*1, out2h, mask = mask)
    tl.store(out_ptr + grad_offset + (cols + 2 * dim_fourth)*1, out1w, mask = mask)
    tl.store(out_ptr + grad_offset + (cols + 3 * dim_fourth)*1, out2w, mask = mask)


class RoPE2D_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos_h, sin_h, cos_w, sin_w, width):
        b, n, hw, head_dim = x.shape
        height = hw // width

        out = torch.empty_like(x)

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        _rope2d_fwd_kernel[b, n*hw](
            x,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            out,
            x.stride(0),
            x.stride(2),
            x.stride(1),
            cos_h.stride(0),
            cos_h.stride(1),
            head_dim,
            b,
            height,
            width,
            n,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(cos_h, sin_h, cos_w, sin_w)
        ctx.width = width
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        cos_h, sin_h, cos_w, sin_w = ctx.saved_tensors
        width = ctx.width
        b, n, hw, head_dim = grad_output.shape
        height = hw // width

        grad_input = torch.empty_like(grad_output)

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        _rope2d_bwd_kernel[b, n*hw](
            grad_output,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            grad_input,
            grad_output.stride(0),
            grad_output.stride(2),
            grad_output.stride(1),
            cos_h.stride(0),
            cos_h.stride(1),
            head_dim,
            b,
            height,
            width,
            n,
            BLOCK_SIZE,
            num_warps=num_warps,
        )

        return grad_input, None, None, None, None, None



def test_rope2d_kernel():
    # Test configurations
    batch_size = 2
    n_heads = 4
    seq_len = 16
    head_dim = 32
    width = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create input tensor on the correct device
    x = torch.randn(batch_size, seq_len, n_heads * head_dim, 
                   requires_grad=True, device=device)
    x_ref = x.clone().detach().requires_grad_(True)
    
    # Get tensors on the correct device
    h = seq_len // width
    w = width
    dim_half = head_dim // 2
    theta = 100 ** (-torch.arange(0, dim_half//2, dtype=torch.float32, device=device) / (dim_half//2))
    
    h_pos = torch.arange(h, dtype=torch.float32, device=device)
    w_pos = torch.arange(w, dtype=torch.float32, device=device)
    
    freqs_h = torch.outer(h_pos, theta)
    freqs_w = torch.outer(w_pos, theta)
    
    cos_h = torch.cos(freqs_h)
    sin_h = torch.sin(freqs_h)
    cos_w = torch.cos(freqs_w)
    sin_w = torch.sin(freqs_w)
    
    x_reshaped = x.view(batch_size, n_heads, seq_len, head_dim)
    kernel_out = RoPE2D_triton.apply(x_reshaped, cos_h, sin_h, cos_w, sin_w, width)
    kernel_out = kernel_out.view(batch_size, seq_len, n_heads * head_dim)

    ref_out = rope2d(x_ref, head_dim, width, n_heads)
    # Test forward pass
    torch.testing.assert_close(ref_out, kernel_out, rtol=1e-5, atol=1e-5)
    
    # Test backward pass
    grad_output = torch.randn_like(ref_out)
    
    # Compute gradients for reference implementation
    ref_out.backward(grad_output)
    ref_grad = x_ref.grad.clone()
    
    # Compute gradients for kernel implementation
    kernel_out.backward(grad_output)
    kernel_grad = x.grad.clone()
    
    # Compare gradients
    torch.testing.assert_close(ref_grad, kernel_grad, rtol=1e-5, atol=1e-5)

def test_rope2d_numerical_gradient():
    # Test with smaller sizes for numerical gradient check
    batch_size = 2
    n_heads = 2
    seq_len = 4  # 2x2 image
    head_dim = 8
    width = 2
    
    x = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.double, requires_grad=True)
    
    h = seq_len // width
    w = width
    dim_half = head_dim // 2
    theta = 100 ** (-torch.arange(0, dim_half//2, dtype=torch.float64) / (dim_half//2))
    
    h_pos = torch.arange(h, dtype=torch.float64)
    w_pos = torch.arange(w, dtype=torch.float64)
    
    freqs_h = torch.outer(h_pos, theta)
    freqs_w = torch.outer(w_pos, theta)
    
    cos_h = torch.cos(freqs_h)
    sin_h = torch.sin(freqs_h)
    cos_w = torch.cos(freqs_w)
    sin_w = torch.sin(freqs_w)

    def func(input_tensor):
        return RoPE2D_triton.apply(input_tensor, cos_h, sin_h, cos_w, sin_w, width)

    # Use gradcheck to verify gradients numerically
    assert gradcheck(func, (x,), eps=1e-6, atol=1e-4)

def test_edge_cases():
    # Test with minimum dimensions
    batch_size = 1
    n_heads = 1
    seq_len = 4  # 2x2 image
    head_dim = 4
    width = 2
    
    x = torch.randn(batch_size, seq_len, n_heads * head_dim, requires_grad=True)
    out = rope2d(x, head_dim, width, n_heads)
    assert out.shape == x.shape
    
    # Test with zero batch size
    x_empty = torch.randn(0, seq_len, n_heads * head_dim, requires_grad=True)
    out_empty = rope2d(x_empty, head_dim, width, n_heads)
    assert out_empty.shape == x_empty.shape
    
    # Test with larger dimensions
    x_large = torch.randn(8, 64, 32 * 128, requires_grad=True)
    out_large = rope2d(x_large, 128, 8, 32)
    assert out_large.shape == x_large.shape

def run_all_tests():
    print("Running RoPE2D tests...")
    test_rope2d_kernel()
    test_rope2d_numerical_gradient()
    test_edge_cases()
    print("All tests passed!")

if __name__ == "__main__":
    run_all_tests()