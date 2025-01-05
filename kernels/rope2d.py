import triton
import triton.language as tl
import torch
from configure import calculate_settings

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
    
    theta = 1 / (100 ** (torch.arange(0, dim_half//2, dtype=torch.float32) / (dim_half)))
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

def get_cis_mat_2d(head_dim, hw, width):
    h = hw // width
    w = width
    
    dim_half = head_dim // 2
    
    theta = 1 / (100 ** (torch.arange(0, dim_half//2, dtype=torch.float32) / (dim_half)))
    
    h_pos = torch.arange(h, dtype=torch.float32)
    w_pos = torch.arange(w, dtype=torch.float32)
    
    freqs_h = torch.outer(h_pos, theta)  
    freqs_w = torch.outer(w_pos, theta) 
    
    cos_h = torch.cos(freqs_h) # h * head_dim/2
    sin_h = torch.sin(freqs_h)
    
    cos_w = torch.cos(freqs_w) # w * head_dim/2
    sin_w = torch.sin(freqs_w)

    return cos_h, sin_h, cos_w, sin_w

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
    # 3D grid: (batch_size, n_heads, height*width)
    b = tl.program_id(0)  # batch index
    n = tl.program_id(1)  # head index
    h_w = tl.program_id(2)  # spatial position index

    # height_coordinate hc = y, width_coordinate wc = x 
    # say h_w = 0 1
    #           2 3
    # so for point 2, y = 1, x = 0
    y = h_w // width
    x = h_w % width
    dim_fourth = head_dim // 4

    inp_offset = (b * inp_stride_batch + n * inp_stride_head + h_w * inp_stride_hw)
    h_offset = (y * cos_stride_hw)
    w_offset = (x * cos_stride_hw)
    cols = tl.arange(0, BLOCK_SIZE)

    mask = cols < dim_fourth
    inp1 = tl.load(inp_ptr + inp_offset + cols, mask=mask)
    inp2 = tl.load(inp_ptr + inp_offset + cols + dim_fourth, mask=mask)
    inp3 = tl.load(inp_ptr + inp_offset + cols + 2 * dim_fourth, mask=mask)
    inp4 = tl.load(inp_ptr + inp_offset + cols + 3 * dim_fourth, mask=mask)

    cos_h = tl.load(cos_h_ptr + h_offset + cols * cos_stride_dim, mask=mask)
    sin_h = tl.load(sin_h_ptr + h_offset + cols * cos_stride_dim, mask=mask)

    cos_w = tl.load(cos_w_ptr + w_offset + cols * cos_stride_dim, mask=mask)
    sin_w = tl.load(sin_w_ptr + w_offset + cols * cos_stride_dim, mask=mask)

    out1h = inp1 * cos_h - inp2 * sin_h
    out2h = inp2 * cos_h + inp1 * sin_h

    out1w = inp3 * cos_w - inp4 * sin_w
    out2w = inp4 * cos_w + inp3 * sin_w

    tl.store(out_ptr + inp_offset + cols, out1h, mask=mask)
    tl.store(out_ptr + inp_offset + cols + dim_fourth, out2h, mask=mask)
    tl.store(out_ptr + inp_offset + cols + 2 * dim_fourth, out1w, mask=mask)
    tl.store(out_ptr + inp_offset + cols + 3 * dim_fourth, out2w, mask=mask)

@triton.jit
def _rope2d_bwd_kernel(
    grad_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    out_ptr,
    grad_stride_batch,
    grad_stride_head,
    grad_stride_hw,
    cos_stride_hw,
    cos_stride_dim,
    head_dim,
    batch_size,
    height, 
    width,
    n_heads,
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (batch_size, n_heads, height*width)
    b = tl.program_id(0)  # batch index
    n = tl.program_id(1)  # head index
    h_w = tl.program_id(2)  # spatial position index

    y = h_w // width
    x = h_w % width
    dim_fourth = head_dim // 4

    grad_offset = (b * grad_stride_batch + n * grad_stride_head + h_w * grad_stride_hw)
    h_offset = (y * cos_stride_hw)
    w_offset = (x * cos_stride_hw)
    cols = tl.arange(0, BLOCK_SIZE)

    mask = cols < dim_fourth
    grad1h = tl.load(grad_ptr + grad_offset + cols * 1, mask=mask)
    grad2h = tl.load(grad_ptr + grad_offset + (cols + dim_fourth)*1, mask=mask)
    grad3w = tl.load(grad_ptr + grad_offset + (cols + 2 * dim_fourth)*1, mask=mask)
    grad4w = tl.load(grad_ptr + grad_offset + (cols + 3 * dim_fourth)*1, mask=mask)

    cos_h = tl.load(cos_h_ptr + h_offset + cols * cos_stride_dim, mask=mask)
    sin_h = tl.load(sin_h_ptr + h_offset + cols * cos_stride_dim, mask=mask)

    cos_w = tl.load(cos_w_ptr + w_offset + cols * cos_stride_dim, mask=mask)
    sin_w = tl.load(sin_w_ptr + w_offset + cols * cos_stride_dim, mask=mask)

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

    tl.store(out_ptr + grad_offset + cols * 1, out1h, mask=mask)
    tl.store(out_ptr + grad_offset + (cols + dim_fourth)*1, out2h, mask=mask)
    tl.store(out_ptr + grad_offset + (cols + 2 * dim_fourth)*1, out1w, mask=mask)
    tl.store(out_ptr + grad_offset + (cols + 3 * dim_fourth)*1, out2w, mask=mask)

class RoPE2D_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos_h, sin_h, cos_w, sin_w, width):
        b, n, hw, head_dim = x.shape
        height = hw // width

        out = torch.empty_like(x)

        BLOCK_SIZE, num_warps = calculate_settings(head_dim//4)


        _rope2d_fwd_kernel[(b, n, hw)](
            x,
            cos_h, sin_h,
            cos_w, sin_w,
            out,
            x.stride(0),
            x.stride(2),
            x.stride(1),
            cos_h.stride(0),
            cos_h.stride(1),
            head_dim,
            b, height, width, n,
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

        BLOCK_SIZE, num_warps = calculate_settings(head_dim//4)

        # Use 3D grid
        _rope2d_bwd_kernel[(b, n, hw)](
            grad_output,
            cos_h, sin_h,
            cos_w, sin_w,
            grad_input,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            cos_h.stride(0),
            cos_h.stride(1),
            head_dim,
            b, height, width, n,
            BLOCK_SIZE,
            num_warps=num_warps,
        )

        return grad_input, None, None, None, None, None

# phew! man that was exhausting, took one whole day to understand and implement this