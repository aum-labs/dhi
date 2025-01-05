import triton
import triton.language as tl
import torch
import triton.driver as driver

@triton.jit
def softmax_kernel(inp_ptr, out_ptr, b, t, c, BLOCK_SIZE : tl.constexpr):

    bid = tl.program_id(0)
    tid = tl.program_id(1)

    if bid >= b or tid >=t:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    offset = bid * t * c + tid * c + cols
    mask = cols < c
    x = tl.load(inp_ptr + offset, mask = mask, other = float('-inf'))
    maxx = tl.max(x, axis = 0)
    x = x - maxx
    expx = tl.exp(x)
    sumx = tl.sum(expx, axis = 0)
    outx = expx/sumx
    tl.store(out_ptr + offset, outx, mask = mask)

def softmax(inp : torch.Tensor):

    b, t, c = inp.shape
    out = torch.zeros_like(inp)
    BLOCK_SIZE = triton.next_power_of_2(c)
    softmax_kernel[(b, t)](
        inp,
        out,
        b,
        t,
        c,
        BLOCK_SIZE = BLOCK_SIZE,
    )
    return out
