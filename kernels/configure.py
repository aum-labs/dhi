import triton
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