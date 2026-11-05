# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open.

Step 1 (library path): m > SKINNY_MAX goes to cuBLAS (torch.mm); the router uses the BF16 hi + lo
split with FP32 accumulation, as SGLang does above 8 tokens.
Step 2 (prototype): m <= SKINNY_MAX uses a Triton swap-AB split-K kernel. The weight rows fill the
tensor-core M slot and the (padded) tokens fill the N slot, so the kernel streams W once at HBM
speed. Split-K spreads the stream over all SMs; a second tiny kernel sums the partials and casts.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl

SKINNY_MAX = 32
TARGET_CTAS = 264  # 2 x 132 SMs on H200


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _splitk_kernel(x_ptr, w_ptr, w2_ptr, ws_ptr, M, N, K, K_PER_SPLIT,
                   BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                   TWO: tl.constexpr, STAGES: tl.constexpr):
    """ws[s, m, n] = sum over this split's k range of x[m, k] * (w + w2)[n, k]  (fp32)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    k_begin = pid_s * K_PER_SPLIT
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k0 in tl.range(k_begin, k_begin + K_PER_SPLIT, BK, num_stages=STAGES):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        xt = tl.load(x_ptr + offs_m[None, :] * K + offs_k[:, None],
                     mask=offs_m[None, :] < M, other=0.0)
        acc = tl.dot(w, xt, acc)
        if TWO:
            w2 = tl.load(w2_ptr + offs_n[:, None] * K + offs_k[None, :])
            acc = tl.dot(w2, xt, acc)
    out_ptrs = ws_ptr + pid_s * M * N + offs_m[None, :] * N + offs_n[:, None]
    tl.store(out_ptrs, acc, mask=offs_m[None, :] < M)


@triton.jit
def _reduce_kernel(ws_ptr, out_ptr, TOTAL, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        acc += tl.load(ws_ptr + s * TOTAL + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _pow2_ceil(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _skinny(x, w, w2, n, out_dtype):
    m, k = x.shape
    # BN is the tensor-core M slot (>=16); BM is the token slot (>=16 for tl.dot).
    bn = 16 if w2 is not None else 32
    bm = 16 if m <= 16 else 32
    bk = 128 if w2 is not None else 256
    n_blocks = n // bn
    splits = min(max(1, k // 512), _pow2_ceil(-(-TARGET_CTAS // n_blocks)))
    while k % (splits * bk):
        splits //= 2
    ws = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _splitk_kernel[(n_blocks, splits)](
        x, w, w2 if w2 is not None else w, ws, m, n, k, k // splits,
        BN=bn, BM=bm, BK=bk, TWO=w2 is not None, STAGES=4, num_warps=4)
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    total = m * n
    _reduce_kernel[(triton.cdiv(total, 1024),)](ws, out, total, SPLITS=splits, BLOCK=1024)
    return out


def prepare(weights: Weights):
    state = dict(weights)
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()                  # [256, 8192], row-major for the Triton kernel
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()            # [8192, 256], for the cuBLAS path
    state["router_lo_t"] = lo.t().contiguous()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer, n, m = cfg["layer"], cfg["n"], x.shape[0]
    if layer == "router":
        if m <= SKINNY_MAX:
            return _skinny(x, state["router_hi"], state["router_lo"], n, torch.float32)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    w = state[layer][:n]
    if m <= SKINNY_MAX:
        return _skinny(x, w, None, n, torch.bfloat16)
    return x @ w.T
