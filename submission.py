# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open.

m > SKINNY_MAX goes to cuBLAS (torch.mm); the router uses the BF16 hi + lo split with FP32
accumulation, as SGLang does above 8 tokens.

m <= SKINNY_MAX uses ONE Triton swap-AB split-K kernel. The weight rows fill the tensor-core M slot
and the (padded) tokens fill the N slot, so the kernel streams W once. Split-K spreads the stream
over all SMs. Each CTA writes its FP32 partial to a workspace, then bumps a per-tile counter; the
last CTA of a tile sums the partials, casts, writes the output and resets the counter to 0 (so the
counters are ready for the next replay). No second launch.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl

SKINNY_MAX = 32
TARGET_CTAS = 264  # 2 x 132 SMs on H200
MAX_TILES = 1024   # counters per call (n // bn <= 512)


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
def _gemm_kernel(x_ptr, w_ptr, w2_ptr, ws_ptr, cnt_ptr, out_ptr, M, N, K, K_PER_SPLIT, SPLITS,
                 BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                 TWO: tl.constexpr, STAGES: tl.constexpr):
    """out[m, n] = sum_k x[m, k] * (w + w2)[n, k], split over k across program_id(1)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    k_begin = pid_s * K_PER_SPLIT
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k0 in tl.range(k_begin, k_begin + K_PER_SPLIT, BK, num_stages=STAGES):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        xr = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                     mask=offs_m[:, None] < M, other=0.0)  # [BM, BK], k contiguous (coalesced)
        xt = tl.trans(xr)
        acc = tl.dot(w, xt, acc)
        if TWO:
            w2 = tl.load(w2_ptr + offs_n[:, None] * K + offs_k[None, :])
            acc = tl.dot(w2, xt, acc)
    tile_off = offs_m[None, :] * N + offs_n[:, None]
    mask = offs_m[None, :] < M
    if SPLITS == 1:
        tl.store(out_ptr + tile_off, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(ws_ptr + pid_s * M * N + tile_off, acc, mask=mask)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_n, 1)  # acq_rel, gpu scope
        if done == SPLITS - 1:
            total = tl.zeros((BN, BM), dtype=tl.float32)
            for s in range(SPLITS):
                total += tl.load(ws_ptr + s * M * N + tile_off, mask=mask, other=0.0,
                                 cache_modifier=".cg")
            tl.store(out_ptr + tile_off, total.to(out_ptr.dtype.element_ty), mask=mask)
            tl.store(cnt_ptr + pid_n, 0)


def _pow2_ceil(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _default_cfg(layer, m, n, k):
    two = layer == "router"
    bn = 16 if two else 32
    bm = 16 if m <= 16 else 32
    bk = 128 if two else 256
    n_blocks = n // bn
    splits = min(max(1, k // 512), _pow2_ceil(-(-TARGET_CTAS // n_blocks)))
    while k % (splits * bk):
        splits //= 2
    return dict(bn=bn, bm=bm, bk=bk, stages=4, warps=4, splits=splits)


def _c(bn, bk, stages, warps, splits):
    return dict(bn=bn, bk=bk, stages=stages, warps=warps, splits=splits)


# (layer, bm) -> config found by an on-GPU sweep of CUDA-graph replays (cold L2)
TUNED = {
    ("qkv_gate", 16): _c(64, 256, 4, 4, 2), ("qkv_gate", 32): _c(64, 256, 4, 4, 2),
    ("o", 16): _c(64, 128, 4, 4, 1),        ("o", 32): _c(64, 128, 4, 4, 1),
    ("pooler", 16): _c(64, 256, 4, 4, 1),   ("pooler", 32): _c(32, 256, 6, 4, 1),
    ("unpooler", 16): _c(64, 256, 4, 4, 1), ("unpooler", 32): _c(64, 256, 4, 4, 1),
    ("router", 16): _c(32, 128, 4, 2, 16),  ("router", 32): _c(16, 128, 4, 4, 16),
}
MIN_CTAS = 128


def _pick(layer, m, n, k):
    cfg = _default_cfg(layer, m, n, k)
    bm = cfg["bm"]
    over = TUNED.get((layer, bm))
    if over:
        cfg = dict(over, bm=bm)
    # a small n (the sliced unpooler) needs split-K to keep the SMs fed
    while (n // cfg["bn"]) * cfg["splits"] < MIN_CTAS and k % (cfg["splits"] * 2 * cfg["bk"]) == 0:
        cfg["splits"] *= 2
    while k % (cfg["splits"] * cfg["bk"]):
        cfg["splits"] //= 2
    return cfg


def _skinny(x, w, w2, cnt, n, out_dtype, cfg):
    m, k = x.shape
    bn, splits = cfg["bn"], cfg["splits"]
    n_blocks = n // bn
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    ws = torch.empty((splits, m, n) if splits > 1 else (1,), dtype=torch.float32, device=x.device)
    _gemm_kernel[(n_blocks, splits)](
        x, w, w2 if w2 is not None else w, ws, cnt, out, m, n, k, k // splits, splits,
        BN=bn, BM=cfg["bm"], BK=cfg["bk"], TWO=w2 is not None, STAGES=cfg["stages"],
        num_warps=cfg["warps"])
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
    # split-K arrival counters; the last CTA of a tile resets its counter, so they stay zero between calls
    state["cnt"] = torch.zeros(MAX_TILES, dtype=torch.int32, device=router.device)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer, n, m = cfg["layer"], cfg["n"], x.shape[0]
    k = x.shape[1]
    if layer == "router":
        if m <= SKINNY_MAX:
            return _skinny(x, state["router_hi"], state["router_lo"], state["cnt"], n,
                           torch.float32, _pick(layer, m, n, k))
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    w = state[layer][:n]
    # qkv_gate: the Triton kernel still loses to cuBLAS here, so it stays on the library path
    if m <= SKINNY_MAX and layer != "qkv_gate":
        return _skinny(x, w, None, state["cnt"], n, torch.bfloat16, _pick(layer, m, n, k))
    return x @ w.T
