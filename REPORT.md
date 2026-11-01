# Report: `muse_bf16_proj`

Written so you can re-learn the problem quickly. Nothing has been run or submitted yet. The repo has no kernel work in it, only the starter files.

## 1. The goal in one paragraph

You are timed against SGLang's kernel on small matrix multiplies from the Muse Spark Open model. It runs on one H200 GPU out of eight. You write `kernel_fn(data, state)`. It must return `x @ W.T`, matching a reference closely, and run faster than SGLang's version. Your score is the geometric mean speedup over 6 loads.

## 2. The math

For every case:

```
out = x @ W.T      x: [m, k] BF16      W: [n, k]      out: [m, n]
```

The output is BF16. The router is the exception: its output is FP32.

There are five weights (all per-GPU sizes):

| layer | W shape [n, k] | dtype | size | notes |
|---|---|---|---|---|
| qkv_gate | 2304 × 8192 | BF16 | 37.7 MB | fused q, k, v and gate |
| o | 8192 × 1024 | BF16 | 16.8 MB | |
| pooler | 4096 × 8192 | BF16 | 67 MB | the biggest cost at small m |
| unpooler | 8192 × 4096 | BF16 | 67 MB | only the first 1024 rows are used when m ≤ 8 |
| router | 256 × 8192 | FP32 | 8.4 MB | picks experts, so it is the most precision-sensitive |

The unpooler detail matters. For m ≤ 8, only the first 1024 rows of the weight are used (`n = 1024`). Above 8 tokens, all 8192 rows are used.

## 3. The shapes that are scored

Six loads, each running all five layers. Each shape has weight 62, which is the number of layers.

| load | m |
|---|---|
| decode 1 | 1 |
| decode 16 | 16 |
| decode 32 | 32 |
| verify 1 | 8 |
| verify 16 | 128 |
| verify 32 | 256 |

- A load's time is the sum of its five layers' times, each times 62.
- Your score is the geometric mean, over the 6 loads, of SGLang's time divided by yours.
- Every load counts equally.

## 4. Where the limit is

The "floor" in TASK.md is the best possible time: either moving the weights at 4.8 TB/s, or doing the math at 989 TFLOP/s.

- **Small m (1 to 32).** These are memory-bound. The time is just reading the weight once. For example, pooler is about 14 µs.
- **m = 128 and 256.** The floors rise a little. Pooler at m = 256 is 17.4 µs. These are closer to compute-bound.
- **Cold L2.** The judge flushes L2 before every timed replay. You cannot count on the weight being cached.
- **Fixed cost.** About 4 µs of graph replay overhead is subtracted from both sides.

Each call is only a few microseconds, so launch overhead and kernel structure matter a lot. Winning means getting close to the floor. For tiny cases like the router and the sliced unpooler, the floor is under 2 µs.

## 5. What SGLang does (`baseline.py`)

- **qkv_gate, o, pooler:** cuBLAS through `_bf16_gemm_dispatch_impl`.
- **unpooler:** the same dispatch. It uses the 1024-row slice up to 8 tokens, and the whole weight above that.
- **router:**
  - Up to 8 tokens: a plain FP32 `F.linear`. This casts `x` to FP32 first.
  - Above 8 tokens: two BF16 GEMMs on a hi/lo split of the weight, accumulated in FP32. The split is `hi = bf16(W)` and `lo = bf16(W − hi)`.

The sliced unpooler's all-gather is not run in the benchmark.

## 6. Correctness rules

A run first has to pass 9 tests. Some use shapes outside the scored ones, such as m = 7, 33, 130, 512 and 1000, and some use an "outliers" input (1% of channels scaled 30×). Checks are relative L2 error against an FP64 reference:

- **Whole output:** at most 1%. For the router, at most 0.01%.
- **Any single row:** at most 5× that bound. Near-zero rows are measured against a tenth of the average row norm.
- **No NaN or Inf.**
- **Probe rows.** About one row in eight of `x` has a single nonzero entry, a power of two. Its output is then just a scaled column of the weights. These rows must be within 0.1%. This stops you from storing weights at lower precision.
- **Router.** BF16 router weights alone fail. The hi + lo split passes at any m. The probe rows are slightly off with the split, but still within 0.1%.

## 7. Rules about how you may do it

**Allowed:**
- Repack or losslessly compress weights in `prepare()`. It runs once, untimed, for up to 300 s.
- Dequantize inside your kernels.
- Use the router hi + lo split with FP32 accumulation.
- Branch on shape and `config`.

**Not allowed:**
- Host syncs inside `kernel_fn`, such as `.item()`, `.cpu()` or printing a tensor.
- Allocating or branching on input values.
- Caching anything derived from `data`. Inputs are fresh every call.
- Storing weights at a different precision.
- Lower-precision activations than SGLang uses.
- Tampering with timing or the harness, starting processes, or using the network.
- Importing the reference.

The final review reruns the top entries, including on nearby shapes and held-out data. Entries that game the judge are removed. The stated aim is real speedups that could be upstreamed.

## 8. Files in the repo

| file | what it is |
|---|---|
| `TASK.md` | the full problem statement |
| `AGENTS.md` | extra notes for a coding agent, including the `okc` commands |
| `baseline.py` | SGLang's kernel, the thing you are timed against |
| `reference.py` | input generator, FP64 reference and checks |
| `checks.py` | the `compare` function used by the checks |
| `submission.py` | your kernel. It is still the plain PyTorch starter, and its first line must stay `# okc: problem=muse_bf16_proj` |

## 9. Workflow

```bash
okc check submission.py --json    # real tests and timing, off the board, no attempt used
okc submit submission.py --json   # counts as an attempt
okc problems --json               # today's limits and what is left
okc leaderboard muse_bf16_proj --json
```

- A person must run `okc login` once in a terminal to accept the rules. An agent cannot do this.
- In a fresh sandbox, set `OKC_TOKEN` from `okc api token` on a signed-in machine.
- There are daily limits on runs and on checks. They reset at 00:00 UTC.
- A run has 10 minutes, including compile time.
- After a passing run, the profile is saved at `profiles/submission.<job>.txt`.
- When you tune, time CUDA graph replays, not eager calls. The two can disagree at these sizes.

## 10. Where things stand

- The repo has only the starter files and a `.gitignore`. There is no real kernel work yet.
- I could not see any past results or leaderboard standing from here. I did not run `okc`.
- Not known to me: whether this sandbox has a GPU, or whether `okc` is logged in.

## 11. Open questions I'm unsure about

- Whether the best approach is a custom kernel for small m, such as split-K GEMV-style kernels that stream the weight at full bandwidth. cuBLAS may pick poor kernels for these shapes. I have not tested this.
- Whether the router at m ≤ 8 can skip the `x.float()` cast and the FP32 GEMM. Using the hi + lo split there would read less data, but whether it stays within the 0.01% tolerance is untested.
- Whether the two router GEMMs above m = 8 can be fused into one pass.
