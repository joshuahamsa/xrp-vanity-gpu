# xrp_vanity_gpu

GPU-accelerated XRP (Ed25519) vanity address search. The GPU derives public
keys from random seeds; the CPU sieves them for prefix matches across all cores.

## Usage

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda run -n rapids-23.12 python xrp_vanity_gpu.py PATTERN [options]
```

`PATTERN` matches the address characters immediately after the leading `r`
(e.g. `Daimyo` matches `rDaimyo...`).

| Option | Default | Meaning |
|---|---|---|
| `--case-sensitive` | off | exact-case prefix match |
| `--batch-size N` | 1048576 | seeds derived per GPU launch |
| `--workers N` | 0 (all cores) | CPU sieve processes |
| `--max-matches N` | 0 (until Ctrl-C) | stop after N hits |
| `--out FILE` | — | append matches to FILE |
| `--stats-interval S` | 5.0 | seconds between throughput lines |
| `--seed-rng-seed N` | random | deterministic run (testing) |

Only the base58 alphabet is legal in `PATTERN`
(`rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz`); `--case-sensitive`
off also accepts upper/lowercase variants.

## Architecture: all-GPU

Both stages run on the GPU as two kernels sharing an on-device pubkey buffer; only
a tiny list of matching indices returns to the host.

- **`pipeline`** (`kernels/pipeline_kernel.cu` + `ed25519_kernel.cu`): seed16 ->
  double SHA-512 + Ed25519 clamp -> scalar-mult base point -> packed 33-byte
  pubkey, written to a device buffer.
- **`sieve_pubkeys`** (`kernels/sieve_kernel.cu`): SHA-256 + RIPEMD-160 ->
  account_id -> base58check address -> prefix compare, `atomicAdd`-ing matching
  indices into a small device list.

The two are compiled as **separate** CuPy modules — combining them into one
translation unit makes ptxas spin for >15 min (the Ed25519 scalarmult plus the
hash chain blow up optimization). Apart, the Ed25519 module is CuPy-cached after
its first build and the sieve module compiles in seconds. The CPU only rebuilds
the rare hits and verifies each against xrpl-py.

The `vanity/sieve.py` Python sieve and its `ParallelSieve` remain for testing —
`test_gpu_sieve.py` asserts the GPU sieve's match set is identical to theirs.

## Throughput (RTX 2060 Super, 20-core CPU)

| Pipeline | Rate |
|---|---|
| Hybrid (GPU derive + readback + Python sieve) | ~0.28M/s |
| **All-GPU sieve** | **~6.1M/s** |

~61x the ~100K/s Java CPU baseline.

## Layout

```
kernels/      CUDA device code (NVRTC-compatible, ASCII-only, no #include)
vanity/       encoding.py, sieve.py, gpu.py, stats.py
tests/        functional tests vs xrpl-py / hashlib / donna vectors
third_party/  vendored ed25519-donna (reference + vector generator)
archive/      parked all-GPU SHA-256 experiment (see archive/README.md)
```

## NVRTC gotchas

- Prepend a PREAMBLE with `typedef unsigned int uint32_t;` (no `stdint.h`).
- ASCII-only source (em-dashes and other non-ASCII break compilation).
- Pass scalar kernel args as `np.uint32(N)`, not `cp.array(N)` — the latter
  causes `cudaErrorIllegalAddress`.
