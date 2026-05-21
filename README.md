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
| `--sieve {c,parallel,serial}` | c | sieve backend (C/OpenMP, multiprocessing, or single-process) |
| `--workers N` | 0 (all cores) | sieve thread/process count |
| `--max-matches N` | 0 (until Ctrl-C) | stop after N hits |
| `--out FILE` | — | append matches to FILE |
| `--stats-interval S` | 5.0 | seconds between throughput lines |
| `--seed-rng-seed N` | random | deterministic run (testing) |

Only the base58 alphabet is legal in `PATTERN`
(`rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz`); `--case-sensitive`
off also accepts upper/lowercase variants.

## Architecture: hybrid GPU/CPU

- **GPU** (`kernels/`, `vanity/gpu.py`): seed16 -> double SHA-512 + Ed25519 clamp
  -> scalar-mult base point -> packed 33-byte pubkey. Compiled once via CuPy NVRTC.
- **CPU** (`vanity/sieve.py`): SHA-256 + RIPEMD-160 -> account_id -> base58check
  address -> prefix match, fanned across all cores by `ParallelSieve`.

The sieve, not the GPU, was the bottleneck. The default `c` backend
(`vanity/csieve.c`) does the whole sieve — SHA-256, RIPEMD-160, base58check,
prefix compare — in C with OpenMP across all cores, compiled on first import and
loaded via `ctypes`. The pure-Python backends (`parallel`, `serial`) remain for
reference and testing.

## Throughput (RTX 2060 Super, 20-core CPU)

| Stage / backend | Rate |
|---|---|
| GPU pipeline | ~4.6M seeds/s |
| Python sieve, single core (`serial`) | ~31K/s |
| Python sieve, 20 cores (`parallel`) | ~283K/s |
| C/OpenMP sieve, 20 cores (`c`) | ~3.7M/s |
| **End-to-end CLI (`c`)** | **~2.2M/s** |

~22x the ~100K/s Java CPU baseline. The serial loop interleaves GPU and sieve;
double-buffering them (planned) lifts the ceiling toward the ~3.7M/s sieve rate.

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
