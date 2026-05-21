# All-GPU Sieve — Design Spec

**Date:** 2026-05-21
**Goal:** Push vanity throughput past the 1M/s floor (target 3M/s) by moving the
entire sieve onto the GPU, so the CPU does almost nothing and pubkeys never leave
device memory. Result: **~6.1M/s** end-to-end.

## Problem

The hybrid pipeline derived pubkeys on-GPU (~4.6M/s) but copied the whole 33-byte
pubkey batch back to the host (~33 MB/batch) and sieved them in Python. The
readback plus the Python sieve were the bottleneck.

## Approach

Do `SHA-256 -> RIPEMD-160 -> base58check -> prefix compare` on the GPU. The
pipeline writes pubkeys into a device buffer; a second kernel reads that buffer,
derives each address, and appends matching indices to a small device list via
`atomicAdd`. Only the tiny index list returns to the host. The CPU rebuilds the
handful of matches and verifies them with xrpl-py.

The historical "SHA-256 doesn't work on GPU" blocker was retired: a clean,
standard SHA-256 device function produces `sha256(b"abc") = ba7816bf...`
correctly. The old failure was an implementation bug, not a CUDA limitation.

## Two separate modules (critical)

The Ed25519 pipeline and the sieve are compiled as **separate** `cp.RawModule`s.
Combining them into one translation unit made ptxas spin for >15 min — the
Ed25519 scalarmult and the inlined hash chain together blow up optimization.
Apart, the Ed25519 module compiles as before (and is CuPy-cached, ~instant after
first build) and the sieve module compiles in seconds. They share the on-device
pubkey buffer by passing the same `cupy` pointer to each kernel.

The hash device functions are also marked `__noinline__` so the `sieve_pubkeys`
kernel body stays small.

## Components

### `kernels/sieve_kernel.cu`
Self-contained, NVRTC-compatible (ASCII, no `#include`, relies on gpu.py's
typedef preamble). Device functions `sv_sha256`, `sv_ripemd160`,
`sv_base58check`, `sv_address` (all `__noinline__`), plus:
- `sha256_test`, `ripemd160_test`, `address_test` — single-thread entry points
  for unit-testing the device functions against hashlib / pycryptodome / the CPU
  encoder.
- `sieve_pubkeys(pubkeys, needle, needle_len, case_sensitive, out_indices,
  out_count, max_out, B)` — one thread per pubkey; derives the address, compares
  the prefix (needle pre-lowercased by the host when case-insensitive), and
  `atomicAdd`s its index into `out_indices` on a hit.

### `vanity/gpu.py`
- `compile_module()` (Ed25519 pipeline) and `compile_sieve_module()` (sieve),
  compiled separately.
- `VanityGpu.sieve_seeds(seeds, needle, case_sensitive) -> np.ndarray` — sets
  seeds, launches `pipeline` (seeds -> device pubkeys) then `sieve_pubkeys`
  (device pubkeys -> indices), reads back the match count + indices. No pubkey
  readback. `out_indices` capacity is 4096 (matches are rare; overflow only
  drops extras within a single huge batch).

### `xrp_vanity_gpu.py`
The loop calls `g.sieve_seeds`; for each returned index it rebuilds the seed and
verifies the address with xrpl-py (`derive_keypair` / `derive_classic_address`),
so every reported match is independently checked.

## Correctness

`tests/test_gpu_sieve.py` runs a batch through both the GPU sieve and the
xrpl-py-verified CPU `sieve.sieve_batch` and asserts identical match-index sets
across several patterns (case-sensitive and insensitive). The device hash
functions are additionally checked against hashlib / pycryptodome.

## Throughput (RTX 2060 Super, 20-core CPU)

| Pipeline | Rate |
|---|---|
| Hybrid (GPU derive + readback + Python sieve) | ~0.28M/s |
| C/OpenMP sieve (other branch) | ~2.2M/s |
| **All-GPU sieve** | **~6.1M/s** |

~61x the ~100K/s Java CPU baseline; 2x past the 3M/s dream.

## Out of scope

- Seed generation is still host-side `rng.bytes`; at 6.1M/s it has not surfaced
  as a bottleneck but is the next thing to check if pushed further.
- Double-buffering / streams (would mainly help the hybrid path; the all-GPU
  path is already kernel-bound).

## Branch

`vanity-gpu-sieve` off `main`. All changes confined to `xrp_vanity_gpu/`.
