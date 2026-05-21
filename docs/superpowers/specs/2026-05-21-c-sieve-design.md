# C-Extension CPU Sieve — Design Spec

**Date:** 2026-05-21
**Goal:** Lift end-to-end vanity throughput from ~283K/s to ≥1M/s (target 3M/s) by replacing the Python sieve with a C/OpenMP sieve, leaving the working GPU pubkey pipeline untouched.

## Problem

The GPU derives pubkeys at ~4.6M/s. The post-GPU sieve (`SHA-256 → RIPEMD-160 → base58check → prefix match`) is the bottleneck: ~31K/s single-core Python, ~283K/s across 20 cores via `multiprocessing`. Per-candidate Python overhead and pure-Python base58 are the wall; no amount of Python parallelism reaches 1M/s.

## Approach

Move the sieve into a single C function compiled to a shared library and called via `ctypes` (no build system, no Python C-API). The algorithm was never slow — Python was. A full base58check in C is ~1µs, so an OpenMP loop over the batch runs ~8M/s, making the pipeline GPU-bound.

**No byte-range filter.** Full, exact base58check per candidate is cheap in C and avoids false-positive/negative bound math. Correctness is trivial to verify against the existing Python implementation.

## Components

### `vanity/csieve.c`
Self-contained C. Public-domain SHA-256 and RIPEMD-160 (no OpenSSL dependency, to match the pure-Python pipeline byte-for-byte). One exported function:

```c
// Returns number of matches found (<= max_out). Writes matching candidate
// indices into out_indices[0..return). OpenMP-parallel over the batch.
int sieve_c(const uint8_t *pubkeys, int b,        // b pubkeys, 33 bytes each
            const char *needle, int needle_len,    // lowercased if !case_sensitive
            int case_sensitive,
            int32_t *out_indices, int max_out);
```

Per candidate `i`:
1. `sha = SHA256(pubkeys[i*33 : i*33+33])`
2. `account_id = RIPEMD160(sha)` (20 bytes)
3. `payload = 0x00 || account_id` (21 bytes)
4. `checksum = SHA256(SHA256(payload))[:4]`
5. `address = base58(payload || checksum)` over the XRPL alphabet, leading-0x00 → leading `r`
6. compare `address[1 : 1+needle_len]` to `needle` (case-folded if `!case_sensitive`)
7. on match, append `i` to a thread-local buffer; buffers merged into `out_indices` after the parallel region.

base58 of the 25-byte payload uses a fixed-width big-integer (the value is ≤ 2^200) divided repeatedly by 58 — a small `uint32_t[]` limb loop. Only the first `needle_len+1` characters are needed for the compare, but computing the full address is simplest and still cheap; emit the full address only on match (Python rebuilds it for reporting, so C need only return the index).

### `vanity/csieve.py`
- On import, compile `csieve.c` to `csieve.so` if missing or stale (mtime check), via `cc -O3 -fopenmp -march=native -shared -fPIC`. Cache the `.so` next to the source.
- Load with `ctypes.CDLL`, declare `sieve_c` signature.
- `CSieve` class with `sieve_batch(pubkeys, seeds, pattern, case_sensitive, first_attempt_index) -> list[Match]`: lowercases `needle` if needed, allocates an `out_indices` buffer (cap e.g. 4096; if full, the batch is rescanned — matches are rare so this never happens in practice), calls `sieve_c`, then for each returned index rebuilds the `Match` (address via `sieve.address_from_pubkey`, seed via `encoding.family_seed_encode`). Same signature/return as `ParallelSieve` so the CLI swaps cleanly.

### CLI (`xrp_vanity_gpu.py`)
Add `--sieve {c,parallel,serial}` (default `c`). `c` → `CSieve`, `parallel` → `ParallelSieve`, `serial` → module `sieve.sieve_batch`. Keep `--workers` (OpenMP thread count for `c`, via `OMP_NUM_THREADS` / a `set_threads` arg).

## Correctness strategy

The C base58check must match the Python pipeline exactly. Test: feed the same random pubkeys to `sieve.sieve_batch` (Python, trusted, verified against xrpl-py) and `CSieve.sieve_batch` (C); assert identical match sets. Also keep the existing xrpl-py oracle test on the Python path. A loose pattern (e.g. `"r"` case-insensitive, ~1/29 hit rate) gives both hits and misses for coverage.

## Throughput expectations

| Stage | Rate |
|---|---|
| GPU pubkey gen | ~4.6M/s |
| C sieve (20 threads) | ~8M/s |
| End-to-end, serial loop | ~2.9M/s |
| End-to-end, double-buffered (Option A of Task 13) | ~4.6M/s |

Serial already clears the 1M floor. Double-buffering (overlap GPU launch with the C sieve of the previous batch) reaches the 3M dream and is a follow-up.

## Out of scope

- All-GPU sieve (separate branch, separate spec).
- Double-buffering (existing Task 13; revisit after this lands).
- Seed generation cost (the per-batch `rng.bytes` may surface as the next bottleneck at multi-M/s; measure, optimize only if it does).

## Branch

`vanity-c-sieve` off `main`. All changes confined to `xrp_vanity_gpu/`.
