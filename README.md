# xrp-vanity-gpu

GPU-accelerated vanity address generator for the XRP Ledger (Ed25519 keys).
A random batch of seeds is turned into XRPL classic addresses entirely on the
GPU, which then reports only the seeds whose address matches your prefix. On an
RTX 2060 Super it sustains **~6.1M addresses/s**.

> **Heads-up:** this tool generates real, spendable XRPL keys. Anything that
> reaches your prefix is a usable account — guard the printed `sEd...` seeds like
> any private key. See [Security](#security).

## Requirements

**An NVIDIA CUDA GPU is mandatory.** Key derivation (Ed25519 scalar
multiplication) runs only on the GPU; there is no CPU-only fallback, so the tool
will not run without a working CUDA device — including on the `csieve` branch,
which only moves the *sieve* to the CPU and still derives keys on the GPU.

| | Requirement | Notes |
|---|---|---|
| GPU | NVIDIA, CUDA-capable (compute ≥ 6.0) | Developed/tested on an **RTX 2060 Super** (8 GB, compute 7.5) |
| VRAM | ~200 MB at the default `--batch-size 1048576` | ~49 B/seed on-device; lower `--batch-size` for smaller cards |
| Driver/CUDA | CUDA 12.x runtime + matching driver | Tested with **CUDA 12.2**. NVRTC compiles the kernels at runtime — **no `nvcc`/CUDA toolkit needed** |
| OS | Linux | Developed on Ubuntu (kernel 5.15). Other CUDA platforms are untested |
| Python | 3.10 | |
| Python deps | `cupy` (built for your CUDA), `xrpl-py`, `pycryptodome`, `numpy` | |
| C compiler | `cc`/`gcc` with OpenMP | **`csieve` branch only**, for the CPU sieve |

No NVIDIA GPU → this tool will not work for you. There is no AMD/Metal/CPU port.

## Setup

The kernels are compiled at runtime with NVRTC, so CuPy needs **both** the CUDA
runtime libraries (incl. `libnvrtc.so`) **and** the CUDA headers. The reliable,
self-contained way to get all of that is to install CuPy from conda-forge, which
bundles a matching CUDA toolkit:

```bash
conda create -n xrpvanity -c conda-forge python=3.10 cupy
conda activate xrpvanity
pip install xrpl-py pycryptodome
```

That's it — `conda-forge`'s `cupy` pulls in the CUDA runtime *and* headers, so
no system CUDA toolkit and no `CUDA_PATH` fiddling is required.

### Alternative: pip wheels

This also works on a machine with no system CUDA toolkit, but you must include
the **nvcc** wheel — without it CuPy can find the GPU yet fails to compile:

```bash
pip install cupy-cuda12x nvidia-cuda-nvcc-cu12 xrpl-py pycryptodome numpy
```

`nvidia-cuda-nvcc-cu12` provides a CUDA root (and pulls in NVRTC) so CuPy can
auto-detect it — no `CUDA_PATH` needed. Common failures from an incomplete set:

- `DynamicLibNotFoundError: Failure finding "libnvrtc.so"` — NVRTC missing.
- `RuntimeError: Failed to auto-detect CUDA root directory` — install
  `nvidia-cuda-nvcc-cu12` (the `nvidia-cuda-nvrtc-cu12` wheel alone is not enough).

Verify CuPy sees your GPU before running:

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

> **First run is slow.** NVRTC compiles the Ed25519 kernel the first time
> (minutes on some setups). CuPy caches the result (`~/.cupy/kernel_cache`), so
> every subsequent run starts in ~1–2 s until the kernel source changes.

## Usage

```bash
python xrp_vanity_gpu.py PATTERN [options]
```

`PATTERN` matches the address characters immediately after the leading `r`
(e.g. `Daimyo` matches `rDaimyo...`).

| Option | Default | Meaning |
|---|---|---|
| `--case-sensitive` | off | exact-case prefix match |
| `--batch-size N` | 1048576 | seeds derived per GPU launch |
| `--max-matches N` | 0 (until Ctrl-C) | stop after N hits |
| `--out FILE` | — | append matches to FILE |
| `--stats-interval S` | 5.0 | seconds between throughput lines |
| `--seed-rng-seed N` | random | deterministic run (for tests) |

Only the XRPL base58 alphabet is legal in `PATTERN`
(`rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz`). With
`--case-sensitive` off, upper/lowercase variants are also accepted.

Each hit prints the address, its `sEd...` family seed, and the attempt index:

```
[2026-05-21T19:28:46] MATCH  rDwnipXUf5QetezXGdnS8WED9bVVvoS3m4  seed=sEdVeQ6F6ubkLaynvmQTcpfAsG364Tx  (attempt 6)
```

Longer / case-sensitive prefixes are exponentially rarer — each extra fixed
character divides the hit rate by ~58.

## How it works

Two GPU kernels share an on-device pubkey buffer; only a tiny list of matching
indices ever returns to the host.

- **`pipeline`** (`kernels/pipeline_kernel.cu` + `ed25519_kernel.cu`): seed16 →
  double SHA-512 + Ed25519 clamp → scalar-mult base point → packed 33-byte
  pubkey, written to a device buffer.
- **`sieve_pubkeys`** (`kernels/sieve_kernel.cu`): SHA-256 + RIPEMD-160 →
  account_id → base58check address → prefix compare, `atomicAdd`-ing matching
  indices into a small device list.

The two are compiled as **separate** CuPy modules. Combined into one translation
unit, ptxas spins for >15 min (the Ed25519 scalarmult plus the hash chain blow up
optimization); apart, the Ed25519 module is cached after first build and the
sieve module compiles in seconds. The CPU only rebuilds the rare hits and
independently verifies each against `xrpl-py`.

Ed25519 is a CUDA port of [ed25519-donna](https://github.com/floodyberry/ed25519-donna);
SHA-256/RIPEMD-160/base58check are self-contained NVRTC-compatible device code.

## Throughput (RTX 2060 Super, 20-core CPU)

| Pipeline | Rate |
|---|---|
| Hybrid (GPU derive + readback + pure-Python sieve) | ~0.28M/s |
| **All-GPU sieve** (`main`) | **~6.1M/s** |
| C/OpenMP CPU sieve (`csieve` branch) | ~2.2M/s |

~61× a ~100K/s single-machine Java CPU baseline.

## Branches

- **`main`** — all-GPU sieve (~6.1M/s). Recommended.
- **`csieve`** — sieve runs on the CPU via a self-contained C/OpenMP extension
  (`vanity/csieve.c`, built on first import). ~2.2M/s, but avoids the slow
  one-time GPU kernel compile. Still requires the GPU for key derivation.

## Layout

```
kernels/      CUDA device code (NVRTC-compatible, ASCII-only, no #include)
vanity/       encoding.py, sieve.py, gpu.py, stats.py
tests/        functional tests vs xrpl-py / hashlib / donna vectors
third_party/  vendored ed25519-donna (reference + vector generator)
archive/      parked early all-GPU SHA-256 experiment (see archive/README.md)
docs/         design specs
```

## Tests

```bash
python -m pytest tests/ -q          # CPU/encoding tests
python -m pytest tests/ -q -m gpu   # GPU correctness (needs a CUDA device)
```

`tests/test_gpu_sieve.py` asserts the GPU sieve's match set is byte-for-byte
identical to the `xrpl-py`-verified CPU reference sieve.

## Security

- Generated `sEd...` seeds are real XRPL private keys. Treat the output (and any
  `--out` file) as secret material; don't paste seeds into untrusted tools.
- Every reported match is re-derived and checked against `xrpl-py` before it is
  printed, so addresses are independently verified — but **you** are responsible
  for safely importing and funding any account you keep.
- Seeds come from NumPy's default RNG (PCG64), which is fine for finding vanity
  addresses but is **not** a cryptographically hardened key-generation path. For
  high-value accounts, generate the final key with a vetted wallet.

## Dev notes (NVRTC gotchas)

- Prepend a preamble with `typedef unsigned int uint32_t;` etc. — NVRTC has no
  `stdint.h`.
- Source must be ASCII-only (em-dashes and other non-ASCII break compilation).
- Pass scalar kernel args as `np.uint32(N)`, not `cp.array(N)` — the latter
  causes `cudaErrorIllegalAddress`.
- Keep the Ed25519 and sieve kernels in separate modules (see "How it works").

## Credits

Ed25519 field/group arithmetic ported from Andrew Moon's public-domain
[ed25519-donna](https://github.com/floodyberry/ed25519-donna).
