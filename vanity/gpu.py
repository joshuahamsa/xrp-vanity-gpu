"""CuPy kernel-loading and launch wrappers."""
from pathlib import Path

import cupy as cp
import numpy as np

KERNELS_DIR = Path(__file__).resolve().parents[1] / "kernels"

# typedef preamble: NVRTC has no stdint.h.
NVRTC_PREAMBLE = """
typedef unsigned char       uint8_t;
typedef unsigned short      uint16_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;
typedef long long           int64_t;
typedef signed char         int8_t;
"""

# Order matters: sha_kernels first (defines sha512_16/sha512_32),
# then ed25519 (depends on typedefs from preamble),
# then pipeline (depends on both).
# The Ed25519 pipeline and the sieve are compiled as SEPARATE modules.
# Combining them makes ptxas spin for >15 min (the Ed25519 scalarmult and the
# hash chain together blow up optimization); apart, each compiles quickly and
# they share the on-device pubkey buffer via a raw pointer.
KERNEL_FILES = ["sha_kernels.cu", "ed25519_kernel.cu", "pipeline_kernel.cu"]
SIEVE_KERNEL_FILES = ["sieve_kernel.cu"]


def _read_kernel_sources(files: list[str]) -> str:
    chunks = [NVRTC_PREAMBLE]
    for name in files:
        chunks.append((KERNELS_DIR / name).read_text(encoding="ascii", errors="replace"))
    return "\n".join(chunks)


def _compile(source: str) -> cp.RawModule:
    try:
        return cp.RawModule(
            code=source,
            backend="nvrtc",
            options=("--std=c++14",),
        )
    except cp.cuda.compiler.CompileException as e:
        numbered = "\n".join(
            f"{i+1:4d}: {line}" for i, line in enumerate(source.splitlines())
        )
        raise RuntimeError(
            f"NVRTC compile failed:\n{e}\n\n--- FULL SOURCE ---\n{numbered}"
        ) from e


def compile_module() -> cp.RawModule:
    """Compile the Ed25519 seed->pubkey pipeline module."""
    return _compile(_read_kernel_sources(KERNEL_FILES))


def compile_sieve_module() -> cp.RawModule:
    """Compile the pubkey->address->match sieve module (no Ed25519)."""
    return _compile(_read_kernel_sources(SIEVE_KERNEL_FILES))


class VanityGpu:
    """Owns the compiled modules and per-batch device/host buffers."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.module = compile_module()
        self.sieve_module = compile_sieve_module()
        self._pipeline = self.module.get_function("pipeline")
        self._sieve = self.sieve_module.get_function("sieve_pubkeys")

        self._d_seeds = cp.zeros(batch_size * 16, dtype=cp.uint8)
        self._d_pubkeys = cp.zeros(batch_size * 33, dtype=cp.uint8)
        try:
            self._h_pubkeys = cp.cuda.alloc_pinned_memory(batch_size * 33)
            self._pinned = True
        except Exception:
            self._h_pubkeys = None
            self._pinned = False

        self._max_out = 4096
        self._d_out_idx = cp.zeros(self._max_out, dtype=cp.uint32)
        self._d_out_cnt = cp.zeros(1, dtype=cp.uint32)
        self._d_needle = cp.zeros(64, dtype=cp.uint8)

        self._threads = 256
        self._blocks = (batch_size + self._threads - 1) // self._threads

    def run_batch(self, seeds: bytes) -> bytes:
        if len(seeds) != self.batch_size * 16:
            raise ValueError(
                f"seeds must be {self.batch_size * 16} bytes, got {len(seeds)}"
            )
        self._d_seeds.set(np.frombuffer(seeds, dtype=np.uint8))
        self._pipeline(
            (self._blocks,), (self._threads,),
            (self._d_seeds, self._d_pubkeys, np.uint32(self.batch_size)),
        )
        if self._pinned:
            cp.cuda.runtime.memcpy(
                int(self._h_pubkeys.ptr),
                int(self._d_pubkeys.data.ptr),
                self.batch_size * 33,
                cp.cuda.runtime.memcpyDeviceToHost,
            )
            return bytes(memoryview(self._h_pubkeys)[: self.batch_size * 33])
        return bytes(cp.asnumpy(self._d_pubkeys))

    def sieve_seeds(self, seeds: bytes, needle: bytes,
                    case_sensitive: bool) -> np.ndarray:
        """Derive + sieve a batch on-GPU; return matching seed indices.

        `needle` must already be lowercased when case_sensitive is False.
        Returns a sorted uint32 ndarray of indices into the batch.
        """
        if len(seeds) != self.batch_size * 16:
            raise ValueError(
                f"seeds must be {self.batch_size * 16} bytes, got {len(seeds)}"
            )
        if len(needle) > 64:
            raise ValueError("needle too long (max 64)")
        self._d_seeds.set(np.frombuffer(seeds, dtype=np.uint8))
        self._d_out_cnt.fill(0)
        if needle:
            self._d_needle[: len(needle)].set(np.frombuffer(needle, dtype=np.uint8))
        self._pipeline(
            (self._blocks,), (self._threads,),
            (self._d_seeds, self._d_pubkeys, np.uint32(self.batch_size)),
        )
        self._sieve(
            (self._blocks,), (self._threads,),
            (self._d_pubkeys, self._d_needle, np.uint32(len(needle)),
             np.uint32(1 if case_sensitive else 0),
             self._d_out_idx, self._d_out_cnt, np.uint32(self._max_out),
             np.uint32(self.batch_size)),
        )
        cnt = int(self._d_out_cnt.get()[0])
        n = min(cnt, self._max_out)
        if n == 0:
            return np.empty(0, dtype=np.uint32)
        idx = cp.asnumpy(self._d_out_idx[:n])
        idx.sort()
        return idx
