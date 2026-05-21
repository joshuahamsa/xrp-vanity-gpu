"""C/OpenMP sieve: compiles csieve.c on demand, calls it via ctypes."""
import ctypes
import os
import subprocess
import threading

import numpy as np

from vanity import encoding, sieve

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "csieve.c")
_SO = os.path.join(_HERE, "libcsieve.so")
_BUILD_LOCK = threading.Lock()


def _ensure_built() -> str:
    with _BUILD_LOCK:
        if (not os.path.exists(_SO)) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
            cmd = [
                "cc", "-O3", "-fopenmp", "-march=native", "-fPIC",
                "-shared", _SRC, "-o", _SO,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _SO


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(_ensure_built())
    lib.sieve_c.restype = ctypes.c_int
    lib.sieve_c.argtypes = [
        ctypes.c_char_p,                    # pubkeys
        ctypes.c_int,                       # b
        ctypes.c_char_p,                    # needle
        ctypes.c_int,                       # needle_len
        ctypes.c_int,                       # case_sensitive
        ctypes.POINTER(ctypes.c_int32),     # out_indices
        ctypes.c_int,                       # max_out
    ]
    return lib


class CSieve:
    """Drop-in for ParallelSieve backed by the C/OpenMP sieve."""

    def __init__(self, n_workers: int | None = None, max_out: int = 4096):
        if n_workers:
            os.environ["OMP_NUM_THREADS"] = str(n_workers)
        self._lib = _load()
        self._max_out = max_out
        self._out = (ctypes.c_int32 * max_out)()

    def sieve_batch(
        self,
        pubkeys: bytes,
        seeds: bytes,
        pattern: str,
        case_sensitive: bool,
        first_attempt_index: int,
    ) -> list[sieve.Match]:
        b = len(pubkeys) // sieve.PUBKEY_LEN
        needle = pattern if case_sensitive else pattern.lower()
        needle_b = needle.encode("ascii")
        n_hits = self._lib.sieve_c(
            pubkeys, b, needle_b, len(needle), 1 if case_sensitive else 0,
            self._out, self._max_out,
        )
        hits: list[sieve.Match] = []
        for j in range(n_hits):
            i = self._out[j]
            pub = pubkeys[i * sieve.PUBKEY_LEN : (i + 1) * sieve.PUBKEY_LEN]
            seed = seeds[i * sieve.SEED_LEN : (i + 1) * sieve.SEED_LEN]
            hits.append(sieve.Match(
                seed_b58=encoding.family_seed_encode(seed),
                address=sieve.address_from_pubkey(pub),
                attempt=first_attempt_index + i,
            ))
        return hits

    def close(self) -> None:
        pass
