"""GPU sieve_pipeline must find exactly the same matches as the CPU sieve."""
import os

import pytest

from vanity import gpu, sieve


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_sieve_matches_cpu_sieve():
    B = 1 << 16
    g = gpu.VanityGpu(batch_size=B)
    seeds = os.urandom(B * 16)
    pubkeys = g.run_batch(seeds)

    for pattern, cs in [("r", False), ("D", True), ("f", True), ("aa", False)]:
        cpu = sieve.sieve_batch(pubkeys, seeds, pattern, cs, 0)
        cpu_idx = sorted(m.attempt for m in cpu)
        needle = (pattern if cs else pattern.lower()).encode("ascii")
        gpu_idx = sorted(int(x) for x in g.sieve_seeds(seeds, needle, cs))
        assert gpu_idx == cpu_idx, (
            f"pattern {pattern!r} cs={cs}: "
            f"gpu={len(gpu_idx)} cpu={len(cpu_idx)} "
            f"diff={set(gpu_idx) ^ set(cpu_idx)}"
        )
        assert len(cpu_idx) > 0, f"no hits for {pattern!r} — weak test"
