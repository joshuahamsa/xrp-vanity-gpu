"""C sieve must match the xrpl-py-verified Python sieve exactly."""
import secrets

from vanity import csieve, encoding, sieve


def _xrplpy_pubkey(seed16: bytes) -> bytes:
    from xrpl.core.keypairs import derive_keypair
    pub_hex, _ = derive_keypair(encoding.family_seed_encode(seed16))
    return bytes.fromhex(pub_hex)


def test_csieve_matches_python_sieve_loose_pattern():
    # "r" case-insensitive hits ~1/29 of addresses -> both hits and misses.
    B = 5000
    seeds = secrets.token_bytes(B * 16)
    pubkeys = secrets.token_bytes(B * 33)  # random pubkeys exercise base58 paths
    cs = csieve.CSieve()
    for pat, cis in [("r", False), ("D", True), ("aa", False)]:
        py = sieve.sieve_batch(pubkeys, seeds, pat, cis, 0)
        c = cs.sieve_batch(pubkeys, seeds, pat, cis, 0)
        assert sorted(c) == sorted(py), f"mismatch for pattern {pat!r} cis={cis}"
    cs.close()


def test_csieve_finds_real_address():
    # Build a real seed whose address starts 'rD', confirm C sieve reports it.
    target = None
    for _ in range(400):
        seed = secrets.token_bytes(16)
        pub = _xrplpy_pubkey(seed)
        if sieve.address_from_pubkey(pub)[1] == "D":
            target = (seed, pub)
            break
    assert target is not None
    seed, pub = target
    other = secrets.token_bytes(16)
    other_pub = _xrplpy_pubkey(other)
    pubkeys = pub + other_pub
    seeds = seed + other
    cs = csieve.CSieve()
    hits = cs.sieve_batch(pubkeys, seeds, "D", True, 100)
    cs.close()
    assert len(hits) == 1
    assert hits[0].address[1] == "D"
    assert hits[0].attempt == 100
    assert hits[0].seed_b58 == encoding.family_seed_encode(seed)
