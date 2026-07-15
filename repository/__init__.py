"""
GameBrains decentralized repository — local, real "blockchain-lite" implementation.

Implements exactly the architecture in `docs/repository-schema.md`, scoped to run entirely on one
machine without any peer-to-peer networking: a content-addressed store standing in for IPFS
(`cas.py`), and a signed, append-only, hash-linked ledger standing in for the "chain" (`ledger.py`).
Core principle carried over from the design doc: never mutate, only add and link.

This is deliberately *not* a full blockchain: there is no consensus mechanism, because our
"transactions" (experiment records) are purely additive between mutually-trusted local runs, not
conflicting claims between untrusted parties — the double-spend problem a consensus protocol
solves does not arise here. What we do provide, with genuine asymmetric cryptography rather than a
toy scheme, is exactly what a blockchain-lite log should: immutability (hash-linking breaks if
anything upstream is edited) and public verifiability (Ed25519 signatures, not a shared secret).
Anchoring a periodic Merkle root to a real public chain remains explicit future work — see
`docs/repository-schema.md` §7 and `CLAUDE.md`.
"""
