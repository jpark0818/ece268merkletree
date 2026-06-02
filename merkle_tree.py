"""
merkle_tree.py  –  Hash-agnostic binary Merkle Tree

Two implementations are provided:

  MerkleTree          – standard CPU implementation; takes an element-wise
                        hash_fn(left: int, right: int) -> int.

  BatchMerkleTree     – level-parallel implementation that processes an
                        entire layer in one call; takes a vectorised
                        batch_hash_fn(lefts, rights) -> array.
                        Compatible with both NumPy (CPU) and CuPy (GPU),
                        enabling GPU acceleration once GPU-native hash
                        kernels are available for Poseidon / Rescue.

Hash adapters (return a hash_fn compatible with MerkleTree):

  make_poseidon_fn()            – wraps hashfunctions.poseidon
  make_rescue_fn(**kwargs)      – wraps hashfunctions.RescuePrime

Usage
-----
    from merkle_tree import MerkleTree, make_poseidon_fn

    h = make_poseidon_fn()
    tree = MerkleTree(leaves=[1, 2, 3, 4], hash_fn=h)
    proof = tree.get_proof(2)
    assert MerkleTree.verify_proof(leaf=3, proof=proof,
                                   root=tree.root, hash_fn=h)
"""

from __future__ import annotations
from typing import Callable, List, Sequence, Tuple

# MerkleTree  (CPU / element-wise)

class MerkleTree:
    """
    Binary Merkle Tree built bottom-up from a list of integer leaves.

    Storage layout (self.layers)
    ─────────────────────────────
        layers[0]    leaf layer, right-padded to the next power of two
        layers[1]    first internal layer
        …
        layers[-1]   [root]

    Padding rule
    ─────────────
    If the leaf count is not a power of two, the last leaf is duplicated
    until the count is a power of two.  This matches the convention used in
    Bitcoin / RFC Merkle proofs and gives every proof the same depth.
    """


    def __init__(
        self,
        leaves:  List[int],
        hash_fn: Callable[[int, int], int],
    ) -> None:
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf.")

        self.hash_fn      = hash_fn
        self.original_len = len(leaves)

        padded = _pad_to_power_of_two(list(leaves))
        self.layers: List[List[int]] = [padded]
        self._build()


    def _build(self) -> None:
        """Iteratively hash adjacent pairs from the leaf layer to the root."""
        current = self.layers[0]
        while len(current) > 1:
            next_layer = [
                self.hash_fn(current[i], current[i + 1])
                for i in range(0, len(current), 2)
            ]
            self.layers.append(next_layer)
            current = next_layer


    @property
    def root(self) -> int:
        """The Merkle root (a single integer field element)."""
        return self.layers[-1][0]

    @property
    def height(self) -> int:
        """Number of edges on any root-to-leaf path (= log₂ of padded size)."""
        return len(self.layers) - 1


    def get_proof(self, index: int) -> List[Tuple[int, str]]:
        """
        Return an inclusion proof for the leaf at position ``index``.

        Returns
        -------
        List of ``(sibling_value, side)`` tuples ordered leaf → root
        (not including the root itself).

        ``side`` encodes the sibling's position relative to the current node:
            'R'  →  sibling is to the RIGHT  →  parent = hash(current, sibling)
            'L'  →  sibling is to the LEFT   →  parent = hash(sibling, current)
        """
        if not (0 <= index < self.original_len):
            raise IndexError(
                f"Leaf index {index} is out of range. "
                f"Tree has {self.original_len} original leaves "
                f"(valid range: 0 – {self.original_len - 1})."
            )

        proof: List[Tuple[int, str]] = []
        idx = index

        for layer in self.layers[:-1]:   # every layer except the root
            if idx % 2 == 0:             # current node is a left child
                sibling_idx = idx + 1    # always valid: padded to power of 2
                side        = 'R'
            else:                        # current node is a right child
                sibling_idx = idx - 1
                side        = 'L'
            proof.append((layer[sibling_idx], side))
            idx //= 2

        return proof


    @staticmethod
    def verify_proof(
        leaf:    int,
        proof:   List[Tuple[int, str]],
        root:    int,
        hash_fn: Callable[[int, int], int],
    ) -> bool:
        """
        Verify an inclusion proof without access to the full tree.

        Walk the proof from leaf toward root, combining with each sibling
        according to its recorded side.  Return True iff the recomputed
        value equals ``root``.
        """
        current = leaf
        for sibling, side in proof:
            if side == 'R':
                current = hash_fn(current, sibling)
            else:
                current = hash_fn(sibling, current)
        return current == root


    def __repr__(self) -> str:
        lines = [
            f"MerkleTree  original_leaves={self.original_len}"
            f"  padded={len(self.layers[0])}  height={self.height}"
        ]
        lines.append(f"  root : {self.root:064x}")
        for lvl, layer in enumerate(self.layers[:-1]):
            preview = "  ".join(f"{v:016x}" for v in layer[:4])
            suffix  = (
                f"  … ({len(layer)} nodes)" if len(layer) > 4
                else f"  ({len(layer)} nodes)"
            )
            lines.append(f"  L{lvl:02d}  : {preview}{suffix}")
        return "\n".join(lines)


# BatchMerkleTree  (level-parallel CPU / GPU)

class BatchMerkleTree:
    """
    Merkle Tree optimised for level-parallel execution on CPU or GPU.
    """

    def __init__(self, leaves, batch_hash_fn: Callable) -> None:
        try:
            import numpy as np
            _arr = np if not _is_cupy(leaves) else __import__("cupy")
        except ImportError:
            import numpy as _arr   # type: ignore

        if len(leaves) == 0:
            raise ValueError("BatchMerkleTree requires at least one leaf.")

        self.batch_hash_fn = batch_hash_fn
        self.original_len  = len(leaves)
        self._xp           = __import__("cupy") if _is_cupy(leaves) else __import__("numpy")

        padded = self._pad(leaves)
        self.layers = [padded]
        self._build()

    def _pad(self, leaves):
        n      = len(leaves)
        target = _next_power_of_two(n)
        if target > n:
            last   = leaves[-1:]
            pad    = self._xp.repeat(last, target - n)
            leaves = self._xp.concatenate([leaves, pad])
        return leaves

    def _build(self) -> None:
        current = self.layers[0]
        while len(current) > 1:
            lefts  = current[0::2]
            rights = current[1::2]
            next_layer = self.batch_hash_fn(lefts, rights)
            self.layers.append(next_layer)
            current = next_layer

    @property
    def root(self) -> int:
        val = self.layers[-1][0]
        return int(val)

    @property
    def height(self) -> int:
        return len(self.layers) - 1

    def get_proof(self, index: int) -> List[Tuple[int, str]]:
        if not (0 <= index < self.original_len):
            raise IndexError(
                f"Leaf index {index} out of range "
                f"(tree has {self.original_len} original leaves)."
            )
        proof: List[Tuple[int, str]] = []
        idx = index
        for layer in self.layers[:-1]:
            if idx % 2 == 0:
                sib_idx = idx + 1
                side    = 'R'
            else:
                sib_idx = idx - 1
                side    = 'L'
            proof.append((int(layer[sib_idx]), side))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(
        leaf:    int,
        proof:   List[Tuple[int, str]],
        root:    int,
        hash_fn: Callable[[int, int], int],
    ) -> bool:
        return MerkleTree.verify_proof(leaf, proof, root, hash_fn)


# Utility functions

def _next_power_of_two(n: int) -> int:
    """Smallest power of two ≥ n (integer arithmetic only)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_to_power_of_two(leaves: List[int]) -> List[int]:
    target = _next_power_of_two(len(leaves))
    while len(leaves) < target:
        leaves.append(leaves[-1])
    return leaves


def _is_cupy(arr) -> bool:
    """Return True if arr is a CuPy array (without importing CuPy)."""
    return type(arr).__module__.startswith("cupy")


# Hash adapters

def make_poseidon_fn() -> Callable[[int, int], int]:
    """
    Return a MerkleTree-compatible hash function backed by Poseidon (BN254).
    """
    from hashfunctions.poseidon import poseidon_init, poseidon_hash as _ph
    poseidon_init()

    def _hash(left: int, right: int) -> int:
        return _ph(left, right)

    _hash.__name__ = "poseidon_hash"
    return _hash


def make_rescue_fn(
    p:              int = 4_294_967_291,   # 2^32 - 5
    m:              int = 3,
    capacity:       int = 1,
    security_level: int = 128,
) -> Callable[[int, int], int]:
    """
    Return a MerkleTree-compatible hash function backed by Rescue-Prime.
    """
    from hashfunctions.RescuePrime import RescuePrime
    rp = RescuePrime(p=p, m=m, capacity=capacity, security_level=security_level)

    def _hash(left: int, right: int) -> int:
        return rp.hash([left, right])[0]

    _hash.__name__ = "rescue_hash"
    return _hash


def make_numpy_batch_fn(
    hash_fn: Callable[[int, int], int],
):
    """
    Wrap an element-wise hash_fn as a NumPy-vectorised batch function
    for use with BatchMerkleTree (CPU-parallel baseline).
    """
    import numpy as np

    vhash = np.vectorize(hash_fn, otypes=[object])

    def _batch(lefts, rights):
        result = vhash(lefts, rights)
        return result.astype(object)

    return _batch