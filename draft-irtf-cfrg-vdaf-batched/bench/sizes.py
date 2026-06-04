"""Measured byte-size accounting for Batch DZK and PCA DZK benchmark objects."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POC_DIR = PACKAGE_ROOT / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

from pca_dzk import encode_public_basis_transcript  # noqa: E402


@dataclass(frozen=True)
class SizeBreakdown:
    """Communication and proof byte counts for one sharded protocol instance."""

    input_share_bytes_by_aggregator: list[int]
    input_shares_total_bytes: int
    public_share_bytes: int
    setup_public_u_bytes: int
    online_total_comm_bytes: int
    total_comm_with_setup_bytes: int
    proof_share_bytes_by_aggregator: list[int]
    proof_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialized_size(obj: Any) -> int:
    """Return the wire byte length of protocol objects used by the benchmark.

    Field elements are counted exactly as their field encoder writes them:
    ``ENCODED_SIZE`` bytes per element. Existing ``bytes`` objects are already
    packed protocol payloads and are counted at their raw length. Lists and
    tuples are fixed-shape protocol containers, so their serialized size is the
    sum of their members with no extra dynamic-length prefix.

    For PCA setup, a NumPy integer matrix is counted using the package's public
    basis transcript encoding, which is the existing deterministic byte encoding
    bound into the setup identifier.
    """

    if isinstance(obj, bytes):
        return len(obj)
    if _is_field_element(obj):
        return int(type(obj).ENCODED_SIZE)
    if isinstance(obj, (list, tuple)):
        return sum(serialized_size(item) for item in obj)
    if isinstance(obj, np.ndarray):
        return len(encode_public_basis_transcript(obj))
    raise TypeError(f"unsupported serialized object type: {type(obj)!r}")


def size_breakdown(
    protocol: str,
    public_share: Any,
    input_shares: list[Any],
    public_u: np.ndarray | None = None,
) -> SizeBreakdown:
    """Compute communication and QC-proof sizes from actual sharded objects."""

    if protocol not in {"batch", "pca"}:
        raise ValueError(f"unsupported protocol {protocol!r}")

    input_share_bytes = [serialized_size(input_share) for input_share in input_shares]
    public_share_bytes = serialized_size(public_share)
    setup_public_u_bytes = serialized_size(public_u) if public_u is not None else 0
    proof_share_bytes = [
        serialized_size(_proof_share(protocol, input_share))
        for input_share in input_shares
    ]
    online_total = sum(input_share_bytes) + public_share_bytes
    return SizeBreakdown(
        input_share_bytes_by_aggregator=input_share_bytes,
        input_shares_total_bytes=sum(input_share_bytes),
        public_share_bytes=public_share_bytes,
        setup_public_u_bytes=setup_public_u_bytes,
        online_total_comm_bytes=online_total,
        total_comm_with_setup_bytes=online_total + setup_public_u_bytes,
        proof_share_bytes_by_aggregator=proof_share_bytes,
        proof_size_bytes=sum(proof_share_bytes),
    )


def _proof_share(protocol: str, input_share: Any) -> Any:
    if protocol == "batch":
        return input_share[1]
    if protocol == "pca":
        return input_share[2]
    raise ValueError(f"unsupported protocol {protocol!r}")


def _is_field_element(obj: Any) -> bool:
    cls = type(obj)
    return (
        hasattr(cls, "ENCODED_SIZE")
        and isinstance(getattr(cls, "ENCODED_SIZE"), int)
        and hasattr(obj, "as_unsigned")
    )
