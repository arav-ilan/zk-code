"""Shared DZK helpers for Batch DZK and PCA DZK."""

from typing import NamedTuple, TypeVar

from vdaf_poc.common import concat, next_power_of_2, vec_add, vec_sub, zeros
from vdaf_poc.field import NttField
from vdaf_poc.flp_bbcggi19 import FlpBBCGGI19, Valid
from vdaf_poc.xof import Xof

F = TypeVar("F", bound=NttField)
X = TypeVar("X", bound=Xof)

USAGE_DZK_RANGE_CHECK = 11
USAGE_DZK_QC_CHALLENGE = 12
USAGE_DZK_QC_PROOF = 13
USAGE_DZK_QC_QUERY = 14
USAGE_DZK_QC_PROOF_SHARE = 15
USAGE_DZK_JOINT_RAND_PART = 16
USAGE_DZK_QUERY_RAND_PART = 17


def chunk_count(chunk_length: int, length: int) -> int:
    """Return the number of padded chunks of size ``chunk_length``."""
    return (length + chunk_length - 1) // chunk_length


class ParallelSumConstraint(NamedTuple):
    """One family of calls to a ``ParallelSum`` FLP gadget."""

    name: str
    num_items: int
    item_length: int
    subcircuit_arity: int
    subcircuit_degree: int = 2


def flp_interpolation_domain(gadget_calls: int) -> int:
    """Return the root-of-unity domain size for ``gadget_calls``."""
    if gadget_calls < 0:
        raise ValueError("gadget_calls must be non-negative")
    return next_power_of_2(1 + gadget_calls)


def flp_gadget_proof_degree(gadget_degree: int, gadget_calls: int) -> int:
    """Return the gadget polynomial degree emitted in a BBCGGI19 proof."""
    return gadget_degree * (flp_interpolation_domain(gadget_calls) - 1)


def parallel_sum_gadget_calls(
    constraint: ParallelSumConstraint,
    chunk_length: int,
) -> int:
    """Return the actual number of ``ParallelSum`` calls made by eval()."""
    _check_parallel_sum_inputs([constraint], chunk_length)
    return constraint.num_items * chunk_count(chunk_length, constraint.item_length)


def parallel_sum_proof_len(
    constraint: ParallelSumConstraint,
    chunk_length: int,
) -> int:
    """Return the FLP proof contribution for a ``ParallelSum`` gadget."""
    gadget_calls = parallel_sum_gadget_calls(constraint, chunk_length)
    domain = flp_interpolation_domain(gadget_calls)
    return (
        constraint.subcircuit_arity * chunk_length
        + constraint.subcircuit_degree * (domain - 1)
        + 1
    )


def validate_parallel_sum_chunk_length(
    field: type[NttField],
    chunk_length: int,
    constraints: list[ParallelSumConstraint],
    context: str,
) -> None:
    """Reject chunks whose FLP interpolation domain exceeds field support."""
    _check_parallel_sum_inputs(constraints, chunk_length)
    for constraint in constraints:
        gadget_calls = parallel_sum_gadget_calls(constraint, chunk_length)
        domain = flp_interpolation_domain(gadget_calls)
        if field.GEN_ORDER % domain != 0:
            degree = flp_gadget_proof_degree(
                constraint.subcircuit_degree,
                gadget_calls,
            )
            raise ValueError(
                f"{context}: {constraint.name} FLP interpolation domain "
                f"P={domain} from gadget_calls={gadget_calls} and "
                f"chunk_length={chunk_length} does not divide field "
                f"GEN_ORDER={field.GEN_ORDER}; gadget polynomial degree is "
                f"{degree}. Use a larger field, fewer vectors, a smaller "
                f"bound/dimension, or automatic chunk selection."
            )


def select_parallel_sum_chunk_length(
    field: type[NttField],
    constraints: list[ParallelSumConstraint],
    context: str,
) -> int:
    """Choose a proof-size-minimizing safe chunk for shared constraints."""
    _check_parallel_sum_inputs(constraints, 1)
    max_chunk_length = max(constraint.item_length for constraint in constraints)
    best: tuple[tuple[int, int, int], int] | None = None

    for chunk_length in _parallel_sum_chunk_candidates(
        max_chunk_length,
        [constraint.item_length for constraint in constraints],
    ):
        try:
            validate_parallel_sum_chunk_length(
                field,
                chunk_length,
                constraints,
                context,
            )
        except ValueError:
            continue

        proof_len = sum(
            parallel_sum_proof_len(constraint, chunk_length)
            for constraint in constraints
        )
        domain_sum = sum(
            flp_interpolation_domain(
                parallel_sum_gadget_calls(constraint, chunk_length)
            )
            for constraint in constraints
        )
        score = (proof_len, domain_sum, chunk_length)
        if best is None or score < best[0]:
            best = (score, chunk_length)

    if best is None:
        minimum_parts = []
        for constraint in constraints:
            minimum_calls = constraint.num_items
            minimum_domain = flp_interpolation_domain(minimum_calls)
            minimum_degree = flp_gadget_proof_degree(
                constraint.subcircuit_degree,
                minimum_calls,
            )
            minimum_parts.append(
                f"{constraint.name}: min_domain={minimum_domain}, "
                f"min_gadget_calls={minimum_calls}, "
                f"min_gadget_degree={minimum_degree}"
            )
        raise ValueError(
            f"{context}: no valid chunk_length in [1, {max_chunk_length}] "
            f"fits field GEN_ORDER={field.GEN_ORDER} ({'; '.join(minimum_parts)}). "
            f"Use a larger field or fewer vectors."
        )

    return best[1]


def _parallel_sum_chunk_candidates(
    max_chunk_length: int,
    item_lengths: list[int],
) -> list[int]:
    candidates = {1, max_chunk_length}
    for item_length in item_lengths:
        chunk_length = 1
        while chunk_length <= max_chunk_length:
            chunks = chunk_count(chunk_length, item_length)
            candidates.add(chunk_length)
            if chunks == 1:
                break
            end = min(max_chunk_length, (item_length - 1) // (chunks - 1))
            if end + 1 <= max_chunk_length:
                candidates.add(end + 1)
            chunk_length = end + 1
    return sorted(candidates)


def _check_parallel_sum_inputs(
    constraints: list[ParallelSumConstraint],
    chunk_length: int,
) -> None:
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    if not constraints:
        raise ValueError("at least one ParallelSum constraint is required")
    for constraint in constraints:
        if constraint.num_items < 1:
            raise ValueError(f"{constraint.name}: num_items must be positive")
        if constraint.item_length < 1:
            raise ValueError(f"{constraint.name}: item_length must be positive")
        if constraint.subcircuit_arity < 1:
            raise ValueError(f"{constraint.name}: subcircuit_arity must be positive")
        if constraint.subcircuit_degree < 1:
            raise ValueError(f"{constraint.name}: subcircuit_degree must be positive")


def bits(value: int, num_bits: int) -> list[int]:
    """Little-endian bit decomposition of ``value`` with exactly ``num_bits`` bits."""
    if value < 0 or value >= 2**num_bits:
        raise ValueError(f"Value {value} out of range [0, {2**num_bits - 1}]")
    return [(value >> j) & 1 for j in range(num_bits)]


def pack_field_elements(elements: list[int], field_bytes: int) -> bytes:
    """Pack integer field-element shares into fixed-width little-endian bytes."""
    out = bytearray()
    for elem in elements:
        out.extend(int(elem).to_bytes(field_bytes, "little"))
    return bytes(out)


def unpack_field_elements(data: bytes, count: int, field_bytes: int) -> list[int]:
    """Unpack fixed-width little-endian field-element shares."""
    elements = []
    for i in range(count):
        start = i * field_bytes
        end = start + field_bytes
        if end <= len(data):
            elements.append(int.from_bytes(data[start:end], "little"))
        else:
            elements.append(0)
    return elements


def encode_transcript_int(value: int) -> bytes:
    """Encode a signed integer with an explicit sign and length."""
    value = int(value)
    sign = b"\x01" if value < 0 else b"\x00"
    magnitude = abs(value)
    byte_len = max(1, (magnitude.bit_length() + 7) // 8)
    return sign + byte_len.to_bytes(4, "big") + magnitude.to_bytes(byte_len, "big")


def encode_transcript_bytes(data: bytes) -> bytes:
    """Length-prefix a transcript byte string."""
    return len(data).to_bytes(8, "big") + data


def encode_qc_first_messages(field: type[F], first_messages: list[list[F]]) -> bytes:
    """Encode QC proof first messages for Fiat-Shamir binding."""
    parts = [encode_transcript_int(len(first_messages))]
    for message in first_messages:
        parts.append(encode_transcript_bytes(field.encode_vec(message)))
    return concat(parts)


def encode_qc_proofs(field: type[F], proofs: list[list[F]]) -> bytes:
    """Encode full QC proof transcripts for query-randomness binding."""
    parts = [encode_transcript_int(len(proofs))]
    for proof in proofs:
        parts.append(encode_transcript_bytes(field.encode_vec(proof)))
    return concat(parts)


def derive_dzk_transcript_seed(
    xof: type[X],
    dst: bytes,
    label: bytes,
    nonce: bytes,
    params: list[int],
    transcript_parts: list[bytes],
) -> bytes:
    """Derive a seed from a structured DZK transcript."""
    encoded_parts = [
        encode_transcript_bytes(label),
        encode_transcript_bytes(dst),
        encode_transcript_bytes(nonce),
        encode_transcript_int(len(params)),
    ]
    encoded_parts.extend(encode_transcript_int(param) for param in params)
    encoded_parts.append(encode_transcript_int(len(transcript_parts)))
    encoded_parts.extend(encode_transcript_bytes(part) for part in transcript_parts)
    return xof.derive_seed(zeros(xof.SEED_SIZE), dst, concat(encoded_parts))


def expand_dzk_joint_rands(
    xof: type[X],
    field: type[F],
    dst: bytes,
    seed: bytes,
    repetitions: int,
    joint_rand_len: int,
) -> list[F]:
    """Expand a transcript-derived seed into per-repetition joint randomness."""
    return xof.expand_into_vec(
        field,
        seed,
        dst,
        b"joint-rand" + encode_transcript_int(repetitions),
        repetitions * joint_rand_len,
    )


def expand_dzk_query_rands(
    xof: type[X],
    field: type[F],
    dst: bytes,
    seed: bytes,
    repetitions: int,
    query_rand_len: int,
) -> list[F]:
    """Expand a transcript-derived seed into per-repetition query randomness."""
    return xof.expand_into_vec(
        field,
        seed,
        dst,
        b"query-rand" + encode_transcript_int(repetitions),
        repetitions * query_rand_len,
    )


def encode_dzk_part_material(agg_id: int, nonce: bytes, material_parts: list[bytes]) -> bytes:
    """Encode one Aggregator's committed transcript material."""
    encoded_parts = [
        encode_transcript_int(agg_id),
        encode_transcript_bytes(nonce),
        encode_transcript_int(len(material_parts)),
    ]
    encoded_parts.extend(encode_transcript_bytes(part) for part in material_parts)
    return concat(encoded_parts)


def flatten_qc_proofs(proofs: list[list[F]]) -> list[F]:
    """Flatten per-repetition QC proofs into one field vector."""
    flat: list[F] = []
    for proof in proofs:
        flat.extend(proof)
    return flat


def unflatten_qc_proofs(
    flat: list[F],
    repetitions: int,
    proof_len: int,
) -> list[list[F]]:
    """Split a flat field vector into per-repetition QC proofs."""
    expected_len = repetitions * proof_len
    if len(flat) != expected_len:
        raise ValueError(
            f"incorrect flattened proof length: got {len(flat)}, "
            f"expected {expected_len}"
        )
    proofs = []
    for rep in range(repetitions):
        start = rep * proof_len
        proofs.append(flat[start:start + proof_len])
    return proofs


class QuadraticConstraintsProof:
    """Parallel FlpBBCGGI19 proofs for the same quadratic constraints."""

    def __init__(self, field: type[NttField], valid: Valid, repetitions: int):
        self.field = field
        self.valid = valid
        self.repetitions = repetitions
        self.flp = FlpBBCGGI19(valid)

    def prove(
        self,
        encoded_meas: list[F],
        prove_rand: list[F],
        joint_rand: list[F],
    ) -> list[list[F]]:
        """Create one QC proof for each repetition."""
        proofs = []
        prove_rand_len = self.flp.PROVE_RAND_LEN
        for rep in range(self.repetitions):
            rep_rand = prove_rand[rep * prove_rand_len:(rep + 1) * prove_rand_len]
            proofs.append(
                self.flp.prove(
                    encoded_meas,
                    rep_rand,
                    self._joint_rand_for_rep(joint_rand, rep),
                )
            )
        return proofs

    def query(
        self,
        encoded_meas: list[F],
        proofs: list[list[F]],
        query_rand: list[F],
        joint_rand: list[F],
        num_shares: int,
    ) -> list[list[F]]:
        """Query each repeated proof and return verifier-message shares."""
        verifiers = []
        query_rand_len = self.flp.QUERY_RAND_LEN
        for rep in range(self.repetitions):
            rep_rand = query_rand[rep * query_rand_len:(rep + 1) * query_rand_len]
            verifiers.append(
                self.flp.query(
                    encoded_meas,
                    proofs[rep],
                    rep_rand,
                    self._joint_rand_for_rep(joint_rand, rep),
                    num_shares,
                )
            )
        return verifiers

    def decide(self, verifiers: list[list[F]]) -> bool:
        """Return true only if every repeated proof verifies."""
        for verifier in verifiers:
            if not self.flp.decide(verifier):
                return False
        return True

    def _joint_rand_for_rep(self, joint_rand: list[F], rep: int) -> list[F]:
        joint_rand_len = self.flp.JOINT_RAND_LEN
        if len(joint_rand) == joint_rand_len:
            return joint_rand
        expected_len = self.repetitions * joint_rand_len
        if len(joint_rand) != expected_len:
            raise ValueError(
                f"incorrect joint randomness length: got {len(joint_rand)}, "
                f"expected {expected_len}"
            )
        start = rep * joint_rand_len
        return joint_rand[start:start + joint_rand_len]

    def proof_first_messages_from_prove_rand(self, prove_rand: list[F]) -> list[list[F]]:
        """Return each repetition's FLP first message from prover randomness."""
        prove_rand_len = self.flp.PROVE_RAND_LEN
        if len(prove_rand) != self.repetitions * prove_rand_len:
            raise ValueError("incorrect prover randomness length")

        messages = []
        for rep in range(self.repetitions):
            messages.append(list(prove_rand[rep * prove_rand_len:(rep + 1) * prove_rand_len]))
        return messages

    def proof_first_messages_from_proofs(self, proofs: list[list[F]]) -> list[list[F]]:
        """Extract each repetition's FLP first message from proof transcripts."""
        if len(proofs) != self.repetitions:
            raise ValueError("incorrect number of QC proof repetitions")

        messages = []
        for proof in proofs:
            if len(proof) != self.flp.PROOF_LEN:
                raise ValueError("incorrect QC proof length")
            offset = 0
            message = []
            for gadget, gadget_calls in zip(
                self.flp.valid.GADGETS,
                self.flp.valid.GADGET_CALLS,
            ):
                P = next_power_of_2(1 + gadget_calls)
                gadget_poly_len = gadget.DEGREE * (P - 1) + 1
                message.extend(proof[offset:offset + gadget.ARITY])
                offset += gadget.ARITY + gadget_poly_len
            messages.append(message)
        return messages


__all__ = [
    "USAGE_DZK_RANGE_CHECK",
    "USAGE_DZK_QC_CHALLENGE",
    "USAGE_DZK_QC_PROOF",
    "USAGE_DZK_QC_QUERY",
    "USAGE_DZK_QC_PROOF_SHARE",
    "USAGE_DZK_JOINT_RAND_PART",
    "USAGE_DZK_QUERY_RAND_PART",
    "ParallelSumConstraint",
    "QuadraticConstraintsProof",
    "bits",
    "chunk_count",
    "derive_dzk_transcript_seed",
    "encode_dzk_part_material",
    "encode_qc_first_messages",
    "encode_qc_proofs",
    "encode_transcript_bytes",
    "encode_transcript_int",
    "expand_dzk_joint_rands",
    "expand_dzk_query_rands",
    "flatten_qc_proofs",
    "flp_gadget_proof_degree",
    "flp_interpolation_domain",
    "pack_field_elements",
    "parallel_sum_gadget_calls",
    "parallel_sum_proof_len",
    "select_parallel_sum_chunk_length",
    "unflatten_qc_proofs",
    "unpack_field_elements",
    "validate_parallel_sum_chunk_length",
    "vec_add",
    "vec_sub",
    "zeros",
]
