"""Batched Differential Zero-Knowledge VDAF prototype.

This module contains the cleaned Batch DZK path refactored from the verified
PINE DZK prototype. It keeps the two-party distributed verifier, transcript
binding, additive proof shares, range-decomposition checks, and local magnitude
cap, without importing the legacy PINE wraparound or single-vector paths.
"""

import math
import time
from typing import Generic, TypeAlias, TypeVar

import numpy as np

from common import (
    USAGE_DZK_JOINT_RAND_PART,
    USAGE_DZK_QC_CHALLENGE,
    USAGE_DZK_QC_PROOF,
    USAGE_DZK_QC_PROOF_SHARE,
    USAGE_DZK_QC_QUERY,
    USAGE_DZK_QUERY_RAND_PART,
    USAGE_DZK_RANGE_CHECK,
    ParallelSumConstraint,
    QuadraticConstraintsProof,
    bits,
    encode_dzk_part_material,
    encode_qc_first_messages,
    encode_qc_proofs,
    expand_dzk_joint_rands,
    expand_dzk_query_rands,
    flatten_qc_proofs,
    pack_field_elements,
    parallel_sum_gadget_calls,
    select_parallel_sum_chunk_length,
    unflatten_qc_proofs,
    unpack_field_elements,
    validate_parallel_sum_chunk_length,
    vec_add,
    vec_sub,
    zeros,
)
from fields import Field64, Field128, NttField
from vdaf_poc.common import concat, to_be_bytes
from vdaf_poc.flp_bbcggi19 import Gadget, Mul, ParallelSum, PolyEval, Valid
from vdaf_poc.xof import Xof, XofTurboShake128

F = TypeVar("F", bound=NttField)
X = TypeVar("X", bound=Xof)

PINE_VERSION = 1


class BatchedDZKValid(Valid[list[list[float]], list[float], F]):
    """Validity circuit for one batched QC proof over ``num_vectors`` vectors."""

    EVAL_OUTPUT_LEN = 1

    def __init__(
        self,
        field: type[F],
        num_vectors: int,
        dimension: int,
        l2_norm_bound: int,
        num_frac_bits: int,
        chunk_length: int | None = None,
        chunk_length_norm_equality: int | None = None,
    ):
        self.field = field
        self.num_vectors = num_vectors
        self.dimension = dimension
        self.l2_norm_bound = l2_norm_bound
        self.num_frac_bits = num_frac_bits

        self.sq_norm_bound = l2_norm_bound**2
        self.num_bits = self.sq_norm_bound.bit_length()
        self.bits_per_vector = 2 * self.num_bits
        self.total_gradient_len = num_vectors * dimension
        self.total_bits_len = num_vectors * self.bits_per_vector
        self.MEAS_LEN = self.total_gradient_len + self.total_bits_len
        self.OUTPUT_LEN = self.total_gradient_len
        self.JOINT_RAND_LEN = 3

        norm_constraint = ParallelSumConstraint(
            "Batch norm equality",
            num_vectors,
            dimension,
            subcircuit_arity=1,
            subcircuit_degree=2,
        )
        bit_constraint = ParallelSumConstraint(
            "Batch bit check",
            num_vectors,
            self.bits_per_vector,
            subcircuit_arity=2,
            subcircuit_degree=2,
        )
        if chunk_length_norm_equality is None:
            chunk_length_norm_equality = select_parallel_sum_chunk_length(
                field,
                [norm_constraint],
                "Batch DZK norm-equality chunk selection",
            )
        else:
            validate_parallel_sum_chunk_length(
                field,
                chunk_length_norm_equality,
                [norm_constraint],
                "Batch DZK norm-equality chunk validation",
            )
        if chunk_length is None:
            chunk_length = select_parallel_sum_chunk_length(
                field,
                [bit_constraint],
                "Batch DZK bit-check chunk selection",
            )
        else:
            validate_parallel_sum_chunk_length(
                field,
                chunk_length,
                [bit_constraint],
                "Batch DZK bit-check chunk validation",
            )

        self.chunk_length = chunk_length
        self.chunk_length_norm_equality = chunk_length_norm_equality

        self.GADGETS: list[Gadget[F]] = [
            ParallelSum(PolyEval([0, 0, 1]), chunk_length_norm_equality),
            ParallelSum(Mul(), chunk_length),
        ]
        self.GADGET_CALLS = [
            parallel_sum_gadget_calls(norm_constraint, chunk_length_norm_equality),
            parallel_sum_gadget_calls(bit_constraint, chunk_length),
        ]

    def eval(self, meas: list[F], joint_rand: list[F], num_shares: int) -> list[F]:
        """Evaluate all vector checks under one random linear combination."""
        self.check_valid_eval(meas, joint_rand)

        shares_inv = self.field(num_shares).inv()
        r_bit_check = joint_rand[0]
        r_final = joint_rand[1]
        rho = joint_rand[2]

        all_gradients = meas[:self.total_gradient_len]
        all_bits = meas[self.total_gradient_len:]

        result = self.field(0)
        rho_power = self.field(1)
        for vec_idx in range(self.num_vectors):
            grad_start = vec_idx * self.dimension
            gradient = all_gradients[grad_start:grad_start + self.dimension]

            bits_start = vec_idx * self.bits_per_vector
            v_bits = all_bits[bits_start:bits_start + self.num_bits]
            u_bits = all_bits[bits_start + self.num_bits:bits_start + self.bits_per_vector]

            computed_sq_norm = self._eval_squared_sum_single(gradient)
            claimed_sq_norm = self.field.decode_from_bit_vector(v_bits)
            norm_eq_check = computed_sq_norm - claimed_sq_norm

            bit_check_result = self._eval_bit_checks_single(
                v_bits + u_bits,
                r_bit_check,
                shares_inv,
            )

            decoded_v = self.field.decode_from_bit_vector(v_bits)
            decoded_u = self.field.decode_from_bit_vector(u_bits)
            range_check = decoded_v + decoded_u - self.field(self.sq_norm_bound) * shares_inv

            vector_check = (
                bit_check_result
                + r_final * norm_eq_check
                + r_final**2 * range_check
            )
            result += rho_power * vector_check
            rho_power *= rho

        return [result]

    def _eval_squared_sum_single(self, gradient: list[F]) -> F:
        gadget = self.GADGETS[0]
        result = self.field(0)
        for i in range(0, len(gradient), self.chunk_length_norm_equality):
            chunk = list(gradient[i:i + self.chunk_length_norm_equality])
            while len(chunk) < self.chunk_length_norm_equality:
                chunk.append(self.field(0))
            result += gadget.eval(self.field, chunk)
        return result

    def _eval_bit_checks_single(
        self,
        bit_values: list[F],
        r_bit_check: F,
        shares_inv: F,
    ) -> F:
        gadget = self.GADGETS[1]
        r_power = self.field(1)
        result = self.field(0)

        for i in range(0, len(bit_values), self.chunk_length):
            inputs: list[F] = []
            for j in range(self.chunk_length):
                idx = i + j
                if idx < len(bit_values):
                    inputs.append(r_power * bit_values[idx])
                    inputs.append(bit_values[idx] - shares_inv)
                    r_power *= r_bit_check
                else:
                    inputs.append(self.field(0))
                    inputs.append(self.field(0))
            result += gadget.eval(self.field, inputs)
        return result

    def encode(self, measurement: list[list[float]]) -> list[F]:
        raise NotImplementedError("Use shard_batched_dzk for batched DZK encoding")

    def truncate(self, meas: list[F]) -> list[F]:
        return meas[:self.total_gradient_len]

    def decode(self, output: list[F], num_measurements: int) -> list[float]:
        decoded = []
        for x in output:
            raw = x.as_unsigned()
            if raw > self.field.MODULUS // 2:
                raw = -(self.field.MODULUS - raw)
            decoded.append(raw / (2**self.num_frac_bits))
        return decoded

    def encode_float_into_field(self, x: float) -> F:
        return self.field(round(x * (2**self.num_frac_bits)))


class DZKParameters:
    """Gaussian and local-cap parameters in encoded, squared-bound space."""

    def __init__(
        self,
        dimension: int,
        norm_bound: int,
        field_modulus: int,
        epsilon: float,
        delta: float,
        num_frac_bits: int,
        qc_repetitions: int = 3,
    ):
        self.dimension = dimension
        self.norm_bound = norm_bound
        self.field_modulus = field_modulus
        self.epsilon = epsilon
        self.delta = delta
        self.num_frac_bits = num_frac_bits
        self.qc_repetitions = qc_repetitions
        self.encoded_norm_bound = norm_bound

        self.c_epsilon_delta = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        self.c_epsilon_delta_half = math.sqrt(2 * math.log(1.25 / (delta / 2))) / epsilon
        self.sigma = self.c_epsilon_delta_half * math.sqrt(self.encoded_norm_bound)

        d = dimension
        sigma_sq = self.sigma**2
        self.Delta = d * sigma_sq * (
            1
            + 2 * math.sqrt(math.log(8 * math.e / delta) / d)
            + 2 * math.log(8 * math.e / delta) / d
        )
        self.Lambda = math.sqrt(self.encoded_norm_bound) + math.sqrt(self.Delta) + math.sqrt(d)

        min_field_size = 4 * self.Lambda**2
        if field_modulus <= min_field_size:
            raise ValueError(
                f"Field size {field_modulus} is too small. "
                f"Requires q > {min_field_size:.0f}"
            )


def sample_truncated_gaussian(dimension: int, sigma: float, Delta: float) -> np.ndarray:
    """Sample from N(0, sigma^2 I) conditioned on ||R||_2^2 <= Delta."""
    max_attempts = 1000
    sqrt_delta = math.sqrt(Delta)
    for _ in range(max_attempts):
        sample = np.random.normal(0, sigma, dimension)
        if np.linalg.norm(sample) <= sqrt_delta:
            return sample

    sample = np.random.normal(0, sigma, dimension)
    norm = np.linalg.norm(sample)
    if norm > sqrt_delta:
        sample = sample * (sqrt_delta / norm)
    return sample


BatchDZKPublicShare: TypeAlias = tuple[
    list[bytes],
    list[bytes],
]

BatchDZKInputShare: TypeAlias = tuple[
    list[F],
    list[list[F]],
    bytes,
    bytes,
    bytes,
    bytes,
]

BatchDZKPrepState: TypeAlias = tuple[
    list[F],
    bytes,
    bytes,
]

BatchDZKPrepShare: TypeAlias = tuple[
    list[list[F]],
    list[F],
    bytes,
    bytes,
]


class BatchDZK(Generic[F, X]):
    """Batch DZK VDAF with genuine two-party distributed verification."""

    test_vec_name = "BatchDZK"

    def __init__(
        self,
        field: type[F],
        xof: type[X],
        num_vectors: int,
        l2_norm_bound: int,
        num_frac_bits: int,
        dimension: int,
        chunk_length: int | None = None,
        chunk_length_norm_equality: int | None = None,
        num_shares: int = 2,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        qc_repetitions: int = 3,
        vdaf_id: int = 0xFFFF3000,
    ):
        if num_shares != 2:
            raise NotImplementedError("Batch DZK supports exactly two aggregators")

        self.xof = xof
        self.ID = vdaf_id
        self.SHARES = num_shares
        self.num_vectors = num_vectors
        self.qc_repetitions = qc_repetitions

        squared_norm_bound = l2_norm_bound**2
        self.dzk_params = DZKParameters(
            dimension=dimension,
            norm_bound=squared_norm_bound,
            field_modulus=field.MODULUS,
            epsilon=epsilon,
            delta=delta,
            num_frac_bits=num_frac_bits,
            qc_repetitions=qc_repetitions,
        )
        self.batched_dzk_valid = BatchedDZKValid(
            field=field,
            num_vectors=num_vectors,
            dimension=dimension,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
        )
        self.qc_prover = QuadraticConstraintsProof(
            field,
            self.batched_dzk_valid,
            qc_repetitions,
        )
        self.flp = self.qc_prover.flp
        self.last_shard_timings: dict[str, float] = {}

    def domain_separation_tag(self, usage: int) -> bytes:
        """Match the PINE v1 domain-separation format used by the reference."""
        return concat([
            to_be_bytes(PINE_VERSION, 1),
            to_be_bytes(self.ID, 4),
            to_be_bytes(usage, 2),
        ])

    def _transcript_params(self) -> list[int]:
        return [
            self.ID,
            self.SHARES,
            self.qc_repetitions,
            self.num_vectors,
            self.batched_dzk_valid.dimension,
            self.batched_dzk_valid.l2_norm_bound,
            self.batched_dzk_valid.num_frac_bits,
            self.batched_dzk_valid.chunk_length,
            self.batched_dzk_valid.chunk_length_norm_equality,
            self.batched_dzk_valid.MEAS_LEN,
            self.batched_dzk_valid.JOINT_RAND_LEN,
            self.qc_prover.flp.PROVE_RAND_LEN,
            self.qc_prover.flp.PROOF_LEN,
        ]

    def _seed_from_parts(self, parts: list[bytes], usage: int) -> bytes:
        if len(parts) != self.SHARES:
            raise ValueError("incorrect number of DZK transcript parts")
        return self.xof.derive_seed(
            zeros(self.xof.SEED_SIZE),
            self.domain_separation_tag(usage),
            concat(parts),
        )

    def _joint_rands(self, joint_rand_seed: bytes) -> list[F]:
        return expand_dzk_joint_rands(
            self.xof,
            self.flp.field,
            self.domain_separation_tag(USAGE_DZK_QC_CHALLENGE),
            joint_rand_seed,
            self.qc_repetitions,
            self.batched_dzk_valid.JOINT_RAND_LEN,
        )

    def _part(
        self,
        agg_id: int,
        blind: bytes,
        nonce: bytes,
        material_parts: list[bytes],
        usage: int,
    ) -> bytes:
        if len(blind) != self.xof.SEED_SIZE:
            raise ValueError("incorrect DZK transcript blind size")
        return self.xof.derive_seed(
            blind,
            self.domain_separation_tag(usage),
            encode_dzk_part_material(agg_id, nonce, material_parts),
        )

    def _replace_part_and_seed(
        self,
        agg_id: int,
        public_parts: list[bytes],
        corrected_part: bytes,
        seed_usage: int,
    ) -> bytes:
        if len(public_parts) != self.SHARES:
            raise ValueError("incorrect number of public DZK transcript parts")
        if agg_id < 0 or agg_id >= self.SHARES:
            raise ValueError("invalid aggregator id")
        corrected_parts = list(public_parts)
        corrected_parts[agg_id] = corrected_part
        return self._seed_from_parts(corrected_parts, seed_usage)

    def _proof_share_rands(self, rand: bytes, nonce: bytes) -> list[list[F]]:
        proof_share_seed = self.xof.derive_seed(
            rand[:self.xof.SEED_SIZE],
            self.domain_separation_tag(USAGE_DZK_QC_PROOF_SHARE),
            b"BatchDZK proof-share" + nonce,
        )
        proof_share_xof = self.xof(
            proof_share_seed,
            self.domain_separation_tag(USAGE_DZK_QC_PROOF_SHARE),
            b"BatchDZK proof-share" + nonce,
        )
        flat = proof_share_xof.next_vec(
            self.flp.field,
            self.qc_repetitions * self.qc_prover.flp.PROOF_LEN,
        )
        return unflatten_qc_proofs(
            flat,
            self.qc_repetitions,
            self.qc_prover.flp.PROOF_LEN,
        )

    def _blinds(self, rand: bytes, nonce: bytes) -> tuple[list[bytes], list[bytes]]:
        blind_seed = self.xof.derive_seed(
            rand[:self.xof.SEED_SIZE],
            self.domain_separation_tag(USAGE_DZK_JOINT_RAND_PART),
            b"BatchDZK blinds" + nonce,
        )
        blind_xof = self.xof(
            blind_seed,
            self.domain_separation_tag(USAGE_DZK_JOINT_RAND_PART),
            b"BatchDZK blinds" + nonce,
        )
        joint_blinds = [blind_xof.next(self.xof.SEED_SIZE) for _ in range(self.SHARES)]
        query_blinds = [blind_xof.next(self.xof.SEED_SIZE) for _ in range(self.SHARES)]
        return joint_blinds, query_blinds

    def _joint_part_material(
        self,
        meas_share: list[F],
        proof_first_messages_share: list[list[F]],
        v_bits_packed: bytes,
        u_bits_packed: bytes,
    ) -> list[bytes]:
        field = self.flp.field
        return [
            field.encode_vec(meas_share),
            encode_qc_first_messages(field, proof_first_messages_share),
            v_bits_packed,
            u_bits_packed,
        ]

    def _query_part_material(
        self,
        meas_share: list[F],
        proof_share: list[list[F]],
        v_bits_packed: bytes,
        u_bits_packed: bytes,
    ) -> list[bytes]:
        field = self.flp.field
        return [
            field.encode_vec(meas_share),
            encode_qc_proofs(field, proof_share),
            v_bits_packed,
            u_bits_packed,
        ]

    def _encoded_measurement_share(self, input_share: BatchDZKInputShare) -> list[F]:
        field = self.flp.field
        meas_share, _proof_share, v_bits_packed, u_bits_packed, _joint_blind, _query_blind = input_share
        field_bytes = (field.MODULUS.bit_length() + 7) // 8
        num_bits = self.batched_dzk_valid.num_bits
        total_bits_per_share = self.num_vectors * num_bits
        v_bits_share = unpack_field_elements(v_bits_packed, total_bits_per_share, field_bytes)
        u_bits_share = unpack_field_elements(u_bits_packed, total_bits_per_share, field_bytes)

        encoded_meas_share: list[F] = list(meas_share)
        for vec_idx in range(self.num_vectors):
            start = vec_idx * num_bits
            end = start + num_bits
            encoded_meas_share.extend(field(bit % field.MODULUS) for bit in v_bits_share[start:end])
            encoded_meas_share.extend(field(bit % field.MODULUS) for bit in u_bits_share[start:end])
        return encoded_meas_share

    def _range_check_share(self, input_share: BatchDZKInputShare) -> list[F]:
        field = self.flp.field
        _meas_share, _proof_share, v_bits_packed, u_bits_packed, _joint_blind, _query_blind = input_share
        field_bytes = (field.MODULUS.bit_length() + 7) // 8
        num_bits = self.batched_dzk_valid.num_bits
        total_bits_per_share = self.num_vectors * num_bits
        v_bits_share = unpack_field_elements(v_bits_packed, total_bits_per_share, field_bytes)
        u_bits_share = unpack_field_elements(u_bits_packed, total_bits_per_share, field_bytes)
        shares_inv = field(self.SHARES).inv()
        bound_share = field(self.batched_dzk_valid.sq_norm_bound) * shares_inv

        checks = []
        for vec_idx in range(self.num_vectors):
            start = vec_idx * num_bits
            end = start + num_bits
            v_bits = [field(bit % field.MODULUS) for bit in v_bits_share[start:end]]
            u_bits = [field(bit % field.MODULUS) for bit in u_bits_share[start:end]]
            decoded_v = field.decode_from_bit_vector(v_bits)
            decoded_u = field.decode_from_bit_vector(u_bits)
            checks.append(decoded_v + decoded_u - bound_share)
        return checks

    def _is_distributed_public_share(self, public_share) -> bool:
        if not isinstance(public_share, tuple) or len(public_share) != 2:
            return False
        joint_parts, query_parts = public_share
        if not isinstance(joint_parts, list) or not isinstance(query_parts, list):
            return False
        return (
            len(joint_parts) == self.SHARES
            and len(query_parts) == self.SHARES
            and all(isinstance(part, bytes) for part in joint_parts)
            and all(isinstance(part, bytes) for part in query_parts)
        )

    def _is_distributed_input_share(self, input_share) -> bool:
        return (
            isinstance(input_share, tuple)
            and len(input_share) == 6
            and isinstance(input_share[0], list)
            and isinstance(input_share[1], list)
            and isinstance(input_share[2], bytes)
            and isinstance(input_share[3], bytes)
            and isinstance(input_share[4], bytes)
            and isinstance(input_share[5], bytes)
        )

    def shard_batched_dzk(
        self,
        measurements: list[list[float]],
        nonce: bytes,
        rand: bytes,
    ) -> tuple[BatchDZKPublicShare, list[BatchDZKInputShare]]:
        """Shard ``num_vectors`` measurements and build additive proof shares."""
        if len(measurements) != self.num_vectors:
            raise ValueError(f"Expected {self.num_vectors} vectors, got {len(measurements)}")

        field = self.flp.field
        timings: dict[str, float] = {}
        all_encoded_gradients: list[F] = []
        all_s0: list[int] = []
        all_s1: list[int] = []
        all_v_bit_shares: list[tuple[list[int], list[int]]] = []
        all_u_bit_shares: list[tuple[list[int], list[int]]] = []

        encode_start = time.perf_counter()
        for vec_idx, measurement in enumerate(measurements):
            if len(measurement) != self.batched_dzk_valid.dimension:
                raise ValueError(
                    f"Vector {vec_idx} has wrong dimension: "
                    f"expected {self.batched_dzk_valid.dimension}, got {len(measurement)}"
                )

            encoded_gradient = [
                self.batched_dzk_valid.encode_float_into_field(x) for x in measurement
            ]
            all_encoded_gradients.extend(encoded_gradient)

            noise = sample_truncated_gaussian(
                self.batched_dzk_valid.dimension,
                self.dzk_params.sigma,
                self.dzk_params.Delta,
            )
            noise_int = [int(r) for r in np.ceil(noise)]
            meas_int = [int(x.as_unsigned()) for x in encoded_gradient]
            s0 = [-r for r in noise_int]
            s1 = [x + r for x, r in zip(meas_int, noise_int)]
            all_s0.extend(s0)
            all_s1.extend(s1)

            squared_norm = sum(x**2 for x in meas_int) % field.MODULUS
            v_bit_shares, u_bit_shares = self._create_bit_shares(
                squared_norm,
                rand,
                nonce,
                vec_idx,
            )
            all_v_bit_shares.append((v_bit_shares[0], v_bit_shares[1]))
            all_u_bit_shares.append((u_bit_shares[0], u_bit_shares[1]))
        timings["encode_and_dp_ms"] = (time.perf_counter() - encode_start) * 1000

        encoding_start = time.perf_counter()
        encoded_meas: list[F] = list(all_encoded_gradients)
        for vec_idx in range(self.num_vectors):
            v_bits_full = self._reconstruct_bits_pair(all_v_bit_shares[vec_idx])
            u_bits_full = self._reconstruct_bits_pair(all_u_bit_shares[vec_idx])
            encoded_meas.extend(field(bit) for bit in v_bits_full)
            encoded_meas.extend(field(bit) for bit in u_bits_full)
        timings["encoding_ms"] = (time.perf_counter() - encoding_start) * 1000

        input_share_start = time.perf_counter()
        field_bytes = (field.MODULUS.bit_length() + 7) // 8
        v_bits_share0: list[int] = []
        v_bits_share1: list[int] = []
        u_bits_share0: list[int] = []
        u_bits_share1: list[int] = []
        for vec_idx in range(self.num_vectors):
            v_bits_share0.extend(all_v_bit_shares[vec_idx][0])
            v_bits_share1.extend(all_v_bit_shares[vec_idx][1])
            u_bits_share0.extend(all_u_bit_shares[vec_idx][0])
            u_bits_share1.extend(all_u_bit_shares[vec_idx][1])

        leader_share = [field(int(x) % field.MODULUS) for x in all_s1]
        leader_v_bits_packed = pack_field_elements(v_bits_share0, field_bytes)
        leader_u_bits_packed = pack_field_elements(u_bits_share0, field_bytes)
        helper_share = [field(int(x) % field.MODULUS) for x in all_s0]
        helper_v_bits_packed = pack_field_elements(v_bits_share1, field_bytes)
        helper_u_bits_packed = pack_field_elements(u_bits_share1, field_bytes)
        timings["input_shares_ms"] = (time.perf_counter() - input_share_start) * 1000

        rand_start = time.perf_counter()
        prove_rand_len = self.qc_prover.flp.PROVE_RAND_LEN
        total_prove_rand_len = self.qc_repetitions * prove_rand_len
        prove_rand_seed = self.xof.derive_seed(
            rand[:self.xof.SEED_SIZE],
            self.domain_separation_tag(USAGE_DZK_QC_PROOF),
            nonce,
        )
        prove_rand_xof = self.xof(
            prove_rand_seed,
            self.domain_separation_tag(USAGE_DZK_QC_PROOF),
            nonce,
        )
        prove_rand = prove_rand_xof.next_vec(field, total_prove_rand_len)

        full_first_messages = self.qc_prover.proof_first_messages_from_prove_rand(prove_rand)
        helper_proof_share = self._proof_share_rands(rand, nonce)
        helper_first_messages = self.qc_prover.proof_first_messages_from_proofs(
            helper_proof_share
        )
        leader_first_messages = [
            vec_sub(full_message, helper_message)
            for full_message, helper_message in zip(full_first_messages, helper_first_messages)
        ]

        joint_blinds, query_blinds = self._blinds(rand, nonce)
        joint_rand_parts = [
            self._part(
                0,
                joint_blinds[0],
                nonce,
                self._joint_part_material(
                    leader_share,
                    leader_first_messages,
                    leader_v_bits_packed,
                    leader_u_bits_packed,
                ),
                USAGE_DZK_JOINT_RAND_PART,
            ),
            self._part(
                1,
                joint_blinds[1],
                nonce,
                self._joint_part_material(
                    helper_share,
                    helper_first_messages,
                    helper_v_bits_packed,
                    helper_u_bits_packed,
                ),
                USAGE_DZK_JOINT_RAND_PART,
            ),
        ]
        joint_rand_seed = self._seed_from_parts(joint_rand_parts, USAGE_DZK_QC_CHALLENGE)
        joint_rand = self._joint_rands(joint_rand_seed)
        timings["randomness_ms"] = (time.perf_counter() - rand_start) * 1000

        qc_start = time.perf_counter()
        qc_proofs = self.qc_prover.prove(encoded_meas, prove_rand, joint_rand)
        timings["qc_proof_ms"] = (time.perf_counter() - qc_start) * 1000

        proof_share_start = time.perf_counter()
        helper_proof_share_flat = flatten_qc_proofs(helper_proof_share)
        leader_proof_share_flat = vec_sub(
            flatten_qc_proofs(qc_proofs),
            helper_proof_share_flat,
        )
        leader_proof_share = unflatten_qc_proofs(
            leader_proof_share_flat,
            self.qc_repetitions,
            self.qc_prover.flp.PROOF_LEN,
        )

        input_shares: list[BatchDZKInputShare] = [
            (
                leader_share,
                leader_proof_share,
                leader_v_bits_packed,
                leader_u_bits_packed,
                joint_blinds[0],
                query_blinds[0],
            ),
            (
                helper_share,
                helper_proof_share,
                helper_v_bits_packed,
                helper_u_bits_packed,
                joint_blinds[1],
                query_blinds[1],
            ),
        ]
        query_rand_parts = [
            self._part(
                0,
                query_blinds[0],
                nonce,
                self._query_part_material(
                    leader_share,
                    leader_proof_share,
                    leader_v_bits_packed,
                    leader_u_bits_packed,
                ),
                USAGE_DZK_QUERY_RAND_PART,
            ),
            self._part(
                1,
                query_blinds[1],
                nonce,
                self._query_part_material(
                    helper_share,
                    helper_proof_share,
                    helper_v_bits_packed,
                    helper_u_bits_packed,
                ),
                USAGE_DZK_QUERY_RAND_PART,
            ),
        ]
        timings["proof_sharing_ms"] = (time.perf_counter() - proof_share_start) * 1000
        self.last_shard_timings = timings
        return (joint_rand_parts, query_rand_parts), input_shares

    def _create_bit_shares(
        self,
        value: int,
        rand: bytes,
        nonce: bytes,
        vec_idx: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        field = self.flp.field
        sq_bound = self.batched_dzk_valid.sq_norm_bound
        num_bits = self.batched_dzk_valid.num_bits
        V = value
        U = sq_bound - value
        if V < 0 or U < 0:
            raise ValueError(f"Value {value} out of range [0, {sq_bound}]")

        v_bits = bits(V, num_bits)
        u_bits = bits(U, num_bits)
        vec_nonce = nonce + vec_idx.to_bytes(4, "little")
        share_rand_seed = self.xof.derive_seed(
            rand[:self.xof.SEED_SIZE],
            self.domain_separation_tag(USAGE_DZK_RANGE_CHECK),
            vec_nonce,
        )
        share_rand_xof = self.xof(
            share_rand_seed,
            self.domain_separation_tag(USAGE_DZK_RANGE_CHECK),
            vec_nonce,
        )

        v_bit_shares: list[list[int]] = [[], []]
        u_bit_shares: list[list[int]] = [[], []]
        for v_bit in v_bits:
            mask = share_rand_xof.next_vec(field, 1)[0]
            mask_int = int(mask.as_unsigned())
            v_bit_shares[0].append(mask_int)
            v_bit_shares[1].append((v_bit - mask_int) % field.MODULUS)
        for u_bit in u_bits:
            mask = share_rand_xof.next_vec(field, 1)[0]
            mask_int = int(mask.as_unsigned())
            u_bit_shares[0].append(mask_int)
            u_bit_shares[1].append((u_bit - mask_int) % field.MODULUS)
        return v_bit_shares, u_bit_shares

    def _reconstruct_bits_pair(self, bit_shares: tuple[list[int], list[int]]) -> list[int]:
        q = self.flp.field.MODULUS
        share0, share1 = bit_shares
        return [(s0 + s1) % q for s0, s1 in zip(share0, share1)]

    def verify_norm_cap_batched(self, shares: list[F], verifier_id: int) -> bool:
        """Locally check every per-vector measurement share against Lambda."""
        field = self.flp.field
        half_mod = field.MODULUS // 2
        dim = self.batched_dzk_valid.dimension

        for vec_idx in range(self.num_vectors):
            start = vec_idx * dim
            vector_share = shares[start:start + dim]
            share_int = []
            for x in vector_share:
                raw = x.as_unsigned()
                share_int.append(raw - field.MODULUS if raw > half_mod else raw)
            if math.sqrt(sum(x**2 for x in share_int)) > self.dzk_params.Lambda:
                return False
        return True

    def prep_init_batched_dzk(
        self,
        agg_id: int,
        public_share: BatchDZKPublicShare,
        input_share: BatchDZKInputShare,
        nonce: bytes,
    ) -> tuple[BatchDZKPrepState[F], BatchDZKPrepShare[F]]:
        """Compute one Aggregator's verifier share from its own input share."""
        if not self._is_distributed_public_share(public_share):
            raise ValueError("expected distributed batched DZK public share")
        if not self._is_distributed_input_share(input_share):
            raise ValueError("expected distributed batched DZK input share")

        public_joint_parts, public_query_parts = public_share
        meas_share, proof_share, v_bits_packed, u_bits_packed, joint_blind, query_blind = input_share

        if len(meas_share) != self.batched_dzk_valid.total_gradient_len:
            raise ValueError("incorrect batched DZK measurement share length")
        if len(proof_share) != self.qc_repetitions:
            raise ValueError("incorrect number of proof-share repetitions")
        for proof in proof_share:
            if len(proof) != self.qc_prover.flp.PROOF_LEN:
                raise ValueError("incorrect proof-share length")
        if not self.verify_norm_cap_batched(meas_share, agg_id):
            raise ValueError("batched DZK norm-cap check failed")

        first_messages_share = self.qc_prover.proof_first_messages_from_proofs(proof_share)
        joint_part = self._part(
            agg_id,
            joint_blind,
            nonce,
            self._joint_part_material(
                meas_share,
                first_messages_share,
                v_bits_packed,
                u_bits_packed,
            ),
            USAGE_DZK_JOINT_RAND_PART,
        )
        corrected_joint_seed = self._replace_part_and_seed(
            agg_id,
            public_joint_parts,
            joint_part,
            USAGE_DZK_QC_CHALLENGE,
        )
        joint_rand = self._joint_rands(corrected_joint_seed)

        query_part = self._part(
            agg_id,
            query_blind,
            nonce,
            self._query_part_material(
                meas_share,
                proof_share,
                v_bits_packed,
                u_bits_packed,
            ),
            USAGE_DZK_QUERY_RAND_PART,
        )
        corrected_query_seed = self._replace_part_and_seed(
            agg_id,
            public_query_parts,
            query_part,
            USAGE_DZK_QC_QUERY,
        )
        query_rand = expand_dzk_query_rands(
            self.xof,
            self.flp.field,
            self.domain_separation_tag(USAGE_DZK_QC_QUERY),
            corrected_query_seed,
            self.qc_repetitions,
            self.qc_prover.flp.QUERY_RAND_LEN,
        )

        encoded_meas_share = self._encoded_measurement_share(input_share)
        verifier_shares = self.qc_prover.query(
            encoded_meas_share,
            proof_share,
            query_rand,
            joint_rand,
            self.SHARES,
        )
        range_check_share = self._range_check_share(input_share)

        return (
            (
                self.batched_dzk_valid.truncate(encoded_meas_share),
                corrected_joint_seed,
                corrected_query_seed,
            ),
            (
                verifier_shares,
                range_check_share,
                joint_part,
                query_part,
            ),
        )

    def prep_shares_to_prep_batched_dzk(
        self,
        prep_shares: list[BatchDZKPrepShare[F]],
    ) -> tuple[bytes, bytes]:
        """Combine verifier shares and fail closed on any invalid check."""
        if len(prep_shares) != self.SHARES:
            raise ValueError("incorrect number of batched DZK prep shares")

        combined_verifiers = [
            self.flp.field.zeros(self.qc_prover.flp.VERIFIER_LEN)
            for _ in range(self.qc_repetitions)
        ]
        combined_range_checks = self.flp.field.zeros(self.num_vectors)
        joint_parts = []
        query_parts = []

        for verifier_shares, range_check_share, joint_part, query_part in prep_shares:
            if len(verifier_shares) != self.qc_repetitions:
                raise ValueError("incorrect verifier-share repetition count")
            if len(range_check_share) != self.num_vectors:
                raise ValueError("incorrect range-check share length")
            for rep in range(self.qc_repetitions):
                combined_verifiers[rep] = vec_add(
                    combined_verifiers[rep],
                    verifier_shares[rep],
                )
            combined_range_checks = vec_add(combined_range_checks, range_check_share)
            joint_parts.append(joint_part)
            query_parts.append(query_part)

        for range_check in combined_range_checks:
            if range_check != self.flp.field(0):
                raise ValueError("batched DZK range decomposition failed")
        if not self.qc_prover.decide(combined_verifiers):
            raise ValueError("batched DZK proof verifier check failed")

        joint_seed = self._seed_from_parts(joint_parts, USAGE_DZK_QC_CHALLENGE)
        query_seed = self._seed_from_parts(query_parts, USAGE_DZK_QC_QUERY)
        return joint_seed, query_seed

    def prep_next_batched_dzk(
        self,
        prep_state: BatchDZKPrepState[F],
        prep_msg: tuple[bytes, bytes],
    ) -> list[F]:
        """Check transcript consistency and return this Aggregator's output share."""
        out_share, corrected_joint_seed, corrected_query_seed = prep_state
        joint_seed, query_seed = prep_msg
        if joint_seed != corrected_joint_seed or query_seed != corrected_query_seed:
            raise ValueError("batched DZK transcript consistency failed")
        return out_share

    def verify_batched_dzk(
        self,
        public_share: BatchDZKPublicShare,
        input_shares: list[BatchDZKInputShare],
        nonce: bytes,
    ) -> bool:
        """Verify via isolated two-Aggregator preparation calls."""
        if not self._is_distributed_public_share(public_share):
            return False
        if len(input_shares) != self.SHARES:
            return False
        if not all(self._is_distributed_input_share(share) for share in input_shares):
            return False

        try:
            prep_states: list[BatchDZKPrepState[F]] = []
            prep_shares: list[BatchDZKPrepShare[F]] = []
            for agg_id, input_share in enumerate(input_shares):
                prep_state, prep_share = self.prep_init_batched_dzk(
                    agg_id,
                    public_share,
                    input_share,
                    nonce,
                )
                prep_states.append(prep_state)
                prep_shares.append(prep_share)

            prep_msg = self.prep_shares_to_prep_batched_dzk(prep_shares)
            for prep_state in prep_states:
                self.prep_next_batched_dzk(prep_state, prep_msg)
        except (AssertionError, IndexError, TypeError, ValueError):
            return False
        return True


class BatchDZK128(BatchDZK[Field128, XofTurboShake128]):
    """Batch DZK instantiated with Field128 and TurboSHAKE128."""

    test_vec_name = "BatchDZK128"

    def __init__(
        self,
        num_vectors: int,
        l2_norm_bound: int,
        num_frac_bits: int,
        dimension: int,
        chunk_length: int | None = None,
        chunk_length_norm_equality: int | None = None,
        num_shares: int = 2,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        qc_repetitions: int = 3,
    ):
        super().__init__(
            Field128,
            XofTurboShake128,
            num_vectors=num_vectors,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            num_shares=num_shares,
            epsilon=epsilon,
            delta=delta,
            qc_repetitions=qc_repetitions,
        )


class BatchDZK64(BatchDZK[Field64, XofTurboShake128]):
    """Batch DZK instantiated with Field64 and TurboSHAKE128."""

    test_vec_name = "BatchDZK64"

    def __init__(
        self,
        num_vectors: int,
        l2_norm_bound: int,
        num_frac_bits: int,
        dimension: int,
        chunk_length: int | None = None,
        chunk_length_norm_equality: int | None = None,
        num_shares: int = 2,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        qc_repetitions: int = 3,
    ):
        super().__init__(
            Field64,
            XofTurboShake128,
            num_vectors=num_vectors,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            num_shares=num_shares,
            epsilon=epsilon,
            delta=delta,
            qc_repetitions=qc_repetitions,
        )


__all__ = [
    "BatchDZK",
    "BatchDZK64",
    "BatchDZK128",
    "BatchedDZKValid",
    "BatchDZKPublicShare",
    "BatchDZKInputShare",
    "BatchDZKPrepState",
    "BatchDZKPrepShare",
    "DZKParameters",
    "sample_truncated_gaussian",
]
