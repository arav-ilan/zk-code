"""PCA Differential Zero-Knowledge VDAF prototype with public setup basis."""

import hashlib
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
    encode_transcript_int,
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
from vdaf_poc.flp_bbcggi19 import Gadget, Mul, ParallelSum, Valid
from vdaf_poc.xof import Xof, XofTurboShake128

F = TypeVar("F", bound=NttField)
X = TypeVar("X", bound=Xof)


def encode_float_vectors(vectors: list[list[float]], num_frac_bits: int) -> np.ndarray:
    """Encode float vectors to fixed-point integers."""
    scale = 2**num_frac_bits
    m = len(vectors)
    d = len(vectors[0]) if m > 0 else 0
    encoded = np.zeros((m, d), dtype=np.int64)
    for i, vec in enumerate(vectors):
        for j, value in enumerate(vec):
            encoded[i, j] = round(value * scale)
    return encoded


def compute_pca_basis(encoded_vectors: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    """Offline helper to compute a PCA basis from public/setup vectors."""
    m, d = encoded_vectors.shape
    mean_vec = np.mean(encoded_vectors, axis=0)
    centered = encoded_vectors - mean_vec
    cov = (centered.T @ centered) / max(m - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    nonzero_mask = eigenvalues > tol
    r = int(np.sum(nonzero_mask))
    if r == 0:
        r = min(m, d)
        return np.eye(d, r), r
    return eigenvectors[:, :r], r


def project_to_coefficients(encoded_vectors: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Project encoded vectors to coefficient space via ``X @ U``."""
    return encoded_vectors @ U


def quantize_pca_basis(U_float: np.ndarray, num_frac_bits: int) -> np.ndarray:
    """Quantize a real-valued PCA basis to integer fixed point."""
    return np.round(U_float * (2**num_frac_bits)).astype(np.int64)


def compute_coefficients_least_squares(X_float: np.ndarray, U_int: np.ndarray) -> np.ndarray:
    """Compute integer-basis coefficients by least squares."""
    a_T, _residuals, _rank, _s = np.linalg.lstsq(
        U_int.astype(np.float64),
        X_float.T,
        rcond=None,
    )
    return a_T.T


def normalize_public_basis(
    public_U: np.ndarray | list[list[int]],
    ambient_dim: int | None = None,
) -> np.ndarray:
    """Validate and normalize the public integer basis matrix ``U``."""
    U_arr = np.asarray(public_U)
    if U_arr.ndim != 2:
        raise ValueError(f"public_U must be a 2D matrix, got shape {U_arr.shape}")

    d, r = U_arr.shape
    if ambient_dim is not None and d != ambient_dim:
        raise ValueError(f"Expected public_U with ambient dimension {ambient_dim}, got {d}")
    if r <= 0:
        raise ValueError("public_U must have at least one coefficient column")

    if not np.issubdtype(U_arr.dtype, np.integer):
        if not np.all(np.isfinite(U_arr)):
            raise ValueError("public_U must contain finite integer values")
        rounded = np.round(U_arr)
        if not np.allclose(U_arr, rounded):
            raise ValueError("public_U must be an integer matrix; quantize it during setup")
        U_arr = rounded.astype(np.int64)
    elif U_arr.dtype != np.int64:
        U_arr = U_arr.astype(np.int64)

    return U_arr


def encode_public_basis_transcript(public_U: np.ndarray) -> bytes:
    """Encode public ``U`` deterministically for setup hashing."""
    U_int = normalize_public_basis(public_U)
    d, r = U_int.shape
    parts = [
        b"PCADZK public U v1",
        encode_transcript_int(d),
        encode_transcript_int(r),
    ]
    for value in U_int.reshape(-1):
        parts.append(encode_transcript_int(int(value)))
    return b"".join(parts)


def public_basis_id(public_U: np.ndarray) -> bytes:
    """Return the setup identifier bound into PCA DZK challenges."""
    return hashlib.sha256(encode_public_basis_transcript(public_U)).digest()


class AnisotropicGaussianParameters:
    """Parameters for anisotropic truncated Gaussian coefficient noise."""

    def __init__(
        self,
        G: np.ndarray,
        norm_bound: float,
        epsilon: float,
        delta: float,
        eta: float = 1e-6,
    ):
        self.G = G
        self.r = G.shape[0]
        self.norm_bound = norm_bound
        self.epsilon = epsilon
        self.delta = delta
        self.eta = eta

        self.c_eps_delta = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        self.sigma = (self.c_eps_delta / 2) * math.sqrt(norm_bound)

        G_reg = G + eta * np.eye(self.r)
        self.G_reg_inv = np.linalg.inv(G_reg)
        self.Sigma_a = (self.sigma**2) * self.G_reg_inv

        t_delta = math.log(8 * math.e / delta)
        trace_sigma = np.trace(self.Sigma_a)
        trace_sigma_sq = np.trace(self.Sigma_a @ self.Sigma_a)
        op_norm_sigma = np.linalg.norm(self.Sigma_a, ord=2)
        self.Delta = (
            trace_sigma
            + 2 * math.sqrt(trace_sigma_sq * t_delta)
            + 2 * op_norm_sigma * t_delta
        )

        try:
            self.L = np.linalg.cholesky(self.Sigma_a)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(self.Sigma_a)
            eigvals = np.maximum(eigvals, 0)
            self.L = eigvecs @ np.diag(np.sqrt(eigvals))

        lambda_min_G = np.min(np.linalg.eigvalsh(G_reg))
        self.A = math.sqrt(norm_bound / lambda_min_G)
        self.Lambda = self.A + math.sqrt(self.Delta) + math.sqrt(self.r)


def sample_anisotropic_truncated_gaussian(
    params: AnisotropicGaussianParameters,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Sample from N(0, Sigma_a), truncated by the configured Euclidean radius."""
    sqrt_delta = math.sqrt(params.Delta)
    for _ in range(max_attempts):
        standard_normal = np.random.standard_normal(params.r)
        sample = params.L @ standard_normal
        if np.linalg.norm(sample) <= sqrt_delta:
            return sample

    standard_normal = np.random.standard_normal(params.r)
    sample = params.L @ standard_normal
    norm = np.linalg.norm(sample)
    if norm > sqrt_delta:
        sample = sample * (sqrt_delta / norm)
    return sample


class PCADZKValid(Valid[list[list[float]], list[float], F]):
    """Validity circuit for ``S = a^T G a`` with range decomposition."""

    EVAL_OUTPUT_LEN = 1

    def __init__(
        self,
        field: type[F],
        num_vectors: int,
        coeff_dim: int,
        G: np.ndarray,
        norm_bound: int,
        num_frac_bits: int,
        chunk_length: int | None = None,
    ):
        self.field = field
        self.num_vectors = num_vectors
        self.coeff_dim = coeff_dim
        self.norm_bound = norm_bound
        self.num_frac_bits = num_frac_bits
        self.G_numpy = G
        self.G = [[field(int(G[i, j])) for j in range(coeff_dim)] for i in range(coeff_dim)]

        self.num_bits = norm_bound.bit_length()
        self.bits_per_vector = 2 * self.num_bits
        self.total_a_len = num_vectors * coeff_dim
        self.total_S_len = num_vectors
        self.total_bits_len = num_vectors * self.bits_per_vector
        self.MEAS_LEN = self.total_a_len + self.total_S_len + self.total_bits_len
        self.OUTPUT_LEN = self.total_a_len
        self.JOINT_RAND_LEN = 4

        quadratic_constraint = ParallelSumConstraint(
            "PCA quadratic form",
            num_vectors,
            coeff_dim,
            subcircuit_arity=2,
            subcircuit_degree=2,
        )
        bit_constraint = ParallelSumConstraint(
            "PCA bit check",
            num_vectors,
            self.bits_per_vector,
            subcircuit_arity=2,
            subcircuit_degree=2,
        )
        constraints = [quadratic_constraint, bit_constraint]
        if chunk_length is None:
            chunk_length = select_parallel_sum_chunk_length(
                field,
                constraints,
                "PCA DZK chunk selection",
            )
        else:
            validate_parallel_sum_chunk_length(
                field,
                chunk_length,
                constraints,
                "PCA DZK chunk validation",
            )

        self.chunk_length = chunk_length

        self.GADGETS: list[Gadget[F]] = [
            ParallelSum(Mul(), chunk_length),
            ParallelSum(Mul(), chunk_length),
        ]
        self.GADGET_CALLS = [
            parallel_sum_gadget_calls(quadratic_constraint, chunk_length),
            parallel_sum_gadget_calls(bit_constraint, chunk_length),
        ]

    def _compute_y_from_a(self, a: list[F]) -> list[F]:
        y = []
        for i in range(self.coeff_dim):
            yi = self.field(0)
            for j in range(self.coeff_dim):
                yi += self.G[i][j] * a[j]
            y.append(yi)
        return y

    def eval(self, meas: list[F], joint_rand: list[F], num_shares: int) -> list[F]:
        """Evaluate the batched PCA validity circuit."""
        self.check_valid_eval(meas, joint_rand)

        shares_inv = self.field(num_shares).inv()
        r_bit_check = joint_rand[0]
        r_final = joint_rand[2]
        rho = joint_rand[3]

        all_a = meas[:self.total_a_len]
        all_S = meas[self.total_a_len:self.total_a_len + self.total_S_len]
        all_bits = meas[self.total_a_len + self.total_S_len:]

        result = self.field(0)
        rho_power = self.field(1)
        for vec_idx in range(self.num_vectors):
            a_start = vec_idx * self.coeff_dim
            a = all_a[a_start:a_start + self.coeff_dim]
            S = all_S[vec_idx]

            bits_start = vec_idx * self.bits_per_vector
            v_bits = all_bits[bits_start:bits_start + self.num_bits]
            u_bits = all_bits[bits_start + self.num_bits:bits_start + self.bits_per_vector]

            y = self._compute_y_from_a(a)
            qf_result = self._eval_quadratic_form(a, y)
            qf_check = S - qf_result

            decoded_v = self.field.decode_from_bit_vector(v_bits)
            range_recon_check = S - decoded_v
            bit_check_result = self._eval_bit_checks(v_bits + u_bits, r_bit_check, shares_inv)
            decoded_u = self.field.decode_from_bit_vector(u_bits)
            range_decomp_check = decoded_v + decoded_u - self.field(self.norm_bound) * shares_inv

            vector_check = (
                bit_check_result
                + r_final * qf_check
                + r_final**2 * range_recon_check
                + r_final**3 * range_decomp_check
            )
            result += rho_power * vector_check
            rho_power *= rho
        return [result]

    def _eval_quadratic_form(self, a: list[F], y: list[F]) -> F:
        gadget = self.GADGETS[0]
        result = self.field(0)
        for i in range(0, len(a), self.chunk_length):
            inputs: list[F] = []
            for j in range(self.chunk_length):
                idx = i + j
                if idx < len(a):
                    inputs.append(a[idx])
                    inputs.append(y[idx])
                else:
                    inputs.append(self.field(0))
                    inputs.append(self.field(0))
            result += gadget.eval(self.field, inputs)
        return result

    def _eval_bit_checks(
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
        raise NotImplementedError("Use shard_pca_dzk for PCA DZK encoding")

    def truncate(self, meas: list[F]) -> list[F]:
        return meas[:self.total_a_len]

    def decode(self, output: list[F], num_measurements: int) -> list[float]:
        decoded = []
        for x in output:
            raw = x.as_unsigned()
            if raw > self.field.MODULUS // 2:
                raw = -(self.field.MODULUS - raw)
            decoded.append(raw / (2**self.num_frac_bits))
        return decoded


PCADZKPublicShare: TypeAlias = tuple[
    list[bytes],
    list[bytes],
]

PCADZKInputShare: TypeAlias = tuple[
    list[F],
    list[F],
    list[list[F]],
    bytes,
    bytes,
    bytes,
    bytes,
]

PCADZKPrepState: TypeAlias = tuple[
    list[F],
    bytes,
    bytes,
]

PCADZKPrepShare: TypeAlias = tuple[
    list[list[F]],
    list[F],
    bytes,
    bytes,
]


class PCADZK(Generic[F, X]):
    """PCA DZK with fixed public integer basis ``U``."""

    test_vec_name = "PCADZK"

    def __init__(
        self,
        field: type[F],
        xof: type[X],
        num_vectors: int,
        ambient_dim: int,
        norm_bound: int,
        num_frac_bits: int,
        num_frac_bits_u: int,
        chunk_length: int | None = None,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        qc_repetitions: int = 3,
        eta: float = 1e-6,
        vdaf_id: int = 0xFFFF4000,
        public_U: np.ndarray | list[list[int]] | None = None,
    ):
        self.field = field
        self.xof = xof
        self.num_vectors = num_vectors
        self.ambient_dim = ambient_dim
        self.norm_bound = norm_bound
        self.num_frac_bits = num_frac_bits
        self.num_frac_bits_u = num_frac_bits_u
        self._requested_chunk_length = chunk_length
        self.chunk_length = chunk_length
        self.epsilon = epsilon
        self.delta = delta
        self.qc_repetitions = qc_repetitions
        self.eta = eta
        self.vdaf_id = vdaf_id
        self.SHARES = 2

        self.coeff_dim = 0
        self.U_int = np.array([], dtype=np.int64)
        self.public_basis_id = b""
        self.G = np.array([])
        self.gaussian_params: AnisotropicGaussianParameters | None = None
        self.pca_valid: PCADZKValid | None = None
        self.qc_prover: QuadraticConstraintsProof | None = None
        self.last_shard_timings: dict[str, float] = {}

        if public_U is not None:
            self.set_public_basis(public_U)

    def domain_separation_tag(self, usage: int) -> bytes:
        """Generate the PCA DZK domain-separation tag."""
        return b"pca_dzk" + self.vdaf_id.to_bytes(4, "big") + usage.to_bytes(2, "big")

    def set_public_basis(self, public_U: np.ndarray | list[list[int]]) -> None:
        """Install public setup basis ``U`` and derive verifier-local state."""
        U_int = normalize_public_basis(public_U, self.ambient_dim)
        self.U_int = U_int
        self.coeff_dim = U_int.shape[1]
        self.public_basis_id = public_basis_id(U_int)
        self.G = U_int.T @ U_int

        self.gaussian_params = AnisotropicGaussianParameters(
            G=self.G.astype(np.float64),
            norm_bound=self.norm_bound,
            epsilon=self.epsilon,
            delta=self.delta,
            eta=self.eta,
        )
        self.pca_valid = PCADZKValid(
            field=self.field,
            num_vectors=self.num_vectors,
            coeff_dim=self.coeff_dim,
            G=self.G,
            norm_bound=self.norm_bound,
            num_frac_bits=self.num_frac_bits,
            chunk_length=self._requested_chunk_length,
        )
        self.chunk_length = self.pca_valid.chunk_length
        self.qc_prover = QuadraticConstraintsProof(
            self.field,
            self.pca_valid,
            self.qc_repetitions,
        )

    def _use_public_basis(self, public_U: np.ndarray | list[list[int]] | None = None) -> None:
        if public_U is not None:
            self.set_public_basis(public_U)
        if self.coeff_dim == 0 or self.public_basis_id == b"":
            raise ValueError("A fixed public_U must be provided at construction or call time")

    def _seed_from_parts(self, parts: list[bytes], usage: int) -> bytes:
        if len(parts) != self.SHARES:
            raise ValueError("incorrect number of PCA DZK transcript parts")
        return self.xof.derive_seed(
            zeros(self.xof.SEED_SIZE),
            self.domain_separation_tag(usage),
            b"".join(parts),
        )

    def _joint_rands(self, joint_rand_seed: bytes) -> list[F]:
        return expand_dzk_joint_rands(
            self.xof,
            self.field,
            self.domain_separation_tag(USAGE_DZK_QC_CHALLENGE),
            joint_rand_seed,
            self.qc_repetitions,
            self.pca_valid.JOINT_RAND_LEN,
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
            raise ValueError("incorrect PCA DZK transcript blind size")
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
            raise ValueError("incorrect number of public PCA DZK transcript parts")
        if agg_id < 0 or agg_id >= self.SHARES:
            raise ValueError("invalid aggregator id")
        corrected_parts = list(public_parts)
        corrected_parts[agg_id] = corrected_part
        return self._seed_from_parts(corrected_parts, seed_usage)

    def _proof_share_rands(self, rand: bytes, nonce: bytes) -> list[list[F]]:
        proof_share_seed = self.xof.derive_seed(
            rand[:self.xof.SEED_SIZE],
            self.domain_separation_tag(USAGE_DZK_QC_PROOF_SHARE),
            b"PCADZK proof-share" + nonce,
        )
        proof_share_xof = self.xof(
            proof_share_seed,
            self.domain_separation_tag(USAGE_DZK_QC_PROOF_SHARE),
            b"PCADZK proof-share" + nonce,
        )
        flat = proof_share_xof.next_vec(
            self.field,
            self.qc_repetitions * self.qc_prover.flp.PROOF_LEN,
        )
        return unflatten_qc_proofs(flat, self.qc_repetitions, self.qc_prover.flp.PROOF_LEN)

    def _blinds(self, rand: bytes, nonce: bytes) -> tuple[list[bytes], list[bytes]]:
        blind_seed = self.xof.derive_seed(
            rand[:self.xof.SEED_SIZE],
            self.domain_separation_tag(USAGE_DZK_JOINT_RAND_PART),
            b"PCADZK blinds" + nonce,
        )
        blind_xof = self.xof(
            blind_seed,
            self.domain_separation_tag(USAGE_DZK_JOINT_RAND_PART),
            b"PCADZK blinds" + nonce,
        )
        joint_blinds = [blind_xof.next(self.xof.SEED_SIZE) for _ in range(self.SHARES)]
        query_blinds = [blind_xof.next(self.xof.SEED_SIZE) for _ in range(self.SHARES)]
        return joint_blinds, query_blinds

    def _joint_part_material(
        self,
        a_share: list[F],
        S_share: list[F],
        proof_first_messages_share: list[list[F]],
        v_bits_packed: bytes,
        u_bits_packed: bytes,
    ) -> list[bytes]:
        return [
            self.public_basis_id,
            self.field.encode_vec(a_share),
            self.field.encode_vec(S_share),
            encode_qc_first_messages(self.field, proof_first_messages_share),
            v_bits_packed,
            u_bits_packed,
        ]

    def _query_part_material(
        self,
        a_share: list[F],
        S_share: list[F],
        proof_share: list[list[F]],
        v_bits_packed: bytes,
        u_bits_packed: bytes,
    ) -> list[bytes]:
        return [
            self.public_basis_id,
            self.field.encode_vec(a_share),
            self.field.encode_vec(S_share),
            encode_qc_proofs(self.field, proof_share),
            v_bits_packed,
            u_bits_packed,
        ]

    def _encoded_measurement_share(self, input_share: PCADZKInputShare) -> list[F]:
        a_share, S_share, _proof_share, v_bits_packed, u_bits_packed, _joint_blind, _query_blind = input_share
        field_bytes = (self.field.MODULUS.bit_length() + 7) // 8
        num_bits = self.pca_valid.num_bits
        total_bits_per_share = self.num_vectors * num_bits
        v_bits_share = unpack_field_elements(v_bits_packed, total_bits_per_share, field_bytes)
        u_bits_share = unpack_field_elements(u_bits_packed, total_bits_per_share, field_bytes)

        encoded_meas_share: list[F] = list(a_share)
        encoded_meas_share.extend(S_share)
        for vec_idx in range(self.num_vectors):
            start = vec_idx * num_bits
            end = start + num_bits
            encoded_meas_share.extend(
                self.field(bit % self.field.MODULUS) for bit in v_bits_share[start:end]
            )
            encoded_meas_share.extend(
                self.field(bit % self.field.MODULUS) for bit in u_bits_share[start:end]
            )
        return encoded_meas_share

    def _range_check_share(self, input_share: PCADZKInputShare) -> list[F]:
        _a_share, _S_share, _proof_share, v_bits_packed, u_bits_packed, _joint_blind, _query_blind = input_share
        field_bytes = (self.field.MODULUS.bit_length() + 7) // 8
        num_bits = self.pca_valid.num_bits
        total_bits_per_share = self.num_vectors * num_bits
        v_bits_share = unpack_field_elements(v_bits_packed, total_bits_per_share, field_bytes)
        u_bits_share = unpack_field_elements(u_bits_packed, total_bits_per_share, field_bytes)
        shares_inv = self.field(self.SHARES).inv()
        bound_share = self.field(self.norm_bound) * shares_inv

        checks = []
        for vec_idx in range(self.num_vectors):
            start = vec_idx * num_bits
            end = start + num_bits
            v_bits = [self.field(bit % self.field.MODULUS) for bit in v_bits_share[start:end]]
            u_bits = [self.field(bit % self.field.MODULUS) for bit in u_bits_share[start:end]]
            decoded_v = self.field.decode_from_bit_vector(v_bits)
            decoded_u = self.field.decode_from_bit_vector(u_bits)
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
            and len(input_share) == 7
            and isinstance(input_share[0], list)
            and isinstance(input_share[1], list)
            and isinstance(input_share[2], list)
            and isinstance(input_share[3], bytes)
            and isinstance(input_share[4], bytes)
            and isinstance(input_share[5], bytes)
            and isinstance(input_share[6], bytes)
        )

    def _check_input_share_lengths(self, input_share: PCADZKInputShare) -> None:
        a_share, S_share, proof_share, _v_bits, _u_bits, _joint_blind, _query_blind = input_share
        if len(a_share) != self.pca_valid.total_a_len:
            raise ValueError("incorrect PCA DZK coefficient share length")
        if len(S_share) != self.pca_valid.total_S_len:
            raise ValueError("incorrect PCA DZK S share length")
        if len(proof_share) != self.qc_repetitions:
            raise ValueError("incorrect number of PCA proof-share repetitions")
        for proof in proof_share:
            if len(proof) != self.qc_prover.flp.PROOF_LEN:
                raise ValueError("incorrect PCA proof-share length")

    def _query_rands_from_seed(self, query_seed: bytes) -> list[F]:
        return expand_dzk_query_rands(
            self.xof,
            self.field,
            self.domain_separation_tag(USAGE_DZK_QC_QUERY),
            query_seed,
            self.qc_repetitions,
            self.qc_prover.flp.QUERY_RAND_LEN,
        )

    def shard_pca_dzk(
        self,
        measurements: list[list[float]],
        nonce: bytes,
        rand: bytes,
        public_U: np.ndarray | list[list[int]] | None = None,
    ) -> tuple[PCADZKPublicShare, list[PCADZKInputShare]]:
        """Project measurements onto the fixed public basis and shard coefficients."""
        self._use_public_basis(public_U)
        if len(measurements) != self.num_vectors:
            raise ValueError(f"Expected {self.num_vectors} vectors, got {len(measurements)}")

        timings: dict[str, float] = {}
        encode_start = time.perf_counter()
        X_int = encode_float_vectors(measurements, self.num_frac_bits)
        if X_int.shape != (self.num_vectors, self.ambient_dim):
            raise ValueError(
                f"Expected measurements with shape {(self.num_vectors, self.ambient_dim)}, "
                f"got {X_int.shape}"
            )
        timings["encode_ms"] = (time.perf_counter() - encode_start) * 1000

        project_start = time.perf_counter()
        a_float = compute_coefficients_least_squares(X_int.astype(np.float64), self.U_int)
        coefficients = np.round(a_float).astype(np.int64)
        timings["project_public_U_ms"] = (time.perf_counter() - project_start) * 1000

        return self.shard_pca_dzk_coefficients(
            coefficients,
            nonce,
            rand,
            public_U=None,
            initial_timings=timings,
        )

    def shard_pca_dzk_coefficients(
        self,
        coefficients: list[list[int]] | np.ndarray,
        nonce: bytes,
        rand: bytes,
        public_U: np.ndarray | list[list[int]] | None = None,
        initial_timings: dict[str, float] | None = None,
    ) -> tuple[PCADZKPublicShare, list[PCADZKInputShare]]:
        """Shard private integer coefficients for the fixed public basis ``U``."""
        self._use_public_basis(public_U)

        coefficients_arr = np.asarray(coefficients)
        if coefficients_arr.shape != (self.num_vectors, self.coeff_dim):
            raise ValueError(
                f"Expected coefficients with shape {(self.num_vectors, self.coeff_dim)}, "
                f"got {coefficients_arr.shape}"
            )
        if not np.issubdtype(coefficients_arr.dtype, np.integer):
            if not np.all(np.isfinite(coefficients_arr)):
                raise ValueError("coefficients must contain finite integer values")
            rounded = np.round(coefficients_arr)
            if not np.allclose(coefficients_arr, rounded):
                raise ValueError("coefficients must be integers")
            coefficients_arr = rounded.astype(np.int64)
        elif coefficients_arr.dtype != np.int64:
            coefficients_arr = coefficients_arr.astype(np.int64)

        timings = dict(initial_timings or {})
        timings["public_U_online_bytes"] = 0
        timings["coeff_dim"] = float(self.coeff_dim)

        dp_start = time.perf_counter()
        all_a0: list[int] = []
        all_a1: list[int] = []
        all_S: list[int] = []
        all_S0: list[int] = []
        all_S1: list[int] = []
        all_v_bit_shares: list[tuple[list[int], list[int]]] = []
        all_u_bit_shares: list[tuple[list[int], list[int]]] = []

        for vec_idx in range(self.num_vectors):
            a = coefficients_arr[vec_idx]
            z_float = sample_anisotropic_truncated_gaussian(self.gaussian_params)
            z = np.ceil(z_float).astype(np.int64)
            all_a0.extend((-z).tolist())
            all_a1.extend((a + z).tolist())

            S = int(a @ self.G @ a)
            all_S.append(S)
            mask_nonce = nonce + vec_idx.to_bytes(4, "little") + b"S"
            mask_seed = self.xof.derive_seed(
                rand[:self.xof.SEED_SIZE],
                self.domain_separation_tag(USAGE_DZK_RANGE_CHECK),
                mask_nonce,
            )
            mask_xof = self.xof(
                mask_seed,
                self.domain_separation_tag(USAGE_DZK_RANGE_CHECK),
                mask_nonce,
            )
            mask_int = int(mask_xof.next_vec(self.field, 1)[0].as_unsigned())
            all_S0.append(mask_int)
            all_S1.append((S - mask_int) % self.field.MODULUS)

            v_shares, u_shares = self._create_bit_shares(S, rand, nonce, vec_idx)
            all_v_bit_shares.append((v_shares[0], v_shares[1]))
            all_u_bit_shares.append((u_shares[0], u_shares[1]))
        timings["dp_shares_ms"] = (time.perf_counter() - dp_start) * 1000

        encoding_start = time.perf_counter()
        encoded_meas: list[F] = []
        for vec_idx in range(self.num_vectors):
            a = coefficients_arr[vec_idx]
            encoded_meas.extend(self.field(int(x) % self.field.MODULUS) for x in a)
        for S in all_S:
            encoded_meas.append(self.field(S % self.field.MODULUS))
        for vec_idx in range(self.num_vectors):
            v_full = self._reconstruct_bits_pair(all_v_bit_shares[vec_idx])
            u_full = self._reconstruct_bits_pair(all_u_bit_shares[vec_idx])
            encoded_meas.extend(self.field(bit) for bit in v_full)
            encoded_meas.extend(self.field(bit) for bit in u_full)
        timings["encoding_ms"] = (time.perf_counter() - encoding_start) * 1000

        field_bytes = (self.field.MODULUS.bit_length() + 7) // 8
        share_start = time.perf_counter()
        v_bits_share0: list[int] = []
        v_bits_share1: list[int] = []
        u_bits_share0: list[int] = []
        u_bits_share1: list[int] = []
        for vec_idx in range(self.num_vectors):
            v_bits_share0.extend(all_v_bit_shares[vec_idx][0])
            v_bits_share1.extend(all_v_bit_shares[vec_idx][1])
            u_bits_share0.extend(all_u_bit_shares[vec_idx][0])
            u_bits_share1.extend(all_u_bit_shares[vec_idx][1])

        v0_a_shares = [self.field(x % self.field.MODULUS) for x in all_a0]
        v0_S_shares = [self.field(x) for x in all_S0]
        v0_v_bits_packed = pack_field_elements(v_bits_share0, field_bytes)
        v0_u_bits_packed = pack_field_elements(u_bits_share0, field_bytes)
        v1_a_shares = [self.field(x % self.field.MODULUS) for x in all_a1]
        v1_S_shares = [self.field(x) for x in all_S1]
        v1_v_bits_packed = pack_field_elements(v_bits_share1, field_bytes)
        v1_u_bits_packed = pack_field_elements(u_bits_share1, field_bytes)
        timings["input_shares_ms"] = (time.perf_counter() - share_start) * 1000

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
        prove_rand = prove_rand_xof.next_vec(self.field, total_prove_rand_len)

        first_messages = self.qc_prover.proof_first_messages_from_prove_rand(prove_rand)
        v1_proof_share = self._proof_share_rands(rand, nonce)
        v1_first_messages = self.qc_prover.proof_first_messages_from_proofs(v1_proof_share)
        v0_first_messages = [
            vec_sub(full_message, helper_message)
            for full_message, helper_message in zip(first_messages, v1_first_messages)
        ]

        joint_blinds, query_blinds = self._blinds(rand, nonce)
        joint_rand_parts = [
            self._part(
                0,
                joint_blinds[0],
                nonce,
                self._joint_part_material(
                    v0_a_shares,
                    v0_S_shares,
                    v0_first_messages,
                    v0_v_bits_packed,
                    v0_u_bits_packed,
                ),
                USAGE_DZK_JOINT_RAND_PART,
            ),
            self._part(
                1,
                joint_blinds[1],
                nonce,
                self._joint_part_material(
                    v1_a_shares,
                    v1_S_shares,
                    v1_first_messages,
                    v1_v_bits_packed,
                    v1_u_bits_packed,
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
        v1_proof_share_flat = flatten_qc_proofs(v1_proof_share)
        v0_proof_share_flat = vec_sub(flatten_qc_proofs(qc_proofs), v1_proof_share_flat)
        v0_proof_share = unflatten_qc_proofs(
            v0_proof_share_flat,
            self.qc_repetitions,
            self.qc_prover.flp.PROOF_LEN,
        )
        input_shares: list[PCADZKInputShare] = [
            (
                v0_a_shares,
                v0_S_shares,
                v0_proof_share,
                v0_v_bits_packed,
                v0_u_bits_packed,
                joint_blinds[0],
                query_blinds[0],
            ),
            (
                v1_a_shares,
                v1_S_shares,
                v1_proof_share,
                v1_v_bits_packed,
                v1_u_bits_packed,
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
                    v0_a_shares,
                    v0_S_shares,
                    v0_proof_share,
                    v0_v_bits_packed,
                    v0_u_bits_packed,
                ),
                USAGE_DZK_QUERY_RAND_PART,
            ),
            self._part(
                1,
                query_blinds[1],
                nonce,
                self._query_part_material(
                    v1_a_shares,
                    v1_S_shares,
                    v1_proof_share,
                    v1_v_bits_packed,
                    v1_u_bits_packed,
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
        num_bits = self.norm_bound.bit_length()
        V = value
        U = self.norm_bound - value
        if V < 0 or U < 0:
            raise ValueError(f"Value {value} out of range [0, {self.norm_bound}]")

        v_bits = bits(V, num_bits)
        u_bits = bits(U, num_bits)
        vec_nonce = nonce + vec_idx.to_bytes(4, "little") + b"bits"
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
            mask = share_rand_xof.next_vec(self.field, 1)[0]
            mask_int = int(mask.as_unsigned())
            v_bit_shares[0].append(mask_int)
            v_bit_shares[1].append((v_bit - mask_int) % self.field.MODULUS)
        for u_bit in u_bits:
            mask = share_rand_xof.next_vec(self.field, 1)[0]
            mask_int = int(mask.as_unsigned())
            u_bit_shares[0].append(mask_int)
            u_bit_shares[1].append((u_bit - mask_int) % self.field.MODULUS)
        return v_bit_shares, u_bit_shares

    def _reconstruct_bits_pair(self, bit_shares: tuple[list[int], list[int]]) -> list[int]:
        share0, share1 = bit_shares
        return [(s0 + s1) % self.field.MODULUS for s0, s1 in zip(share0, share1)]

    def verify_magnitude_check(self, a_shares: list[F], verifier_id: int) -> bool:
        """Locally check coefficient shares against the configured Lambda."""
        if self.gaussian_params is None:
            return False

        half_mod = self.field.MODULUS // 2
        Lambda = self.gaussian_params.Lambda
        for vec_idx in range(self.num_vectors):
            start = vec_idx * self.coeff_dim
            for x in a_shares[start:start + self.coeff_dim]:
                raw = x.as_unsigned()
                val = raw - self.field.MODULUS if raw > half_mod else raw
                if abs(val) > Lambda:
                    return False
        return True

    def prep_init_pca_dzk(
        self,
        agg_id: int,
        public_share: PCADZKPublicShare,
        input_share: PCADZKInputShare,
        nonce: bytes,
    ) -> tuple[PCADZKPrepState[F], PCADZKPrepShare[F]]:
        """Compute one Aggregator's verifier share from its own input share."""
        if self.coeff_dim == 0 or self.public_basis_id == b"":
            raise ValueError("fixed public_U is not configured")
        if not self._is_distributed_public_share(public_share):
            raise ValueError("expected distributed PCA DZK public share")
        if not self._is_distributed_input_share(input_share):
            raise ValueError("expected distributed PCA DZK input share")
        if agg_id < 0 or agg_id >= self.SHARES:
            raise ValueError("invalid aggregator id")

        self._check_input_share_lengths(input_share)
        public_joint_parts, public_query_parts = public_share
        a_share, S_share, proof_share, v_bits_packed, u_bits_packed, joint_blind, query_blind = input_share

        if not self.verify_magnitude_check(a_share, agg_id):
            raise ValueError("PCA DZK magnitude check failed")

        first_messages_share = self.qc_prover.proof_first_messages_from_proofs(proof_share)
        joint_part = self._part(
            agg_id,
            joint_blind,
            nonce,
            self._joint_part_material(
                a_share,
                S_share,
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
                a_share,
                S_share,
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
        query_rand = self._query_rands_from_seed(corrected_query_seed)

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
                self.pca_valid.truncate(encoded_meas_share),
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

    def prep_shares_to_prep_pca_dzk(
        self,
        prep_shares: list[PCADZKPrepShare[F]],
    ) -> tuple[bytes, bytes]:
        """Combine verifier shares and fail closed on any invalid check."""
        if len(prep_shares) != self.SHARES:
            raise ValueError("incorrect number of PCA DZK prep shares")

        combined_verifiers = [
            self.field.zeros(self.qc_prover.flp.VERIFIER_LEN)
            for _ in range(self.qc_repetitions)
        ]
        combined_range_checks = self.field.zeros(self.num_vectors)
        joint_parts = []
        query_parts = []
        for verifier_shares, range_check_share, joint_part, query_part in prep_shares:
            if len(verifier_shares) != self.qc_repetitions:
                raise ValueError("incorrect PCA verifier-share repetition count")
            if len(range_check_share) != self.num_vectors:
                raise ValueError("incorrect PCA range-check share length")
            for rep in range(self.qc_repetitions):
                combined_verifiers[rep] = vec_add(
                    combined_verifiers[rep],
                    verifier_shares[rep],
                )
            combined_range_checks = vec_add(combined_range_checks, range_check_share)
            joint_parts.append(joint_part)
            query_parts.append(query_part)

        for range_check in combined_range_checks:
            if range_check != self.field(0):
                raise ValueError("PCA DZK range decomposition failed")
        if not self.qc_prover.decide(combined_verifiers):
            raise ValueError("PCA DZK proof verifier check failed")

        joint_seed = self._seed_from_parts(joint_parts, USAGE_DZK_QC_CHALLENGE)
        query_seed = self._seed_from_parts(query_parts, USAGE_DZK_QC_QUERY)
        return joint_seed, query_seed

    def prep_next_pca_dzk(
        self,
        prep_state: PCADZKPrepState[F],
        prep_msg: tuple[bytes, bytes],
    ) -> list[F]:
        """Check transcript consistency and return this Aggregator's output share."""
        out_share, corrected_joint_seed, corrected_query_seed = prep_state
        joint_seed, query_seed = prep_msg
        if joint_seed != corrected_joint_seed or query_seed != corrected_query_seed:
            raise ValueError("PCA DZK transcript consistency failed")
        return out_share

    def verify_pca_dzk(
        self,
        public_share: PCADZKPublicShare,
        input_shares: list[PCADZKInputShare],
        nonce: bytes,
        public_U: np.ndarray | list[list[int]] | None = None,
    ) -> bool:
        """Verify PCA DZK via isolated two-Aggregator preparation calls."""
        try:
            self._use_public_basis(public_U)
        except ValueError:
            return False

        if not self._is_distributed_public_share(public_share):
            return False
        if len(input_shares) != self.SHARES:
            return False
        if not all(self._is_distributed_input_share(input_share) for input_share in input_shares):
            return False

        try:
            prep_states: list[PCADZKPrepState[F]] = []
            prep_shares: list[PCADZKPrepShare[F]] = []
            for agg_id, input_share in enumerate(input_shares):
                prep_state, prep_share = self.prep_init_pca_dzk(
                    agg_id,
                    public_share,
                    input_share,
                    nonce,
                )
                prep_states.append(prep_state)
                prep_shares.append(prep_share)

            prep_msg = self.prep_shares_to_prep_pca_dzk(prep_shares)
            for prep_state in prep_states:
                self.prep_next_pca_dzk(prep_state, prep_msg)
        except (AssertionError, IndexError, TypeError, ValueError):
            return False
        return True


class PCADZK128(PCADZK[Field128, XofTurboShake128]):
    """PCA DZK instantiated with Field128 and TurboSHAKE128."""

    test_vec_name = "PCADZK128"

    def __init__(
        self,
        num_vectors: int,
        ambient_dim: int,
        norm_bound: int,
        num_frac_bits: int,
        num_frac_bits_u: int,
        chunk_length: int | None = None,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        qc_repetitions: int = 3,
        public_U: np.ndarray | list[list[int]] | None = None,
        eta: float = 1e-6,
    ):
        super().__init__(
            Field128,
            XofTurboShake128,
            num_vectors=num_vectors,
            ambient_dim=ambient_dim,
            norm_bound=norm_bound,
            num_frac_bits=num_frac_bits,
            num_frac_bits_u=num_frac_bits_u,
            chunk_length=chunk_length,
            epsilon=epsilon,
            delta=delta,
            qc_repetitions=qc_repetitions,
            eta=eta,
            public_U=public_U,
        )


class PCADZK64(PCADZK[Field64, XofTurboShake128]):
    """PCA DZK instantiated with Field64 and TurboSHAKE128."""

    test_vec_name = "PCADZK64"

    def __init__(
        self,
        num_vectors: int,
        ambient_dim: int,
        norm_bound: int,
        num_frac_bits: int,
        num_frac_bits_u: int,
        chunk_length: int | None = None,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        qc_repetitions: int = 3,
        public_U: np.ndarray | list[list[int]] | None = None,
        eta: float = 1e-6,
    ):
        super().__init__(
            Field64,
            XofTurboShake128,
            num_vectors=num_vectors,
            ambient_dim=ambient_dim,
            norm_bound=norm_bound,
            num_frac_bits=num_frac_bits,
            num_frac_bits_u=num_frac_bits_u,
            chunk_length=chunk_length,
            epsilon=epsilon,
            delta=delta,
            qc_repetitions=qc_repetitions,
            eta=eta,
            public_U=public_U,
        )


__all__ = [
    "PCADZK",
    "PCADZK64",
    "PCADZK128",
    "PCADZKValid",
    "PCADZKPublicShare",
    "PCADZKInputShare",
    "PCADZKPrepState",
    "PCADZKPrepShare",
    "AnisotropicGaussianParameters",
    "compute_coefficients_least_squares",
    "compute_pca_basis",
    "encode_float_vectors",
    "encode_public_basis_transcript",
    "normalize_public_basis",
    "project_to_coefficients",
    "public_basis_id",
    "quantize_pca_basis",
    "sample_anisotropic_truncated_gaussian",
]
