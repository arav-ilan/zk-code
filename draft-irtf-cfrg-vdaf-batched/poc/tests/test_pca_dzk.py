import inspect
import os

import numpy as np

from common import USAGE_DZK_JOINT_RAND_PART, USAGE_DZK_QUERY_RAND_PART
from pca_dzk import (
    PCADZK128,
    AnisotropicGaussianParameters,
    compute_pca_basis,
    encode_float_vectors,
    project_to_coefficients,
    sample_anisotropic_truncated_gaussian,
)


def make_axis_public_basis(ambient_dim: int, coeff_dim: int, num_frac_bits_u: int) -> np.ndarray:
    U = np.zeros((ambient_dim, coeff_dim), dtype=np.int64)
    scale = 2**num_frac_bits_u
    for i in range(coeff_dim):
        U[i, i] = scale
    return U


def _recompute_pca_public_share(vdaf, input_shares, nonce):
    joint_parts = []
    query_parts = []
    for agg_id, input_share in enumerate(input_shares):
        (
            a_share,
            S_share,
            proof_share,
            v_bits_packed,
            u_bits_packed,
            joint_blind,
            query_blind,
        ) = input_share
        first = vdaf.qc_prover.proof_first_messages_from_proofs(proof_share)
        joint_parts.append(
            vdaf._part(
                agg_id,
                joint_blind,
                nonce,
                vdaf._joint_part_material(
                    a_share,
                    S_share,
                    first,
                    v_bits_packed,
                    u_bits_packed,
                ),
                USAGE_DZK_JOINT_RAND_PART,
            )
        )
        query_parts.append(
            vdaf._part(
                agg_id,
                query_blind,
                nonce,
                vdaf._query_part_material(
                    a_share,
                    S_share,
                    proof_share,
                    v_bits_packed,
                    u_bits_packed,
                ),
                USAGE_DZK_QUERY_RAND_PART,
            )
        )
    return joint_parts, query_parts


def test_encoding_and_pca_helpers():
    vectors = [[0.5, -0.25, 0.125], [1.0, -1.0, 0.0]]
    encoded = encode_float_vectors(vectors, 10)
    assert encoded.tolist() == [[512, -256, 128], [1024, -1024, 0]]

    subspace_vectors = np.array(
        [[1, 2, 0], [3, 1, 0], [-1, 4, 0], [2, -1, 0]],
        dtype=np.int64,
    )
    U, r = compute_pca_basis(subspace_vectors)
    assert r == 2
    assert U.shape == (3, 2)
    assert np.allclose(U[2, :], 0, atol=1e-10)

    coefficients = project_to_coefficients(subspace_vectors, U)
    reconstructed = coefficients @ U.T
    assert np.allclose(reconstructed[:, 2], 0, atol=1e-6)


def test_anisotropic_parameters_and_sampling():
    G = np.array([[2.0, 0.5], [0.5, 1.0]])
    params = AnisotropicGaussianParameters(G, norm_bound=100.0, epsilon=1.0, delta=1e-6)
    assert np.all(np.linalg.eigvalsh(params.Sigma_a) > 0)
    assert params.Delta > 0
    assert params.Lambda > 0

    np.random.seed(11)
    sqrt_delta = np.sqrt(params.Delta)
    for _ in range(20):
        sample = sample_anisotropic_truncated_gaussian(params)
        assert np.linalg.norm(sample) <= sqrt_delta * 1.001


def test_honest_distributed_verify_public_u():
    np.random.seed(42)
    num_vectors = 3
    ambient_dim = 10
    coeff_dim = 4
    num_frac_bits = 10
    num_frac_bits_u = 5
    norm_bound = 2**20
    public_U = make_axis_public_basis(ambient_dim, coeff_dim, num_frac_bits_u)
    expected_G = public_U.T @ public_U

    measurements = []
    for _ in range(num_vectors):
        v = np.random.randn(ambient_dim)
        v = v / (np.linalg.norm(v) + 1e-10) * 0.9
        measurements.append(v.tolist())

    prover = PCADZK128(
        num_vectors=num_vectors,
        ambient_dim=ambient_dim,
        norm_bound=norm_bound,
        num_frac_bits=num_frac_bits,
        num_frac_bits_u=num_frac_bits_u,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    verifier = PCADZK128(
        num_vectors=num_vectors,
        ambient_dim=ambient_dim,
        norm_bound=norm_bound,
        num_frac_bits=num_frac_bits,
        num_frac_bits_u=num_frac_bits_u,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    assert prover.U_int is public_U
    assert verifier.U_int is public_U
    assert np.array_equal(prover.G, expected_G)

    nonce = os.urandom(16)
    public_share, input_shares = prover.shard_pca_dzk(measurements, nonce, os.urandom(64))
    assert len(public_share) == 2
    assert [len(share) for share in input_shares] == [7, 7]
    assert verifier.verify_pca_dzk(public_share, input_shares, nonce)


def test_coefficients_input_protocol():
    public_U = make_axis_public_basis(4, 2, 0)
    vdaf = PCADZK128(
        num_vectors=2,
        ambient_dim=4,
        norm_bound=20,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_pca_dzk_coefficients(
        np.array([[1, 2], [2, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )
    assert vdaf.verify_pca_dzk(public_share, input_shares, nonce)


def test_magnitude_check_rejects_out_of_cap_share():
    public_U = make_axis_public_basis(3, 2, 0)
    vdaf = PCADZK128(
        num_vectors=1,
        ambient_dim=3,
        norm_bound=10,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )

    bad_input_shares = list(input_shares)
    bad_leader = list(bad_input_shares[0])
    bad_a_shares = list(bad_leader[0])
    too_large = int(np.ceil(vdaf.gaussian_params.Lambda)) + 1
    bad_a_shares[0] = vdaf.field(too_large)
    bad_leader[0] = bad_a_shares
    bad_input_shares[0] = tuple(bad_leader)

    assert not vdaf.verify_magnitude_check(bad_a_shares, 0)
    assert not vdaf.verify_pca_dzk(public_share, bad_input_shares, nonce)


def test_wrong_public_basis_rejects():
    public_U = make_axis_public_basis(3, 2, 0)
    wrong_U = public_U.copy()
    wrong_U[2, 1] = 1

    prover = PCADZK128(
        num_vectors=1,
        ambient_dim=3,
        norm_bound=10,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    wrong_verifier = PCADZK128(
        num_vectors=1,
        ambient_dim=3,
        norm_bound=10,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=wrong_U,
    )
    nonce = os.urandom(16)
    public_share, input_shares = prover.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )
    assert not wrong_verifier.verify_pca_dzk(public_share, input_shares, nonce)


def test_out_of_bound_coefficients_reject_distributed_verify():
    public_U = make_axis_public_basis(3, 2, 0)
    vdaf = PCADZK128(
        num_vectors=1,
        ambient_dim=3,
        norm_bound=10,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )

    tampered = list(input_shares)
    leader = list(tampered[0])
    leader_a = list(leader[0])
    leader_a[0] += vdaf.field(3)
    leader[0] = leader_a
    tampered[0] = tuple(leader)

    reconstructed = [tampered[0][0][i] + tampered[1][0][i] for i in range(2)]
    reconstructed_ints = [
        x.as_unsigned() if x.as_unsigned() < vdaf.field.MODULUS // 2
        else x.as_unsigned() - vdaf.field.MODULUS
        for x in reconstructed
    ]
    qf = int(np.array(reconstructed_ints) @ vdaf.G @ np.array(reconstructed_ints))
    assert qf > vdaf.norm_bound

    tampered_public = _recompute_pca_public_share(vdaf, tampered, nonce)
    assert not vdaf.verify_pca_dzk(tampered_public, tampered, nonce)


def test_prep_init_share_isolation():
    public_U = make_axis_public_basis(3, 2, 0)
    vdaf = PCADZK128(
        num_vectors=1,
        ambient_dim=3,
        norm_bound=10,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )

    signature = inspect.signature(vdaf.prep_init_pca_dzk)
    assert list(signature.parameters) == ["agg_id", "public_share", "input_share", "nonce"]

    state0, prep0 = vdaf.prep_init_pca_dzk(0, public_share, input_shares[0], nonce)
    try:
        vdaf.prep_shares_to_prep_pca_dzk([prep0])
        assert False, "single prep share must not decide"
    except ValueError:
        pass

    state1, prep1 = vdaf.prep_init_pca_dzk(1, public_share, input_shares[1], nonce)
    prep_msg = vdaf.prep_shares_to_prep_pca_dzk([prep0, prep1])
    assert vdaf.prep_next_pca_dzk(state0, prep_msg)
    assert vdaf.prep_next_pca_dzk(state1, prep_msg)


def test_legacy_public_share_shape_rejected():
    public_U = make_axis_public_basis(3, 2, 0)
    vdaf = PCADZK128(
        num_vectors=1,
        ambient_dim=3,
        norm_bound=10,
        num_frac_bits=0,
        num_frac_bits_u=0,
        chunk_length=4,
        qc_repetitions=1,
        public_U=public_U,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )
    old_public_share = (public_share[0][0], input_shares[0][2])
    assert not vdaf.verify_pca_dzk(old_public_share, input_shares, nonce)
