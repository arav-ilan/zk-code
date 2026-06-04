import inspect
import os

import numpy as np

from batch_dzk import BatchDZK, BatchDZK128
from common import (
    USAGE_DZK_JOINT_RAND_PART,
    USAGE_DZK_QC_CHALLENGE,
    USAGE_DZK_QC_PROOF,
    USAGE_DZK_QUERY_RAND_PART,
    pack_field_elements,
)
from fields import Field32
from vdaf_poc.xof import XofTurboShake128


def _recompute_batch_public_share(vdaf, input_shares, nonce):
    joint_parts = []
    query_parts = []
    for agg_id, input_share in enumerate(input_shares):
        (
            meas_share,
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
                    meas_share,
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
                    meas_share,
                    proof_share,
                    v_bits_packed,
                    u_bits_packed,
                ),
                USAGE_DZK_QUERY_RAND_PART,
            )
        )
    return joint_parts, query_parts


def test_honest_distributed_accepts():
    np.random.seed(1)
    vdaf = BatchDZK128(
        num_vectors=2,
        l2_norm_bound=32,
        num_frac_bits=0,
        dimension=2,
        chunk_length=2,
        chunk_length_norm_equality=2,
        qc_repetitions=1,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_batched_dzk(
        [[3.0, 4.0], [5.0, 12.0]],
        nonce,
        os.urandom(64),
    )

    assert len(public_share) == 2
    assert [len(share) for share in input_shares] == [6, 6]
    assert vdaf.verify_batched_dzk(public_share, input_shares, nonce)


def test_auto_chunk_handles_bits12_m8_regression():
    np.random.seed(13)
    vdaf = BatchDZK128(
        num_vectors=8,
        l2_norm_bound=2**12,
        num_frac_bits=0,
        dimension=2,
        qc_repetitions=1,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_batched_dzk(
        [[1.0, 2.0] for _ in range(8)],
        nonce,
        os.urandom(64),
    )

    assert vdaf.verify_batched_dzk(public_share, input_shares, nonce)


def test_auto_chunk_reports_field_capacity_limit():
    try:
        BatchDZK(
            field=Field32,
            xof=XofTurboShake128,
            num_vectors=Field32.GEN_ORDER,
            l2_norm_bound=2,
            num_frac_bits=0,
            dimension=1,
            qc_repetitions=1,
        )
        assert False, "unsupported FLP domain must fail during construction"
    except ValueError as exc:
        message = str(exc)

    assert "no valid chunk_length" in message
    assert "field GEN_ORDER" in message


def test_out_of_bound_tampered_vector_rejects():
    np.random.seed(2)
    vdaf = BatchDZK128(
        num_vectors=1,
        l2_norm_bound=32,
        num_frac_bits=0,
        dimension=1,
        chunk_length=1,
        chunk_length_norm_equality=1,
        qc_repetitions=1,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_batched_dzk([[31.0]], nonce, os.urandom(64))

    tampered = list(input_shares)
    leader = list(tampered[0])
    leader_share = list(leader[0])
    leader_share[0] += vdaf.flp.field(2)
    leader[0] = leader_share
    tampered[0] = tuple(leader)

    reconstructed = tampered[0][0][0] + tampered[1][0][0]
    assert reconstructed.as_unsigned() == 33
    tampered_public = _recompute_batch_public_share(vdaf, tampered, nonce)
    assert not vdaf.verify_batched_dzk(tampered_public, tampered, nonce)


def test_honest_shard_rejects_out_of_bound_measurement():
    vdaf = BatchDZK128(
        num_vectors=1,
        l2_norm_bound=32,
        num_frac_bits=0,
        dimension=1,
        chunk_length=1,
        chunk_length_norm_equality=1,
        qc_repetitions=1,
    )
    try:
        vdaf.shard_batched_dzk([[33.0]], os.urandom(16), os.urandom(64))
        assert False, "out-of-bound honest sharding must fail closed"
    except ValueError:
        pass


def _sqrt_mod_prime(n: int, p: int) -> int | None:
    n %= p
    if n == 0:
        return 0
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    q = p - 1
    s = 0
    while q % 2 == 0:
        s += 1
        q //= 2
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    while t != 1:
        i = 1
        t2i = pow(t, 2, p)
        while t2i != 1:
            i += 1
            t2i = pow(t2i, 2, p)
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * c) % p
        r = (r * b) % p
    return r


def _solve_attack_bits(vdaf, joint_rand, computed_sq_norm):
    q = vdaf.flp.field.MODULUS
    sq_bound = vdaf.batched_dzk_valid.sq_norm_bound
    num_bits = vdaf.batched_dzk_valid.num_bits
    r_bit = joint_rand[0].as_unsigned()
    r_final = joint_rand[1].as_unsigned()
    a = pow(r_bit, num_bits, q)

    A = (1 + a) % q
    B = (-1 + a * (1 - 2 * sq_bound) - r_final) % q
    C = (a * (sq_bound * sq_bound - sq_bound) + r_final * computed_sq_norm) % q
    disc = (B * B - 4 * A * C) % q
    sqrt_disc = _sqrt_mod_prime(disc, q)
    if sqrt_disc is None:
        return None
    inv = pow(2 * A, -1, q)
    root = ((-B + sqrt_disc) * inv) % q
    return root, (sq_bound - root) % q


def test_grinding_attack_legacy_shape_rejected():
    nonce = b"BATCH-DZK-ATTACK"
    vdaf = BatchDZK(
        field=Field32,
        xof=XofTurboShake128,
        num_vectors=1,
        l2_norm_bound=32,
        num_frac_bits=0,
        dimension=1,
        chunk_length=1,
        chunk_length_norm_equality=1,
        num_shares=2,
        epsilon=1.0,
        delta=1e-6,
        qc_repetitions=3,
    )
    out_of_bound = 33
    computed_sq_norm = out_of_bound * out_of_bound

    while True:
        client_rand = os.urandom(2 * XofTurboShake128.SEED_SIZE)
        joint_rand_seed = XofTurboShake128.derive_seed(
            client_rand[XofTurboShake128.SEED_SIZE:2 * XofTurboShake128.SEED_SIZE],
            vdaf.domain_separation_tag(USAGE_DZK_QC_CHALLENGE),
            nonce,
        )
        joint_rand_xof = XofTurboShake128(
            joint_rand_seed,
            vdaf.domain_separation_tag(USAGE_DZK_QC_CHALLENGE),
            nonce,
        )
        joint_rand = joint_rand_xof.next_vec(
            Field32,
            vdaf.qc_repetitions * vdaf.batched_dzk_valid.JOINT_RAND_LEN,
        )
        solved = _solve_attack_bits(vdaf, joint_rand[:3], computed_sq_norm)
        if solved is not None:
            v0, u0 = solved
            break

    num_bits = vdaf.batched_dzk_valid.num_bits
    v_bits = [Field32(v0)] + [Field32(0) for _ in range(num_bits - 1)]
    u_bits = [Field32(u0)] + [Field32(0) for _ in range(num_bits - 1)]
    encoded_meas = [Field32(out_of_bound)] + v_bits + u_bits
    assert vdaf.batched_dzk_valid.eval(encoded_meas, joint_rand[:3], 1)[0] == Field32(0)

    prove_rand_seed = XofTurboShake128.derive_seed(
        client_rand[:XofTurboShake128.SEED_SIZE],
        vdaf.domain_separation_tag(USAGE_DZK_QC_PROOF),
        nonce,
    )
    prove_rand_xof = XofTurboShake128(
        prove_rand_seed,
        vdaf.domain_separation_tag(USAGE_DZK_QC_PROOF),
        nonce,
    )
    prove_rand = prove_rand_xof.next_vec(
        Field32,
        vdaf.qc_repetitions * vdaf.qc_prover.flp.PROVE_RAND_LEN,
    )
    qc_proofs = vdaf.qc_prover.prove(encoded_meas, prove_rand, joint_rand)
    field_bytes = (Field32.MODULUS.bit_length() + 7) // 8
    zero_bits = [0 for _ in range(num_bits)]
    old_public_share = (joint_rand_seed, qc_proofs)
    old_input_shares = [
        (
            [Field32(out_of_bound // 2)],
            pack_field_elements([v0] + zero_bits[1:], field_bytes),
            pack_field_elements([u0] + zero_bits[1:], field_bytes),
        ),
        (
            [Field32(out_of_bound - out_of_bound // 2)],
            pack_field_elements(zero_bits, field_bytes),
            pack_field_elements(zero_bits, field_bytes),
        ),
    ]

    assert not vdaf.verify_batched_dzk(old_public_share, old_input_shares, nonce)


def test_prep_init_share_isolation():
    np.random.seed(3)
    vdaf = BatchDZK128(
        num_vectors=1,
        l2_norm_bound=32,
        num_frac_bits=0,
        dimension=1,
        chunk_length=1,
        chunk_length_norm_equality=1,
        qc_repetitions=1,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_batched_dzk([[7.0]], nonce, os.urandom(64))

    signature = inspect.signature(vdaf.prep_init_batched_dzk)
    assert list(signature.parameters) == ["agg_id", "public_share", "input_share", "nonce"]

    state0, prep0 = vdaf.prep_init_batched_dzk(0, public_share, input_shares[0], nonce)
    try:
        vdaf.prep_shares_to_prep_batched_dzk([prep0])
        assert False, "single prep share must not decide"
    except ValueError:
        pass

    state1, prep1 = vdaf.prep_init_batched_dzk(1, public_share, input_shares[1], nonce)
    prep_msg = vdaf.prep_shares_to_prep_batched_dzk([prep0, prep1])
    assert vdaf.prep_next_batched_dzk(state0, prep_msg)
    assert vdaf.prep_next_batched_dzk(state1, prep_msg)


def test_batch_dzk128_equivalent_runs():
    np.random.seed(4)
    vdaf = BatchDZK128(
        num_vectors=1,
        l2_norm_bound=128,
        num_frac_bits=2,
        dimension=3,
        chunk_length=2,
        chunk_length_norm_equality=2,
        qc_repetitions=2,
    )
    nonce = os.urandom(16)
    public_share, input_shares = vdaf.shard_batched_dzk(
        [[1.25, -2.0, 3.5]],
        nonce,
        os.urandom(64),
    )
    assert vdaf.verify_batched_dzk(public_share, input_shares, nonce)
