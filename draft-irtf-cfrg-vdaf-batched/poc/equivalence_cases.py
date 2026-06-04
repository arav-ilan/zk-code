"""Print accept/reject equivalence cases for the new package or originals."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
POC_DIR = Path(__file__).resolve().parent


def make_axis_public_basis(ambient_dim: int, coeff_dim: int, num_frac_bits_u: int) -> np.ndarray:
    U = np.zeros((ambient_dim, coeff_dim), dtype=np.int64)
    scale = 2**num_frac_bits_u
    for i in range(coeff_dim):
        U[i, i] = scale
    return U


def sqrt_mod_prime(n: int, p: int) -> int | None:
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


def solve_attack_bits(vdaf: Any, joint_rand: list[Any], computed_sq_norm: int):
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
    sqrt_disc = sqrt_mod_prime(disc, q)
    if sqrt_disc is None:
        return None
    inv = pow(2 * A, -1, q)
    root = ((-B + sqrt_disc) * inv) % q
    return root, (sq_bound - root) % q


def run_new() -> None:
    sys.path.insert(0, str(POC_DIR))
    from batch_dzk import BatchDZK, BatchDZK128
    from common import (
        USAGE_DZK_JOINT_RAND_PART,
        USAGE_DZK_QC_CHALLENGE,
        USAGE_DZK_QC_PROOF,
        USAGE_DZK_QUERY_RAND_PART,
        pack_field_elements,
    )
    from fields import Field32
    from pca_dzk import PCADZK128
    from vdaf_poc.xof import XofTurboShake128

    def recompute_batch_public_share(vdaf, input_shares, nonce):
        joint_parts = []
        query_parts = []
        for agg_id, input_share in enumerate(input_shares):
            meas_share, proof_share, v_bits_packed, u_bits_packed, joint_blind, query_blind = input_share
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

    def recompute_pca_public_share(vdaf, input_shares, nonce):
        joint_parts = []
        query_parts = []
        for agg_id, input_share in enumerate(input_shares):
            a_share, S_share, proof_share, v_bits_packed, u_bits_packed, joint_blind, query_blind = input_share
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

    np.random.seed(101)
    nonce = b"N" * 16
    batch = BatchDZK128(1, 32, 0, 1, 1, 1, qc_repetitions=1)
    public_share, input_shares = batch.shard_batched_dzk([[31.0]], nonce, os.urandom(64))
    print("new_batch_honest_accept", batch.verify_batched_dzk(public_share, input_shares, nonce))

    tampered = list(input_shares)
    leader = list(tampered[0])
    leader_share = list(leader[0])
    leader_share[0] += batch.flp.field(2)
    leader[0] = leader_share
    tampered[0] = tuple(leader)
    tampered_public = recompute_batch_public_share(batch, tampered, nonce)
    print("new_batch_out_of_bound_reject", not batch.verify_batched_dzk(tampered_public, tampered, nonce))

    U = make_axis_public_basis(3, 2, 0)
    pca = PCADZK128(1, 3, 10, 0, 0, 4, qc_repetitions=1, public_U=U)
    public_share, input_shares = pca.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )
    print("new_pca_honest_accept", pca.verify_pca_dzk(public_share, input_shares, nonce))

    tampered = list(input_shares)
    leader = list(tampered[0])
    leader_a = list(leader[0])
    leader_a[0] += pca.field(3)
    leader[0] = leader_a
    tampered[0] = tuple(leader)
    tampered_public = recompute_pca_public_share(pca, tampered, nonce)
    print("new_pca_out_of_bound_coeff_reject", not pca.verify_pca_dzk(tampered_public, tampered, nonce))

    attack_nonce = b"BATCH-DZK-ATTACK"
    attack = BatchDZK(Field32, XofTurboShake128, 1, 32, 0, 1, 1, 1, qc_repetitions=3)
    print("new_grinding_attack_reject", _run_grinding_case(
        attack,
        Field32,
        XofTurboShake128,
        USAGE_DZK_QC_CHALLENGE,
        USAGE_DZK_QC_PROOF,
        pack_field_elements,
        attack_nonce,
        repetitions_times_joint=True,
    ))


def run_original() -> None:
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != POC_DIR]
    sys.path.insert(0, str(ROOT / "draft-irtf-cfrg-vdaf-11/poc"))
    sys.path.insert(0, str(ROOT / "draft-chen-cfrg-vdaf-pine/poc"))

    from field import Field32
    from vdaf_pca_dzk import PinePCADZK128
    from vdaf_pine_dzk import (
        PineBatchedDZK,
        PineBatchedDZK128,
        USAGE_DZK_JOINT_RAND_PART,
        USAGE_DZK_QC_CHALLENGE,
        USAGE_DZK_QC_PROOF,
        USAGE_DZK_QUERY_RAND_PART,
        pack_field_elements,
    )
    from vdaf_poc.xof import XofTurboShake128

    def recompute_batch_public_share(vdaf, input_shares, nonce):
        joint_parts = []
        query_parts = []
        for agg_id, input_share in enumerate(input_shares):
            meas_share, proof_share, v_bits_packed, u_bits_packed, joint_blind, query_blind = input_share
            first = vdaf.qc_prover.proof_first_messages_from_proofs(proof_share)
            joint_parts.append(
                vdaf._batched_dzk_part(
                    agg_id,
                    joint_blind,
                    nonce,
                    vdaf._batched_dzk_joint_part_material(
                        meas_share,
                        first,
                        v_bits_packed,
                        u_bits_packed,
                    ),
                    USAGE_DZK_JOINT_RAND_PART,
                )
            )
            query_parts.append(
                vdaf._batched_dzk_part(
                    agg_id,
                    query_blind,
                    nonce,
                    vdaf._batched_dzk_query_part_material(
                        meas_share,
                        proof_share,
                        v_bits_packed,
                        u_bits_packed,
                    ),
                    USAGE_DZK_QUERY_RAND_PART,
                )
            )
        return joint_parts, query_parts

    def recompute_pca_public_share(vdaf, input_shares, nonce):
        joint_parts = []
        query_parts = []
        for agg_id, input_share in enumerate(input_shares):
            a_share, S_share, proof_share, v_bits_packed, u_bits_packed, joint_blind, query_blind = input_share
            first = vdaf.qc_prover.proof_first_messages_from_proofs(proof_share)
            joint_parts.append(
                vdaf._pca_dzk_part(
                    agg_id,
                    joint_blind,
                    nonce,
                    vdaf._pca_dzk_joint_part_material(
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
                vdaf._pca_dzk_part(
                    agg_id,
                    query_blind,
                    nonce,
                    vdaf._pca_dzk_query_part_material(
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

    np.random.seed(101)
    nonce = b"N" * 16
    batch = PineBatchedDZK128(1, 32, 0, 1, 1, 1, 2, qc_repetitions=1)
    public_share, input_shares = batch.shard_batched_dzk([[31.0]], nonce, os.urandom(64))
    print("orig_batch_honest_accept", batch.verify_batched_dzk(public_share, input_shares, nonce))

    tampered = list(input_shares)
    leader = list(tampered[0])
    leader_share = list(leader[0])
    leader_share[0] += batch.flp.field(2)
    leader[0] = leader_share
    tampered[0] = tuple(leader)
    tampered_public = recompute_batch_public_share(batch, tampered, nonce)
    print("orig_batch_out_of_bound_reject", not batch.verify_batched_dzk(tampered_public, tampered, nonce))

    U = make_axis_public_basis(3, 2, 0)
    pca = PinePCADZK128(1, 3, 10, 0, 0, 4, qc_repetitions=1, public_U=U)
    public_share, input_shares = pca.shard_pca_dzk_coefficients(
        np.array([[1, 1]], dtype=np.int64),
        nonce,
        os.urandom(64),
    )
    print("orig_pca_honest_accept", pca.verify_pca_dzk(public_share, input_shares, nonce))

    tampered = list(input_shares)
    leader = list(tampered[0])
    leader_a = list(leader[0])
    leader_a[0] += pca.field(3)
    leader[0] = leader_a
    tampered[0] = tuple(leader)
    tampered_public = recompute_pca_public_share(pca, tampered, nonce)
    print("orig_pca_out_of_bound_coeff_reject", not pca.verify_pca_dzk(tampered_public, tampered, nonce))

    attack_nonce = b"BATCH-DZK-ATTACK"
    attack = PineBatchedDZK(
        field=Field32,
        xof=XofTurboShake128,
        num_vectors=1,
        l2_norm_bound=32,
        num_frac_bits=0,
        dimension=1,
        chunk_length=1,
        chunk_length_norm_equality=1,
        num_shares=2,
        num_proofs=1,
        num_proofs_norm_equality=1,
        epsilon=1.0,
        delta=1e-6,
        qc_repetitions=3,
    )
    print("orig_grinding_attack_reject", _run_grinding_case(
        attack,
        Field32,
        XofTurboShake128,
        USAGE_DZK_QC_CHALLENGE,
        USAGE_DZK_QC_PROOF,
        pack_field_elements,
        attack_nonce,
        repetitions_times_joint=False,
    ))


def _run_grinding_case(
    attack: Any,
    field: Any,
    xof: Any,
    challenge_usage: int,
    proof_usage: int,
    pack_field_elements,
    nonce: bytes,
    repetitions_times_joint: bool,
) -> bool:
    out_of_bound = 33
    computed_sq_norm = out_of_bound * out_of_bound
    while True:
        client_rand = os.urandom(2 * xof.SEED_SIZE)
        joint_rand_seed = xof.derive_seed(
            client_rand[xof.SEED_SIZE:2 * xof.SEED_SIZE],
            attack.domain_separation_tag(challenge_usage),
            nonce,
        )
        joint_rand_xof = xof(
            joint_rand_seed,
            attack.domain_separation_tag(challenge_usage),
            nonce,
        )
        joint_len = attack.batched_dzk_valid.JOINT_RAND_LEN
        if repetitions_times_joint:
            joint_len *= attack.qc_repetitions
        joint_rand = joint_rand_xof.next_vec(field, joint_len)
        solved = solve_attack_bits(
            attack,
            joint_rand[:attack.batched_dzk_valid.JOINT_RAND_LEN],
            computed_sq_norm,
        )
        if solved is not None:
            v0, u0 = solved
            break

    num_bits = attack.batched_dzk_valid.num_bits
    v_bits = [field(v0)] + [field(0) for _ in range(num_bits - 1)]
    u_bits = [field(u0)] + [field(0) for _ in range(num_bits - 1)]
    encoded_meas = [field(out_of_bound)] + v_bits + u_bits
    prove_rand_seed = xof.derive_seed(
        client_rand[:xof.SEED_SIZE],
        attack.domain_separation_tag(proof_usage),
        nonce,
    )
    prove_rand_xof = xof(
        prove_rand_seed,
        attack.domain_separation_tag(proof_usage),
        nonce,
    )
    prove_rand = prove_rand_xof.next_vec(
        field,
        attack.qc_repetitions * attack.qc_prover.flp.PROVE_RAND_LEN,
    )
    qc_proofs = attack.qc_prover.prove(encoded_meas, prove_rand, joint_rand)
    field_bytes = (field.MODULUS.bit_length() + 7) // 8
    zero_bits = [0 for _ in range(num_bits)]
    old_public_share = (joint_rand_seed, qc_proofs)
    old_input_shares = [
        (
            [field(out_of_bound // 2)],
            pack_field_elements([v0] + zero_bits[1:], field_bytes),
            pack_field_elements([u0] + zero_bits[1:], field_bytes),
        ),
        (
            [field(out_of_bound - out_of_bound // 2)],
            pack_field_elements(zero_bits, field_bytes),
            pack_field_elements(zero_bits, field_bytes),
        ),
    ]
    return not attack.verify_batched_dzk(old_public_share, old_input_shares, nonce)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"new", "original"}:
        raise SystemExit("usage: equivalence_cases.py {new|original}")
    if sys.argv[1] == "new":
        run_new()
    else:
        run_original()


if __name__ == "__main__":
    main()
