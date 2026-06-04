"""Timing harness for measured Batch DZK and PCA DZK benchmark trials."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POC_DIR = PACKAGE_ROOT / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

from batch_dzk import BatchDZK  # noqa: E402
from fields import Field64, Field128  # noqa: E402
from pca_dzk import PCADZK, encode_public_basis_transcript  # noqa: E402
from sizes import SizeBreakdown, size_breakdown  # noqa: E402
from vdaf_poc.xof import XofTurboShake128  # noqa: E402


FIELD_CLASSES = {
    "Field64": Field64,
    "Field128": Field128,
}


def run_config(config: dict[str, Any], warmups: int, trials: int, seed: int) -> dict[str, Any]:
    """Run warmups and timed trials for one benchmark configuration."""

    if trials < 1:
        raise ValueError("trials must be >= 1")
    if warmups < 0:
        raise ValueError("warmups must be >= 0")

    config_seed = _config_seed(config, seed)
    for warmup_idx in range(warmups):
        _run_trial(config, trial_index=-(warmup_idx + 1), seed=config_seed + warmup_idx)

    trial_records = []
    for trial_idx in range(trials):
        trial_seed = config_seed + 10_000 + trial_idx
        trial_records.append(
            _run_trial(config, trial_index=trial_idx, seed=trial_seed)
        )

    first = trial_records[0]
    return {
        "config_id": _config_id(config),
        "protocol": config["protocol"],
        "params": first["params"],
        "sizes": first["sizes"],
        "stats": _summarize_trials(trial_records),
        "trials": trial_records,
    }


def compute_size_config(config: dict[str, Any]) -> dict[str, Any]:
    """Compute deterministic byte sizes for one configuration without timing."""

    protocol = config["protocol"]
    if protocol == "batch":
        return _compute_batch_size_config(config)
    if protocol == "pca":
        return _compute_pca_size_config(config)
    raise ValueError(f"unknown protocol {protocol!r}")


def stats(values: list[float]) -> dict[str, float]:
    """Return mean, median, and p95 for a non-empty sample."""

    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(np.asarray(values, dtype=float), 95)),
    }


def _run_trial(config: dict[str, Any], trial_index: int, seed: int) -> dict[str, Any]:
    np.random.seed(seed % (2**32 - 1))
    protocol = config["protocol"]
    if protocol == "batch":
        return _run_batch_trial(config, trial_index, seed)
    if protocol == "pca":
        return _run_pca_trial(config, trial_index, seed)
    raise ValueError(f"unknown protocol {protocol!r}")


def _run_batch_trial(
    config: dict[str, Any],
    trial_index: int,
    seed: int,
) -> dict[str, Any]:
    field = _field_class(config["field"])
    measurements = _make_measurements(
        seed=seed,
        rows=config["m"],
        cols=config["d"],
    )
    nonce = _deterministic_bytes(seed, b"batch nonce", 16)
    rand = _deterministic_bytes(
        seed,
        b"batch rand",
        2 * XofTurboShake128.SEED_SIZE,
    )
    vdaf = BatchDZK(
        field,
        XofTurboShake128,
        num_vectors=config["m"],
        l2_norm_bound=2 ** config["norm_bound_bits"],
        num_frac_bits=config["num_frac_bits"],
        dimension=config["d"],
        chunk_length=config["chunk_length"],
        chunk_length_norm_equality=config["chunk_length_norm_equality"],
        num_shares=2,
        epsilon=config["epsilon"],
        delta=config["delta"],
        qc_repetitions=config["qc_repetitions"],
    )

    prove_start = time.perf_counter()
    public_share, input_shares = vdaf.shard_batched_dzk(measurements, nonce, rand)
    prove_ms = (time.perf_counter() - prove_start) * 1000

    sizes = size_breakdown("batch", public_share, input_shares).to_dict()
    verify = _time_batch_verify(vdaf, public_share, input_shares, nonce)

    return _trial_record(
        config=config,
        field=field,
        vdaf=vdaf,
        protocol="batch",
        trial_index=trial_index,
        seed=seed,
        prove_ms=prove_ms,
        prove_phases_ms=vdaf.last_shard_timings,
        verify=verify,
        sizes=sizes,
    )


def _run_pca_trial(
    config: dict[str, Any],
    trial_index: int,
    seed: int,
) -> dict[str, Any]:
    field = _field_class(config["field"])
    public_u = _make_axis_public_u(
        ambient_dim=config["d"],
        coeff_dim=config["r"],
        num_frac_bits_u=config["num_frac_bits_u"],
    )
    measurements = _make_measurements(
        seed=seed,
        rows=config["m"],
        cols=config["d"],
    )
    nonce = _deterministic_bytes(seed, b"pca nonce", 16)
    rand = _deterministic_bytes(
        seed,
        b"pca rand",
        2 * XofTurboShake128.SEED_SIZE,
    )
    vdaf = PCADZK(
        field,
        XofTurboShake128,
        num_vectors=config["m"],
        ambient_dim=config["d"],
        norm_bound=2 ** config["norm_bound_bits"],
        num_frac_bits=config["num_frac_bits"],
        num_frac_bits_u=config["num_frac_bits_u"],
        chunk_length=config["chunk_length"],
        epsilon=config["epsilon"],
        delta=config["delta"],
        qc_repetitions=config["qc_repetitions"],
        public_U=public_u,
    )

    prove_start = time.perf_counter()
    public_share, input_shares = vdaf.shard_pca_dzk(measurements, nonce, rand)
    prove_ms = (time.perf_counter() - prove_start) * 1000

    sizes = size_breakdown("pca", public_share, input_shares, public_u).to_dict()
    verify = _time_pca_verify(vdaf, public_share, input_shares, nonce)

    return _trial_record(
        config=config,
        field=field,
        vdaf=vdaf,
        protocol="pca",
        trial_index=trial_index,
        seed=seed,
        prove_ms=prove_ms,
        prove_phases_ms=vdaf.last_shard_timings,
        verify=verify,
        sizes=sizes,
        public_u_shape=list(public_u.shape),
    )


def _time_batch_verify(vdaf: BatchDZK, public_share: Any, input_shares: list[Any], nonce: bytes) -> dict[str, Any]:
    prep_states = []
    prep_shares = []
    phases: dict[str, float] = {}

    total_start = time.perf_counter()
    for agg_id, input_share in enumerate(input_shares):
        phase_start = time.perf_counter()
        prep_state, prep_share = vdaf.prep_init_batched_dzk(
            agg_id,
            public_share,
            input_share,
            nonce,
        )
        phases[f"prep_init_agg{agg_id}_ms"] = (time.perf_counter() - phase_start) * 1000
        prep_states.append(prep_state)
        prep_shares.append(prep_share)

    phase_start = time.perf_counter()
    prep_msg = vdaf.prep_shares_to_prep_batched_dzk(prep_shares)
    phases["combine_decide_ms"] = (time.perf_counter() - phase_start) * 1000

    phase_start = time.perf_counter()
    for prep_state in prep_states:
        vdaf.prep_next_batched_dzk(prep_state, prep_msg)
    phases["consistency_ms"] = (time.perf_counter() - phase_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "ok": True,
        "verify_ms": total_ms,
        "phases_ms": phases,
    }


def _compute_batch_size_config(config: dict[str, Any]) -> dict[str, Any]:
    field = _field_class(config["field"])
    vdaf = BatchDZK(
        field,
        XofTurboShake128,
        num_vectors=config["m"],
        l2_norm_bound=2 ** config["norm_bound_bits"],
        num_frac_bits=config["num_frac_bits"],
        dimension=config["d"],
        chunk_length=config["chunk_length"],
        chunk_length_norm_equality=config["chunk_length_norm_equality"],
        num_shares=2,
        epsilon=config["epsilon"],
        delta=config["delta"],
        qc_repetitions=config["qc_repetitions"],
    )
    field_bytes = (field.MODULUS.bit_length() + 7) // 8
    proof_share_bytes = (
        config["qc_repetitions"]
        * vdaf.qc_prover.flp.PROOF_LEN
        * field.ENCODED_SIZE
    )
    input_share_bytes = (
        vdaf.batched_dzk_valid.total_gradient_len * field.ENCODED_SIZE
        + proof_share_bytes
        + 2 * vdaf.num_vectors * vdaf.batched_dzk_valid.num_bits * field_bytes
        + 2 * XofTurboShake128.SEED_SIZE
    )
    public_share_bytes = 2 * vdaf.SHARES * XofTurboShake128.SEED_SIZE
    sizes = _size_breakdown_from_counts(
        input_share_bytes=input_share_bytes,
        public_share_bytes=public_share_bytes,
        setup_public_u_bytes=0,
        proof_share_bytes=proof_share_bytes,
        shares=vdaf.SHARES,
    )
    return {
        "config_id": _config_id(config),
        "protocol": config["protocol"],
        "params": _params_for_result(config, field, vdaf, public_u_shape=None),
        "sizes": sizes,
    }


def _compute_pca_size_config(config: dict[str, Any]) -> dict[str, Any]:
    field = _field_class(config["field"])
    public_u = _make_axis_public_u(
        ambient_dim=config["d"],
        coeff_dim=config["r"],
        num_frac_bits_u=config["num_frac_bits_u"],
    )
    vdaf = PCADZK(
        field,
        XofTurboShake128,
        num_vectors=config["m"],
        ambient_dim=config["d"],
        norm_bound=2 ** config["norm_bound_bits"],
        num_frac_bits=config["num_frac_bits"],
        num_frac_bits_u=config["num_frac_bits_u"],
        chunk_length=config["chunk_length"],
        epsilon=config["epsilon"],
        delta=config["delta"],
        qc_repetitions=config["qc_repetitions"],
        public_U=public_u,
    )
    field_bytes = (field.MODULUS.bit_length() + 7) // 8
    proof_share_bytes = (
        config["qc_repetitions"]
        * vdaf.qc_prover.flp.PROOF_LEN
        * field.ENCODED_SIZE
    )
    input_share_bytes = (
        vdaf.pca_valid.total_a_len * field.ENCODED_SIZE
        + vdaf.pca_valid.total_S_len * field.ENCODED_SIZE
        + proof_share_bytes
        + 2 * vdaf.num_vectors * vdaf.pca_valid.num_bits * field_bytes
        + 2 * XofTurboShake128.SEED_SIZE
    )
    public_share_bytes = 2 * vdaf.SHARES * XofTurboShake128.SEED_SIZE
    sizes = _size_breakdown_from_counts(
        input_share_bytes=input_share_bytes,
        public_share_bytes=public_share_bytes,
        setup_public_u_bytes=len(encode_public_basis_transcript(public_u)),
        proof_share_bytes=proof_share_bytes,
        shares=vdaf.SHARES,
    )
    return {
        "config_id": _config_id(config),
        "protocol": config["protocol"],
        "params": _params_for_result(
            config,
            field,
            vdaf,
            public_u_shape=list(public_u.shape),
        ),
        "sizes": sizes,
    }


def _size_breakdown_from_counts(
    input_share_bytes: int,
    public_share_bytes: int,
    setup_public_u_bytes: int,
    proof_share_bytes: int,
    shares: int,
) -> dict[str, Any]:
    input_share_bytes_by_aggregator = [input_share_bytes for _ in range(shares)]
    proof_share_bytes_by_aggregator = [proof_share_bytes for _ in range(shares)]
    input_shares_total_bytes = sum(input_share_bytes_by_aggregator)
    online_total = input_shares_total_bytes + public_share_bytes
    return SizeBreakdown(
        input_share_bytes_by_aggregator=input_share_bytes_by_aggregator,
        input_shares_total_bytes=input_shares_total_bytes,
        public_share_bytes=public_share_bytes,
        setup_public_u_bytes=setup_public_u_bytes,
        online_total_comm_bytes=online_total,
        total_comm_with_setup_bytes=online_total + setup_public_u_bytes,
        proof_share_bytes_by_aggregator=proof_share_bytes_by_aggregator,
        proof_size_bytes=sum(proof_share_bytes_by_aggregator),
    ).to_dict()


def _time_pca_verify(vdaf: PCADZK, public_share: Any, input_shares: list[Any], nonce: bytes) -> dict[str, Any]:
    prep_states = []
    prep_shares = []
    phases: dict[str, float] = {}

    total_start = time.perf_counter()
    for agg_id, input_share in enumerate(input_shares):
        phase_start = time.perf_counter()
        prep_state, prep_share = vdaf.prep_init_pca_dzk(
            agg_id,
            public_share,
            input_share,
            nonce,
        )
        phases[f"prep_init_agg{agg_id}_ms"] = (time.perf_counter() - phase_start) * 1000
        prep_states.append(prep_state)
        prep_shares.append(prep_share)

    phase_start = time.perf_counter()
    prep_msg = vdaf.prep_shares_to_prep_pca_dzk(prep_shares)
    phases["combine_decide_ms"] = (time.perf_counter() - phase_start) * 1000

    phase_start = time.perf_counter()
    for prep_state in prep_states:
        vdaf.prep_next_pca_dzk(prep_state, prep_msg)
    phases["consistency_ms"] = (time.perf_counter() - phase_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "ok": True,
        "verify_ms": total_ms,
        "phases_ms": phases,
    }


def _trial_record(
    config: dict[str, Any],
    field: Any,
    vdaf: Any,
    protocol: str,
    trial_index: int,
    seed: int,
    prove_ms: float,
    prove_phases_ms: dict[str, float],
    verify: dict[str, Any],
    sizes: dict[str, Any],
    public_u_shape: list[int] | None = None,
) -> dict[str, Any]:
    params = _params_for_result(config, field, vdaf, public_u_shape)
    timing_phases = {
        key: value
        for key, value in prove_phases_ms.items()
        if key.endswith("_ms")
    }
    return {
        "trial": trial_index,
        "seed": int(seed),
        "params": params,
        "sizes": sizes,
        "prove_ms": float(prove_ms),
        "prove_phases_ms": {key: float(value) for key, value in timing_phases.items()},
        "verify_ms": float(verify["verify_ms"]),
        "verify_phases_ms": {
            key: float(value) for key, value in verify["phases_ms"].items()
        },
        "verify_ok": bool(verify["ok"]),
    }


def _params_for_result(
    config: dict[str, Any],
    field: Any,
    vdaf: Any,
    public_u_shape: list[int] | None,
) -> dict[str, Any]:
    params = dict(config)
    params.update(
        {
            "field_modulus": int(field.MODULUS),
            "field_encoded_size": int(field.ENCODED_SIZE),
            "xof": "XofTurboShake128",
            "shares": 2,
            "proof_len_field_elements": int(vdaf.qc_prover.flp.PROOF_LEN),
            "verifier_len_field_elements": int(vdaf.qc_prover.flp.VERIFIER_LEN),
            "prove_rand_len_field_elements": int(vdaf.qc_prover.flp.PROVE_RAND_LEN),
            "query_rand_len_field_elements": int(vdaf.qc_prover.flp.QUERY_RAND_LEN),
        }
    )
    if config["protocol"] == "batch":
        params["chunk_length"] = int(vdaf.batched_dzk_valid.chunk_length)
        params["chunk_length_norm_equality"] = int(
            vdaf.batched_dzk_valid.chunk_length_norm_equality
        )
        params["l2_norm_bound"] = int(2 ** config["norm_bound_bits"])
        params["squared_norm_bound"] = int(vdaf.batched_dzk_valid.sq_norm_bound)
        params["range_bits"] = int(vdaf.batched_dzk_valid.num_bits)
        params["meas_len_field_elements"] = int(vdaf.batched_dzk_valid.MEAS_LEN)
    else:
        params["chunk_length"] = int(vdaf.pca_valid.chunk_length)
        params["pca_norm_bound"] = int(2 ** config["norm_bound_bits"])
        params["range_bits"] = int(vdaf.pca_valid.num_bits)
        params["coeff_dim"] = int(vdaf.coeff_dim)
        params["public_u_shape"] = public_u_shape
        params["meas_len_field_elements"] = int(vdaf.pca_valid.MEAS_LEN)
    return params


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    prove_phases = sorted(
        {
            phase
            for trial in trials
            for phase in trial["prove_phases_ms"].keys()
        }
    )
    verify_phases = sorted(
        {
            phase
            for trial in trials
            for phase in trial["verify_phases_ms"].keys()
        }
    )
    return {
        "trials": len(trials),
        "prove_ms": stats([trial["prove_ms"] for trial in trials]),
        "verify_ms": stats([trial["verify_ms"] for trial in trials]),
        "prove_phases_ms": {
            phase: stats([
                trial["prove_phases_ms"][phase]
                for trial in trials
                if phase in trial["prove_phases_ms"]
            ])
            for phase in prove_phases
        },
        "verify_phases_ms": {
            phase: stats([
                trial["verify_phases_ms"][phase]
                for trial in trials
                if phase in trial["verify_phases_ms"]
            ])
            for phase in verify_phases
        },
        "all_verify_ok": all(trial["verify_ok"] for trial in trials),
    }


def _make_measurements(seed: int, rows: int, cols: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 3, size=(rows, cols), dtype=np.int64)
    return values.astype(float).tolist()


def _make_axis_public_u(
    ambient_dim: int,
    coeff_dim: int,
    num_frac_bits_u: int,
) -> np.ndarray:
    if coeff_dim > ambient_dim:
        raise ValueError("PCA coeff_dim must be <= ambient_dim")
    public_u = np.zeros((ambient_dim, coeff_dim), dtype=np.int64)
    scale = 2 ** num_frac_bits_u
    for idx in range(coeff_dim):
        public_u[idx, idx] = scale
    return public_u


def _deterministic_bytes(seed: int, label: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hashlib.sha256(
                seed.to_bytes(16, "big", signed=False)
                + label
                + counter.to_bytes(4, "big")
            ).digest()
        )
        counter += 1
    return bytes(out[:length])


def _field_class(name: str) -> Any:
    try:
        return FIELD_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported field {name!r}") from exc


def _config_id(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _config_seed(config: dict[str, Any], base_seed: int) -> int:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**63 - 1)
