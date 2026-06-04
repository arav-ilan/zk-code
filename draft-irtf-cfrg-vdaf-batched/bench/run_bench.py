#!/usr/bin/env python3
"""Run measured Batch DZK and PCA DZK benchmark sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import os

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = BENCH_DIR.parent
POC_DIR = PACKAGE_ROOT / "poc"
for _path in (BENCH_DIR, POC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from timing import compute_size_config, run_config  # noqa: E402


PRESET_DEFAULTS = {
    "smoke": {"trials": 3, "warmups": 1},
    "full": {"trials": 3, "warmups": 1},
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = args.timestamp or os.environ.get("BENCH_TIMESTAMP") or _default_timestamp()
    trials = args.trials if args.trials is not None else PRESET_DEFAULTS[args.preset]["trials"]
    warmups = args.warmups if args.warmups is not None else PRESET_DEFAULTS[args.preset]["warmups"]
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    all_configs = build_preset(args.preset)
    configs = select_config_shard(all_configs, args.shard_count, args.shard_index)
    json_path, csv_path, default_jsonl_path = output_paths(args.out, args.preset, timestamp)
    jsonl_path = Path(args.jsonl) if args.jsonl is not None else default_jsonl_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sizes_only:
        print(
            f"Computing deterministic sizes for {len(configs)} configs: "
            f"preset={args.preset}, shard={args.shard_index}/{args.shard_count}",
            file=sys.stderr,
            flush=True,
        )
        results = [compute_size_config(config) for config in configs]
        results = sorted(results, key=_result_sort_key)
        payload = {
            "schema_version": 1,
            "mode": "sizes-only",
            "preset": args.preset,
            "preset_config_count": len(all_configs),
            "shard": shard_info(args.shard_count, args.shard_index, len(configs)),
            "timestamp": timestamp,
            "env": env_info(
                workers=0,
                warmups=0,
                trials=0,
                seed=args.seed,
                config_count=len(configs),
            ),
            "configs": results,
        }
        write_json(json_path, payload)
        write_sizes_csv(csv_path, payload)

        print_size_summary_table(results)
        print(f"\nWrote JSON: {json_path}")
        print(f"Wrote CSV:  {csv_path}")
        return 0

    print(
        f"Running {len(configs)} configs: preset={args.preset}, "
        f"trials={trials}, warmups={warmups}, workers={workers}, "
        f"shard={args.shard_index}/{args.shard_count}, "
        f"jsonl={jsonl_path}",
        file=sys.stderr,
        flush=True,
    )
    results = run_all_configs(
        configs,
        warmups=warmups,
        trials=trials,
        workers=workers,
        seed=args.seed,
        jsonl_path=jsonl_path,
    )
    results = sorted(results, key=_result_sort_key)

    payload = {
        "schema_version": 1,
        "preset": args.preset,
        "preset_config_count": len(all_configs),
        "shard": shard_info(args.shard_count, args.shard_index, len(configs)),
        "timestamp": timestamp,
        "env": env_info(
            workers=workers,
            warmups=warmups,
            trials=trials,
            seed=args.seed,
            config_count=len(configs),
        ),
        "configs": results,
    }
    write_json(json_path, payload)
    write_csv(csv_path, payload)

    print_summary_table(results)
    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote JSONL: {jsonl_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=None, help="process-pool workers; default cpu_count")
    parser.add_argument("--trials", type=int, default=None, help="timed trials per config")
    parser.add_argument("--warmups", type=int, default=None, help="warmup trials per config")
    parser.add_argument("--seed", type=int, default=20260604, help="base deterministic RNG seed")
    parser.add_argument(
        "--sizes-only",
        action="store_true",
        help="compute deterministic communication/proof sizes without timing",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="output timestamp; also accepted via BENCH_TIMESTAMP",
    )
    parser.add_argument(
        "--out",
        default=str(BENCH_DIR / "results"),
        help="output directory or .json/.csv file path",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="incremental JSONL path for timing results; default is beside JSON/CSV",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="split preset configs into this many index-modulo shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="run only configs whose preset index modulo shard-count equals this index",
    )
    return parser.parse_args(argv)


def build_preset(name: str) -> list[dict[str, Any]]:
    if name == "smoke":
        return _smoke_configs()
    if name == "full":
        return _full_configs()
    raise ValueError(f"unknown preset {name!r}")


def select_config_shard(
    configs: list[dict[str, Any]],
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
    if shard_count == 1:
        return configs
    return [
        config
        for idx, config in enumerate(configs)
        if idx % shard_count == shard_index
    ]


def shard_info(shard_count: int, shard_index: int, config_count: int) -> dict[str, int]:
    return {
        "count": shard_count,
        "index": shard_index,
        "config_count": config_count,
    }


def _base_config() -> dict[str, Any]:
    return {
        "epsilon": 1.0,
        "delta": 1e-6,
        "num_frac_bits": 0,
        "num_frac_bits_u": 0,
    }


def _smoke_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    fields = ["Field64", "Field128"]
    dimensions = [4, 16]
    batch_sizes = [1, 4]
    norm_bits_values = [8]
    repetitions_values = [1]

    for field in fields:
        for d in dimensions:
            for m in batch_sizes:
                for norm_bits in norm_bits_values:
                    for repetitions in repetitions_values:
                        configs.append(_batch_config(field, d, m, norm_bits, repetitions))
                        configs.append(_pca_config(field, d, m, r=2, norm_bits=norm_bits, repetitions=repetitions))
    return configs


def _full_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    fields = ["Field64", "Field128"]
    dimensions = [16, 64, 128]
    batch_sizes = [1, 4, 8]
    norm_bits_values = [8, 12]
    repetitions_values = [1, 3]

    for field in fields:
        for d in dimensions:
            for m in batch_sizes:
                for norm_bits in norm_bits_values:
                    for repetitions in repetitions_values:
                        configs.append(_batch_config(field, d, m, norm_bits, repetitions))
                        for r in _pca_ranks_for_dimension(d):
                            configs.append(_pca_config(field, d, m, r, norm_bits, repetitions))
    return configs


def _batch_config(
    field: str,
    d: int,
    m: int,
    norm_bits: int,
    repetitions: int,
) -> dict[str, Any]:
    config = _base_config()
    config.update(
        {
            "protocol": "batch",
            "field": field,
            "d": d,
            "m": m,
            "r": None,
            "norm_bound_bits": norm_bits,
            "qc_repetitions": repetitions,
            "chunk_length": None,
            "chunk_length_norm_equality": min(16, max(1, d)),
        }
    )
    return config


def _pca_config(
    field: str,
    d: int,
    m: int,
    r: int,
    norm_bits: int,
    repetitions: int,
) -> dict[str, Any]:
    config = _base_config()
    config.update(
        {
            "protocol": "pca",
            "field": field,
            "d": d,
            "m": m,
            "r": r,
            "norm_bound_bits": norm_bits,
            "qc_repetitions": repetitions,
            "chunk_length": min(16, max(1, r, 2 * norm_bits)),
            "chunk_length_norm_equality": None,
        }
    )
    return config


def _pca_ranks_for_dimension(d: int) -> list[int]:
    return [r for r in (2, 4, 8) if r <= d]


def run_all_configs(
    configs: list[dict[str, Any]],
    warmups: int,
    trials: int,
    workers: int,
    seed: int,
    jsonl_path: Path | None = None,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    jsonl_file = jsonl_path.open("a", encoding="utf-8") if jsonl_path is not None else None
    try:
        return _run_all_configs(
            configs=configs,
            warmups=warmups,
            trials=trials,
            workers=workers,
            seed=seed,
            jsonl_file=jsonl_file,
        )
    finally:
        if jsonl_file is not None:
            jsonl_file.close()


def _run_all_configs(
    configs: list[dict[str, Any]],
    warmups: int,
    trials: int,
    workers: int,
    seed: int,
    jsonl_file: Any | None,
) -> list[dict[str, Any]]:
    if workers == 1:
        results = []
        for idx, config in enumerate(configs, start=1):
            result = _run_config_with_start_line(
                idx,
                len(configs),
                config,
                warmups,
                trials,
                seed,
            )
            results.append(result)
            _record_completed_config(idx, len(configs), result, jsonl_file)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_config_with_start_line,
                idx,
                len(configs),
                config,
                warmups,
                trials,
                seed,
            )
            for idx, config in enumerate(configs, start=1)
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            _record_completed_config(idx, len(futures), result, jsonl_file)
    return results


def _run_config_with_start_line(
    idx: int,
    total: int,
    config: dict[str, Any],
    warmups: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    print(_config_line("starting", idx, total, config), file=sys.stderr, flush=True)
    return run_config(config, warmups, trials, seed)


def _record_completed_config(
    idx: int,
    total: int,
    result: dict[str, Any],
    jsonl_file: Any | None,
) -> None:
    if jsonl_file is not None:
        json.dump(result, jsonl_file, sort_keys=True)
        jsonl_file.write("\n")
        jsonl_file.flush()
        os.fsync(jsonl_file.fileno())
    print(_progress_line(idx, total, result), file=sys.stderr, flush=True)


def _progress_line(idx: int, total: int, result: dict[str, Any]) -> str:
    params = result["params"]
    stats = result["stats"]
    r_value = "-" if params["r"] is None else params["r"]
    return (
        f"completed {idx}/{total}: "
        f"{_config_descriptor(result['protocol'], params, r_value)} "
        f"prove={stats['prove_ms']['mean']:.3f}ms "
        f"verify={stats['verify_ms']['mean']:.3f}ms "
        f"verify_ok={stats['all_verify_ok']}"
    )


def _config_line(
    label: str,
    idx: int,
    total: int,
    config: dict[str, Any],
) -> str:
    r_value = "-" if config["r"] is None else config["r"]
    return (
        f"{label} {idx}/{total}: "
        f"{_config_descriptor(config['protocol'], config, r_value)}"
    )


def _config_descriptor(
    protocol: str,
    params: dict[str, Any],
    r_value: Any,
) -> str:
    return (
        f"{protocol} field={params['field']} d={params['d']} "
        f"m={params['m']} r={r_value} "
        f"bits={params['norm_bound_bits']} t={params['qc_repetitions']}"
    )


def output_paths(out_arg: str, preset: str, timestamp: str) -> tuple[Path, Path, Path]:
    out = Path(out_arg)
    if out.suffix == ".json":
        return out, out.with_suffix(".csv"), out.with_suffix(".jsonl")
    if out.suffix == ".csv":
        return out.with_suffix(".json"), out, out.with_suffix(".jsonl")
    if out.suffix == ".jsonl":
        return out.with_suffix(".json"), out.with_suffix(".csv"), out
    return (
        out / f"{preset}-{timestamp}.json",
        out / f"{preset}-{timestamp}.csv",
        out / f"{preset}-{timestamp}.jsonl",
    )


def env_info(
    workers: int,
    warmups: int,
    trials: int,
    seed: int,
    config_count: int,
) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "workers": workers,
        "warmups": warmups,
        "trials": trials,
        "seed": seed,
        "config_count": config_count,
        "package_root": str(PACKAGE_ROOT),
        "cwd": os.getcwd(),
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [_csv_row(result) for result in payload["configs"]]
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sizes_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [_size_csv_row(result) for result in payload["configs"]]
    fieldnames = _size_csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    params = result["params"]
    sizes = result["sizes"]
    stats = result["stats"]
    row: dict[str, Any] = {
        "config_id": result["config_id"],
        "protocol": result["protocol"],
        "field": params["field"],
        "d": params["d"],
        "m": params["m"],
        "r": "" if params["r"] is None else params["r"],
        "norm_bound_bits": params["norm_bound_bits"],
        "qc_repetitions": params["qc_repetitions"],
        "field_encoded_size": params["field_encoded_size"],
        "range_bits": params["range_bits"],
        "input_share_bytes_by_aggregator": ";".join(
            str(x) for x in sizes["input_share_bytes_by_aggregator"]
        ),
        "input_shares_total_bytes": sizes["input_shares_total_bytes"],
        "public_share_bytes": sizes["public_share_bytes"],
        "setup_public_u_bytes": sizes["setup_public_u_bytes"],
        "online_total_comm_bytes": sizes["online_total_comm_bytes"],
        "total_comm_with_setup_bytes": sizes["total_comm_with_setup_bytes"],
        "proof_share_bytes_by_aggregator": ";".join(
            str(x) for x in sizes["proof_share_bytes_by_aggregator"]
        ),
        "proof_size_bytes": sizes["proof_size_bytes"],
        "trials": stats["trials"],
        "all_verify_ok": stats["all_verify_ok"],
    }
    _add_stats_columns(row, "prove_ms", stats["prove_ms"])
    _add_stats_columns(row, "verify_ms", stats["verify_ms"])
    for phase, phase_stats in stats["prove_phases_ms"].items():
        _add_stats_columns(row, f"prove_phase_{phase}", phase_stats)
    for phase, phase_stats in stats["verify_phases_ms"].items():
        _add_stats_columns(row, f"verify_phase_{phase}", phase_stats)
    return row


def _size_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    params = result["params"]
    sizes = result["sizes"]
    return {
        "config_id": result["config_id"],
        "protocol": result["protocol"],
        "field": params["field"],
        "d": params["d"],
        "m": params["m"],
        "r": "" if params["r"] is None else params["r"],
        "norm_bound_bits": params["norm_bound_bits"],
        "qc_repetitions": params["qc_repetitions"],
        "field_encoded_size": params["field_encoded_size"],
        "range_bits": params["range_bits"],
        "meas_len_field_elements": params["meas_len_field_elements"],
        "proof_len_field_elements": params["proof_len_field_elements"],
        "verifier_len_field_elements": params["verifier_len_field_elements"],
        "prove_rand_len_field_elements": params["prove_rand_len_field_elements"],
        "query_rand_len_field_elements": params["query_rand_len_field_elements"],
        "input_share_bytes_by_aggregator": ";".join(
            str(x) for x in sizes["input_share_bytes_by_aggregator"]
        ),
        "input_shares_total_bytes": sizes["input_shares_total_bytes"],
        "public_share_bytes": sizes["public_share_bytes"],
        "setup_public_u_bytes": sizes["setup_public_u_bytes"],
        "online_total_comm_bytes": sizes["online_total_comm_bytes"],
        "total_comm_with_setup_bytes": sizes["total_comm_with_setup_bytes"],
        "proof_share_bytes_by_aggregator": ";".join(
            str(x) for x in sizes["proof_share_bytes_by_aggregator"]
        ),
        "proof_size_bytes": sizes["proof_size_bytes"],
    }


def _add_stats_columns(row: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
    row[f"{prefix}_mean"] = values["mean"]
    row[f"{prefix}_median"] = values["median"]
    row[f"{prefix}_p95"] = values["p95"]


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "config_id",
        "protocol",
        "field",
        "d",
        "m",
        "r",
        "norm_bound_bits",
        "qc_repetitions",
        "field_encoded_size",
        "range_bits",
        "input_share_bytes_by_aggregator",
        "input_shares_total_bytes",
        "public_share_bytes",
        "setup_public_u_bytes",
        "online_total_comm_bytes",
        "total_comm_with_setup_bytes",
        "proof_share_bytes_by_aggregator",
        "proof_size_bytes",
        "prove_ms_mean",
        "prove_ms_median",
        "prove_ms_p95",
        "verify_ms_mean",
        "verify_ms_median",
        "verify_ms_p95",
        "trials",
        "all_verify_ok",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    return preferred + extra


def _size_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "config_id",
        "protocol",
        "field",
        "d",
        "m",
        "r",
        "norm_bound_bits",
        "qc_repetitions",
        "field_encoded_size",
        "range_bits",
        "meas_len_field_elements",
        "proof_len_field_elements",
        "verifier_len_field_elements",
        "prove_rand_len_field_elements",
        "query_rand_len_field_elements",
        "input_share_bytes_by_aggregator",
        "input_shares_total_bytes",
        "public_share_bytes",
        "setup_public_u_bytes",
        "online_total_comm_bytes",
        "total_comm_with_setup_bytes",
        "proof_share_bytes_by_aggregator",
        "proof_size_bytes",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    return preferred + extra


def print_summary_table(results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        params = result["params"]
        sizes = result["sizes"]
        stats = result["stats"]
        rows.append(
            {
                "protocol": "Batch" if result["protocol"] == "batch" else "PCA",
                "field": params["field"],
                "d": str(params["d"]),
                "m": str(params["m"]),
                "r": "-" if params["r"] is None else str(params["r"]),
                "bits": str(params["norm_bound_bits"]),
                "t": str(params["qc_repetitions"]),
                "total B": f"{sizes['total_comm_with_setup_bytes']:,}",
                "setup U B": f"{sizes['setup_public_u_bytes']:,}",
                "proof B": f"{sizes['proof_size_bytes']:,}",
                "prove ms": f"{stats['prove_ms']['mean']:.3f}",
                "verify ms": f"{stats['verify_ms']['mean']:.3f}",
            }
        )
    _print_table(rows)


def print_size_summary_table(results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        params = result["params"]
        sizes = result["sizes"]
        rows.append(
            {
                "protocol": "Batch" if result["protocol"] == "batch" else "PCA",
                "field": params["field"],
                "d": str(params["d"]),
                "m": str(params["m"]),
                "r": "-" if params["r"] is None else str(params["r"]),
                "bits": str(params["norm_bound_bits"]),
                "t": str(params["qc_repetitions"]),
                "total B": f"{sizes['total_comm_with_setup_bytes']:,}",
                "setup U B": f"{sizes['setup_public_u_bytes']:,}",
                "proof B": f"{sizes['proof_size_bytes']:,}",
            }
        )
    _print_table(rows)


def _print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No results")
        return
    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def _result_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    params = result["params"]
    return (
        result["protocol"],
        params["field"],
        params["d"],
        params["m"],
        params["r"] if params["r"] is not None else -1,
        params["norm_bound_bits"],
        params["qc_repetitions"],
    )


def _default_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
