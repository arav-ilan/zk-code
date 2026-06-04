# Batch DZK / PCA DZK Benchmarks

This directory contains measured benchmarks for the native Batch DZK and PCA
DZK implementations in `../poc/`. The suite shards actual protocol objects,
times actual verifier prep calls, and writes machine-readable JSON and CSV
results.

## Run

From the repository root:

```sh
./norm/bin/python draft-irtf-cfrg-vdaf-batched/bench/run_bench.py --preset smoke
```

For the larger sweep:

```sh
./norm/bin/python draft-irtf-cfrg-vdaf-batched/bench/run_bench.py --preset full --workers 40
```

Useful options:

- `--workers N`: process-pool workers. The default is `os.cpu_count()`.
- `--trials N`: timed trials per config. Preset default is 3.
- `--warmups N`: untimed warmups per config. Preset default is 1.
- `--seed N`: base deterministic seed. Default is `20260604`.
- `--timestamp VALUE`: output timestamp. `BENCH_TIMESTAMP` is also accepted.
- `--out PATH`: output directory, `.json` path, or `.csv` path.

By default, results are written to:

```text
draft-irtf-cfrg-vdaf-batched/bench/results/<preset>-<timestamp>.json
draft-irtf-cfrg-vdaf-batched/bench/results/<preset>-<timestamp>.csv
```

Generated JSON/CSV files under `bench/results/` are ignored by git.

## Metrics

`sizes.py` provides the single `serialized_size(obj)` helper used by the suite.
It counts the actual sharded objects returned by `shard_batched_dzk` and
`shard_pca_dzk`:

- Field elements: `type(element).ENCODED_SIZE` bytes, matching
  `field.encode_vec`.
- `bytes`: raw length, for already-packed bit-share bytes, blinds, and public
  transcript parts.
- Fixed-shape lists/tuples: sum of member sizes, with no extra dynamic-length
  prefix.
- PCA setup `U`: `len(encode_public_basis_transcript(U))`, marked as setup
  bytes and not online per-batch bytes.

The JSON and CSV include:

- `input_share_bytes_by_aggregator`
- `input_shares_total_bytes`
- `public_share_bytes`
- `setup_public_u_bytes`
- `online_total_comm_bytes`
- `total_comm_with_setup_bytes`
- `proof_share_bytes_by_aggregator`
- `proof_size_bytes`

`proof_size_bytes` is the additive QC proof material contained inside the input
shares, so it is a subset of input-share communication, not an additional
term in the communication total.

`timing.py` measures:

- Proving time: wall-clock around `shard_batched_dzk` or `shard_pca_dzk`.
- Proving phases: protocol-provided `_ms` keys from `last_shard_timings`.
- Verification time: explicit distributed verification, including both
  `prep_init_*` calls, `prep_shares_to_prep_*` combine/decide, and
  `prep_next_*` transcript consistency checks.
- Statistics: mean, median, and p95 over the timed trials.

## Presets

`smoke` is intended to validate the harness quickly:

- Protocols: Batch DZK and PCA DZK.
- Fields: `Field64`, `Field128`.
- Dimensions `d`: 4, 16.
- Batch sizes `m`: 1, 4.
- PCA rank `r`: 2.
- Norm-bound bits: 8.
- QC repetitions `t`: 1.
- Default trials/warmups: 3 timed trials, 1 warmup.
- Total configs: 16.

`full` is the default paper-oriented pure-Python sweep:

- Protocols: Batch DZK and PCA DZK.
- Fields: `Field64`, `Field128`.
- Dimensions `d`: 16, 64, 128.
- Batch sizes `m`: 1, 4, 8.
- PCA ranks `r`: 2, 4, 8.
- Norm-bound bits: 8, 12.
- QC repetitions `t`: 1, 3.
- Default trials/warmups: 3 timed trials, 1 warmup.
- Total configs: 288.

The full preset intentionally does not silently cap larger cases. It is still
pure Python and can take tens of minutes to a few hours depending on core count
and machine speed. For calibration, run `--preset full --workers 1 --trials 1
--warmups 0` after reducing the sweep in code or use the smoke preset first.

## Parallelism

`run_bench.py` schedules one complete config per worker process. A config's
warmups and timed trials run sequentially inside that worker; there is no
intra-config parallelism. This keeps each timed run isolated to one worker
process while allowing independent configs to run across a 40-core machine.

The runner sets common NumPy/BLAS thread-count environment variables to `1`
unless the caller already set them, avoiding nested BLAS thread fanout when many
worker processes are active.

## Output

The JSON file contains environment info, every config's parameters, measured
size breakdown, per-trial timings, and aggregate stats. The CSV is one flattened
row per config with scalar size fields and timing stats. The CLI also prints a
summary table with total communication, setup `U`, proof size, proving mean, and
verification mean.

The benchmark imports only the package, Python standard library modules, and
NumPy. It does not require Sage.
