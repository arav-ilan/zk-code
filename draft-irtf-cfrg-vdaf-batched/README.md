# draft-irtf-cfrg-vdaf-batched

Clean Sage-free proof-of-concept package for the Batch DZK and PCA DZK
contributions refactored from the verified PINE prototype.

## Dependencies

Runtime dependencies are:

- `numpy`
- `pycryptodomex`
- vendored `vdaf_poc` under `poc/vdaf_poc`

The vendored `vdaf_poc.field` module has been replaced with a pure-Python
finite-field implementation and pure-Python polynomial interpolation.

## Run Tests

From the repository root:

```sh
./norm/bin/python -m pytest draft-irtf-cfrg-vdaf-batched/poc/tests
```

## Layout

- `poc/fields.py`: Sage-free field classes, including PINE custom fields.
- `poc/common.py`: shared transcript encoding, proof splitting, and QC helpers.
- `poc/batch_dzk.py`: native Batch DZK implementation.
- `poc/pca_dzk.py`: native PCA DZK implementation with public setup basis `U`.
- `poc/equivalence_cases.py`: new/original accept-reject comparison runner.
- `poc/vdaf_poc/`: vendored VDAF proof-of-concept dependency.
- `bench/`: placeholder for a later benchmarking task.
