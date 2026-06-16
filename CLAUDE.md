# irides-cabinet

Signal processing filter library for quantitative finance, published as the `irides` package on PyPI.
This cabinet is the authoritative source — it is rsync'd one-way into a separate `irides` GitLab repo
(irides-dev/irides). Never make changes directly in that repo; they will be overwritten on the next sync.

## Package Structure

- `src/irides/design/` — filter design parameters, analog and digital
  - Filter families: `bessel_level`, `bessel_inline_slope`, `bessel_inline_curvature`,
    `mbox_level`, `mbox_inline_slope`, `mbox_inline_curvature`,
    `polyema_level`, `polyema_inline_slope`, `polyema_inline_curvature`,
    `polyema_composite_slope`, `damped_oscillator`, jump-vol variants
- `src/irides/filter_signatures/` — resulting filter signatures (analog + digital)
- `src/irides/filter_signature_blocks/` — digital signature blocks (jump volatility, returns volatility)
- `src/irides/operators/` — signal operators: unit step, ideal delay, first/second difference
- `src/irides/resources/` — core containers: wireframes, transfer functions, pole resources,
  discrete sequences, enumerations
- `src/irides/tools/` — design tools, A→D conversion, impulse response builders,
  transfer function builders
- `src/irides/test_signals/` — test signal fixtures

## Conventions

- Every analog design has a matching digital counterpart; keep the two in structural parity
- Tests mirror the src layout under `tests/`
- Packaging: `pyproject.toml` + `setup.py`; build with `python -m build`

## Publishing

Sync to the irides GitLab repo is done via the rsync workflow documented in
`../agent-docs/irides-cabinet/rsync-to-irides-method.md`.
