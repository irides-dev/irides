# PyPI Upload Instructions

This document provides instructions for uploading the irides package to TestPyPI and PyPI.

## Prerequisites

1. Ensure you have a TestPyPI account: https://test.pypi.org/account/register/
2. Ensure you have a PyPI account: https://pypi.org/account/register/
3. Configure your PyPI credentials in `~/.pypirc` or be prepared to enter them when prompted:

```
[testpypi]
username = __token__
password = your-test-pypi-token

[pypi]
username = __token__
password = your-pypi-token
```

## Upload Process

### 1. Upload to TestPyPI First

It's recommended to upload to TestPyPI first to verify everything works correctly:

```bash
./upload-to-testpypi
```

After uploading, verify the package is available at: https://test.pypi.org/project/irides/

You can test installation from TestPyPI with:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ irides
```

### 2. Upload to PyPI

Once you've verified the package works correctly on TestPyPI, upload to the main PyPI repository:

```bash
./upload-to-pypi
```

After uploading, verify the package is available at: https://pypi.org/project/irides/

You can install from PyPI with:

```bash
pip install irides
```

## Troubleshooting

- If you get an error about the package version already existing, you'll need to update the version number in:
  - `pyproject.toml`
  - `setup.py`
  - `src/irides/__init__.py`
  - `conda-forge.recipe/meta.yaml`

- If you encounter authentication issues, check your PyPI token and ensure it has the correct permissions.

- For other issues, refer to the twine documentation: https://twine.readthedocs.io/
