# Changelog

The package's contents are based on _Signal Processing for Algorithmic Trading_ (to be published) by Jay Damask. 

## 1.0.4

- Fixed missing `__init__.py` in `filter_signature_blocks` and `resources/special_designs`, making these subpackages properly importable.
- Fixed `np.trapz` → `np.trapezoid` (removed in numpy 2.x) in curvature filter tests.
- Fixed `TypeError` in `analog_to_digital_conversion_tools.py` caused by scipy passing 0-dimensional arrays to `np.outer` under Python 3.13.
- Pinned GitLab CI to Python 3.12.
- Switched license from Apache 2.0 to MIT.
- Dropped support for Python < 3.10.
- Added `scipy` as an explicit dependency.

## 1.0.2

Unit-test tweaks to align with the py 3.13.


## 1.0.1

First release of package and source code. This package was used for my book. I completely expect to heavily revise this package's API while preserving the underlying functionality.

## 0.0.2

Skeletal package release. 




