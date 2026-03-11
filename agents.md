# Agents Guide

This repository exposes one public inference API in `/Users/mayank/Documents/style-transfer-for-high-fidelity-time-series-augmentation/infer.py`:

- `style_transfer(...)`

## Transfer Methods

`style_transfer` supports these methods via `method=`:

- `"diffusion"` (default): learned DiffTSST model, requires length 128.
- `"wavelet"`: wavelet baseline (`wavelet_style_transfer`).
- `"multiscale"`: moving-average baseline (`stitching`, alias `sticthing`).

All methods return the same dict schema:

- `normalized_style`
- `normalized_content`
- `normalized_generated`
- `denorm`

## Edit Points

To modify baseline behavior, edit only these functions in `style-transfer-for-high-fidelity-time-series-augmentation/infer.py`:

- `wavelet_style_transfer(...)`
- `multi_scale_decompose_1d(...)`
- `stitching(...)` / `sticthing(...)`

To add a new baseline method while preserving the same infer script:

1. Add a new function in `infer.py` that takes `content_series` and `style_series` and returns a 1D numpy array.
2. Add a new `method` branch inside `style_transfer(...)`.
3. Keep output keys unchanged so downstream code keeps working.
4. Update `README.md` with method-specific args and an example.

## Dependencies

- Wavelet baseline depends on `pywavelets` (imported as `pywt`) and is declared in `style-transfer-for-high-fidelity-time-series-augmentation/pyproject.toml`.
