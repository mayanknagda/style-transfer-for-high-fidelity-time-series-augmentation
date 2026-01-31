# Time Series Style Transfer (DiffTSST)

This repo provides a **single, simple function** for time series style transfer, powered by a diffusion model.
It implements the method from the paper **“Style Transfer for High-Fidelity Time Series Augmentation” (DiffTSST)**,
presented at **ECML PKDD 2025 – Workshop & Tutorial Track (SynDAiTE)**.

You give:
- a **style reference** series,
- a **content reference** series,
- how you want to **denormalize** the output.

You get a dict containing:
- normalized style series
- normalized content series
- normalized generated output
- denormalized output (content-based, style-based, or none)

---

## Quickstart (uv)

From the repo root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

That installs **all dependencies** declared in `pyproject.toml`.

---

## One-Function API

The main entrypoint is `style_transfer` in `infer.py`.

```python
import numpy as np
from infer import style_transfer

# Example input (must be length 128)
x = np.linspace(0, 4 * np.pi, 128)
content = np.sin(x)
style = np.random.randn(128) + np.arange(128) * 0.08

result = style_transfer(
    style_series=style,
    content_series=content,
    denorm="content",     # "content", "style", or "none"
    num_steps=500,         # diffusion steps
    c_c=1.0,               # content weight
    c_s=0.0,               # style-content weight
)

print(result.keys())
# dict_keys(['normalized_style', 'normalized_content', 'normalized_generated', 'denorm'])

print(result["denorm"].shape)  # (128,)
```

---

## Function Reference

### `style_transfer(...)`

```python
style_transfer(
    style_series,
    content_series,
    denorm="content",
    checkpoint_path="checkpoints/",
    num_steps=500,
    c_c=1.0,
    c_s=0.0,
    device=None,
    model=None,
)
```

**Arguments**
- `style_series` (array-like, length 128): Style reference time series.
- `content_series` (array-like, length 128): Content reference time series.
- `denorm` (str): How to denormalize the generated output:
  - `"content"` → scale by content mean/std
  - `"style"` → scale by style mean/std
  - `"none"` → keep normalized output
- `checkpoint_path` (str): Folder containing model checkpoints (e.g., `model_iter50000.pth`).
- `num_steps` (int): Diffusion steps for sampling (must be <= 500 for the provided checkpoint).
- `c_c` (float): Weight for content preservation.
- `c_s` (float): Weight for style-content influence.
- `device` (torch.device or str or None): Explicit device override (e.g., `"cpu"`, `"cuda:0"`).
- `model` (torch.nn.Module or None): Optional pre-loaded model to reuse across calls.

**Returns**
A `dict` with:
- `normalized_style`: z-scored style series (np.ndarray, shape `(128,)`).
- `normalized_content`: z-scored content series (np.ndarray, shape `(128,)`).
- `normalized_generated`: generated output in normalized space (np.ndarray, shape `(128,)`).
- `denorm`: generated output after denormalization (np.ndarray, shape `(128,)`).

---

## Notes & Constraints

- **Fixed length**: the model currently expects **length 128** series.
- For longer series, split into overlapping windows of length 128 and stitch results.
- If you want a smoothed version of content, use a flat style series.
- `c_c + c_s = 1` is recommended but not required.
- Checkpoints must live in `checkpoints/` and be named like `model_iterXXXXX.pth`.

---

## Minimal Plot (optional)

```python
import matplotlib.pyplot as plt
from infer import style_transfer

result = style_transfer(style, content, denorm="content")
plt.plot(content, label="content", linestyle="dashed")
plt.plot(style, label="style", linestyle="dotted")
plt.plot(result["denorm"], label="generated")
plt.legend()
plt.show()
```

---

## Troubleshooting

- **"No checkpoint found"**: ensure `checkpoints/` contains a `.pth` file.
- **Shape errors**: both input series must be 1D arrays of length 128.

---

## Citation

If you use this codebase, please cite the paper:

**Style Transfer for High-Fidelity Time Series Augmentation**  
ECML PKDD 2025 – Workshop & Tutorial Track (SynDAiTE)

See `CITATION.cff` for machine-readable citation metadata.
