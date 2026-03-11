import os
import numpy as np
import torch
import torch.nn.functional as F
import pywt
from model import MultiResUNet1D


########################
# Utils
########################
# ================= MULTI-SCALE DECOMPOSITION (BATCH LEVEL) =================
def multi_scale_decompose(
    x,
    num_scales=3,
    kernel_sizes=[3, 5, 15],
    shift_range=10,
    mask_ratio=0.1,
    mode="train",
):
    """
    Multi-scale decomposition of a batch of time series using averaging filters.
    Applies augmentations:
      - Random shifts (circular)
      - Random masking (zeros)
    """
    assert len(kernel_sizes) == num_scales

    batch_size, channels, length = x.shape
    current = x.clone()
    bands = []

    for i in range(num_scales):
        k = kernel_sizes[i]
        pad_left = (k - 1) // 2
        pad_right = k // 2
        weight = (
            torch.ones((channels, 1, k), device=x.device) / k
        )  # Maintain batch shape

        current_padded = F.pad(current, (pad_left, pad_right), mode="replicate")
        smooth = F.conv1d(current_padded, weight, groups=channels)

        residual = current - smooth
        if mode == "train":
            residual = augment_style_band(
                residual, shift_range, mask_ratio
            )  # Apply augmentations
        bands.append(residual)
        current = smooth

    return {"content": current, "bands": bands}


def augment_style_band(band, shift_range=10, mask_ratio=0.05):
    """
    Applies augmentations to the style bands:
    1. Randomly shifts the band along the time axis.
    2. Randomly masks a fraction of the band.
    """
    batch_size, channels, length = band.shape

    # ========== Random Shift ==========
    shift_amount = torch.randint(
        -shift_range, shift_range + 1, (batch_size,), device=band.device
    )
    for i in range(batch_size):
        band[i] = torch.roll(band[i], shifts=int(shift_amount[i]), dims=-1)

    # ========== Random Masking ==========
    mask = torch.ones_like(band)
    num_masked = int(mask_ratio * length)

    for i in range(batch_size):
        mask_idx = torch.randperm(length, device=band.device)[:num_masked]
        mask[i, :, mask_idx] = 0  # Set random locations to 0

    return band * mask


########################
# 1) Noise schedule
########################
def prepare_noise_schedule(T=500, beta_start=1e-4, beta_end=0.02):
    """
    Create a linear schedule of betas from beta_start to beta_end,
    and compute alpha_cumprod for each t in [0..T-1].
    """
    betas = torch.linspace(beta_start, beta_end, T)  # shape (T,)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)  # shape (T,)
    return betas, alpha_cumprod


########################
# 2) Single-step reverse update
########################
@torch.no_grad()
def p_sample(model, x_t, t, betas, alpha_cumprod, content_band, style_bands, device):
    """
    Perform one reverse diffusion step:
      x_{t-1} = 1/sqrt(alpha_t) [ x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t} * eps_theta ]
                 + sigma_t * z  (if t>0)
    Where eps_theta = model(x_t, t, content, style_bands) [the predicted noise].

    Args:
      model:           your UNet model
      x_t:             (batch, 1, 100) current noisy sample at time t
      t:               python int for the current time step
      betas:           (T,)  the beta schedule
      alpha_cumprod:   (T,)  the cumulative product of (1-beta) up to t
      content_band:    (batch, 1, 100)
      style_bands:     list of (batch, 1, 100)
      device:          torch device

    Returns:
      x_{t-1}: (batch,1,100) the denoised sample at time t-1
    """
    # Gather the relevant scalars
    beta_t = betas[t]  # scalar
    alpha_t = 1.0 - beta_t  # scalar
    alpha_bar_t = alpha_cumprod[t]  # scalar

    # Model predicts noise
    t_tensor = torch.tensor([t], device=device).repeat(x_t.size(0))
    eps_pred = model(
        x_t, t_tensor.unsqueeze(-1).float(), content_band, style_bands
    )  # shape (B,1,100)

    # Coefficients
    one_minus_alpha_t = (1.0 - alpha_t).sqrt()  # sqrt(1 - alpha_t)
    one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()  # sqrt(1 - alpha_bar_t)
    # We broadcast to match (batch,1,100)
    alpha_t_sqrt = alpha_t.sqrt()
    alpha_bar_t_sqrt = alpha_bar_t.sqrt()

    # Compute the "predicted x_{t-1} mean"
    # mean = 1/sqrt(alpha_t) * [ x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t)* eps_pred ]
    # shape is (batch,1,100)
    mean = (1.0 / alpha_t_sqrt) * (x_t - (beta_t / one_minus_alpha_bar_t) * eps_pred)

    if t > 0:
        # sample random noise
        z = torch.randn_like(x_t)
        sigma_t = beta_t.sqrt()  # often used in DDPM
        x_prev = mean + sigma_t * z
    else:
        # at t=0, we don't add noise
        x_prev = mean

    return x_prev


########################
# 3) Full sampling loop
########################
@torch.no_grad()
def ddpm_sample(model, content_band, style_bands, device, T=500):
    """
    We start from pure noise x_T ~ N(0,1), then sample x_{t-1} from x_t with p_sample,
    for t=T-1 down to 0.
    """
    batch_size = content_band.size(0)
    length = content_band.size(2)

    # Prepare betas/alpha schedules.
    # (IMPORTANT: Must match the schedule used in training)
    betas, alpha_cumprod = prepare_noise_schedule(T=T)
    betas = betas.to(device)
    alpha_cumprod = alpha_cumprod.to(device)

    # Start from pure noise
    x_t = torch.randn((batch_size, 1, length), device=device)

    for t in reversed(range(T)):
        x_t = p_sample(
            model, x_t, t, betas, alpha_cumprod, content_band, style_bands, device
        )
    return x_t


########################
# 4) Putting it together
########################
def load_latest_checkpoint(model, checkpoint_path="checkpoints/", device="cpu"):
    checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in checkpoints directory.")

    latest_checkpoint = max(
        checkpoints, key=lambda f: int(f.split("iter")[-1].split(".pth")[0])
    )
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_path, latest_checkpoint),
            map_location=device,
        )
    )
    print(f"Loaded checkpoint: {latest_checkpoint}")
    return model


def normalize(series):
    mean = np.mean(series)
    std = np.std(series) + 1e-8
    return (series - mean) / std, mean, std


def denormalize(series, mean, std):
    return series * std + mean


########################
# Baseline methods
########################
def wavelet_style_transfer(content_series, style_series, wavelet="haar", level=3):
    """
    Wavelet baseline: keep the low-frequency content component from `content_series`
    and high-frequency style components from `style_series`.
    """
    content_series = _to_1d_float_array(content_series, "content_series")
    style_series = _to_1d_float_array(style_series, "style_series")
    _validate_same_length(content_series, style_series)

    content_coeffs = pywt.wavedec(content_series, wavelet, level=level)
    style_coeffs = pywt.wavedec(style_series, wavelet, level=level)
    combined_coeffs = [content_coeffs[0]] + style_coeffs[1:]
    reconstructed_series = pywt.waverec(combined_coeffs, wavelet)
    return reconstructed_series[: len(content_series)]


def multi_scale_decompose_1d(x, num_scales=3, kernel_sizes=None):
    """
    1D multi-scale decomposition baseline using moving average filters.
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 15]
    if len(kernel_sizes) != num_scales:
        raise ValueError("Number of scales must match kernel_sizes length.")

    x = _to_1d_float_array(x, "x")
    current = x.copy()
    bands = []

    for k in kernel_sizes:
        smooth = np.convolve(current, np.ones(k) / k, mode="same")
        residual = current - smooth
        bands.append(residual)
        current = smooth

    return {"content": current, "bands": bands}


def stitching(content_series, style_series, num_scales=1, kernel_sizes=None):
    """
    Multi-scale baseline: use content low-frequency signal from content and
    style residual bands from style.
    """
    if kernel_sizes is None:
        kernel_sizes = [3]

    content_series = _to_1d_float_array(content_series, "content_series")
    style_series = _to_1d_float_array(style_series, "style_series")
    _validate_same_length(content_series, style_series)

    content_decomp = multi_scale_decompose_1d(content_series, num_scales, kernel_sizes)
    style_decomp = multi_scale_decompose_1d(style_series, num_scales, kernel_sizes)
    return content_decomp["content"] + sum(style_decomp["bands"])


def sticthing(content_series, style_series, num_scales=1, kernel_sizes=None):
    """
    Backward-compatible alias for the requested baseline name.
    """
    return stitching(
        content_series=content_series,
        style_series=style_series,
        num_scales=num_scales,
        kernel_sizes=kernel_sizes,
    )


def _generate_series_norm(
    content_series, style_series, model, device, c_c=1, c_s=0, num_steps=500
):
    """
    Generate diffusion output in normalized space.
    """
    if np.isclose(c_c + c_s, 0.0):
        raise ValueError("c_c + c_s must be non-zero for diffusion.")

    # Convert input series to tensors
    content_tensor = (
        torch.tensor(content_series, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    style_tensor = (
        torch.tensor(style_series, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Step 1: Multi-scale decomposition
    content_decomp = multi_scale_decompose(content_tensor, mode="infer")
    style_decomp = multi_scale_decompose(style_tensor, mode="infer")

    # Mix low-frequency content and keep style residual bands as-is.
    content_band = (c_c * content_decomp["content"] + c_s * style_decomp["content"]) / (
        c_c + c_s
    )
    x_0 = ddpm_sample(model, content_band, style_decomp["bands"], device, T=num_steps)

    return x_0.squeeze().cpu().numpy()


def plot_generated_series(generated, content, style, output_path="generated.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(content, label="Content Series", linestyle="dashed")
    plt.plot(style, label="Style Series", linestyle="dotted")
    plt.plot(generated, label="Generated Series", linewidth=2)
    plt.legend()
    plt.title("Style Transfer via Diffusion Model")
    plt.savefig(output_path)
    plt.show()

def _to_1d_float_array(series, name):
    array = np.asarray(series, dtype=np.float32).squeeze()
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like, got shape {array.shape}.")
    return array


def _validate_series_length(content_series, style_series, expected_length=128):
    if content_series.shape[0] != expected_length:
        raise ValueError(
            f"content_series must be length {expected_length}, got {content_series.shape[0]}."
        )
    if style_series.shape[0] != expected_length:
        raise ValueError(
            f"style_series must be length {expected_length}, got {style_series.shape[0]}."
        )


def _validate_same_length(content_series, style_series):
    if content_series.shape[0] != style_series.shape[0]:
        raise ValueError(
            "content_series and style_series must have the same length, got "
            f"{content_series.shape[0]} and {style_series.shape[0]}."
        )


@torch.no_grad()
def style_transfer(
    style_series,
    content_series,
    denorm="content",
    method="diffusion",
    checkpoint_path="checkpoints/",
    num_steps=500,
    c_c=1.0,
    c_s=0.0,
    wavelet="haar",
    level=3,
    num_scales=1,
    kernel_sizes=[3],
    device=None,
    model=None,
):
    """
    Simple, high-level API for time series style transfer.

    Args:
        style_series (array-like): Style reference time series (length 128).
        content_series (array-like): Content reference time series (length 128).
        denorm (str): How to denormalize the generated output. One of:
            - "content": scale by the content series mean/std
            - "style": scale by the style series mean/std
            - "none": keep normalized output (no denorm)
        method (str): Transfer method. One of:
            - "diffusion" (default)
            - "wavelet"
            - "multiscale"
        checkpoint_path (str): Folder containing model checkpoints (*.pth).
        num_steps (int): Number of diffusion steps used during sampling.
        c_c (float): Weight for content preservation.
        c_s (float): Weight for style-content influence.
        wavelet (str): Wavelet family used when `method="wavelet"`.
        level (int): Wavelet decomposition depth when `method="wavelet"`.
        num_scales (int): Number of smoothing levels when `method="multiscale"`.
        kernel_sizes (list[int]|None): Moving-average kernels for `method="multiscale"`.
        device (torch.device|str|None): Optional device override.
        model (torch.nn.Module|None): Optional pre-loaded model instance.

    Returns:
        dict with:
            - normalized_style (np.ndarray): z-scored style series.
            - normalized_content (np.ndarray): z-scored content series.
            - normalized_generated (np.ndarray): generated output in normalized space.
            - denorm (np.ndarray): denormalized output based on `denorm`.
    """
    if denorm not in {"content", "style", "none"}:
        raise ValueError("denorm must be 'content', 'style', or 'none'.")
    if method not in {"diffusion", "wavelet", "multiscale"}:
        raise ValueError("method must be 'diffusion', 'wavelet', or 'multiscale'.")
    if method == "diffusion" and num_steps > 500:
        raise ValueError("num_steps must be <= 500 for the provided checkpoints.")

    style_series = _to_1d_float_array(style_series, "style_series")
    content_series = _to_1d_float_array(content_series, "content_series")
    _validate_same_length(content_series, style_series)

    # Full-series normalization for user-facing outputs and baseline methods
    normalized_style, style_mean, style_std = normalize(style_series)
    normalized_content, content_mean, content_std = normalize(content_series)

    if method == "wavelet":
        normalized_generated = wavelet_style_transfer(
            content_series=normalized_content,
            style_series=normalized_style,
            wavelet=wavelet,
            level=level,
        )
    elif method == "multiscale":
        if kernel_sizes is None:
            kernel_sizes = [3]
        normalized_generated = stitching(
            content_series=normalized_content,
            style_series=normalized_style,
            num_scales=num_scales,
            kernel_sizes=kernel_sizes,
        )
    else:
        _validate_series_length(content_series, style_series, expected_length=128)
        if np.isclose(c_c + c_s, 0.0):
            raise ValueError("c_c + c_s must be non-zero for diffusion.")
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if model is None:
            model = MultiResUNet1D(T=500).to(device)
            model = load_latest_checkpoint(model, checkpoint_path, device)
        model.eval()

        normalized_generated = _generate_series_norm(
            content_series=normalized_content,
            style_series=normalized_style,
            model=model,
            device=device,
            c_c=c_c,
            c_s=c_s,
            num_steps=num_steps,
        )

    if denorm == "none":
        denorm_output = normalized_generated
    elif denorm == "content":
        denorm_output = denormalize(normalized_generated, content_mean, content_std)
    else:
        denorm_output = denormalize(normalized_generated, style_mean, style_std)

    return {
        "normalized_style": normalized_style,
        "normalized_content": normalized_content,
        "normalized_generated": normalized_generated,
        "denorm": denorm_output,
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(0)

    # Define your content and style series.
    # The model currently supports only fixed-length series of size 128.
    # For longer sequences, split them into overlapping patches of length 128.

    # To generate a smoothed version of a series, use a flat style series (e.g., a constant line).
    # The weights c_c and c_s control how much content is inherited from each source:
    #   - c_c: weight of the content series
    #   - c_s: weight of the style's content component
    # It is recommended that c_c + c_s = 1.
    # For example, c_c = 1 and c_s = 0 means full content preservation, no style content influence.

    x = np.linspace(0, np.pi * 4, 128)
    content_series = np.sin(x)
    style_series = np.random.randn(128) + np.arange(128) * 0.08
    c_c = 1
    c_s = 0

    # Simple, high-level API (style first, then content)
    result = style_transfer(
        style_series=style_series,
        content_series=content_series,
        denorm="content",
        c_c=c_c,
        c_s=c_s,
    )
    plot_generated_series(result["denorm"], content_series, style_series)
