"""GPU optimization utilities for maximum throughput.

Enables Flash Attention, TF32, AMP bf16, and other CUDA optimizations.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def enable_gpu_optimizations(
    enable_tf32: bool = True,
    enable_flash_attention: bool = True,
    enable_cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
) -> dict:
    """Enable GPU optimizations for maximum throughput.

    Args:
        enable_tf32: Enable TF32 for faster matmul/convolutions on Ampere+ GPUs
        enable_flash_attention: Enable Flash Attention 2 via environment variable
        enable_cudnn_benchmark: Enable cuDNN autotuning for faster convolutions
        matmul_precision: Set matrix multiplication precision

    Returns:
        Dict with optimization settings applied
    """
    settings = {}

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU optimizations")
        return settings

    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    logger.info(f"GPU: {device_name} (compute capability {compute_capability[0]}.{compute_capability[1]})")

    # Enable TF32 for Ampere+ GPUs (compute capability >= 8.0)
    if enable_tf32 and compute_capability[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings["tf32"] = True
        logger.info("Enabled TF32 for matmul and cuDNN")
    else:
        settings["tf32"] = False

    torch.set_float32_matmul_precision(matmul_precision)
    settings["matmul_precision"] = matmul_precision
    logger.info(f"Set matmul precision to '{matmul_precision}'")

    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        settings["cudnn_benchmark"] = True
        logger.info("Enabled cuDNN benchmark mode")

    if enable_flash_attention:
        os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "1"
        os.environ["FLASH_ATTENTION_FORCE_BUILD"] = "FALSE"
        settings["flash_attention"] = True
        logger.info("Enabled Flash Attention 2 via environment variable")

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        settings["sdpa"] = True
        logger.info("Enabled PyTorch 2.0 SDPA (Scaled Dot Product Attention)")

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"Total GPU memory: {total_memory:.1f} GB")

    return settings


def get_optimal_dtype() -> torch.dtype:
    """Get optimal dtype for current GPU.

    Returns:
        torch.bfloat16 for Ampere+ GPUs, torch.float16 otherwise
    """
    if not torch.cuda.is_available():
        return torch.float32

    compute_capability = torch.cuda.get_device_capability(0)

    if compute_capability[0] >= 8:
        return torch.bfloat16
    else:
        return torch.float16


def create_autocast_context(dtype: Optional[torch.dtype] = None):
    """Create an autocast context for mixed precision inference.

    Args:
        dtype: Override dtype (default: auto-detect optimal)

    Returns:
        torch.cuda.amp.autocast context manager
    """
    if dtype is None:
        dtype = get_optimal_dtype()

    return torch.cuda.amp.autocast(dtype=dtype)


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Apply inference optimizations to a model.

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model
    """
    model.train(False)

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile with reduce-overhead mode")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    return model


_initialized = False


def auto_init():
    """Auto-initialize GPU optimizations on first call."""
    global _initialized
    if not _initialized and torch.cuda.is_available():
        enable_gpu_optimizations()
        _initialized = True


def get_device(device: Optional[str] = None) -> str:
    """Get device string with automatic detection.

    Args:
        device: Explicit device string (cuda, cpu, etc.)
                If None, auto-detects: cuda if available, else cpu

    Returns:
        Device string (e.g., "cuda", "cpu", "cuda:0")
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_device(device: Optional[str] = None) -> torch.device:
    """Get torch.device object with automatic detection.

    Args:
        device: Explicit device string

    Returns:
        torch.device object
    """
    return torch.device(get_device(device))


def move_to_device(tensor_or_model, device: Optional[str] = None):
    """Move tensor or model to device with auto-detection.

    Args:
        tensor_or_model: PyTorch tensor or model
        device: Target device (auto-detected if None)

    Returns:
        Moved tensor or model
    """
    target_device = get_device(device)
    return tensor_or_model.to(target_device)


def get_optimal_config() -> dict:
    """Get optimal configuration for current hardware.

    Returns:
        Dict with device, dtype, and other optimal settings
    """
    config = {
        "device": get_device(),
        "dtype": get_optimal_dtype(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        config["compute_capability"] = torch.cuda.get_device_capability(0)
        config["gpu_name"] = torch.cuda.get_device_name(0)
        config["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        config["supports_bf16"] = torch.cuda.is_bf16_supported()

    return config
