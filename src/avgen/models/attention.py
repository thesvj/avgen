"""Attention kernel dispatch: one call site, several backends, no silent math.

Every attention in avgen goes through :func:`attention`, which is a thin
wrapper over :func:`torch.nn.functional.scaled_dot_product_attention`. The
wrapper exists for three reasons, none of them cosmetic.

**Backend choice is architecture-dependent and PyTorch will not make it for
you.** The dispatcher picks the first backend whose constraints the call
satisfies, and its default priority is not the same as the fastest priority on
every generation of hardware. cuDNN's fused attention is the fastest path on
Hopper and Blackwell and is the only one with tuned ``sm_100`` kernels; on
Ampere it has historically been slower than FlashAttention and has shipped
correctness regressions. So the ordering is chosen per architecture, and the
list always ends in ``MATH`` so that an unusual dtype or head dimension
degrades to a correct answer instead of raising.

**A padding mask silently disables the fast kernels.** FlashAttention and the
cuDNN path accept ``is_causal`` and nothing else; hand them an arbitrary
``attn_mask`` and the dispatcher falls back to the memory-efficient or math
backend, which materialises an ``O(seq^2)`` score matrix. At 100k video tokens
that is a 40 GB tensor per head. :func:`key_padding_mask` therefore returns
``None`` when every key is valid, which is the common case for a bucketed
loader, and that ``None`` is what keeps the fast kernel engaged.

**Video streams are frequently empty.** A text-to-video batch carries a
zero-length audio stream, and a kernel handed a zero-length sequence is at best
wasted launch latency and at worst an assertion. The wrapper short-circuits it.

Nothing here touches ``torch.cuda`` at import time: the architecture probe runs
on first use and caches, so the package still imports on a CPU-only machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "attention",
    "attention_backends",
    "device_arch",
    "key_padding_mask",
    "sdpa_context",
]

#: Architecture name by compute-capability floor, highest first. Consumer
#: Blackwell reports ``sm_120`` and shares the ``sm_100`` kernel family, so the
#: table is read as "at least this capability".
_ARCH_BY_CAPABILITY: tuple[tuple[int, str], ...] = (
    (100, "blackwell"),
    (90, "hopper"),
    (89, "ada"),
    (80, "ampere"),
    (0, "legacy"),
)

#: Backend priority per architecture. ``MATH`` is last everywhere and is never
#: removed: it is the only backend that accepts every dtype, head dimension, and
#: mask, and losing it turns an unusual configuration into a hard failure.
_BACKENDS_BY_ARCH: dict[str, tuple[SDPBackend, ...]] = {
    "blackwell": (
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ),
    "hopper": (
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ),
    # Pre-Hopper cuDNN attention is deliberately excluded: it is not faster than
    # FlashAttention there and has shipped silent-wrong-result regressions.
    "ada": (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ),
    "ampere": (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ),
    "legacy": (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH),
    "cpu": (SDPBackend.MATH,),
}

#: Memoised architecture per device index. Populated on first use so that
#: importing avgen never initialises CUDA.
_ARCH_CACHE: dict[int, str] = {}


def device_arch(device: torch.device | str | None = None) -> str:
    """Classify a device into an attention-kernel architecture family.

    Args:
        device: Device to classify. ``None`` means the current default device,
            which is CUDA only if CUDA is both built and available.

    Returns:
        One of ``"blackwell"``, ``"hopper"``, ``"ada"``, ``"ampere"``,
        ``"legacy"``, or ``"cpu"``. ``"cpu"`` is returned for every
        non-CUDA device, including MPS and XPU, because the backend list for
        those is the conservative one either way.
    """
    if device is None:
        available = torch.cuda.is_available()
        resolved = torch.device("cuda" if available else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return "cpu"

    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    cached = _ARCH_CACHE.get(index)
    if cached is not None:
        return cached

    major, minor = torch.cuda.get_device_capability(index)
    capability = major * 10 + minor
    arch = next(name for floor, name in _ARCH_BY_CAPABILITY if capability >= floor)
    _ARCH_CACHE[index] = arch
    return arch


def attention_backends(
    device: torch.device | str | None = None,
    *,
    allow_flash: bool = True,
) -> tuple[SDPBackend, ...]:
    """Return the backend priority list for a device.

    Args:
        device: Device the attention will run on.
        allow_flash: Set false to exclude FlashAttention. Useful when a model
            needs an attention bias that FlashAttention cannot express and you
            would rather not pay for the dispatcher discovering that.

    Returns:
        Backends in descending preference, always ending in a universal
        fallback.
    """
    backends = _BACKENDS_BY_ARCH[device_arch(device)]
    if not allow_flash:
        backends = tuple(b for b in backends if b is not SDPBackend.FLASH_ATTENTION)
    return backends


def sdpa_context(
    device: torch.device | str | None = None,
    *,
    backends: Sequence[SDPBackend] | None = None,
) -> AbstractContextManager[None]:
    """Return a context manager pinning the attention backend priority.

    On CPU this is a no-op. Constraining the CPU dispatcher buys nothing —
    there is effectively one kernel — and an over-narrow list turns a working
    CPU smoke test into a hard error on an older build, which is exactly the
    path that must never break.

    Args:
        device: Device the attention will run on.
        backends: Explicit priority list, overriding the per-architecture
            default.

    Returns:
        A context manager to wrap the attention call in.
    """
    if device_arch(device) == "cpu":
        return nullcontext()
    chosen = tuple(backends) if backends is not None else attention_backends(device)
    return sdpa_kernel(list(chosen))


def key_padding_mask(
    mask: torch.Tensor | None,
    *,
    assume_dense: bool = False,
) -> torch.Tensor | None:
    """Expand a key-validity mask into an SDPA mask, or ``None`` if unneeded.

    Returning ``None`` when every key is valid is not an optimisation detail,
    it is the difference between a fused kernel and an ``O(seq^2)`` score
    matrix: FlashAttention and cuDNN reject an explicit ``attn_mask`` outright,
    so an all-true mask costs the entire long-sequence advantage while changing
    no numbers.

    The all-valid test reads a device tensor into Python, which is one
    host-device synchronisation. Call this **once per forward pass** and reuse
    the result across every block; never call it inside the block loop. A
    loader that guarantees dense batches can pass ``assume_dense=True`` and skip
    the sync entirely.

    Args:
        mask: ``(batch, keys)`` boolean validity, or ``None``.
        assume_dense: Skip the check and assert that every key is valid.

    Returns:
        A ``(batch, 1, 1, keys)`` boolean mask broadcastable over heads and
        queries, or ``None`` when no masking is required.

    Raises:
        ValueError: If ``mask`` is not rank 2.
        TypeError: If ``mask`` is not boolean.
    """
    if mask is None or assume_dense:
        return None
    if mask.ndim != 2:
        raise ValueError(
            f"key mask must have rank 2 (batch, keys); got {tuple(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise TypeError(f"key mask must be bool; got {mask.dtype}")
    if mask.numel() == 0 or bool(mask.all()):
        return None
    return mask[:, None, None, :]


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    enable_gqa: bool = False,
    backends: Sequence[SDPBackend] | None = None,
) -> torch.Tensor:
    """Run scaled dot-product attention on the best available backend.

    Args:
        query: ``(batch, heads, queries, head_dim)``.
        key: ``(batch, kv_heads, keys, head_dim)``.
        value: ``(batch, kv_heads, keys, head_dim)``.
        attn_mask: Broadcastable boolean or additive mask, or ``None``. Prefer
            ``None``; see :func:`key_padding_mask`.
        dropout_p: Attention dropout. Video diffusion transformers train
            without it, so this defaults off.
        is_causal: Whether to apply a causal mask. Diffusion video models are
            bidirectional; this exists for autoregressive-in-time variants.
        enable_gqa: Whether ``key``/``value`` carry fewer heads than ``query``
            and should be broadcast inside the kernel rather than by an
            explicit ``repeat_interleave`` that materialises the full K/V.
        backends: Explicit backend priority, overriding the per-device default.

    Returns:
        ``(batch, heads, queries, head_dim)`` attention output.
    """
    # An absent modality is a zero-length sequence, not a ``None``. Short-circuit
    # rather than launching a kernel on an empty tensor: some backends assert,
    # and the rest waste a launch.
    if query.shape[-2] == 0 or key.shape[-2] == 0:
        return torch.zeros_like(query)

    with sdpa_context(query.device, backends=backends):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
