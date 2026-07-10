"""
wan/modules/attention.py — 跨平台 Flash Attention 封装

支持后端 (按优先级):
    NPU:  mindiesd / torch_npu.npu_fusion_attention → SDPA 回退
    CUDA: Flash Attention 3 → Flash Attention 2 → SDPA 回退
    CPU:  SDPA

环境变量:
    WAN_FA_VERSION=0  强制禁用 FA, 使用 SDPA
    WAN_FA_VERSION=2  强制使用 FA2 (CUDA only)
    WAN_FA_VERSION=3  强制使用 FA3 (CUDA only)
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

__all__ = [
    "flash_attention",
    "attention",
    "get_attention_backend",
]

# ── 模块级后端检测 ──────────────────────────────────
_FA3_AVAILABLE = False
_FA2_AVAILABLE = False
_MINDIESD_AVAILABLE = False
_NPU_FA_AVAILABLE = False
_ATTN_BACKEND: str = "sdpa"  # fa3 / fa2 / mindiesd / npu_fa / sdpa
_USER_FA_VERSION = os.getenv("WAN_FA_VERSION", "")


def _detect_backends() -> None:
    """一次性检测所有可用的 attention 后端 (结果缓存在模块全局变量)."""
    global _FA3_AVAILABLE, _FA2_AVAILABLE, _MINDIESD_AVAILABLE
    global _NPU_FA_AVAILABLE, _ATTN_BACKEND

    if _ATTN_BACKEND != "sdpa":
        return  # 已检测过

    # ── 用户显式禁用 FA ──
    if _USER_FA_VERSION == "0":
        _ATTN_BACKEND = "sdpa"
        return

    # ── 检测 NPU ──
    try:
        import torch_npu
        _npu_ok = torch_npu.npu.is_available()
    except (ImportError, AssertionError, RuntimeError):
        _npu_ok = False

    if _npu_ok:
        # mindiesd: Ascend 官方 flash attention 库
        try:
            import importlib.util
            if importlib.util.find_spec("mindiesd"):
                import mindiesd  # noqa: F401
                _MINDIESD_AVAILABLE = True
        except (ImportError, ModuleNotFoundError):
            pass

        # torch_npu 内置融合注意力 (CANN 新版)
        if not _MINDIESD_AVAILABLE:
            try:
                if hasattr(torch_npu, "npu_fusion_attention"):
                    _NPU_FA_AVAILABLE = True
            except Exception:
                pass

        if _MINDIESD_AVAILABLE:
            _ATTN_BACKEND = "mindiesd"
        elif _NPU_FA_AVAILABLE:
            _ATTN_BACKEND = "npu_fa"
        else:
            _ATTN_BACKEND = "sdpa"
        return

    # ── 检测 CUDA ──
    try:
        _cuda_ok = torch.cuda.is_available()
    except (AssertionError, RuntimeError):
        _cuda_ok = False

    if not _cuda_ok:
        _ATTN_BACKEND = "sdpa"
        return

    # FA3
    if _USER_FA_VERSION in ("", "3"):
        try:
            import flash_attn_interface
            _FA3_AVAILABLE = True
            _ATTN_BACKEND = "fa3"
            return
        except (ImportError, ModuleNotFoundError):
            pass

    # FA2
    if _USER_FA_VERSION in ("", "2", "3"):
        try:
            import flash_attn
            _FA2_AVAILABLE = True
            _ATTN_BACKEND = "fa2"
            return
        except (ImportError, ModuleNotFoundError):
            pass

    _ATTN_BACKEND = "sdpa"


def get_attention_backend() -> str:
    """返回当前使用的 attention 后端名称."""
    _detect_backends()
    return _ATTN_BACKEND


# ── NPU Flash Attention 实现 ────────────────────────

def _npu_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Ascend NPU 上的 flash attention.

    优先使用 mindiesd, 备选 torch_npu.npu_fusion_attention.
    """
    _detect_backends()

    b, lq, nq, c1 = q.shape
    _, lk, nk, c2 = v.shape
    head_dim = c1

    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # mindiesd 路径
    if _MINDIESD_AVAILABLE:
        import mindiesd
        # mindiesd 使用与 flash_attn 兼容的接口
        return mindiesd.flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    # torch_npu.npu_fusion_attention 路径
    if _NPU_FA_AVAILABLE:
        import torch_npu
        # NPU 融合注意力需要 [B, N, S, D] 格式
        q_npu = q.transpose(1, 2).contiguous()  # [B, Nq, Lq, C1]
        k_npu = k.transpose(1, 2).contiguous()  # [B, Nk, Lk, C1]
        v_npu = v.transpose(1, 2).contiguous()  # [B, Nk, Lk, C2]

        out = torch_npu.npu_fusion_attention(
            q_npu, k_npu, v_npu,
            head_num=nq,
            input_layout="BNSD",
            scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous()  # back to [B, Lq, Nq, C2]

    raise RuntimeError("NPU FA not available (should not reach here)")


# ── 通用 flash_attention 函数 ───────────────────────

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: str | None = None,
) -> torch.Tensor:
    """
    跨平台 flash attention.

    Args:
        q: [B, Lq, Nq, C1]
        k: [B, Lk, Nk, C1]
        v: [B, Lk, Nk, C2]
        ... (其他参数与原始 flash_attn 接口兼容)
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes, f"dtype must be one of {half_dtypes}, got {dtype}"

    _detect_backends()
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # varlen 预处理
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32,
                              device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32,
                              device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # ── NPU 路径 ──
    if _ATTN_BACKEND in ("mindiesd", "npu_fa"):
        # NPU FA 目前不支持 varlen cu_seqlens, 所以 reshape 回 dense
        q_dense = q.reshape(b, lq, -1, q.size(-1))
        k_dense = k.reshape(b, lk, -1, k.size(-1))
        v_dense = v.reshape(b, lk, -1, v.size(-1))
        x = _npu_flash_attention(
            q_dense, k_dense, v_dense,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        return x.reshape(b * lq, -1, x.size(-1)).unflatten(0, (b, lq)).type(out_dtype)

    # ── FA3 路径 ──
    if _ATTN_BACKEND == "fa3":
        import flash_attn_interface
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
        return x.type(out_dtype)

    # ── FA2 路径 ──
    if _ATTN_BACKEND == "fa2":
        import flash_attn
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
        return x.type(out_dtype)

    raise RuntimeError(f"Unknown attention backend: {_ATTN_BACKEND}")


# ── 高层 attention 函数 (含回退逻辑) ────────────────

_WARNED_FA_DISABLED = False


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: str | None = None,
) -> torch.Tensor:
    """高层 attention: 自动选择 FA3/FA2/mindiesd/SDPA.

    当用户指定 version='0' 或没有可用 FA 后端时, 回退到 PyTorch SDPA.
    """
    global _WARNED_FA_DISABLED
    _detect_backends()

    # 用户显式禁用 FA
    if version == '0' or _USER_FA_VERSION == '0':
        if version == '0' and not _WARNED_FA_DISABLED:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                _logger.info("Flash Attention DISABLED by user, using SDPA.")
            _WARNED_FA_DISABLED = True

    # 有硬件加速且未禁用
    elif _ATTN_BACKEND not in ("sdpa",):
        try:
            return flash_attention(
                q=q, k=k, v=v,
                q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=version,
            )
        except Exception as e:
            _logger.warning("FA call failed (%s), falling back to SDPA: %s",
                           _ATTN_BACKEND, e)

    # ── SDPA 回退路径 ──
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention. "
            "It can have a significant impact on performance."
        )

    attn_mask = None

    if q.dim() == 3:
        q_sdpa = q.unsqueeze(0).transpose(1, 2).to(dtype)
        k_sdpa = k.unsqueeze(0).transpose(1, 2).to(dtype)
        v_sdpa = v.unsqueeze(0).transpose(1, 2).to(dtype)
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).squeeze(0).contiguous()
    else:
        q_sdpa = q.transpose(1, 2).to(dtype)
        k_sdpa = k.transpose(1, 2).to(dtype)
        v_sdpa = v.transpose(1, 2).to(dtype)
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous()

    return out
