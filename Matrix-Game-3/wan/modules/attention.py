import math
import torch
try:
    import flash_attn_interface
    import os
    WAN_FA_VERSION = os.getenv("WAN_FA_VERSION", "3")
    if WAN_FA_VERSION == "3":
        FLASH_ATTN_3_AVAILABLE = True
    else:
        FLASH_ATTN_3_AVAILABLE = False
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# ── NPU mindiesd 检测 (参照 vllm-omni FlashAttentionImpl.forward_fa_npu) ──
_MINDIESD_AVAILABLE = False
try:
    from mindiesd import attention_forward as _mindiesd_attention_forward
    _MINDIESD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

# ── NPU npu_fusion_attention 检测 (Ascend 原生 FA) ──
_NPU_FA_AVAILABLE = False
try:
    import torch_npu
    if hasattr(torch_npu, 'npu_fusion_attention'):
        _npu_fusion_attention = torch_npu.npu_fusion_attention
        _NPU_FA_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AssertionError):
    pass

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'FLASH_ATTN_3_AVAILABLE',
    'FLASH_ATTN_2_AVAILABLE',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    ...
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type in ('cuda', 'npu') and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    # ── 无 FA 可用时，直接用 SDPA (在 varlen 转换前处理, 避免 shape 问题) ──
    _fa_available = FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE  # NVIDIA only
    _fa_disabled = (version == '0')

    if _fa_disabled or not _fa_available:
        # NPU native FA path (torch_npu.npu_fusion_attention, dense → BSND)
        if _NPU_FA_AVAILABLE and not _fa_disabled:
            try:
                head_num = q.size(2)
                scale = softmax_scale or (1.0 / math.sqrt(q.size(-1)))
                q_bsnd = q.transpose(1, 2).contiguous()
                k_bsnd = k.transpose(1, 2).contiguous()
                v_bsnd = v.transpose(1, 2).contiguous()
                x = _npu_fusion_attention(
                    q_bsnd, k_bsnd, v_bsnd,
                    head_num,
                    "BSND",
                    scale=scale,
                    keep_prob=1.0,
                )[0]
                return x.transpose(1, 2).contiguous().type(out_dtype)
            except Exception:
                pass  # npu_fusion_attention failed, fall through

        # NPU mindiesd path (dense → BNSD → mindiesd.attention_forward)
        if _MINDIESD_AVAILABLE and not _fa_disabled:
            try:
                q_bnsd = q.transpose(1, 2).contiguous()
                k_bnsd = k.transpose(1, 2).contiguous()
                v_bnsd = v.transpose(1, 2).contiguous()
                x = _mindiesd_attention_forward(
                    q_bnsd, k_bnsd, v_bnsd,
                    attn_mask=None,
                    opt_mode="manual",
                    op_type="fused_attn_score",
                    layout="BNSD",
                )
                return x.transpose(1, 2).contiguous().type(out_dtype)
            except Exception:
                pass  # mindiesd failed, fall through to SDPA

        # SDPA fallback — 直接在原始 dense 输入上运行
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using SDPA. '
                'It can have a significant impact on performance.'
            )

        if q.dim() == 3:
            q_sdpa = q.unsqueeze(0).transpose(1, 2).to(dtype)
            k_sdpa = k.unsqueeze(0).transpose(1, 2).to(dtype)
            v_sdpa = v.unsqueeze(0).transpose(1, 2).to(dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=causal, dropout_p=dropout_p)
            out = out.transpose(1, 2).squeeze(0).contiguous()
        else:
            q_sdpa = q.transpose(1, 2).to(dtype)
            k_sdpa = k.transpose(1, 2).to(dtype)
            v_sdpa = v.transpose(1, 2).to(dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=causal, dropout_p=dropout_p)
            out = out.transpose(1, 2).contiguous()

        return out.type(out_dtype)

    # ── 以下是原始 FA3/FA2 路径 (varlen 格式), 完全不变 ──

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if (version is None or version == 3 or version == "3") and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
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
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
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
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)


_WARNED_FA_DISABLED = False


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    global _WARNED_FA_DISABLED
    if version != '0' and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )
    else:
        if version == '0':
            if not _WARNED_FA_DISABLED:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print('DEBUG: Flash attention is DISABLED by user, using scaled_dot_product_attention (SDPA).')
                _WARNED_FA_DISABLED = True
        elif q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )

        attn_mask = None

        if q.dim() == 3:
            q = q.unsqueeze(0).transpose(1, 2).to(dtype)
            k = k.unsqueeze(0).transpose(1, 2).to(dtype)
            v = v.unsqueeze(0).transpose(1, 2).to(dtype)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

            out = out.transpose(1, 2).squeeze(0).contiguous()
        else:
            q = q.transpose(1, 2).to(dtype)
            k = k.transpose(1, 2).to(dtype)
            v = v.transpose(1, 2).to(dtype)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

            out = out.transpose(1, 2).contiguous()

        return out
