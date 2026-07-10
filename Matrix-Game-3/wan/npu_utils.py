"""
wan/npu_utils.py — 集中式 NPU 设备管理 (参照 vllm-omni NPUOmniPlatform 模式)

本模块负责:
  1. 安全检测 NPU/CUDA/CPU 后端
  2. 统一设备创建和管理
  3. 分布式后端选择 (hccl/nccl/gloo)
  4. NPU 一次性初始化 (allow_internal_format 等)
  5. Attention 后端检测 (mindiesd / flash_attn / sdpa)

用法:
    from wan.npu_utils import get_device, is_npu_available, synchronize
    device = get_device(0)
"""

from __future__ import annotations

import importlib.util
import logging
import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)

# ── 缓存 ────────────────────────────────────────────
_NPU_AVAILABLE: bool | None = None        # None = 未检测
_NPU_COUNT: int = 0
_DEVICE_TYPE: str = "cpu"                 # "npu" / "cuda" / "cpu"
_INITIALIZED: bool = False

# Attention 后端枚举值
ATTN_BACKEND_NONE = "none"           # 未检测到任何加速后端
ATTN_BACKEND_MINDIESD = "mindiesd"   # Ascend flash attention (torch_npu)
ATTN_BACKEND_FA3 = "fa3"             # NVIDIA Flash Attention 3
ATTN_BACKEND_FA2 = "fa2"             # NVIDIA Flash Attention 2
ATTN_BACKEND_SDPA = "sdpa"           # PyTorch SDPA (通用回退)
_ATTN_BACKEND: str | None = None     # None = 未检测


# ═══════════════════════════════════════════════════
# 1. 后端检测
# ═══════════════════════════════════════════════════

def _detect_backend() -> tuple[bool, int, str]:
    """一次检测 NPU / CUDA / CPU, 缓存结果."""
    global _NPU_AVAILABLE, _NPU_COUNT, _DEVICE_TYPE

    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE, _NPU_COUNT, _DEVICE_TYPE

    # 1) 尝试 NPU
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            _NPU_AVAILABLE = True
            _NPU_COUNT = torch_npu.npu.device_count()
            _DEVICE_TYPE = "npu"
            _logger.info("Backend: NPU (Ascend), %d device(s)", _NPU_COUNT)
            return _NPU_AVAILABLE, _NPU_COUNT, _DEVICE_TYPE
    except ImportError:
        pass
    except (AssertionError, RuntimeError) as e:
        _logger.debug("NPU detection ignored: %s", e)

    # 2) 尝试 CUDA
    try:
        if torch.cuda.is_available():
            _NPU_AVAILABLE = False
            _NPU_COUNT = torch.cuda.device_count()
            _DEVICE_TYPE = "cuda"
            _logger.info("Backend: CUDA, %d device(s)", _NPU_COUNT)
            return _NPU_AVAILABLE, _NPU_COUNT, _DEVICE_TYPE
    except (AssertionError, RuntimeError) as e:
        _logger.debug("CUDA detection ignored: %s", e)

    # 3) CPU
    _NPU_AVAILABLE = False
    _NPU_COUNT = 0
    _DEVICE_TYPE = "cpu"
    _logger.info("Backend: CPU (no accelerator)")
    return _NPU_AVAILABLE, _NPU_COUNT, _DEVICE_TYPE


def is_npu_available() -> bool:
    """NPU 是否可用."""
    result, _, _ = _detect_backend()
    return result


def is_cuda_available() -> bool:
    """CUDA 是否可用 (且 NPU 不可用时)."""
    _, _, dtype = _detect_backend()
    return dtype == "cuda"


def get_device_type() -> str:
    """返回 "npu" / "cuda" / "cpu"."""
    _, _, dtype = _detect_backend()
    return dtype


# ═══════════════════════════════════════════════════
# 2. 设备管理
# ═══════════════════════════════════════════════════

def get_device(device_id: int | None = None) -> torch.device:
    """统一获取设备对象.

    Args:
        device_id: 设备编号. None 时返回不带 index 的设备.

    Returns:
        torch.device("npu:0") / torch.device("cuda:0") / torch.device("cpu")
    """
    _detect_backend()
    if _DEVICE_TYPE == "npu":
        if device_id is not None:
            return torch.device("npu", device_id)
        return torch.device("npu")
    elif _DEVICE_TYPE == "cuda":
        if device_id is not None:
            return torch.device("cuda", device_id)
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_count() -> int:
    """可用加速器数量."""
    _, count, _ = _detect_backend()
    return count


def get_device_name(device_id: int = 0) -> str:
    """获取设备名称."""
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        import torch_npu
        return torch_npu.npu.get_device_name(device_id)
    elif dtype == "cuda":
        return torch.cuda.get_device_name(device_id)
    else:
        return "cpu"


def set_device(device_id: int) -> None:
    """设置当前设备."""
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        import torch_npu
        torch_npu.npu.set_device(device_id)
    elif dtype == "cuda":
        torch.cuda.set_device(device_id)


def synchronize() -> None:
    """同步设备."""
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        import torch_npu
        torch_npu.npu.synchronize()
    elif dtype == "cuda":
        torch.cuda.synchronize()


def empty_cache() -> None:
    """清空设备缓存."""
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        import torch_npu
        torch_npu.npu.empty_cache()
    elif dtype == "cuda":
        torch.cuda.empty_cache()


def get_free_memory(device: torch.device | None = None) -> int:
    """获取设备空闲内存 (bytes)."""
    if device is None:
        device = get_device(0)
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        free, _ = torch.npu.mem_get_info(device)
        return free
    elif dtype == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return free
    return 0


def get_total_memory(device_id: int = 0) -> int:
    """获取设备总内存 (bytes)."""
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        props = torch.npu.get_device_properties(device_id)
        return props.total_memory
    elif dtype == "cuda":
        props = torch.cuda.get_device_properties(device_id)
        return props.total_memory
    return 0


# ═══════════════════════════════════════════════════
# 3. 分布式
# ═══════════════════════════════════════════════════

def get_dist_backend() -> str:
    """返回分布式后端: "hccl" / "nccl" / "gloo"."""
    _, _, dtype = _detect_backend()
    if dtype == "npu":
        return "hccl"
    elif dtype == "cuda":
        return "nccl"
    else:
        return "gloo"


def init_distributed(rank: int, world_size: int, local_rank: int,
                     master_addr: str = "127.0.0.1",
                     master_port: int = 29500) -> None:
    """初始化分布式进程组 (自动选择后端)."""
    set_device(local_rank)

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))

    backend = get_dist_backend()
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    _logger.info("Distributed initialized: backend=%s rank=%d world=%d",
                 backend, rank, world_size)


# ═══════════════════════════════════════════════════
# 4. NPU 一次性初始化
# ═══════════════════════════════════════════════════

def init_npu() -> None:
    """NPU 环境一次性初始化.

    在 import 完 torch 之后、加载模型之前调用.
    幂等 — 多次调用只执行一次.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    if not is_npu_available():
        _INITIALIZED = True
        return

    import torch_npu

    # FRACTAL_NZ 内部格式 — Ascend 量化权重优化
    if hasattr(torch.npu.config, "allow_internal_format"):
        torch.npu.config.allow_internal_format = True

    _INITIALIZED = True
    _logger.info("NPU initialized (allow_internal_format enabled)")


# ═══════════════════════════════════════════════════
# 5. Autocast
# ═══════════════════════════════════════════════════

def get_autocast_context(device_type: str | None = None,
                         dtype: torch.dtype = torch.bfloat16,
                         enabled: bool = True):
    """获取平台适配的 autocast context.

    Args:
        device_type: "npu" / "cuda". None 时自动检测.
        dtype: 目标 dtype (默认 bfloat16).
        enabled: 是否启用 autocast.

    Returns:
        Context manager (autocast 或 nullcontext).
    """
    if not enabled:
        return nullcontext()

    if device_type is None:
        device_type = get_device_type()

    if device_type == "npu":
        try:
            return torch.npu.amp.autocast(dtype=dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            _logger.warning("NPU autocast unavailable: %s", exc)
            return nullcontext()
    elif device_type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return nullcontext()


# ═══════════════════════════════════════════════════
# 6. Attention 后端检测 (参照 vllm-omni 的 DiffusionAttentionBackendEnum)
# ═══════════════════════════════════════════════════

def _detect_attention_backend() -> str:
    """检测最佳可用的 attention 后端, 结果缓存."""
    global _ATTN_BACKEND
    if _ATTN_BACKEND is not None:
        return _ATTN_BACKEND

    # 用户显式禁用
    fa_env = os.getenv("WAN_FA_VERSION", "")
    if fa_env == "0":
        _ATTN_BACKEND = ATTN_BACKEND_SDPA
        _logger.info("Attention backend: SDPA (user disabled FA via WAN_FA_VERSION=0)")
        return _ATTN_BACKEND

    device_type = get_device_type()

    if device_type == "npu":
        # Ascend: mindiesd (参照 vllm-omni)
        if importlib.util.find_spec("mindiesd"):
            try:
                from mindiesd import attention_forward  # noqa: F401
                _ATTN_BACKEND = ATTN_BACKEND_MINDIESD
                _logger.info("Attention backend: mindiesd")
                return _ATTN_BACKEND
            except ImportError:
                pass

        _ATTN_BACKEND = ATTN_BACKEND_SDPA
        _logger.info("Attention backend: SDPA (mindiesd not available)")
        return _ATTN_BACKEND

    elif device_type == "cuda":
        # NVIDIA: Flash Attention 3 > 2 > SDPA
        try:
            import flash_attn_interface
            _ATTN_BACKEND = ATTN_BACKEND_FA3
            _logger.info("Attention backend: Flash Attention 3")
            return _ATTN_BACKEND
        except ImportError:
            pass

        try:
            import flash_attn
            _ATTN_BACKEND = ATTN_BACKEND_FA2
            _logger.info("Attention backend: Flash Attention 2")
            return _ATTN_BACKEND
        except ImportError:
            pass

        _ATTN_BACKEND = ATTN_BACKEND_SDPA
        _logger.info("Attention backend: SDPA (no FA available)")
        return _ATTN_BACKEND

    else:
        # CPU
        _ATTN_BACKEND = ATTN_BACKEND_SDPA
        return _ATTN_BACKEND


def get_attention_backend() -> str:
    """获取当前 attention 后端."""
    return _detect_attention_backend()


def has_fast_attention() -> bool:
    """是否有硬件加速 attention (FA3/FA2/mindiesd)."""
    backend = _detect_attention_backend()
    return backend in (ATTN_BACKEND_MINDIESD, ATTN_BACKEND_FA3, ATTN_BACKEND_FA2)


# ═══════════════════════════════════════════════════
# 7. 环境信息
# ═══════════════════════════════════════════════════

def print_env_info() -> None:
    """打印 NPU/CUDA 环境信息."""
    _detect_backend()
    _detect_attention_backend()

    lines = [
        "=" * 55,
        "  Matrix-Game-3.0  Device Environment",
        "=" * 55,
        f"  Device type:     {_DEVICE_TYPE}",
        f"  Device count:    {_NPU_COUNT}",
        f"  Device name:     {get_device_name(0) if _NPU_COUNT > 0 else 'N/A'}",
        f"  Dist backend:    {get_dist_backend()}",
        f"  Attention:       {_ATTN_BACKEND}",
        f"  NPU init:        {_INITIALIZED}",
        "=" * 55,
    ]
    print("\n".join(lines), flush=True)
