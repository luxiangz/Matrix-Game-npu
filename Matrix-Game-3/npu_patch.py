"""
npu_patch.py — Matrix-Game-3.0  NPU (Ascend) 适配补丁

在 generate.py 的最开头导入此模块即可完成 NPU 适配。
用法:
    import npu_patch   # 必须在所有其他 torch 导入之前

环境变量控制:
    WAN_FA_VERSION=0      禁用 Flash Attention (NPU 不支持)
    WAN_DISABLE_INT8=1    禁用 Triton INT8 量化
    WAN_DISABLE_COMPILE=1 禁用 torch.compile
"""

import os
import sys
import functools
import logging

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 0. 环境变量预设 (在其他模块导入前生效)
# ──────────────────────────────────────────────────────────
os.environ.setdefault("WAN_FA_VERSION", "0")
os.environ.setdefault("WAN_DISABLE_INT8", "1")
os.environ.setdefault("WAN_DISABLE_COMPILE", "1")


# ──────────────────────────────────────────────────────────
# 1. 检测并导入 torch_npu
# ──────────────────────────────────────────────────────────
_HAS_NPU = False
try:
    import torch_npu
    _HAS_NPU = torch_npu.npu.is_available()
    _NPU_COUNT = torch_npu.npu.device_count() if _HAS_NPU else 0
    _logger.info(f"NPU detected: {_NPU_COUNT} device(s)")
except ImportError:
    torch_npu = None
    _logger.warning("torch_npu not installed. Running on CPU/CUDA fallback.")


# ──────────────────────────────────────────────────────────
# 2. CUDA → NPU 设备映射
# ──────────────────────────────────────────────────────────
_ORIGINAL_CUDA_FUNCS = {}


def _patch_torch_cuda():
    """将 torch.cuda.* 常用函数重定向到 torch_npu"""
    import torch

    if not _HAS_NPU:
        return

    # 保存原始函数引用
    for name in ["current_device", "device_count", "set_device",
                 "synchronize", "Stream", "Event", "set_stream"]:
        if hasattr(torch.cuda, name):
            _ORIGINAL_CUDA_FUNCS[name] = getattr(torch.cuda, name)

    # 替换
    torch.cuda.current_device = torch_npu.npu.current_device
    torch.cuda.device_count = torch_npu.npu.device_count
    torch.cuda.set_device = torch_npu.npu.set_device
    torch.cuda.synchronize = torch_npu.npu.synchronize
    torch.cuda.Stream = torch_npu.npu.Stream
    torch.cuda.Event = torch_npu.npu.Event

    # 设备名称映射
    _original_torch_device = torch.device

    @functools.wraps(_original_torch_device)
    def _patched_device(arg, *args, **kwargs):
        if isinstance(arg, str) and arg.startswith("cuda"):
            arg = arg.replace("cuda", "npu")
        return _original_torch_device(arg, *args, **kwargs)

    torch.device = _patched_device

    _logger.info("torch.cuda → torch_npu device mapping applied.")


def _restore_torch_cuda():
    """恢复 CUDA 映射 (仅用于测试)"""
    import torch
    for name, func in _ORIGINAL_CUDA_FUNCS.items():
        setattr(torch.cuda, name, func)
    _logger.info("torch.cuda mapping restored.")


# ──────────────────────────────────────────────────────────
# 3. Flash Attention 补丁
# ──────────────────────────────────────────────────────────
def _patch_flash_attention():
    """NPU 不支持 flash_attn, 确保回退策略生效"""
    import importlib

    # 标记模块为不可用
    for mod_name in ["flash_attn", "flash_attn_interface"]:
        try:
            importlib.import_module(mod_name)
        except (ImportError, ModuleNotFoundError):
            pass  # 预期行为

    _logger.info("Flash Attention disabled (FA version=0, using SDPA fallback).")


# ──────────────────────────────────────────────────────────
# 4. Triton 内核补丁
# ──────────────────────────────────────────────────────────
def _patch_triton_kernels():
    """禁用 Triton INT8 GEMM 内核"""
    # 预先设置环境变量, 让 wan/modules/model.py 中的 _get_triton_kernels() 返回 None
    os.environ["WAN_TRITON_KERNELS_PATH"] = ""  # 触发 FileNotFoundError → 优雅降级
    _logger.info("Triton INT8 kernels disabled (NPU unsupported).")


# ──────────────────────────────────────────────────────────
# 5. torch.compile 补丁
# ──────────────────────────────────────────────────────────
_ORIGINAL_COMPILE = None


def _patch_torch_compile():
    """torch.compile 默认 CUDA 后端, NPU 上需用 no-op"""
    global _ORIGINAL_COMPILE
    import torch

    if os.environ.get("WAN_DISABLE_COMPILE") == "1":
        if hasattr(torch, "compile"):
            _ORIGINAL_COMPILE = torch.compile

            @functools.wraps(torch.compile)
            def _noop_compile(model, *args, **kwargs):
                _logger.info("torch.compile skipped (NPU compat mode).")
                return model

            torch.compile = _noop_compile
            _logger.info("torch.compile patched to no-op.")


def _restore_torch_compile():
    """恢复 torch.compile"""
    import torch
    if _ORIGINAL_COMPILE is not None:
        torch.compile = _ORIGINAL_COMPILE


# ──────────────────────────────────────────────────────────
# 6. 分布式 HCCL 初始化
# ──────────────────────────────────────────────────────────
def init_npu_distributed(rank, world_size, local_rank,
                         master_addr="127.0.0.1", master_port=29500):
    """包装 NPU 上的分布式初始化 (HCCL 后端)"""
    import torch
    import torch.distributed as dist

    if not _HAS_NPU:
        raise RuntimeError("NPU not available for distributed init.")

    torch_npu.npu.set_device(local_rank)

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))

    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    _logger.info(f"NPU distributed initialized: rank={rank}, world={world_size}")


# ──────────────────────────────────────────────────────────
# 7. 工具函数
# ──────────────────────────────────────────────────────────
def get_device(device_id=0):
    """获取正确的设备对象"""
    if _HAS_NPU:
        return torch_npu.npu.device(device_id) if hasattr(torch_npu.npu, 'device') else \
               __import__('torch').device(f"npu:{device_id}")
    else:
        return __import__('torch').device(f"cuda:{device_id}")


def get_device_str(device_id=0):
    """获取设备字符串"""
    return "npu" if _HAS_NPU else "cuda"


def sync_device():
    """同步设备"""
    if _HAS_NPU:
        torch_npu.npu.synchronize()
    else:
        import torch
        torch.cuda.synchronize()


def print_env_info():
    """打印环境信息"""
    info_lines = ["=" * 50, "Matrix-Game-3.0 NPU Environment Info", "=" * 50]
    if _HAS_NPU:
        info_lines.append(f"  NPU count:     {_NPU_COUNT}")
        info_lines.append(f"  NPU name:      {torch_npu.npu.get_device_name(0)}")
        info_lines.append(f"  torch_npu ver: {torch_npu.__version__}")
        info_lines.append(f"  CANN ver:      {getattr(torch_npu, '_CANN_VERSION', 'unknown')}")
    else:
        info_lines.append("  NPU:           NOT AVAILABLE")
    info_lines.append(f"  FA version:    {os.environ.get('WAN_FA_VERSION', 'unset')}")
    info_lines.append(f"  Disable INT8:  {os.environ.get('WAN_DISABLE_INT8', 'unset')}")
    info_lines.append(f"  Disable Comp:  {os.environ.get('WAN_DISABLE_COMPILE', 'unset')}")
    info_lines.append("=" * 50)
    print("\n".join(info_lines), flush=True)


# ──────────────────────────────────────────────────────────
# 8. 自动执行补丁
# ──────────────────────────────────────────────────────────
_patch_torch_cuda()
_patch_flash_attention()
_patch_triton_kernels()
_patch_torch_compile()
