# Matrix-Game-3.0 NPU 部署指南

## 目录
- [1. 概述](#1-概述)
- [2. 环境准备](#2-环境准备)
- [3. 权重准备](#3-权重准备)
- [4. NPU 适配改造](#4-npu-适配改造)
- [5. 单卡推理](#5-单卡推理)
- [6. 多卡推理](#6-多卡推理)
- [7. 常用参数说明](#7-常用参数说明)
- [8. 常见问题排查](#8-常见问题排查)

---

## 1. 概述

**Matrix-Game-3.0** 是 Skywork AI 开源的记忆增强交互式世界模型，基于 Wan2.2 基础模型构建，支持 720p 实时长视频生成。

### 模型规格
| 项目 | 规格 |
|------|------|
| 模型参数量 | 5B (DiT: dim=5120, 40层, 40头) |
| 推理精度 | bfloat16 |
| 分辨率 | 704×1280 (720p) |
| 帧率 | 16 FPS |
| 文本编码器 | umT5-XXL |
| VAE | Wan2.2 VAE / MG-LightVAE |

### 权重文件清单

权重目录 `/data1/weights/Matrix-Game-3.0` 应包含以下文件：

```
/data1/weights/Matrix-Game-3.0/
├── base_model/                    # 基础模型 (50步推理)
│   └── (DiT 模型权重文件)
├── base_distilled_model/          # 蒸馏模型 (3步推理, 推荐)
│   └── (DiT 蒸馏权重文件)
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 文本编码器
├── google/umt5-xxl/                   # T5 tokenizer (或自动下载)
├── Wan2.2_VAE.pth                     # Wan2.2 VAE 权重
├── MG-LightVAE.pth                    # MG-LightVAE 权重 (剪枝率50%)
└── MG-LightVAE_v2.pth                 # MG-LightVAE v2 权重 (剪枝率75%)
```

---

## 2. 环境准备

### 2.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| NPU | 1× Ascend 910B (64GB) | 8× Ascend 910B |
| 内存 | 64 GB | 128 GB |
| 操作系统 | Linux (openEuler/Ubuntu) | openEuler 22.03+ |

### 2.2 安装 CANN 与 PyTorch

```bash
# 1. 安装 CANN (Ascend Computing)  toolkit
# 请从华为官网下载对应版本的 Ascend-cann-toolkit
# 参考: https://www.hiascend.com/software/cann

# 2. 安装 torch_npu
pip install torch-npu==2.10.0

# 3. 安装 torchvision (NPU 兼容版本)
pip install torchvision==0.25.0

# 4. 验证 NPU 可用
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available()); print(torch_npu.npu.device_count())"
```

### 2.3 创建虚拟环境

```bash
conda create -n matrix-game-3.0 python=3.12 -y
conda activate matrix-game-3.0
```

### 2.4 安装依赖

```bash
# 进入项目目录
cd /path/to/Matrix-Game-3

# 安装基础依赖
pip install -r requirements.txt

# NPU 环境额外依赖 (替换 CUDA 特定包)
pip install torch-npu==2.10.0

# 以下包在 NPU 上可能需要特殊处理:
# - flash_attn: Ascend 上需使用 torch_npu 内置的融合注意力
# - triton: NPU 不支持，INT8 量化需改用 NPU 原生方案
```

---

## 3. 权重准备

### 3.1 从 HuggingFace 下载

```bash
pip install "huggingface_hub[cli]"

# 下载所有权重到指定目录
huggingface-cli download Skywork/Matrix-Game-3.0 \
  --local-dir /data1/weights/Matrix-Game-3.0

# 仅下载蒸馏模型 (推理推荐)
huggingface-cli download Skywork/Matrix-Game-3.0 \
  --local-dir /data1/weights/Matrix-Game-3.0 \
  --include "base_distilled_model/*" "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.2_VAE.pth" "MG-LightVAE*.pth"
```

### 3.2 手动准备 (已有权重文件)

```bash
# 确保权重目录结构正确
mkdir -p /data1/weights/Matrix-Game-3.0
# 将所有权重文件复制/软链接到此目录
cp -r /your/weights/* /data1/weights/Matrix-Game-3.0/
```

### 3.3 目录权限

```bash
chmod -R 755 /data1/weights/Matrix-Game-3.0
```

---

## 4. NPU 适配改造

> **重要**: 原始代码为 NVIDIA CUDA GPU 编写。在 Ascend NPU 上部署需要做以下适配。

### 4.1 核心问题清单

| 原始代码 | CUDA 依赖 | NPU 替代方案 |
|----------|----------|-------------|
| `torch.cuda.set_device()` | CUDA 设备管理 | `torch_npu.npu.set_device()` |
| `dist.init_process_group(backend="nccl")` | NCCL 通信 | `backend="hccl"` |
| `flash_attn` / `flash_attn_interface` | Flash Attention | `torch_npu.npu_fusion_attention()` |
| `wan/triton_kernels.py` | Triton JIT | NPU 不支持, 禁用 INT8 或使用量化方案 |
| `torch.compile` (VAE) | CUDA compile | 暂时禁用 |
| `device = torch.device(f"cuda:{id}")` | CUDA 设备 | `torch.device(f"npu:{id}")` |

### 4.2 应用 NPU 适配补丁

在项目根目录创建 `npu_patch.py` 适配脚本：

```python
# npu_patch.py — NPU 设备适配
import os
import torch
import torch.distributed as dist

# --- 环境变量设置 (在导入其他模块前设置) ---
os.environ.setdefault("WAN_FA_VERSION", "0")          # 禁用 Flash Attention
os.environ.setdefault("WAN_DISABLE_INT8", "1")        # 禁用 Triton INT8
os.environ.setdefault("WAN_DISABLE_COMPILE", "1")     # 禁用 torch.compile

def patch_device():
    """将 cuda 设备调用重定向到 npu"""
    import torch_npu
    
    # 替换 torch.cuda 为 torch_npu
    if not hasattr(torch, '_original_cuda'):
        torch._original_cuda = torch.cuda
    
    # 设备字符串映射
    torch.cuda.current_device = torch_npu.npu.current_device
    torch.cuda.device_count = torch_npu.npu.device_count
    torch.cuda.set_device = torch_npu.npu.set_device
    torch.cuda.synchronize = torch_npu.npu.synchronize
    torch.cuda.Stream = torch_npu.npu.Stream

def get_npu_device(device_id=0):
    """获取 NPU 设备"""
    return torch.device(f"npu:{device_id}")

def init_npu_distributed(rank, world_size, local_rank):
    """初始化 NPU 分布式 (HCCL 后端)"""
    import torch_npu
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
```

### 4.3 修改 `generate.py` 关键位置

#### 4.3.1 分布式初始化 (第 102 行)

```python
# 原始代码
torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size)

# NPU 适配
import torch_npu
torch_npu.npu.set_device(local_rank)
dist.init_process_group(
    backend="hccl",
    init_method="env://",
    rank=rank,
    world_size=world_size)
```

#### 4.3.2 设备引用 (多处 `torch.cuda.*`)

全局替换策略：
```python
# 在 generate.py 开头添加
import torch_npu
# 将 cuda 同步调用替换
# torch.cuda.synchronize() → torch_npu.npu.synchronize()
```

#### 4.3.3 Flash Attention 禁用

```bash
# 启动时设置环境变量
export WAN_FA_VERSION=0
```

或者在命令行添加 `--fa_version 0`。

### 4.4 修改 `pipeline/inference_interactive_pipeline.py`

#### 4.4.1 设备创建 (第 113 行)

```python
# 原始
self.device = torch.device(f"cuda:{device_id}")

# NPU 适配
self.device = torch.device(f"npu:{device_id}")
```

#### 4.4.2 VAE 加载 (vae_config.py 第 48 行)

```python
# 原始
device = torch.device(f"cuda:{device_id}")

# NPU 适配
device = torch.device(f"npu:{device_id}")
```

### 4.5 注意力模块适配 (`wan/modules/attention.py`)

Flash Attention 在 NPU 上不可用，会自动回退到 PyTorch SDPA：

```python
# 设置环境变量禁用 Flash Attention
export WAN_FA_VERSION=0
```

此时 `flash_attention()` 函数会检测到既没有 FA3 也没有 FA2，自动走 SDPA 路径。

### 4.6 INT8 量化处理

NPU 不支持 Triton 内核，INT8 量化需禁用：

```bash
# 启动时不要加 --use_int8 参数
```

如果需要量化加速，建议等待 NPU 原生量化方案或使用 Ascend 的 `torch_npu.npu_quantize` API。

---

## 5. 单卡推理

### 5.1 启动命令 (交互模式)

```bash
# 激活环境
conda activate matrix-game-3.0

# 设置环境变量
export WAN_FA_VERSION=0

# 运行推理
python generate.py \
  --size 704*1280 \
  --ckpt_dir /data1/weights/Matrix-Game-3.0 \
  --fa_version 0 \
  --num_iterations 12 \
  --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --save_name test \
  --seed 42 \
  --interactive \
  --vae_type mg_lightvae \
  --lightvae_pruning_rate 0.5 \
  --output_dir ./output
```

### 5.2 启动命令 (非交互模式, 自动动作)

```bash
python generate.py \
  --size 704*1280 \
  --ckpt_dir /data1/weights/Matrix-Game-3.0 \
  --fa_version 0 \
  --num_iterations 12 \
  --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --save_name test \
  --seed 42 \
  --vae_type mg_lightvae \
  --lightvae_pruning_rate 0.5 \
  --output_dir ./output
```

### 5.3 使用基础模型 (更高质量, 更慢)

```bash
python generate.py \
  --size 704*1280 \
  --ckpt_dir /data1/weights/Matrix-Game-3.0 \
  --fa_version 0 \
  --use_base_model \
  --num_iterations 12 \
  --num_inference_steps 50 \
  --sample_guide_scale 5.0 \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape..." \
  --save_name test_base \
  --seed 42 \
  --output_dir ./output
```

---

## 6. 多卡推理

### 6.1 使用 torchrun 启动 (HCCL 后端)

```bash
#!/bin/bash
# run_npu.sh — NPU 多卡启动脚本

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# NPU 数量
NUM_NPUS=8

# 环境变量
export WAN_FA_VERSION=0

torchrun \
  --nproc_per_node=${NUM_NPUS} \
  --master_port=29500 \
  generate.py \
  --size 704*1280 \
  --dit_fsdp \
  --t5_fsdp \
  --ckpt_dir /data1/weights/Matrix-Game-3.0 \
  --fa_version 0 \
  --num_iterations 12 \
  --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --save_name test_multinpu \
  --seed 42 \
  --interactive \
  --vae_type mg_lightvae \
  --lightvae_pruning_rate 0.5 \
  --output_dir ./output \
  --ulysses_size ${NUM_NPUS}
```

### 6.2 使用 HCCL 后端的 rank 表启动

```bash
# 方式1: 使用 torchrun (推荐)
torchrun --nproc_per_node=8 --master_port=29500 generate.py ...

# 方式2: 手动设置环境变量 (8卡示例)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=8

for i in $(seq 0 7); do
  export RANK=$i
  export LOCAL_RANK=$i
  python generate.py ... &
done
wait
```

### 6.3 异步 VAE (需要额外 NPU)

```bash
# 7 个 NPU 做扩散, 1 个 NPU 做 VAE 解码
torchrun --nproc_per_node=7 generate.py \
  ... \
  --use_async_vae \
  --async_vae_warmup_iters 1
```

> **注意**: 异步 VAE 在 NPU 上需要 `torch.multiprocessing` 的 `spawn` 模式 (已内置)，请确保足够的 NPU 内存。

---

## 7. 常用参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--size` | str | `1280*704` | 输出分辨率 (宽*高) |
| `--ckpt_dir` | str | 必填 | **权重目录路径，设为 `/data1/weights/Matrix-Game-3.0`** |
| `--fa_version` | str | `None` | Flash Attention 版本。NPU 上必须设为 `0` |
| `--num_iterations` | int | `12` | 迭代块数。总帧数 = 57 + (n-1)*40 |
| `--num_inference_steps` | int | `50` | 采样步数。蒸馏模型用 `3`，基础模型用 `50` |
| `--use_base_model` | flag | False | 使用基础模型 (需50步采样) |
| `--vae_type` | str | `mg_lightvae_v2` | VAE 类型: `wan`/`mg_lightvae`/`mg_lightvae_v2` |
| `--lightvae_pruning_rate` | float | None | LightVAE 剪枝率: mg_lightvae用0.5, v2用0.75 |
| `--interactive` | flag | False | 交互模式 (逐块手动输入 WASD+鼠标) |
| `--seed` | int | `42` | 随机种子 |
| `--output_dir` | str | `./output` | 输出目录 |
| `--save_name` | str | `generated_video` | 输出文件名 |
| `--dit_fsdp` | flag | False | DiT 使用 FSDP 分片 (多卡推荐) |
| `--t5_fsdp` | flag | False | T5 使用 FSDP 分片 (多卡推荐) |
| `--ulysses_size` | int | `1` | 序列并行度 (多卡时设为 NPU 数量) |
| `--t5_cpu` | flag | False | T5 放在 CPU 上 (节省 NPU 显存) |
| `--use_int8` | flag | False | INT8 量化 (**NPU 暂不支持，禁用**) |
| `--compile_vae` | flag | False | torch.compile VAE (**NPU 暂不支持，禁用**) |
| `--use_async_vae` | flag | False | 异步 VAE 解码 (多卡时加速) |

---

## 8. 常见问题排查

### 8.1 NPU 不可用

```bash
# 验证 NPU 环境
python -c "
import torch
import torch_npu
print('NPU available:', torch_npu.npu.is_available())
print('NPU count:', torch_npu.npu.device_count())
print('NPU name:', torch_npu.npu.get_device_name(0))
"
```

如果返回 False，检查：
1. CANN toolkit 版本与 torch-npu 版本是否匹配
2. `npu-smi info` 是否显示 NPU 设备
3. 驱动是否正常加载

### 8.2 HCCL 初始化失败

```
RuntimeError: HCCL error
```

排查：
```bash
# 检查 HCCL 环境
env | grep -i hccl
env | grep -i ascend

# 确保单卡模式下未设置分布式参数
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
```

### 8.3 显存不足 (OOM)

```bash
# 策略1: T5 放 CPU
python generate.py ... --t5_cpu

# 策略2: 减少迭代数
python generate.py ... --num_iterations 4

# 策略3: 使用更轻量的 VAE
python generate.py ... --vae_type mg_lightvae_v2 --lightvae_pruning_rate 0.75
```

### 8.4 Flash Attention 报错

```
ModuleNotFoundError: No module named 'flash_attn'
ModuleNotFoundError: No module named 'flash_attn_interface'
```

**解决**: 这是正常的，设置 `--fa_version 0` 使用 SDPA 回退。

```bash
export WAN_FA_VERSION=0
# 或在命令行添加 --fa_version 0
```

### 8.5 Triton 报错

```
Warning: Failed to load Triton kernels for Int8 quantization
```

**解决**: 不要使用 `--use_int8` 参数。NPU 不支持 Triton。

### 8.6 VAE 加载失败

确认权重目录中存在对应文件：
```bash
ls -la /data1/weights/Matrix-Game-3.0/Wan2.2_VAE.pth
ls -la /data1/weights/Matrix-Game-3.0/MG-LightVAE.pth
ls -la /data1/weights/Matrix-Game-3.0/MG-LightVAE_v2.pth
```

### 8.7 T5 模型/Tokenizer 加载失败

```bash
# 检查 T5 权重
ls -la /data1/weights/Matrix-Game-3.0/models_t5_umt5-xxl-enc-bf16.pth

# Tokenizer 需要从 HuggingFace 下载，确保网络连通
# 或手动下载后放到指定路径
huggingface-cli download google/umt5-xxl --local-dir /data1/weights/Matrix-Game-3.0/google/umt5-xxl
```

### 8.8 torch_npu 与 torch 版本不匹配

```bash
# 检查版本兼容性
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch_npu; print('torch_npu:', torch_npu.__version__)"

# 当前项目需要 torch==2.10.0
# 请安装对应 CANN 版本的 torch_npu
```

---

## 附录: 一键 NPU 启动脚本

项目根目录已提供 `run_npu.sh` 脚本, 可直接使用:

```bash
# 赋予执行权限
chmod +x run_npu.sh

# 单卡运行
bash run_npu.sh

# 多卡运行 (修改脚本中的 NUM_NPUS 变量)
vim run_npu.sh  # 修改 NUM_NPUS=8
bash run_npu.sh
```

### 快速验证部署

```bash
# 最小化测试 (2个迭代, 约97帧)
python generate.py \
  --size 704*1280 \
  --ckpt_dir /data1/weights/Matrix-Game-3.0 \
  --fa_version 0 \
  --num_iterations 2 \
  --num_inference_steps 3 \
  --image demo_images/001/image.png \
  --prompt "A test scene." \
  --save_name quick_test \
  --seed 42 \
  --vae_type mg_lightvae \
  --lightvae_pruning_rate 0.5 \
  --output_dir ./output

# 成功后输出文件: ./output/quick_test.mp4
```
