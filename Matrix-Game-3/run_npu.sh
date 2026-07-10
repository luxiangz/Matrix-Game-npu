#!/usr/bin/env bash
# =============================================================================
# run_npu.sh — Matrix-Game-3.0  NPU (Ascend) 推理启动脚本
# =============================================================================
# 用法:
#   单卡:  bash run_npu.sh
#   多卡:  修改 NUM_NPUS 后运行, 或 export NUM_NPUS=8 && bash run_npu.sh
#   交互:  bash run_npu.sh --interactive
#   基础模型: bash run_npu.sh --use_base_model
#
# 权重路径: /data1/weights/Matrix-Game-3.0
# =============================================================================

set -euo pipefail

# ─── 项目路径 ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ─── NPU 设置 ───
NUM_NPUS="${NUM_NPUS:-1}"                        # NPU 数量, 默认 1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ─── 权重路径 ───
CKPT_DIR="${CKPT_DIR:-/data1/weights/Matrix-Game-3.0}"

# ─── NPU 环境变量 ───
export WAN_FA_VERSION=0                          # 禁用 Flash Attention
export WAN_DISABLE_INT8=1                        # 禁用 Triton INT8
export WAN_DISABLE_COMPILE=1                     # 禁用 torch.compile

# ─── 输出目录 ───
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
SAVE_NAME="${SAVE_NAME:-generated_video}"
SEED="${SEED:-42}"

# ─── 默认参数 ───
DEFAULT_IMAGE="demo_images/001/image.png"
DEFAULT_PROMPT="A colorful, animated cityscape with a gas station and various buildings."
DEFAULT_STEPS=3                                  # 蒸馏模型 3 步
DEFAULT_ITERATIONS=12                            # 57 + 11*40 = 497 帧
DEFAULT_VAE_TYPE="mg_lightvae"
DEFAULT_VAE_PRUNE="0.5"

# ─── 解析用户额外参数 ───
EXTRA_ARGS=()
USE_INTERACTIVE=false
USE_BASE_MODEL=false

for arg in "$@"; do
    case "$arg" in
        --interactive)
            USE_INTERACTIVE=true
            ;;
        --use_base_model)
            USE_BASE_MODEL=true
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# ─── 运行时推断 ───
if [ "$USE_BASE_MODEL" = true ]; then
    DEFAULT_STEPS=50
    echo "[INFO] Using base model (50 inference steps, higher quality)."
else
    echo "[INFO] Using distilled model (3 inference steps, fast generation)."
fi

# ─── 检查权重目录 ───
check_weights() {
    local missing=0
    local files=(
        "base_distilled_model"
        "models_t5_umt5-xxl-enc-bf16.pth"
        "Wan2.2_VAE.pth"
    )
    for f in "${files[@]}"; do
        if [ ! -e "${CKPT_DIR}/${f}" ]; then
            echo "❌ Missing: ${CKPT_DIR}/${f}"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        echo ""
        echo "请确保权重目录 ${CKPT_DIR} 包含所有必需文件。"
        echo "下载命令: huggingface-cli download Skywork/Matrix-Game-3.0 --local-dir ${CKPT_DIR}"
        exit 1
    fi
    echo "✅ 所有权重文件检查通过"
}

# ─── 检查 NPU 环境 ───
check_npu() {
    if ! python -c "import torch_npu; assert torch_npu.npu.is_available()" 2>/dev/null; then
        echo "❌ NPU 不可用，请检查:"
        echo "   1. CANN toolkit 是否安装"
        echo "   2. torch-npu 是否安装"
        echo "   3. npu-smi info 是否显示设备"
        echo ""
        echo "验证命令:"
        echo "   python -c \"import torch_npu; print(torch_npu.npu.is_available())\""
        exit 1
    fi
    echo "✅ NPU 环境检查通过"
}

# ─── 打印启动信息 ───
print_banner() {
    echo "=============================================="
    echo "  Matrix-Game-3.0  NPU Inference"
    echo "=============================================="
    echo "  NPU count:      ${NUM_NPUS}"
    echo "  Checkpoint:     ${CKPT_DIR}"
    echo "  Output:         ${OUTPUT_DIR}/${SAVE_NAME}.mp4"
    echo "  Seed:           ${SEED}"
    echo "  Steps:          ${DEFAULT_STEPS}"
    echo "  Iterations:     ${DEFAULT_ITERATIONS}"
    echo "  VAE:            ${DEFAULT_VAE_TYPE} (prune=${DEFAULT_VAE_PRUNE})"
    echo "  Interactive:    ${USE_INTERACTIVE}"
    echo "  Base model:     ${USE_BASE_MODEL}"
    echo "=============================================="
}

# ─── 构建参数列表 ───
build_args() {
    local args=(
        --size 704*1280
        --ckpt_dir "${CKPT_DIR}"
        --fa_version 0
        --num_iterations "${DEFAULT_ITERATIONS}"
        --num_inference_steps "${DEFAULT_STEPS}"
        --image "${DEFAULT_IMAGE}"
        --prompt "${DEFAULT_PROMPT}"
        --save_name "${SAVE_NAME}"
        --seed "${SEED}"
        --vae_type "${DEFAULT_VAE_TYPE}"
        --lightvae_pruning_rate "${DEFAULT_VAE_PRUNE}"
        --output_dir "${OUTPUT_DIR}"
    )

    if [ "$USE_INTERACTIVE" = true ]; then
        args+=(--interactive)
    fi

    if [ "$USE_BASE_MODEL" = true ]; then
        args+=(--use_base_model)
        args+=(--sample_guide_scale 5.0)
    fi

    # 多卡参数
    if [ "$NUM_NPUS" -gt 1 ]; then
        args+=(--dit_fsdp)
        args+=(--t5_fsdp)
        args+=(--ulysses_size "${NUM_NPUS}")
        # 异步 VAE: 需要额外的 NPU
        if [ "$NUM_NPUS" -ge 2 ]; then
            args+=(--use_async_vae)
            args+=(--async_vae_warmup_iters 1)
        fi
    fi

    # 追加用户自定义参数
    args+=("${EXTRA_ARGS[@]}")

    echo "${args[@]}"
}

# ─── 主流程 ───
main() {
    check_npu
    check_weights
    print_banner

    ARGS=($(build_args))

    if [ "$NUM_NPUS" -gt 1 ]; then
        echo ""
        echo "[INFO] 多卡启动: torchrun --nproc_per_node=${NUM_NPUS}"
        echo ""
        torchrun \
            --nproc_per_node="${NUM_NPUS}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            generate.py \
            "${ARGS[@]}"
    else
        echo ""
        echo "[INFO] 单卡启动: python generate.py"
        echo ""
        python generate.py "${ARGS[@]}"
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "  ✅ 推理完成!"
        echo "  输出: ${OUTPUT_DIR}/${SAVE_NAME}.mp4"
        echo "=============================================="
    else
        echo ""
        echo "=============================================="
        echo "  ❌ 推理失败 (exit code: ${exit_code})"
        echo "  请查看上方错误日志, 参考 DEPLOY.md 排查"
        echo "=============================================="
    fi
    exit $exit_code
}

main
