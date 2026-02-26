#!/bin/bash

# 定义硬编码列表
# ATTACKS=("figstep" "hades" "jood" "mml" "siuo")
# ATTACKS=("siuo" "mml" "jood")
# ATTACKS=("cs_dj")
ATTACKS=("sd35_figstep")

# 定义模型列表
MODELS=(
    "gemma-3-12b-it" "gemma-3-27b-it" "gemma-3-4b-it" 
    "GLM-4.6V-Flash" "InternVL3_5-8B" "Kimi-VL-A3B-Instruct" 
    "llava-onevision-qwen2-7b-ov-hf" "llava-v1.6-mistral-7b-hf" 
    "Qwen3-VL-8B-Instruct" "Step3-VL-10B" 
    "deepseek-vl2" "Qwen3-VL-30B-A3B-Instruct"
)

JUDGE_MODEL="gpt-oss-120b"
GPU_ID="0"
PORT=8002

for ATK in "${ATTACKS[@]}"; do
    echo "------------------------------------------------"
    echo "🚀 Starting Judge for Attack: $ATK"
    echo "------------------------------------------------"
    
    # 直接调用 Python 脚本
    # 假设你的 Python 脚本内部会遍历一个硬编码的 MODEL_LIST
    python batch_judge_auto.py \
        --attack "$ATK" \
        --judge_model "$JUDGE_MODEL" \
        --gpu "$GPU_ID" \
        --port "$PORT"
        
    echo "✅ Finished Attack: $ATK"
done