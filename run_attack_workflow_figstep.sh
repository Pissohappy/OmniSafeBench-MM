#!/usr/bin/env bash
set -euo pipefail

# 批量测试某一个攻击手法在不同模型上的可运行工作流
# 逻辑：
# 1) 先生成该 attack 的 test cases（可指定 sample50 配置）
# 2) 批量拉起 vLLM，逐模型执行 response_generation
# 3) （可选）拉起 judge 模型，对所有已生成响应做 evaluation

ATTACK="figstep"
GPU="3"
EVAL_PORT=8070
JUDGE_PORT=8080
JUDGE_MODEL="gpt-oss-120b"
# "gemma-3-27b-it", "deepseek-vl2", "GLM-4.1V-9B-Thinking", "Qwen3-VL-30B-A3B-Instruct", "Kimi-VL-A3B-Instruct"
MODELS="Kimi-VL-A3B-Instruct,deepseek-vl2,Qwen3-VL-30B-A3B-Instruct,GLM-4.1V-9B-Thinking,gemma-3-27b-it"

# 使用 sample50 的推荐配置（你可替换）
GENERAL_CONFIG="config/general_config_${ATTACK}_sample.yaml"
MODEL_CONFIG="config/model_config.yaml"
TEST_CASES_FILE="output_sample/test_cases/${ATTACK}/test_cases.jsonl"

# 可选：把 judge 结果和 response 分离到非默认目录时可以改这个
# 例如 output_runs/exp_20260227/responses/None
RESPONSE_ROOT="output_sample/responses/None"

# 三个阶段可分离设计，默认都要跑一遍
SKIP_GENERATE=0
SKIP_RESPONSE=0
SKIP_JUDGE=0

usage() {
  cat <<USAGE
Usage: bash run_attack_workflow.sh [options]

Options:
  --attack <name>              攻击名称，默认: ${ATTACK}
  --models <m1,m2,...>         逗号分隔模型列表
  --gpu <id or ids>            CUDA_VISIBLE_DEVICES，默认: ${GPU}
  --eval-port <port>           batch_eval_auto 使用端口，默认: ${EVAL_PORT}
  --judge-port <port>          batch_judge_auto 使用端口，默认: ${JUDGE_PORT}
  --judge-model <name>         启用 judge 阶段时必填
  --general-config <path>      通用配置，默认: ${GENERAL_CONFIG}
  --model-config <path>        模型配置，默认: ${MODEL_CONFIG}
  --test-cases-file <path>     指定 response 阶段输入测试集；默认 output/test_cases/<attack>/test_cases.jsonl
  --response-root <path>       judge 阶段读取响应根目录，默认: ${RESPONSE_ROOT}
  --skip-generate              跳过 test_case_generation
  --skip-judge                 跳过 evaluation/judge
USAGE
}



while [[ $# -gt 0 ]]; do
  case "$1" in
    --attack) ATTACK="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --eval-port) EVAL_PORT="$2"; shift 2 ;;
    --judge-port) JUDGE_PORT="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --general-config) GENERAL_CONFIG="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --test-cases-file) TEST_CASES_FILE="$2"; shift 2 ;;
    --response-root) RESPONSE_ROOT="$2"; shift 2 ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    --skip-judge) SKIP_JUDGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$TEST_CASES_FILE" ]]; then
  TEST_CASES_FILE="output/test_cases/${ATTACK}/test_cases.jsonl"
fi

echo "============================================================"
echo "[Workflow] attack=${ATTACK}"
echo "[Workflow] models=${MODELS}"
echo "[Workflow] gpu=${GPU}"
echo "[Workflow] test_cases_file=${TEST_CASES_FILE}"
echo "[Workflow] general_config=${GENERAL_CONFIG}"
echo "[Workflow] model_config=${MODEL_CONFIG}"
echo "============================================================"

if [[ ${SKIP_GENERATE} -eq 0 ]]; then
  echo "=== Step 1: Generate test cases ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python run_pipeline.py \
    --config "${GENERAL_CONFIG}" \
    --model-config "$(basename "${MODEL_CONFIG}")" \
    --stage test_case_generation
else
  echo "=== Step 1: Skip test case generation ==="
fi

if [[ ${SKIP_RESPONSE} -eq 0 ]]; then
  echo "=== Step 2: Batch response generation across models ==="
  python batch_eval_auto.py \
    --attack "${ATTACK}" \
    --port "${EVAL_PORT}" \
    --gpu "${GPU}" \
    --models "${MODELS}" \
    --test-cases-file "${TEST_CASES_FILE}" \
    --base-gen-config "${GENERAL_CONFIG}" \
    --base-mod-config "${MODEL_CONFIG}"
else
  echo "=== Step 2: Skip response generation ==="
fi

if [[ ${SKIP_JUDGE} -eq 0 ]]; then
  if [[ -z "${JUDGE_MODEL}" ]]; then
    echo "❌ judge 阶段未跳过，但未提供 --judge-model"
    exit 1
  fi
  echo "=== Step 3: Judge evaluation ==="
  python batch_judge_auto.py \
    --attack "${ATTACK}" \
    --judge_model "${JUDGE_MODEL}" \
    --port "${JUDGE_PORT}" \
    --gpu "${GPU}" \
    --models "${MODELS}" \
    --response-root "${RESPONSE_ROOT}" \
    --base-gen-config "${GENERAL_CONFIG}" \
    --base-mod-config "${MODEL_CONFIG}"
else
  echo "=== Step 3: Skip judge evaluation ==="
fi

echo "✅ Workflow done."
