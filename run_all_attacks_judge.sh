#!/bin/bash
# 1. 指定需要运行的硬编码攻击列表
ATTACKS=("advbench" "arttextfigstep" "figstep" "holisafe" "jailbreakv28k" "mmsafetybench" "mossbench" "mssbench" "sd35_figstep" "spa_vl")

EVAL_PORT=8053
JUDGE_PORT=8035
GPU=3
JUDGE_MODEL="gpt-oss-120b"
TEST_CASES_DIR="output_sample/test_cases"

echo "=== Starting full pipeline over selected attacks ==="

# 2. 修改循环逻辑：遍历数组而非遍历目录
for attack in "${ATTACKS[@]}"; do
    # 自动拼接该攻击对应的目录路径
    dir="${TEST_CASES_DIR}/${attack}/"
    test_cases_file="${dir}test_cases.jsonl"

    # 检查该攻击对应的文件夹和文件是否存在
    if [ ! -f "$test_cases_file" ]; then
        echo "=== [$attack] Skipping: $test_cases_file not found ==="
        continue
    fi
    
    echo ""
    echo "================================================================"
    echo "=== [$attack] Step 2: Run judge ==="
    echo "================================================================"
    
    # 运行判断脚本
    python batch_judge_auto.py \
        --attack "$attack" \
        --judge_model "$JUDGE_MODEL" \
        --port $JUDGE_PORT \
        --gpu $GPU \
        || echo "⚠️  Judge failed for $attack, continuing..."

    # 发送通知
    echo "=== [$attack] Send judge notification ==="
    python /mnt/disk1/szchen/monitor.py "Judge Finished: $attack" "Batch judging completed for attack: $attack" || true

done

echo ""
echo "=== All selected attacks done ==="
python /mnt/disk1/szchen/monitor.py "All Done" "Full pipeline completed for: ${ATTACKS[*]}"