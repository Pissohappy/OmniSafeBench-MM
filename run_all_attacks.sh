#!/bin/bash
# 遍历 output_sample/test_cases/ 下的所有攻击数据集，依次运行 eval 和 judge 两个阶段。
# 外部数据集（无需注册到 plugins.yaml）通过 --test-cases-file 直接指定 test_cases.jsonl。

EVAL_PORT=8053
JUDGE_PORT=8035
GPU=3
JUDGE_MODEL="gpt-oss-120b"
TEST_CASES_DIR="output_sample/test_cases"
ATTACKS=("advbench" "arttextfigstep" "figstep" "holisafe" "jailbreakv28k" "mmsafetybench" "mossbench" "mssbench" "sd35_figstep" "spa_vl")

echo "=== Starting full pipeline over all attacks in $TEST_CASES_DIR ==="

for dir in "$TEST_CASES_DIR"/*/; do
    attack=$(basename "$dir")
    test_cases_file="${dir}test_cases.jsonl"

    if [ ! -f "$test_cases_file" ]; then
        echo "=== [$attack] Skipping: no test_cases.jsonl found ==="
        continue
    fi

    echo ""
    echo "================================================================"
    echo "=== [$attack] Step 1: Run batch evaluation ==="
    echo "================================================================"
    python batch_eval_auto.py \
        --attack "$attack" \
        --port $EVAL_PORT \
        --gpu $GPU \
        --test-cases-file "$test_cases_file" \
        || echo "⚠️  Eval failed for $attack, continuing..."

    echo "=== [$attack] Send eval notification ==="
    python /mnt/disk1/szchen/monitor.py "Eval Finished: $attack" "Batch evaluation completed for attack: $attack" || true

    echo ""
    echo "================================================================"
    echo "=== [$attack] Step 2: Run judge ==="
    echo "================================================================"
    python batch_judge_auto.py \
        --attack "$attack" \
        --judge_model "$JUDGE_MODEL" \
        --port $JUDGE_PORT \
        --gpu $GPU \
        || echo "⚠️  Judge failed for $attack, continuing..."

    echo "=== [$attack] Send judge notification ==="
    python /mnt/disk1/szchen/monitor.py "Judge Finished: $attack" "Batch judging completed for attack: $attack" || true

done

echo ""
echo "=== All attacks done ==="
python /mnt/disk1/szchen/monitor.py "All Done" "Full pipeline completed over all attacks in $TEST_CASES_DIR"
