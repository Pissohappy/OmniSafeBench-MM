#!/bin/bash
set -e  # 遇到错误就停止脚本

echo "=== Step 2: Run batch evaluation ==="
python batch_eval_auto.py --attack arttextfigstep --port 8053 --gpu 3

echo "=== Step 2: Send notification email ==="
python /mnt/disk1/szchen/monitor.py "Eval Finished" "batch evaluation completed successfully."

echo "=== Step 3: Run judge ==="
python batch_judge_auto.py --attack arttextfigstep --judge_model "gpt-oss-120b" --port 8035 --gpu 3

echo "=== Step 3: Send notification email ==="
python /mnt/disk1/szchen/monitor.py "Judge Finished" "Batch judging completed successfully."

echo "=== All done ==="