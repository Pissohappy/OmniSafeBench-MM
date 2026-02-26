#!/bin/bash
set -e  # 遇到错误就停止脚本

# 我修改了generate_test_cases.py中176行
# return self._generate_test_cases_parallel(
#                     attack_name,
#                     attack_config,
#                     combinations,
#                     batch_size,
#                     output_file_path,
#                     image_save_dir,
#                     # max_workers_override=policy.max_workers,
#                     max_workers_override=1
#                 )

# echo "=== Step 1: Generate test cases ==="
# CUDA_VISIBLE_DEVICES=1 python run_pipeline.py --config config/general_config_sd35_figstep.yaml --stage test_case_generation

# echo "=== Step 1: Send notification email ==="
# python /mnt/disk1/szchen/monitor.py "test generated Finished" "Test case generation successfully."


echo "=== Step 2: Run batch evaluation ==="
python batch_eval_auto.py --attack sd35_figstep --port 8072 --gpu 1

echo "=== Step 2: Send notification email ==="
python /mnt/disk1/szchen/monitor.py "Eval Finished" "Test case generation and batch evaluation completed successfully."

echo "=== Step 3: Run judge ==="
python batch_judge_auto.py --attack sd35_figstep --judge_model "gpt-oss-120b" --port 8072 --gpu 1

echo "=== Step 2: Send notification email ==="
python /mnt/disk1/szchen/monitor.py "Eval Finished" "Test case generation and batch evaluation completed successfully."

echo "=== All done ==="