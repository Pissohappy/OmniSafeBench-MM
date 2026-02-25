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

echo "=== Step 1: Generate test cases ==="
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py --config config/general_config.yaml --stage test_case_generation

echo "=== Step 3: Send notification email ==="
python /mnt/disk1/szchen/monitor.py "test generated Finished" "Test case generation successfully."


echo "=== Step 2: Run batch evaluation ==="
python batch_eval_auto.py --attack cs_dj --port 8072 --gpu 1

echo "=== Step 3: Send notification email ==="
python /mnt/disk1/szchen/monitor.py "Pipeline Finished" "Test case generation and batch evaluation completed successfully."

echo "=== All done ==="