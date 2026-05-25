# Scripts Index

本目录放可复用的数据转换、采样、分析脚本。根目录下仍保留了一些历史批处理脚本，它们大多绑定具体服务器路径、GPU、端口和通知方式；推荐优先使用 `run_pipeline.py`。

## Recommended Entry

```bash
python run_pipeline.py --config config/general_config.yaml --full
python run_pipeline.py --config config/general_config.yaml --stage test_case_generation
python run_pipeline.py --config config/general_config.yaml --stage response_generation --test-cases-file <test_cases.jsonl>
python run_pipeline.py --config config/general_config.yaml --stage evaluation --input-file <responses.jsonl>
```

## Data Conversion

这些脚本把外部 benchmark 转成本仓库标准 `TestCase` JSONL：

- `convert_advbenchm_to_testcases.py`
- `convert_hades_to_testcases.py`
- `convert_holisafe_to_testcases.py`
- `convert_jailbreakv_to_testcases.py`
- `convert_jailbreakv28k_to_testcases.py`
- `convert_mmbench_to_testcases.py`
- `convert_mmmu_to_testcases.py`
- `convert_mmsafetybench_to_testcases.py`
- `convert_mossbench_to_testcases.py`
- `convert_mssbench_to_testcases.py`
- `convert_siuo_to_testcases.py`
- `convert_spavl_to_testcases.py`

常用辅助：

- `filter_mmmu_single_image.py`: 过滤 MMMU 单图样本。
- `filter_complete_arttext_testcases.py`: 过滤完整 ArtText 样本。
- `compact_arttext_testcases.py`: 压缩/整理 ArtText test cases。
- `sample_testcases.py`: 从已有 test cases 中采样到 `output_sample/test_cases/<dataset>/`。

## Analysis And Reports

- `compare_attack_results.py`: 对比多个 evaluation JSONL 的 ASR 和分数。
- `analyze_axis_effects.py`: 分析维度影响。
- `analyze_arttext_asr_by_dimension.py`: 按 ArtText 维度分析 ASR。
- `analyze_arttext_font_distribution.py`: 统计 ArtText 字体分布。
- `report_arttext_asr_visuals.py`: 生成 ArtText ASR 可视化报告。
- `backfill_reasoning_fields.py`: 给历史 response 补 reasoning/final answer 字段。
- `test_judge_stability.py`: judge 稳定性测试。

## Legacy Root Scripts

以下根目录脚本是历史实验/服务器编排脚本，保留用于复现实验，但不作为通用入口：

- `batch_eval_*.py`: 批量启动目标模型 vLLM 服务并生成 responses。
- `batch_judge_*.py`: 批量启动 judge 服务并生成 evaluations。
- `run_attack_workflow*.sh`: 特定 attack 的端到端服务器工作流。
- `run_all_attacks*.sh`: 遍历已有 test cases 的批量工作流。
- `run0224.sh`、`run0225.sh`、`run0304.sh`、`run_sd35_figstep_0227.sh`: 日期命名的历史实验脚本。
- `sh_batch_*.sh`: 固定 attack/GPU/端口的小批量脚本。
- `check_model.py`: vLLM 兼容性检查脚本。
- `debug_figstep.py`、`debug.ipynb`: 调试脚本。

运行 legacy 脚本前请检查：

- `VLLM_PYTHON_PATH`、`KIMI_PYTHON_PATH`
- `MODELS_ROOT`
- GPU id 和端口
- `config/model_config*.yaml` 中的模型名和 base URL
- 邮件通知、API key、本地绝对路径

不要把新的实验结果、日志、大数据或私有配置提交到 Git。
