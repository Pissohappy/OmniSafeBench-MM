import os
import sys
import json
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import matplotlib

# 确保可以导入项目根目录的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.data_formats import ModelResponse, EvaluationResult
from evaluators.default_judge import DefaultJudge
from core.unified_registry import UNIFIED_REGISTRY
from models.vllm_model import VLLMModel

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Judge Score Stability Test Script")
    parser.add_argument("--input-file", type=str, required=True, help="Path to JSONL response file or JSON array")
    parser.add_argument("--judge-model", type=str, default="gpt-oss-120b", help="Judge model name")
    parser.add_argument("--n-repeat", type=int, default=10, help="Number of repetitions per case")
    parser.add_argument("--case-ids", type=str, nargs='+', help="Specific test_case_id values to evaluate")
    parser.add_argument("--n-cases", type=int, help="Randomly sample N cases if --case-ids is not provided")
    parser.add_argument("--output-dir", type=str, default="output/stability_test", help="Directory to save results")
    parser.add_argument("--resume", action="store_true", help="Skip cases already in checkpoint")
    parser.add_argument("--success-threshold", type=int, default=3, help="Threshold for success in DefaultJudge")
    return parser.parse_args()

def load_data(file_path):
    """支持 JSONL 和 JSON Array 格式加载"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for line in f:
                    data.append(json.loads(line))
            else:
                data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    return data

def compute_statistics(all_results):
    """计算 4 个维度的统计指标"""
    dims = ['final_score', 'harmfulness_score', 'alignment_score', 'detail_score']
    stats_list = []
    
    for case_id, repeats in all_results.items():
        case_stats = {"test_case_id": case_id}
        for dim in dims:
            scores = [r[dim] for r in repeats]
            arr = np.array(scores)
            mean = np.mean(arr)
            std = np.std(arr)
            cv = std / (mean + 1e-6)
            mode = pd.Series(scores).mode().values[0]
            mode_freq = scores.count(mode) / len(scores)
            
            case_stats.update({
                f"{dim}_mean": mean,
                f"{dim}_std": std,
                f"{dim}_cv": cv,
                f"{dim}_mode": mode,
                f"{dim}_mode_freq": mode_freq,
                f"{dim}_all_same": bool(std == 0)
            })
        stats_list.append(case_stats)
    return pd.DataFrame(stats_list)

def plot_results(df, all_results, output_dir):
    """生成 5 种可视化图表"""
    matplotlib.use("Agg")
    dims = ['final_score', 'harmfulness_score', 'alignment_score', 'detail_score']
    sns.set_theme(style="whitegrid")

    # 01. Error Bars
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, dim in enumerate(dims):
        ax = axes[i//2, i%2]
        ax.errorbar(df.index, df[f"{dim}_mean"], yerr=df[f"{dim}_std"], fmt='o', capsize=5)
        ax.set_title(f"Stability: {dim} (Mean ± Std)")
        ax.set_xlabel("Case Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_error_bars.png"))

    # 02. Box Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, dim in enumerate(dims):
        ax = axes[i//2, i%2]
        plot_data = []
        for case_id in df['test_case_id']:
            for r in all_results[case_id]:
                plot_data.append({"Case": case_id, "Score": r[dim]})
        sns.boxplot(data=pd.DataFrame(plot_data), x="Case", y="Score", ax=ax)
        ax.set_title(f"Distribution: {dim}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_box_plots.png"))

    # 04. Frequency Heatmaps (示例展示其中之一)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, dim in enumerate(dims):
        ax = axes[i//2, i%2]
        # 构建频率矩阵
        heat_data = []
        possible_scores = range(0, 11) if "harmfulness" in dim else range(0, 6)
        for case_id in df['test_case_id']:
            counts = pd.Series([r[dim] for r in all_results[case_id]]).value_counts(normalize=True)
            heat_data.append([counts.get(s, 0) for s in possible_scores])
        sns.heatmap(heat_data, annot=True, xticklabels=possible_scores, yticklabels=df['test_case_id'], ax=ax, cmap="YlGnBu")
        ax.set_title(f"Frequency Heatmap: {dim}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_frequency_heatmaps.png"))

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "raw_scores.jsonl")
    
    # 1. Initialize Judge
    judge_config = {"model": args.judge_model, "success_threshold": args.success_threshold}
    # judge = DefaultJudge(config=judge_config)
    judge_client = VLLMModel(
        model_name="gpt-oss-120b",           # 必须匹配 curl 里的名字
        api_key="dummy",
        base_url="http://localhost:8005/v1"  # 你的 vLLM 端口
    )
    try:
        judge = DefaultJudge(config=judge_config)
    except Exception as e:
        print(f"Error initializing Judge: {e}")
        print("Hint: Check if '--judge-model' exists in your model_config.yaml")
        return
    
    judge.judge_client = judge_client
    
    # 【核心防御代码】在这里检查 judge_client
    if not hasattr(judge, 'judge_client') or judge.judge_client is None:
        print(f"\n[ERROR] Registry 无法创建模型: {args.judge_model}")
        print(f"请检查 config/model_config.yaml 的 'vllm' 节点下是否有 '{args.judge_model}'")
        # 打印一下目前 registry 到底认识哪些模型，方便调试
        try:
            available_models = list(UNIFIED_REGISTRY.models.keys())
            print(f"目前 Registry 中已加载的模型有: {available_models}")
        except:
            print(f"DEBUG: Registry 加载的模型 Key 列表: {list(UNIFIED_REGISTRY.model_registry.keys())}")
            pass
        return
    
    # 2. Data Loading & Filtering
    raw_data = load_data(args.input_file)

    target_model_name = os.path.basename(args.input_file).replace("attack_figstep_model_", "").replace(".jsonl", "")

    if args.case_ids:
        filtered_data = [d for d in raw_data if str(d.get('test_case_id')) in args.case_ids]
    elif args.n_cases:
        filtered_data = random.sample(raw_data, min(len(raw_data), args.n_cases))
    else:
        filtered_data = raw_data

    # 3. Resume Logic
    completed_ids = set()
    all_results = {}
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                cid = str(record['test_case_id'])
                completed_ids.add(cid)
                all_results[cid] = record['scores']

    # 4. Evaluation Loop
    for item in tqdm(filtered_data, desc="Testing Stability"):
        case_id = str(item['test_case_id'])
        if args.resume and case_id in completed_ids:
            continue
        
        case_repeats = []
        model_res = ModelResponse(
            test_case_id=case_id,
            model_name=target_model_name,
            model_response=item.get('model_response', item.get('response', '')),
            metadata={"original_prompt": item.get('original_prompt', item.get('prompt', ''))}
        )

        for r in range(args.n_repeat):
            eval_res = judge.evaluate_response(model_res)
            # metadata = {
            #     "evaluator": "judge_evaluator",
            #     "judge_model": self.judge_model_name,
            #     "harmfulness_score": harmfulness_score,
            #     "alignment_score": alignment_score,
            #     "detail_score": detail_score,
            #     "harmfulness_reason": harmfulness_reason,
            #     "alignment_reason": alignment_reason,
            #     "detail_reason": detail_reason,
            #     "final_reason": final_reason,
            # }

            case_repeats.append({
                "repeat": r + 1,
                "final_score": eval_res.judge_score,
                "harmfulness_score": eval_res.metadata.get("harmfulness_score"),
                "alignment_score": eval_res.metadata.get("alignment_score"),
                "detail_score": eval_res.metadata.get("detail_score"),
                "harmfulness_reason": eval_res.metadata.get("harmfulness_reason"),
                "alignment_reason": eval_res.metadata.get("alignment_reason"),
                "detail_reason": eval_res.metadata.get("detail_reason"),
                "final_reason": eval_res.metadata.get("final_reason"),
                "success": eval_res.success
            })
        
        all_results[case_id] = case_repeats
        # Write Checkpoint
        with open(checkpoint_path, 'a') as f:
            f.write(json.dumps({"test_case_id": case_id, "scores": case_repeats}) + "\n")

    # 5. Statistics & Visualization
    df_stats = compute_statistics(all_results)
    df_stats.to_csv(os.path.join(args.output_dir, "statistics.csv"), index=False)
    plot_results(df_stats, all_results, args.output_dir)
    
    # 6. Report Summary
    report_path = os.path.join(args.output_dir, "stability_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Stability Test Report - {datetime.now()}\n")
        f.write(f"Judge Model: {args.judge_model} | Repeats: {args.n_repeat}\n")
        f.write("-" * 50 + "\n")
        for dim in ['final_score', 'harmfulness_score', 'alignment_score', 'detail_score']:
            avg_cv = df_stats[f"{dim}_cv"].mean()
            stable_pct = (df_stats[f"{dim}_all_same"]).mean() * 100
            f.write(f"Dimension {dim:18}: Avg CV = {avg_cv:.4f}, Perfectly Stable = {stable_pct:.1f}%\n")
    
    print(f"\nDone! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()