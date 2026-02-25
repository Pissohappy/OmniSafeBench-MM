import subprocess
import time
import os
import yaml
import shutil
import argparse

# --- 配置区 ---
# 基础路径配置
BASE_CONFIG = "config/general_config.yaml"
RESPONSE_ROOT = "output/responses/None" # 推理结果根目录
LOG_DIR = "logs/eval_parallel_logs"
MODEL_NAMES = [
    # "deepseek-vl2", 
    "Qwen3-VL-8B-Instruct", 
    # "gemma-3-4b-it", 
    # "Qwen3-VL-30B-A3B-Instruct",
    "Kimi-VL-A3B-Instruct", 
    "GLM-4.6V-Flash",
    "Step3-VL-10B", 
    "Youtu-VL-4B-Instruct",
    "gemma-3-27b-it",
    "llava-onevision-qwen2-7b-ov-hf",
    "InternVL3_5-8B-Flash"
]

def run_attack_eval(attack_name):
    """
    为一个特定的攻击手法运行所有模型的 Evaluation
    """
    print(f"🔥 开始并行评测攻击手法: {attack_name}")
    
    # 1. 为该攻击手法创建独立的临时配置文件 (方案1 核心)
    temp_config_path = f"config/temp_eval_{attack_name}.yaml"
    shutil.copy(BASE_CONFIG, temp_config_path)
    
    # 2. 修改副本内容 (固定 Judge 为 gpt-4o-mini)
    with open(temp_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['evaluation']['evaluators'] = ["default_judge"]
    if 'evaluator_params' not in config['evaluation']:
        config['evaluation']['evaluator_params'] = {}
    config['evaluation']['evaluator_params']['default_judge'] = {"model": "gpt-4o-mini"}
    
    with open(temp_config_path, 'w') as f:
        yaml.safe_dump(config, f)

    # 3. 循环该攻击下的所有模型
    for model in MODEL_NAMES:
        response_file = os.path.join(RESPONSE_ROOT, f"attack_{attack_name}_model_{model}.jsonl")
        
        if not os.path.exists(response_file):
            print(f"⚠️ [{attack_name}] 找不到模型 {model} 的响应结果，跳过...")
            continue

        log_file = os.path.join(LOG_DIR, f"eval_{attack_name}_{model}.log")
        
        print(f"⚖️  [{attack_name}] 正在评测模型: {model}")
        
        try:
            with open(log_file, "w") as log_fd:
                subprocess.run([
                    "python", "run_pipeline.py",
                    "--config", temp_config_path,
                    "--stage", "evaluation",
                    "--input-file", response_file
                ], stdout=log_fd, stderr=log_fd, check=True)
            print(f"✅ [{attack_name}] {model} 评测完成")
        except Exception as e:
            print(f"❌ [{attack_name}] {model} 运行失败，详情请看日志: {log_file}")

    # 4. 任务结束清理
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == "__main__":
    # 使用命令行参数来启动，方便在多个终端窗口运行不同的攻击
    # 比如：python this_script.py --attack figstep
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, required=True, help="要评测的攻击手法名称")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    run_attack_eval(args.attack)


# TEST_CASES_PATH = "/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/output/test_cases/hades/test_cases.jsonl"
# VLLM_PORT = 8008
# GENERAL_CONFIG = "config/general_config.yaml"
# LOG_DIR = "logs/hades/vllm_logs" # 新增日志存放目录

# # --- 邮件通知配置 ---
# def send_email_notification(model_name, status, details=""):
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 465
#     sender_email = "chenshunzhang823@gmail.com"
#     password = "noiuuflcwrmyalbf" 
#     to_email = "chenshunzhang823@gmail.com"

#     subject = f"VLM Eval: {model_name} - {status}"
#     content = f"模型: {model_name}\n状态: {status}\n详情: {details}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"

#     msg = MIMEText(content, 'plain', 'utf-8')
#     msg['Subject'] = Header(subject, 'utf-8')
#     msg['From'] = f"H200 Server <{sender_email}>"
#     msg['To'] = to_email

#     try:
#         with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
#             server.login(sender_email, password)
#             server.sendmail(sender_email, [to_email], msg.as_string())
#         print(f"📧 通知邮件已发送 (Status: {status})")
#     except Exception as e:
#         print(f"⚠️ 邮件发送失败: {e}")

# # --- 功能函数 ---
# def update_general_config(model_name):
#     with open(GENERAL_CONFIG, 'r') as f:
#         config = yaml.safe_load(f)
#     config['response_generation']['models'] = [model_name]
#     with open(GENERAL_CONFIG, 'w') as f:
#         yaml.safe_dump(config, f, default_flow_style=False)

# def wait_for_vllm(port, timeout=400):
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             response = requests.get(f"http://localhost:{port}/v1/models")
#             if response.status_code == 200:
#                 return True
#         except:
#             pass
#         time.sleep(5)
#     return False

# # --- 主循环 ---
# os.makedirs(LOG_DIR, exist_ok=True)
# print(f"🎬 开始批量评测任务，共 {len(MODEL_NAMES)} 个模型...")

# for i, model in enumerate(MODEL_NAMES):
#     model_path = os.path.join(MODELS_ROOT, model)
#     log_file_path = os.path.join(LOG_DIR, f"{model}_vllm.log")
    
#     print(f"\n" + "="*50)
#     print(f"📦 [{i+1}/{len(MODEL_NAMES)}] 当前模型: {model}")
#     print(f"📝 vLLM 日志将写入: {log_file_path}")
    
#     update_general_config(model)
#     tp_size = 2 if ("30B" in model or "27b" in model) else 1

#     # 关键：将 stdout 和 stderr 重定向到文件
#     vllm_log_fd = open(log_file_path, "w")
#     vllm_cmd = (
#         f"{VLLM_PYTHON_PATH} -m vllm.entrypoints.openai.api_server "
#         f"--model {model_path} --served-model-name {model} "
#         f"--port {VLLM_PORT} --trust-remote-code --dtype bfloat16 "
#         f"--tensor-parallel-size {tp_size} --gpu-memory-utilization 0.8"
#     )

#     # --- 【修改点 1: 启动】 ---
#     # 使用 preexec_fn=os.setsid 为这一家子进程创建一个“组 ID”
#     vllm_process = subprocess.Popen(
#         vllm_cmd, 
#         shell=True, 
#         stdout=vllm_log_fd, 
#         stderr=vllm_log_fd,
#         preexec_fn=os.setsid  
#     )
    
#     # vllm_process = subprocess.Popen(vllm_cmd, shell=True, stdout=vllm_log_fd, stderr=vllm_log_fd)

#     print(f"⏳ 正在启动 vLLM 并加载权重...")
#     if wait_for_vllm(VLLM_PORT):
#         print(f"✅ 服务就绪！开始运行推理流水线...")
#         try:
#             # 运行推理逻辑
#             start_eval = time.time()
#             subprocess.run([
#                 "python", "run_pipeline.py",
#                 "--config", GENERAL_CONFIG,
#                 "--stage", "response_generation",
#                 "--test-cases-file", TEST_CASES_PATH
#             ], check=True)
            
#             duration = round((time.time() - start_eval) / 60, 2)
#             send_email_notification(model, "Success", f"推理耗时: {duration} mins")
            
#         except subprocess.CalledProcessError as e:
#             print(f"❌ 推理失败: {e}")
#             send_email_notification(model, "Failed", f"Pipeline 运行出错:\n{str(e)}")
#     else:
#         print(f"❌ 模型启动超时！请检查日志: {log_file_path}")
#         send_email_notification(model, "Timeout", "vLLM 服务未能成功启动")

#     # # 清理
#     # print(f"🛑 正在清理 {model} 进程...")
#     # vllm_process.terminate()
#     # vllm_log_fd.close()
#     # subprocess.run(f"fuser -k {VLLM_PORT}/tcp", shell=True, stderr=subprocess.DEVNULL)
#     # time.sleep(15)

#     # --- 【修改点 2: 精确清理】 ---
#     print(f"🛑 正在精确清理 {model} 进程树...")
#     try:
#         # 获取该进程组 ID 并将其全部杀掉，这样绝不会误伤别人
#         pgid = os.getpgid(vllm_process.pid)
#         os.killpg(pgid, signal.SIGKILL) 
#     except Exception as e:
#         print(f"清理过程中出现小异常（可能进程已结束）: {e}")

#     vllm_log_fd.close()
    
#     # 辅助清理：只杀掉占用你指定端口的进程（双重保险）
#     subprocess.run(f"fuser -k {VLLM_PORT}/tcp", shell=True, stderr=subprocess.DEVNULL)
    
#     # --- 【修改点 3: 显存安全等待】 ---
#     # 不要固定死等 15 秒，改用显存监控
#     print("⏳ 等待显存完全释放...")
#     for _ in range(30): # 最多等 5 分钟
#         # 检查 GPU 0 的显存（因为你 TP=2 至少会占 GPU 0）
#         gpu_check = subprocess.check_output(
#             f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0", 
#             shell=True
#         ).decode().strip()
        
#         if int(gpu_check) < 10000: # 如果显存占用小于 10GB (H200 很空的状态)
#             print(f"✅ 显存已释放 ({gpu_check} MiB)，准备下一个模型。")
#             break
#         time.sleep(10)

# print("\n🎉 所有任务已完成！")
# send_email_notification("All Models", "Finished", "批量评测任务全部跑完啦！")