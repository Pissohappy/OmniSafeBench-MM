import signal
import subprocess
import time
import os
import yaml
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# --- 配置区 ---
# Evaluation 阶段通常不需要 vLLM 服务，但需要确保你的 OPENAI_API_KEY 已设置在环境变量或 config 中
GENERAL_CONFIG = "config/general_config.yaml"
# 推理结果存放的根目录，根据 README 的目录结构推断
RESPONSE_ROOT = "output/responses/None" # 假设 defense 为 None
MODEL_NAMES = [
    "deepseek-vl2"
    # "Qwen3-VL-8B-Instruct", 
    # "Kimi-VL-A3B-Instruct", 
    # "GLM-4.6V-Flash",
    # "Step3-VL-10B", 
    # "Youtu-VL-4B-Instruct",
    # "gemma-3-4b-it",
    # "gemma-3-27b-it",
    # "llava-onevision-qwen2-7b-ov-hf",
    # "InternVL3_5-8B-Flash"
]
ATTACK_METHOD = "figstep" # 你跑的 attack 类型

# --- 邮件通知配置 ---
def send_email_notification(model_name, status, details=""):
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "chenshunzhang823@gmail.com"
    password = "noiuuflcwrmyalbf" 
    to_email = "chenshunzhang823@gmail.com"

    subject = f"VLM Eval(Judge): {model_name} - {status}"
    content = f"模型: {model_name}\n状态: {status}\n详情: {details}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"

    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = f"H200 Server <{sender_email}>"
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        print(f"📧 通知邮件已发送 (Status: {status})")
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")

# --- 功能函数 ---
def setup_eval_config():
    """
    统一配置评测插件为 gpt-4o-mini
    """
    with open(GENERAL_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置评测器
    config['evaluation']['evaluators'] = ["default_judge"]
    # 假设你在 model_config.yaml 中已经定义了 gpt-4o-mini 
    # 或者在这里强制指定参数
    if 'evaluator_params' not in config['evaluation']:
        config['evaluation']['evaluator_params'] = {}
    
    # 这一步很关键：确保调用的是 gpt-4o-mini
    config['evaluation']['evaluator_params']['model'] = "gpt-4o-mini"
    
    with open(GENERAL_CONFIG, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

# --- 主循环 ---
print(f"🎬 开始批量 [Evaluation] 任务，共 {len(MODEL_NAMES)} 个模型...")

# 1. 先全局配置 Judge 模型
setup_eval_config()

for i, model in enumerate(MODEL_NAMES):
    # 根据 README 拼接 response 文件的路径
    # 格式通常为: attack_<attack>_model_<model>.jsonl
    response_file = os.path.join(RESPONSE_ROOT, f"attack_{ATTACK_METHOD}_model_{model}.jsonl")
    
    print(f"\n" + "="*50)
    print(f"⚖️ [{i+1}/{len(MODEL_NAMES)}] 评测模型响应: {model}")
    
    if not os.path.exists(response_file):
        print(f"⚠️ 未找到响应文件，跳过: {response_file}")
        continue

    try:
        start_eval = time.time()
        print(f"🚀 正在运行评测流水线 (Judge: gpt-4o-mini)...")
        
        # 核心命令：切换到 evaluation 阶段，并指定 --input-file
        subprocess.run([
            "python", "run_pipeline.py",
            "--config", GENERAL_CONFIG,
            "--stage", "evaluation",
            "--input-file", response_file
        ], check=True)
        
        duration = round((time.time() - start_eval) / 60, 2)
        print(f"✅ {model} 评测完成！耗时: {duration} mins")
        send_email_notification(model, "Eval Success", f"评测耗时: {duration} mins\n输入文件: {response_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 评测失败: {e}")
        send_email_notification(model, "Eval Failed", f"Evaluation 运行出错:\n{str(e)}")

print("\n🎉 所有评测任务已完成！")
send_email_notification("All Models Eval", "Finished", "所有模型的 GPT-4o-mini 评测已跑完。")