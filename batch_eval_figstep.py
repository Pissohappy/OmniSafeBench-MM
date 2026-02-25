import argparse
import os
import signal
import smtplib
import subprocess
import time
from email.header import Header
from email.mime.text import MIMEText

import requests
import yaml

# --- 配置区 ---
VLLM_PYTHON_PATH = "/mnt/disk1/szchen/miniconda3/envs/vllm_env/bin/python"
MODELS_ROOT = "/mnt/disk1/weights/vlm"
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
    "InternVL3_5-8B-Flash",
]
VLLM_PORT = 8008
GENERAL_CONFIG = "config/general_config.yaml"
LOG_DIR = "logs/vllm_logs"


# --- 邮件通知配置 ---
def send_email_notification(model_name, status, details=""):
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "chenshunzhang823@gmail.com"
    password = "noiuuflcwrmyalbf"
    to_email = "chenshunzhang823@gmail.com"

    subject = f"VLM Eval: {model_name} - {status}"
    content = (
        f"模型: {model_name}\n状态: {status}\n详情: {details}\n"
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = f"H200 Server <{sender_email}>"
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        print(f"📧 通知邮件已发送 (Status: {status})")
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")


# --- 功能函数 ---
def update_general_config(model_name):
    with open(GENERAL_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    config["response_generation"]["models"] = [model_name]
    with open(GENERAL_CONFIG, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def wait_for_vllm(port, timeout=400):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/v1/models")
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量响应生成脚本（支持 attack 参数化，避免重复维护多份脚本）"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="figstep",
        help="测试用例目录名，如 figstep / sd35_figstep",
    )
    parser.add_argument(
        "--test-cases-path",
        type=str,
        default=None,
        help="可选：显式指定 test_cases.jsonl 路径；默认按 output/test_cases/<attack>/test_cases.jsonl 组装",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_cases_path = args.test_cases_path or os.path.join(
        "output", "test_cases", args.attack, "test_cases.jsonl"
    )

    # --- 主循环 ---
    os.makedirs(LOG_DIR, exist_ok=True)
    print(
        f"🎬 开始批量评测任务，共 {len(MODEL_NAMES)} 个模型，"
        f"attack={args.attack}, test_cases={test_cases_path}"
    )

    for i, model in enumerate(MODEL_NAMES):
        model_path = os.path.join(MODELS_ROOT, model)
        log_file_path = os.path.join(LOG_DIR, f"{args.attack}_{model}_vllm.log")

        print(f"\n" + "=" * 50)
        print(f"📦 [{i + 1}/{len(MODEL_NAMES)}] 当前模型: {model}")
        print(f"📝 vLLM 日志将写入: {log_file_path}")

        update_general_config(model)
        tp_size = 2 if ("30B" in model or "27b" in model) else 1

        # 关键：将 stdout 和 stderr 重定向到文件
        vllm_log_fd = open(log_file_path, "w")
        vllm_cmd = (
            f"{VLLM_PYTHON_PATH} -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} --served-model-name {model} "
            f"--port {VLLM_PORT} --trust-remote-code --dtype bfloat16 "
            f"--tensor-parallel-size {tp_size} --gpu-memory-utilization 0.8"
        )

        vllm_process = subprocess.Popen(
            vllm_cmd,
            shell=True,
            stdout=vllm_log_fd,
            stderr=vllm_log_fd,
            preexec_fn=os.setsid,
        )

        print("⏳ 正在启动 vLLM 并加载权重...")
        if wait_for_vllm(VLLM_PORT):
            print("✅ 服务就绪！开始运行推理流水线...")
            try:
                start_eval = time.time()
                subprocess.run(
                    [
                        "python",
                        "run_pipeline.py",
                        "--config",
                        GENERAL_CONFIG,
                        "--stage",
                        "response_generation",
                        "--test-cases-file",
                        test_cases_path,
                    ],
                    check=True,
                )

                duration = round((time.time() - start_eval) / 60, 2)
                send_email_notification(model, "Success", f"推理耗时: {duration} mins")

            except subprocess.CalledProcessError as e:
                print(f"❌ 推理失败: {e}")
                send_email_notification(model, "Failed", f"Pipeline 运行出错:\n{str(e)}")
        else:
            print(f"❌ 模型启动超时！请检查日志: {log_file_path}")
            send_email_notification(model, "Timeout", "vLLM 服务未能成功启动")

        print(f"🛑 正在精确清理 {model} 进程树...")
        try:
            pgid = os.getpgid(vllm_process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception as e:
            print(f"清理过程中出现小异常（可能进程已结束）: {e}")

        vllm_log_fd.close()

        subprocess.run(f"fuser -k {VLLM_PORT}/tcp", shell=True, stderr=subprocess.DEVNULL)

        print("⏳ 等待显存完全释放...")
        for _ in range(30):
            gpu_check = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0",
                shell=True,
            ).decode().strip()

            if int(gpu_check) < 10000:
                print(f"✅ 显存已释放 ({gpu_check} MiB)，准备下一个模型。")
                break
            time.sleep(10)

    print("\n🎉 所有任务已完成！")
    send_email_notification("All Models", "Finished", f"{args.attack} 批量评测任务全部跑完啦！")
