import subprocess
import time
import os
import yaml
import shutil
import argparse
import signal
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# --- 基础基础路径配置（根据你的服务器环境检查） ---
VLLM_PYTHON_PATH = "/mnt/disk1/szchen/miniconda3/envs/vllm_env/bin/python"
KIMI_PYTHON_PATH = "/mnt/disk1/szchen/miniconda3/envs/kimi_env/bin/python"
MODELS_ROOT = "/mnt/disk1/weights/vlm"
BASE_GEN_CONFIG = "config/general_config.yaml"
BASE_MOD_CONFIG = "config/model_config.yaml"
LOG_DIR = "logs/parallel_vllm_logs"

# --- 引擎配置映射 ---
# 1 代表使用 V1 引擎 (Experimental)，0 代表强制回退到 V0 (Stable)
MODEL_ENGINE_MAP = {
    # "deepseek-vl2": "0",
    # "InternVL3_5-8B": "0",
    # "GLM-4.6V-Flash": "0",
    # "llava-onevision-qwen2-7b-ov-hf": "1", # LLaVA 系列通常对 V1 兼容较好
    "default": "1" # 其他默认尝试 V1
}

# --- 显存/长度配置映射 (新增) ---
MODEL_MAX_LEN_MAP = {
    "deepseek-vl2": 4096,
}

# --- 邮件通知函数 ---
def send_email_notification(model_name, attack_name, status, details=""):
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "chenshunzhang823@gmail.com"
    password = "noiuuflcwrmyalbf"  # 应用专用密码
    to_email = "chenshunzhang823@gmail.com"

    subject = f"VLM HADES: {attack_name} | {model_name} - {status}"
    content = (
        f"【任务状态更新】\n"
        f"攻击手法: {attack_name}\n"
        f"测试模型: {model_name}\n"
        f"当前状态: {status}\n"
        f"详细信息: {details}\n"
        f"通知时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = f"VLM Monitor <{sender_email}>"
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        print(f"📧 [{model_name}] 状态邮件已发送")
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")

def run_response_pipeline(
    attack_name,
    base_port,
    gpu_id,
    model_list,
    test_cases_file,
    base_gen_config=BASE_GEN_CONFIG,
    base_mod_config=BASE_MOD_CONFIG,
    models_root=MODELS_ROOT,
    vllm_python_path=VLLM_PYTHON_PATH,
    kimi_python_path=KIMI_PYTHON_PATH,
):
    # 1. 准备环境变量和目录
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 2. 定义临时配置路径
    tmp_mod_filename = f"model_config_{attack_name}_{gpu_id}.yaml"
    tmp_gen_filename = f"general_config_{attack_name}_{gpu_id}.yaml"
    tmp_mod_path = os.path.join("config", tmp_mod_filename)
    tmp_gen_path = os.path.join("config", tmp_gen_filename)

    for model in model_list:
        model_path = os.path.join(models_root, model)
        vllm_log_file = os.path.join(LOG_DIR, f"{attack_name}_{model}_vllm.log")

        engine_version = MODEL_ENGINE_MAP.get(model, MODEL_ENGINE_MAP["default"])
        env["VLLM_USE_V1"] = engine_version

        current_python = kimi_python_path if model == "Kimi-VL-A3B-Instruct" else vllm_python_path
        
        print(f"\n" + "█"*60)
        print(f"🚀 正在启动: {model} | 端口: {base_port} | GPU: {gpu_id} | VLLM_USE_V1: {engine_version}")
        print(f"█" + "━"*59)

        # --- 第一步：定制 model_config (修改端口) ---
        with open(base_mod_config, 'r') as f:
            m_cfg = yaml.safe_load(f)
        
        new_url = f"http://localhost:{base_port}/v1"
        if 'providers' in m_cfg and 'vllm' in m_cfg['providers']:
            m_cfg['providers']['vllm']['base_url'] = new_url
            if 'models' in m_cfg['providers']['vllm']:
                for m_key in m_cfg['providers']['vllm']['models']:
                    m_cfg['providers']['vllm']['models'][m_key]['base_url'] = new_url
        
        with open(tmp_mod_path, 'w') as f:
            yaml.safe_dump(m_cfg, f)

        # --- 第二步：定制 general_config (设置生成长度限制) ---
        with open(base_gen_config, 'r') as f:
            g_cfg = yaml.safe_load(f)
        
        g_cfg['response_generation']['models'] = [model]
        # 在这里强制注入 max_tokens，防止死循环输出
        if 'model_kwargs' not in g_cfg['response_generation']:
            g_cfg['response_generation']['model_kwargs'] = {}
        g_cfg['response_generation']['model_kwargs']['max_tokens'] = 512
        g_cfg['response_generation']['model_kwargs']['temperature'] = 0.0 # 保持评测的一致性

        with open(tmp_gen_path, 'w') as f:
            yaml.safe_dump(g_cfg, f)

        # --- 第三步：清理并启动 vLLM ---
        # 启动前强杀端口占用，防止启动失败
        subprocess.run(f"fuser -k {base_port}/tcp", shell=True, stderr=subprocess.DEVNULL)
        
        vllm_log_fd = open(vllm_log_file, "w")
        tp_size = len(str(gpu_id).split(','))

        custom_max_len = MODEL_MAX_LEN_MAP.get(model)
        max_len_arg = f"--max-model-len {custom_max_len}" if custom_max_len else "--max-model-len 8192"
        
        # 这里的 --max-model-len 2048 是为了防止极端情况下模型占用过大 KV Cache
        vllm_cmd = (
            f"{current_python} -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} --served-model-name {model} "
            f"--port {base_port} --trust-remote-code --dtype bfloat16 "
            f"--tensor-parallel-size {tp_size} --gpu-memory-utilization 0.8 "
            f"{max_len_arg}"
            # f"--max-model-len 8192" 
        )

        vllm_process = subprocess.Popen(
            vllm_cmd, shell=True, stdout=vllm_log_fd, stderr=vllm_log_fd, 
            preexec_fn=os.setsid, env=env
        )

        # --- 第四步：健康检查 (等待服务 Ready) ---
        ready = False
        print(f"⏳ 等待 vLLM 服务启动...")
        for i in range(1, 61): # 最多等待 10 分钟
            try:
                if requests.get(f"http://localhost:{base_port}/v1/models", timeout=5).status_code == 200:
                    ready = True
                    print(f"✅ 服务已就绪 (耗时 {i*10}s)")
                    break
            except:
                pass
            if i % 6 == 0: print(f"   ...已等待 {i*10}s")
            time.sleep(10)

        # --- 第五步：运行推理流水线 ---
        if ready:
            try:
                start_time = time.time()
                subprocess.run([
                    "python", "run_pipeline.py",
                    "--config", tmp_gen_path,
                    "--model-config", tmp_mod_filename,
                    "--stage", "response_generation",
                    "--test-cases-file", test_cases_file
                ], check=True, env=env)
                
                duration = round((time.time() - start_time) / 60, 2)
                send_email_notification(model, attack_name, "Success", f"耗时: {duration} mins")
            except Exception as e:
                print(f"❌ 推理运行失败: {e}")
                send_email_notification(model, attack_name, "Failed", str(e))
        else:
            print(f"❌ 服务启动超时，跳过模型 {model}")
            send_email_notification(model, attack_name, "Timeout", "vLLM 启动超过 10 分钟无响应")

        # --- 第六步：彻底清理现场 ---
        print(f"🧹 清理 {model} 进程及端口...")
        try:
            os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
        except:
            pass
        vllm_log_fd.close()
        subprocess.run(f"fuser -k {base_port}/tcp", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(10) # 给显存一点释放时间

    # 全部结束后清理临时文件
    for f in [tmp_mod_path, tmp_gen_path]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AUTO Attack Batch Evaluator")
    parser.add_argument("--attack", type=str, default="figstep", help="攻击名称")
    parser.add_argument("--port", type=int, required=True, help="起始端口号")
    parser.add_argument("--gpu", type=str, required=True, help="指定的 GPU ID (如 0 或 0,1)")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="逗号分隔的模型列表；不传则使用脚本内默认列表",
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        default="",
        help="测试用例文件路径；默认 output/test_cases/<attack>/test_cases.jsonl",
    )
    parser.add_argument("--base-gen-config", type=str, default=BASE_GEN_CONFIG)
    parser.add_argument("--base-mod-config", type=str, default=BASE_MOD_CONFIG)
    parser.add_argument("--models-root", type=str, default=MODELS_ROOT)
    parser.add_argument("--vllm-python", type=str, default=VLLM_PYTHON_PATH)
    parser.add_argument("--kimi-python", type=str, default=KIMI_PYTHON_PATH)
    args = parser.parse_args()

    # 你要跑的模型列表
    MODELS_TO_RUN = [
        # "Qwen3-VL-8B-Instruct", 
        "Kimi-VL-A3B-Instruct", 
        # "GLM-4.6V-Flash",
        "GLM-4.1V-9B-Thinking",
        # "Step3-VL-10B", 

        "gemma-3-27b-it",
        # "gemma-3-12b-it",
        # "llava-onevision-qwen2-7b-ov-hf",
        # "InternVL3_5-8B",

        "Qwen3-VL-30B-A3B-Instruct",

        
        # "llava-v1.6-mistral-7b-hf"

        # "Youtu-VL-4B-Instruct", 不支持vllm
        "deepseek-vl2",  #存在bug
        # "Llama-4-Scout-17B-16E-Instruct", 需要TP=2两张卡

    ]

    if args.models.strip():
        MODELS_TO_RUN = [m.strip() for m in args.models.split(",") if m.strip()]

    test_cases_file = args.test_cases_file or f"output/test_cases/{args.attack}/test_cases.jsonl"

    run_response_pipeline(
        args.attack,
        args.port,
        args.gpu,
        MODELS_TO_RUN,
        test_cases_file,
        base_gen_config=args.base_gen_config,
        base_mod_config=args.base_mod_config,
        models_root=args.models_root,
        vllm_python_path=args.vllm_python,
        kimi_python_path=args.kimi_python,
    )
