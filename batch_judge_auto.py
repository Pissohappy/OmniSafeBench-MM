import subprocess
import time
import os
import yaml
import argparse
import signal
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# --- 路径与环境配置 (保持与 Eval 一致) ---
VLLM_PYTHON_PATH = "/mnt/disk1/szchen/miniconda3/envs/vllm_env/bin/python"
MODELS_ROOT = "/mnt/disk1/weights/llm"
BASE_GEN_CONFIG = "config/general_config.yaml"
BASE_MOD_CONFIG = "config/model_config.yaml"
RESPONSE_ROOT = "output/responses/None"
LOG_DIR = "logs/judge_vllm_logs"

# --- 邮件通知函数 ---
def send_email_notification(model_name, attack_name, status, details=""):
    sender_email = "chenshunzhang823@gmail.com"
    password = "noiuuflcwrmyalbf" 
    to_email = "chenshunzhang823@gmail.com"

    subject = f"VLM HADES (Judge): {attack_name} | {model_name} - {status}"
    content = (
        f"【Judge 任务状态更新】\n"
        f"攻击手法: {attack_name}\n"
        f"待评测模型: {model_name}\n"
        f"状态: {status}\n"
        f"详情: {details}\n"
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = f"VLM Judge Monitor <{sender_email}>"
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        print(f"📧 [{model_name}] 状态邮件已发送")
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")

def run_judge_vllm_pipeline(
    attack_name,
    judge_model_name,
    base_port,
    gpu_id,
    target_models,
    response_root=RESPONSE_ROOT,
    base_gen_config=BASE_GEN_CONFIG,
    base_mod_config=BASE_MOD_CONFIG,
    models_root=MODELS_ROOT,
    vllm_python_path=VLLM_PYTHON_PATH,
):
    """
    judge_model_name: 作为裁判的模型名称 (如 'Llama-3-70B-Instruct')
    target_models: 需要被评测的响应模型列表
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. 准备 Judge 模型的服务
    judge_model_path = os.path.join(models_root, judge_model_name)
    vllm_log_file = os.path.join(LOG_DIR, f"judge_{judge_model_name}_vllm.log")
    
    # 定义临时配置
    tmp_mod_filename = f"model_config_judge_{gpu_id}.yaml"
    tmp_gen_filename = f"general_config_judge_{gpu_id}.yaml"
    tmp_mod_path = os.path.join("config", tmp_mod_filename)
    tmp_gen_path = os.path.join("config", tmp_gen_filename)

    # --- 第一步：启动 vLLM Judge 服务 ---
    subprocess.run(f"fuser -k {base_port}/tcp", shell=True, stderr=subprocess.DEVNULL)
    vllm_log_fd = open(vllm_log_file, "w")
    tp_size = len(str(gpu_id).split(','))

    vllm_cmd = (
        f"{vllm_python_path} -m vllm.entrypoints.openai.api_server "
        f"--model {judge_model_path} --served-model-name {judge_model_name} "
        f"--port {base_port} --trust-remote-code --dtype bfloat16 "
        f"--tensor-parallel-size {tp_size} --gpu-memory-utilization 0.8 "
        f"--max-model-len 8192"
    )

    print(f"🚀 启动 Judge 模型服务: {judge_model_name} | Port: {base_port} | GPU: {gpu_id}")
    vllm_process = subprocess.Popen(
        vllm_cmd, shell=True, stdout=vllm_log_fd, stderr=vllm_log_fd, 
        preexec_fn=os.setsid, env=env
    )

    # 健康检查
    ready = False
    for i in range(1, 61):
        try:
            if requests.get(f"http://localhost:{base_port}/v1/models", timeout=5).status_code == 200:
                ready = True; break
        except: pass
        if i % 6 == 0: print(f"⏳ 等待 Judge 服务就绪... ({i*10}s)")
        time.sleep(10)

    if not ready:
        print("❌ Judge 模型启动失败")
        os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
        return

    # --- 第二步：遍历目标模型进行评测 ---
    try:
        for model in target_models:
            response_file = os.path.join(response_root, f"attack_{attack_name}_model_{model}.jsonl")
            if not os.path.exists(response_file):
                print(f"⚠️ 跳过：找不到响应文件 {response_file}")
                continue

            print(f"\n⚖️ 正在评测模型响应: {model}")

            # 定制临时 model_config (指向本地 Judge 端口)
            with open(base_mod_config, 'r') as f:
                m_cfg = yaml.safe_load(f)
            
            new_url = f"http://localhost:{base_port}/v1"
            # 这里关键：需要确保 your_pipeline 支持通过配置指定 judge 模型
            # 假设逻辑是修改 default_judge 的配置
            if 'providers' in m_cfg and 'vllm' in m_cfg['providers']:
                m_cfg['providers']['vllm']['base_url'] = new_url
                if 'models' not in m_cfg['providers']['vllm']:
                    m_cfg['providers']['vllm']['models'] = {}
                m_cfg['providers']['vllm']['models'][judge_model_name] = {'base_url': new_url}

            with open(tmp_mod_path, 'w') as f:
                yaml.safe_dump(m_cfg, f)

            # 定制临时 general_config
            with open(base_gen_config, 'r') as f:
                g_cfg = yaml.safe_load(f)
            
            # g_cfg['evaluation'] = {
            #     'evaluators': ["default_judge"],
            #     'evaluator_params': {
            #         'model': judge_model_name, # 指定使用刚刚启动的模型
            #         'base_url': new_url
            #     }
            # }
            g_cfg['evaluation'] = {
                'evaluators': ["default_judge"],
                'evaluator_params': {
                    'default_judge': {
                        'model': judge_model_name,
                        'base_url': new_url,
                        'api_key': "EMPTY",
                        'temperature': 0.0,
                        'max_tokens': 2048
                    }
                }
            }


            with open(tmp_gen_path, 'w') as f:
                yaml.safe_dump(g_cfg, f)

            # 运行 Pipeline
            start_time = time.time()
            subprocess.run([
                "python", "run_pipeline.py",
                "--config", tmp_gen_path,
                "--model-config", tmp_mod_filename, # 注意：这里要传文件名而非路径，取决于你 pipeline 的实现
                "--stage", "evaluation",
                "--input-file", response_file
            ], check=True, env=env)

            duration = round((time.time() - start_time) / 60, 2)
            send_email_notification(model, attack_name, "Judge Success", f"耗时: {duration} mins")

    finally:
        # --- 第三步：清理现场 ---
        print(f"🧹 关闭 Judge 服务并清理临时文件...")
        os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
        vllm_log_fd.close()
        for f in [tmp_mod_path, tmp_gen_path]:
            if os.path.exists(f): os.remove(f)
        subprocess.run(f"fuser -k {base_port}/tcp", shell=True, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="figstep")
    parser.add_argument("--judge_model", type=str, required=True, help="用作裁判的模型文件夹名")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="逗号分隔的待评测模型列表；不传则使用脚本内默认列表",
    )
    parser.add_argument("--response-root", type=str, default=RESPONSE_ROOT)
    parser.add_argument("--base-gen-config", type=str, default=BASE_GEN_CONFIG)
    parser.add_argument("--base-mod-config", type=str, default=BASE_MOD_CONFIG)
    parser.add_argument("--models-root", type=str, default=MODELS_ROOT)
    parser.add_argument("--vllm-python", type=str, default=VLLM_PYTHON_PATH)
    args = parser.parse_args()

    # 待评测的列表
    # ALL_MODELS = [
    #     "gemma-3-12b-it", "gemma-3-27b-it", "gemma-3-4b-it",
    #     "GLM-4.6V-Flash", "InternVL3_5-8B", "Kimi-VL-A3B-Instruct",
    #     "llava-onevision-qwen2-7b-ov-hf", "llava-v1.6-mistral-7b-hf",
    #     "Qwen3-VL-8B-Instruct", "qwen3-vl-8b", "Step3-VL-10B",
    #     "deepseek-vl2", "Qwen3-VL-30B-A3B-Instruct"
    # ]

    # ALL_MODELS = [
    #     "gemma-3-27b-it", "deepseek-vl2", "GLM-4.1V-9B-Thinking", "Qwen3-VL-30B-A3B-Instruct", "Kimi-VL-A3B-Instruct"
    # ]

    ALL_MODELS = ["GLM-4.1V-9B-Thinking"]

    if args.models.strip():
        ALL_MODELS = [m.strip() for m in args.models.split(",") if m.strip()]

    # 2. 【新增逻辑】在调用前进行路径检查，筛选出真实存在的文件
    MODELS_TO_CHECK = []
    for model in ALL_MODELS:
        # 拼接路径，检查该 attack + model 的组合是否存在
        response_file = os.path.join(args.response_root, f"attack_{args.attack}_model_{model}.jsonl")
        if os.path.exists(response_file):
            MODELS_TO_CHECK.append(model)
        else:
            # 仅打印提示，不加入待判定列表
            print(f"📌 [File Not Found] 跳过 {model}，因为找不到文件: {response_file}")

    # 3. 如果筛选后的列表不为空，则启动 Judge 流程
    if MODELS_TO_CHECK:
        print(f"✅ 找到以下有效模型待判定 (Attack: {args.attack}): {MODELS_TO_CHECK}")
        run_judge_vllm_pipeline(
            args.attack,
            args.judge_model,
            args.port,
            args.gpu,
            MODELS_TO_CHECK,
            response_root=args.response_root,
            base_gen_config=args.base_gen_config,
            base_mod_config=args.base_mod_config,
            models_root=args.models_root,
            vllm_python_path=args.vllm_python,
        )
    else:
        print(f"❌ [Skipping] 攻击方式 {args.attack} 下没有找到任何可用的响应文件，脚本退出。")

    # run_judge_vllm_pipeline(args.attack, args.judge_model, args.port, args.gpu, MODELS_TO_CHECK)
