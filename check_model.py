import os
import json
import gc
import time
import torch
import smtplib
import threading
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from PIL import Image
from transformers import AutoConfig
from vllm import LLM, SamplingParams

# ================= 路径与配置 =================
MODEL_ROOT = "/mnt/disk1/weights/vlm"
OUTPUT_JSON = "/mnt/disk1/szchen/vllm_compatibility_results.json"

MODEL_LIST = [
    "Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Thinking", 
    "Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-30B-A3B-Thinking",
    "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "deepseek-vl2", "deepseek-vl2-small", "deepseek-vl2-tiny",
    "InternVL3_5-8B", "InternVL3_5-8B-Flash", "InternVL3_5-14B-Flash",
    "llava-v1.6-mistral-7b", "llava-onevision-qwen2-7b-si", 
    "llava-onevision-qwen2-7b-ov", "llava-onevision-qwen2-7b-high-res",
    "Kimi-V1.5", "Kimi-V2", "Step3-VL-10B", "Youtu-VL-4B", 
    "GLM-4.6V-Flash", "Llama-4-Scout-17B-16E-Instruct", "Llama-Guard-4-12B"
]

SPECIAL_CONFIGS = {
    "Qwen3-VL-30B-A3B-Instruct": {"tp": 1, "gpu_util": 0.80},
    "Qwen3-VL-30B-A3B-Thinking": {"tp": 1, "gpu_util": 0.80},
    "gemma-3-27b-it": {"tp": 1, "gpu_util": 0.80},
}

# ================= 显存监控工具 =================
class PeakVRAMTracker:
    def __init__(self, device_id=0, interval=0.5):
        self.device_id = device_id
        self.interval = interval
        self.peak_mem = 0
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            # 获取已分配显存和保留显存的总和
            curr_mem = torch.cuda.max_memory_reserved(self.device_id) / 1024**3
            if curr_mem > self.peak_mem:
                self.peak_mem = curr_mem
            time.sleep(self.interval)

    def start(self):
        torch.cuda.reset_peak_memory_stats(self.device_id)
        self.peak_mem = 0
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return round(self.peak_mem, 2)

# ================= 邮件发送函数 =================
def send_email_notification(model_name, attack_name, status, details=""):
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "chenshunzhang823@gmail.com"
    password = "noiuuflcwrmyalbf" 
    to_email = "chenshunzhang823@gmail.com"

    subject = f"VLLM Compatibility: {attack_name} | {model_name} - {status}"
    content = (
        f"【任务信息】\n"
        f"测试类型: {attack_name}\n"
        f"测试模型: {model_name}\n"
        f"当前状态: {status}\n"
        f"详细信息: {details}\n"
        f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"测试显卡: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}\n"
    )

    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = f"VLM Tester <{sender_email}>"
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        print(f"📧 [{model_name}] 报告已发送 (Status: {status})")
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")

# ================= 核心测试逻辑 =================
def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(5)

def run_full_test(model_name, tp=1, gpu_util=0.9):
    model_path = os.path.join(MODEL_ROOT, model_name)
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # 启动显存监控
    tracker = PeakVRAMTracker()
    tracker.start()
    
    start_time = time.time()
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            max_model_len=2048,
            enforce_eager=True,
            disable_log_stats=True
        )
        
        outputs = llm.generate(
            prompts=[{"prompt": "Describe this image briefy.", "multi_modal_data": {"image": test_image}}],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=10)
        )
        
        duration = round(time.time() - start_time, 2)
        res_text = outputs[0].outputs[0].text
        
        # 停止监控并获取峰值
        peak_vram = tracker.stop()
        
        del llm
        cleanup_gpu()
        
        details = f"耗时: {duration}s | 峰值显存: {peak_vram} GB | 输出预览: {res_text[:30]}"
        return {"status": "SUCCESS", "details": details, "vram": peak_vram}
        
    except Exception as e:
        peak_vram = tracker.stop()
        cleanup_gpu()
        return {"status": "FAILED", "details": f"Error: {str(e)[:200]} | 崩溃前显存: {peak_vram} GB", "vram": peak_vram}

def main():
    final_report = []
    
    print(f"开始测试，共 {len(MODEL_LIST)} 个模型...")
    
    for name in MODEL_LIST:
        conf = SPECIAL_CONFIGS.get(name, {"tp": 1, "gpu_util": 0.9})
        print(f"---> Testing: {name}")
        
        result = run_full_test(name, tp=conf['tp'], gpu_util=conf['gpu_util'])
        
        final_report.append({
            "model": name,
            "status": result["status"],
            "vram": result.get("vram", 0)
        })
        
        # 发送单体邮件
        send_email_notification(
            model_name=name, 
            attack_name="VLLM_VRAM_Test", 
            status=result["status"], 
            details=result["details"]
        )

    # 汇总
    success_list = [item["model"] for item in final_report if item["status"] == "SUCCESS"]
    summary = (
        f"测试统计: 成功 {len(success_list)} / 总数 {len(MODEL_LIST)}\n\n"
        f"【详细清单】:\n" + 
        "\n".join([f"- {item['model']}: {item['status']} ({item['vram']} GB)" for item in final_report])
    )
    send_email_notification("OVERALL_MODELS", "COMPATIBILITY_FINAL", "FINISHED", summary)

if __name__ == "__main__":
    main()