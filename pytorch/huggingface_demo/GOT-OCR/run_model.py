import uuid
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import torch
from transformers import AutoModel, AutoTokenizer

# 视频帧提取函数
def sample_frames(file_path, num_frames):
    try:
        video = cv2.VideoCapture(file_path)  # 打开本地视频文件
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print("视频中没有帧。")
            return []

        interval = max(total_frames // num_frames, 1)  # 确保间隔至少为 1
        frames = []
        
        for i in range(total_frames):
            ret, frame = video.read()
            if not ret:
                continue
            if i % interval == 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil_img)

        video.release()
        return frames
    except Exception as e:
        print(f"发生错误：{e}")
        return []

# 加载模型和分词器
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, low_cpu_mem_usage=True, device_map='musa', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().musa()
    return model, tokenizer

# OCR 推理
def ocr_inference(model, tokenizer, image):
    res = model.chat(tokenizer, image, ocr_type='ocr')
    return res

def main():
    # 设置视频文件路径（确保文件存在于同一级目录）
    video_1_path = "./cats_1.mp4"

    # 提取视频中的帧
    video_frames = sample_frames(video_1_path, 6)

    # 新建存储图片的文件夹
    save_dir = "./jpg_generate"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果文件夹不存在，则创建

    # 加载模型
    model_dir = os.path.abspath(os.path.dirname(__file__))  # 假设模型和脚本在同一目录
    model, tokenizer = load_model(model_dir)

    # 获取GPU数量
    gpu_count = 8

    # 对每一帧进行OCR推理
    for idx, frame in enumerate(video_frames):
        # 保存帧作为临时图片文件
        frame_path = os.path.join(save_dir, f"frame_{idx + 1}.jpg")
        frame.save(frame_path)

        # 推理
        start_time = time.time()
        ocr_result = ocr_inference(model, tokenizer, frame_path)
        end_time = time.time()

        # 计算性能指标
        total_time = end_time - start_time
        encoded_res = tokenizer.encode(ocr_result, return_tensors="pt")
        num_tokens = encoded_res.size(1)
        tokens_per_second = num_tokens / total_time if total_time > 0 else 0

        max_memory_usage = 0
        max_memory_utilization = 0
        for i in range(gpu_count):
            memory_allocated = torch.musa.memory_allocated(i)
            memory_reserved = torch.musa.memory_reserved(i)
            max_memory_usage = max(max_memory_usage, memory_allocated)
            max_memory_utilization = max(max_memory_utilization, memory_allocated / memory_reserved if memory_reserved > 0 else 0)

        # 输出性能指标和OCR结果
        print(f"推理时间 (帧 {idx + 1}): {total_time:.2f} s")
        print(f"总token数 (帧 {idx + 1}): {num_tokens}")
        print(f"每秒生成token数 (帧 {idx + 1}): {tokens_per_second:.2f}")
        print(f"GPU数量: {gpu_count}")
        print(f"输入图片大小: {frame.size}")
        print(f"最大显存使用量: {max_memory_usage / (1024 ** 2):.2f} MB")
        print(f"最大显存利用率: {max_memory_utilization * 100:.2f}%")
        print(f"OCR 结果 (帧 {idx + 1}): {ocr_result}")

if __name__ == "__main__":
    main()