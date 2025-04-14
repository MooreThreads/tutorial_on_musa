import gradio as gr
import requests
import json
import argparse
import time
import gradio_musa


def parse_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Start the vLLM server.")

    # 添加命令行参数
    parser.add_argument(
        "--ip", 
        type=str, 
        default="0.0.0.0",  # 如果没有传入--ip，使用默认值
        help="IP address to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port", 
        type=str, 
        default="8000",  # 如果没有传入--port，使用默认值
        help="Port number to use (default: 8000)"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        help="Model Name"
    )

    # 解析传入的参数
    args = parser.parse_args()
    return args

args = parse_args()
# 配置 vLLM 推理服务的地址和模型名
VLLM_API_URL = f"http://{args.ip}:{args.port}/v1/chat/completions"
MODEL_NAME = args.model_name


# ✅ 流式请求函数
def chat_with_model_streaming(user_input, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True  # ✅ 启用流式输出
    }

    history = history or []  # 初始化历史记录
    bot_response = ""  # 存储逐步生成的回答

    # ✅ 记录开始时间
    start_time = time.time()
    token_count = 0  # ✅ 记录生成的 Token 数量
    first_token_time = None

    try:
        # ✅ 使用 requests 的流式请求
        with requests.post(VLLM_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # ✅ 逐块解析流式响应
            for chunk in response.iter_lines():
                if chunk:
                    chunk_str = chunk.decode("utf-8").strip()
                    if chunk_str.startswith("data: "):
                        chunk_data = chunk_str[6:]  # 去掉 "data: " 前缀
                        if chunk_data != "[DONE]":
                            try:
                                chunk_json = json.loads(chunk_data)
                                delta = chunk_json["choices"][0]["delta"]
                                if "content" in delta:
                                    bot_response += delta["content"]
                                    # ✅ 逐步更新聊天记录
                                    token_count += 1  # ✅ 每个 Token 计数  
                                    if first_token_time is None and token_count > 0:
                                        first_token_time = time.time()

                                    yield history + [(user_input, bot_response)], "", "推理中..."
                            except json.JSONDecodeError:
                                pass
            # ✅ 记录结束时间 & 计算时长
            first_token_latency = first_token_time - start_time if first_token_time is not None else 0
            elapsed_time = time.time() - first_token_time
            tps = token_count / elapsed_time if elapsed_time > 0 else 0  # ✅ 计算 Tokens Per Second
            speed_text = f"⏳ 首字延迟: {first_token_latency:.2f} | ⏱️  耗时: {elapsed_time:.2f} 秒 | 🔢 Tokens: {token_count} | ⚡ 速度: {tps:.2f} TPS" # ⏳
            yield history + [(user_input, bot_response)], "", speed_text  # ✅ 返回推理速度

    except Exception as e:
        bot_response = f"❌ 推理失败: {str(e)}"
        yield history + [(user_input, bot_response)], ""



# ✅ 清除聊天记录 & 计时器
def clear_chat():
    return [], "", "⏳ 首字延迟: 0.00 秒 | ⏱️  耗时: 0.00 秒 | 🔢 Tokens: 0 | ⚡ 速度: 0.00 TPS"  # ✅ 清空所有 UI

# 构建 Gradio 界面
with gradio_musa.Blocks() as demo:
    # gr.Markdown("## 💬 Web UI 接入 vLLM 模型（流式输出）")
    chatbot = gr.Chatbot(label="Running on MTT S4000")
    msg_input = gr.Textbox(placeholder="请输入你的问题", label="输入...", lines=1, autofocus=True)

    speed_display = gr.Textbox(label="推理速度", value="⏳ 首字延迟: 0.00 秒 | ⏱️  耗时: 0.00 秒 | 🔢 Tokens: 0 | ⚡ 速度: 0.00 TPS", interactive=False)  # >✅ 显示推理速度

    # clear = gr.Button("清除")
    # submit = gr.Button("提交")
    with gr.Row():
        submit_btn = gr.Button(value="提交")
        clear_btn = gr.Button("清除历史")  # ✅ 添加清除按钮

    # ✅ 使用流式函数
    msg_input.submit(chat_with_model_streaming, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, speed_display]) # ✅ 按 Enter 触发
    submit_btn.click(chat_with_model_streaming, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, speed_display]) # ✅ 按按钮触发
    clear_btn.click(clear_chat, inputs=[], outputs=[chatbot, msg_input, speed_display])  # ✅ 清除聊天 & 计时

demo.queue()  # ✅ 允许流式数据传输
demo.launch(server_name=args.ip)