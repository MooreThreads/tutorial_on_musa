import gradio as gr
import requests
import json
import argparse


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
                                    yield history + [(user_input, bot_response)], ""
                            except json.JSONDecodeError:
                                pass

    except Exception as e:
        bot_response = f"❌ 推理失败: {str(e)}"
        yield history + [(user_input, bot_response)], ""

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Web UI 接入 vLLM 模型（流式输出）")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(placeholder="请输入你的问题", label="输入")
    clear = gr.Button("清除")
    submit = gr.Button("提交")

    # ✅ 使用流式函数
    submit.click(chat_with_model_streaming, [txt, chatbot], [chatbot, txt])
    txt.submit(chat_with_model_streaming, [txt, chatbot], [chatbot, txt])
    clear.click(lambda: ([], ""), [], [chatbot, txt])

demo.launch(server_name=args.ip)