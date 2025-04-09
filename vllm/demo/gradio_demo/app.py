import gradio as gr
import requests
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

# 请求函数
def chat_with_model(user_input, history):

    global IP, PORT, MODEL_NAME
    VLLM_API_URL = f"http://{IP}:{PORT}/v1/chat/completions"

    # 构造 messages（支持上下文）
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # for user_msg, bot_msg in history:
    #     messages.append({"role": "user", "content": user_msg})
    #     messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": user_input})

    # 构造请求 payload
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True
    }

    try:
        # 调用 OpenAI 格式的 API
        response = requests.post(VLLM_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content']
    except Exception as e:
        answer = f"❌ 推理失败: {str(e)}"

    # history.append((user_input, answer))
    return history + [(user_input, answer)], ""


def create_webui(ip):
    with gr.Blocks() as demo:
        gr.Markdown("## 💬 Web UI 接入 vLLM 模型")
        chatbot = gr.Chatbot()
        txt = gr.Textbox(placeholder="请输入你的问题", label="输入")
        clear = gr.Button("清除")
        submit = gr.Button("提交")

        submit.click(chat_with_model, [txt, chatbot], [chatbot, txt])
        txt.submit(chat_with_model, [txt, chatbot], [chatbot, txt])
        clear.click(lambda: ([], ""), [], [chatbot, txt])

    demo.launch(server_name=ip,)


def main():
    args = parse_args()
    global IP, PORT, MODEL_NAME
    IP = args.ip
    PORT = args.port
    MODEL_NAME = args.model_name

    create_webui(ip=IP)

if __name__ == "__main__":
    main()

