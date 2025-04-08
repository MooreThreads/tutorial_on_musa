import gradio as gr
import gradio_musa
import time  # ✅ 计时
from openai import OpenAI
import os


model_path="/workspace/qwq-32b_mtt"
print(model_path)
infer_url="http://10.10.142.39:22019/v1"
print(infer_url)

# ✅ 创建 OpenAI 客户端
client = OpenAI(base_url=infer_url, api_key="EMPTY")
# ✅ 处理对话请求（支持流式 + 计算推理速度）
def chat_with_openai(user_input, history):
    history = history or []  # 初始化聊天记录
    messages =[]
    # ✅ 构造 messages 结构
    messages = [{"role": "system", "content": "You are a chatbot. You can help people with their questions about anything. Always enclose latex snippets with dollar signs! For example, $$\phi$$"}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        answer = bot_msg[bot_msg.find("[💭思考完毕]") + len("[💭思考完毕]"):]
        messages.append({"role": "assistant", "content": answer})


    messages.append({"role": "user", "content": user_input})
    print(messages)

    # ✅ 记录开始时间
    start_time = time.time()
    token_count = 0  # ✅ 记录生成的 Token 数量

    # ✅ 发送请求
    try:
        response = client.chat.completions.create(
            model=model_path, # /data/mtt/models_convert/DeepSeek-R1-Distill-Llama-70B-converted", # converted_model_dir,  # ✅ 你的模型路径
            messages=messages,
            stream=True  # ✅ 启用流式输出
        )

        bot_response = ""
        for chunk in response:
            chunk_data = chunk.model_dump()  # ✅ 解析响应数据
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    bot_response += content
                    bot_response = bot_response.replace('<think>', '[💭开始思考]')
                    bot_response = bot_response.replace('</think>', '[💭思考完毕]')
                    token_count += 1  # ✅ 每个 Token 计数
                    yield history + [(user_input, bot_response)], "", "推理中..."

        # ✅ 记录结束时间 & 计算时长
        elapsed_time = time.time() - start_time
        tps = token_count / elapsed_time if elapsed_time > 0 else 0  # ✅ 计算 Tokens Per Second

        speed_text = f"⏱️  耗时: {elapsed_time:.2f} 秒 | 🔢 Tokens: {token_count} | ⚡ 速度: {tps:.2f} TPS"

        yield history + [(user_input, bot_response)], "", speed_text  # ✅ 返回推理速度

    except Exception as e:
        yield history + [(user_input, f"❌ 请求失败: {e}")], "", "⏱️  推理失败"

# ✅ 清除聊天记录 & 计时器
def clear_chat():
    return [], "", "⏱️  耗时: 0.00 秒 | 🔢 Tokens: 0 | ⚡ 速度: 0.00 TPS"  # ✅ 清空所有 UI

# ✅ 创建 Gradio 界面
with gradio_musa.Blocks() as demo:
    #gr.Markdown("### 💬 Deepseek-R1-Distill-Llama-70B 推理模型")

    chatbot = gr.Chatbot(label="Running on MTT S4000")  # ✅ 显示对话历史
    msg_input = gr.Textbox(label="输入你的问题", placeholder="请输入...", lines=1, autofocus=True)  # ✅ 支持>多行输入
    speed_display = gr.Textbox(label="推理速度", value="⏱️  耗时: 0.00 秒 | 🔢 Tokens: 0 | ⚡ 速度: 0.00 TPS", interactive=False)  # >✅ 显示推理速度

    with gr.Row():
        submit_btn = gr.Button(value="提交")
        clear_btn = gr.Button("清除历史")  # ✅ 添加清除按钮

    # ✅ 绑定事件
    msg_input.submit(chat_with_openai, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, speed_display])  # ✅ 按 Enter 触发
    submit_btn.click(chat_with_openai, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, speed_display])  # ✅ 按按钮触发
    clear_btn.click(clear_chat, inputs=[], outputs=[chatbot, msg_input, speed_display])  # ✅ 清除聊天 & 计时

demo.queue()  # ✅ 允许流式数据传输
demo.launch(server_name="0.0.0.0")  # ✅ 启动 Gradio 界面
