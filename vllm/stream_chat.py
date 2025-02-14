import requests
import json
import time
import readline
import threading
import sys
import argparse

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

HISTORY_FILE = ".chat_history"
try:
    readline.read_history_file(HISTORY_FILE)
except FileNotFoundError:
    pass  


thinking_flag = False  
stop_thinking = threading.Event() 

def thinking_animation():
    dots = 0
    while not stop_thinking.is_set():  
        dots = (dots % 6) + 1   
        sys.stdout.write(f"\rAI: Thinking{'.' * dots}   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 20 + "\r")  
    sys.stdout.flush()

def stream_vllm_response(messages, model="deepseek_test"):
    global thinking_flag
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    token_count = 0
    start_time = time.time()
    
    with requests.post(VLLM_API_URL, headers=headers, json=payload, stream=True) as response:
        found_think_tag = False  
        thinking_flag = True 
        stop_thinking.clear()

       
        thinking_thread = threading.Thread(target=thinking_animation)
        thinking_thread.start()

        output_buffer = "" 

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8")[6:])  
                    if "choices" in data and data["choices"]:
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            if not found_think_tag and "</think>" in token:
                                found_think_tag = True
                                token = token.split("</think>", 1)[-1]  # delete `<think>... </think>` for deepseek

                            if found_think_tag:
                                if thinking_flag:  
                                    stop_thinking.set() 
                                    thinking_thread.join()  
                                    thinking_flag = False
                                print(token, end="", flush=True)
                            token_count += 1
                except json.JSONDecodeError:
                    continue
    
    elapsed_time = time.time() - start_time
    tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\n\n[Model Metrics] Tokens: {token_count}, Time: {elapsed_time:.2f}s, Tokens/s: {tokens_per_sec:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive chat client for vLLM with streaming.")
    parser.add_argument("--model-id",dest="model_id", type=str, required=True, help="Specify the serve model name ")
    args = parser.parse_args()
    
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

    while True:
        try:
            user_input = input("\nUser: ") 
            readline.write_history_file(HISTORY_FILE) 
        except (KeyboardInterrupt, EOFError):
            print("\n聊天结束。")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("聊天结束。")
            break

        messages.append({"role": "user", "content": user_input})  # 记录用户输入
        stream_vllm_response(messages,args.model_id)  # 交互式流式输出
        messages.append({"role": "assistant", "content": ""})  # 记录 AI 回复

