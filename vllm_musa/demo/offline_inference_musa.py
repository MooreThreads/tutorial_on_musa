import argparse
from vllm import LLM, SamplingParams

def generate_from_prompts(model: str, prompts=None, max_tokens=100, temperature=0.0):
    """
    使用指定的模型生成文本。

    参数:
    - model: 模型名称（如 "Qwen/Qwen2.5-0.5B-Instruct"）
    - prompts: 输入提示词列表
    - max_tokens: 最大生成长度
    - temperature: 控制随机性（0 表示完全确定）
    """
    if prompts is None:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
    
    # 创建 sampling 参数对象
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    # 创建 LLM 实例
    llm = LLM(model=model)

    # 生成文本
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM text generation with a specified model.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path (e.g., Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic)")
    parser.add_argument("--prompts", type=str, nargs="+", help="Optional custom list of prompts")

    args = parser.parse_args()

    generate_from_prompts(
        model=args.model,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
