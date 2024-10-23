import vllm
from vllm import LLM, SamplingParams
import configparser
import pathlib
import argparse
import os


def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser(description="Generate answer from LLM")
    parser.add_argument(
        "-ckpt",
        "--checkpoint-path",
        required=True,
        help="Path to the model or checkpoint who must be the one converted by MT-Transformer",
    )

    args = parser.parse_args()

    config_path = pathlib.Path(args.checkpoint_path + "/config.ini")

    if config_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        config_name = "config"
        model_type = cfg.get(config_name, "model_name")
        tp_size = int(cfg.get(config_name, "tensor_para_size"))
    else:
        raise ValueError(f"The model path must be the one converted by MT-Transformer")

    print(f">>>>>>>>>>>>>>>> model type from the config is {model_type}")
    print(f">>>>>>>>>>>>>>>> tensor parallel size is {tp_size}")

    llm = LLM(
        model=args.checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
        device="musa",
        block_size=64,
        max_num_seqs=128,
        max_model_len=2048,
        max_num_batched_tokens=2048,
    )

    prompts = [
        "什么是牛顿第三定律？",
        "如何保持良好的睡眠？",
        "我想去北京，有哪些景点值得去？",
    ]

    tokenizer = llm.get_tokenizer()

    if model_type == "yayi":
        os.environ["VLLM_NO_USAGE_STATS"] = "1"

    # Try to use the apply_chat_template from the Transformers Library.
    # However, due to the absence of certain model templates,
    # this might lead to incorrect responses. It’s essential to verify
    # that the selected template aligns with the model’s architecture and
    # expected input format to ensure accurate outputs. If a template
    # mismatch occurs, it could result in inconsistencies or unexpected
    # behavior in the generated answers. Therefore, consider reviewing or
    # customizing the template when necessary to better match the specific
    # model in use.For more details, refer to the official Transformers
    # documentation: https://huggingface.co/docs/transformers/main/chat_templating
    prompts_tmp = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts_tmp.append(text)
    print(f">>>>>>>>>>>> prompts is {prompts_tmp}")

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1024)
    outputs = llm.generate(prompts_tmp, sampling_params)
    print(f"\n")
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {idx}: {prompt}, \nGenerated text: {generated_text}\n\n")


if __name__ == "__main__":
    main()
