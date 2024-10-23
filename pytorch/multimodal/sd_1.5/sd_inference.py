import argparse

import torch
import torch_musa
from PIL import Image
from diffusers import AutoPipelineForText2Image

torch.backends.mudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="the model path of stable diffusion",
    )
    parser.add_argument(
        "--batch_size", default=512, type=int, help="batch size to be used"
    )

    return parser.parse_args()


def _main() -> None:
    args = parse_args()
    model_name_or_path = args.model_path
    batch_size = args.batch_size

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("musa")

    prompts = ["a photo of an astronaut riding a horse on mars"] * batch_size
    pipe_obj = pipeline_text2image(prompt=prompts, width=640, height=480)

    for itx, image in enumerate(pipe_obj.images):
        image.save(f"output_{itx}.png")


if __name__ == "__main__":
    _main()
