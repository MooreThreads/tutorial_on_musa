import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_11_0", "hi")
dataset.save_to_disk("./common_voice_11_0")
