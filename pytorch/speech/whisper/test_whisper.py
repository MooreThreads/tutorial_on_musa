import os
import torch
import evaluate
import argparse

from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--models_dir", type=str, help="Path to models weights")
parser.add_argument("--data_dir", type=str, help="Path to datasets")

voice_path = parser.data_dir  #'/data/wwx/whisper/common_voice_11_0'
weights = parser.models_dir
common_voice = load_dataset(path=voice_path, data_files={"test": os.path.join(voice_path, "test/data-00000-of-00001.arrow")}, split="test")
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained(weights)
tokenizer = WhisperTokenizer.from_pretrained(weights, language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained(weights, language="Hindi", task="transcribe")

metric = evaluate.load("wer")

model = WhisperForConditionalGeneration.from_pretrained("./whisper-small").to("musa")

def map_to_pred(batch):
    
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = tokenizer.decode(processor.tokenizer(batch['sentence']).input_ids, skip_special_tokens=True)

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("musa"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = tokenizer.decode(processor.tokenizer(transcription).input_ids, skip_special_tokens=True)
    return batch

result = common_voice.take(10).map(map_to_pred)
print(100 * metric.compute(references=result["reference"], predictions=result["prediction"]))


