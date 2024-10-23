#!/bin/bash


MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=./stable-diffusion-v1-5 \
  --train_data_dir=./KonyconiStyle/10_boho \
  --output_dir="output_lora_weights"  \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --resolution=512