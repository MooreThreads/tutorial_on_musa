# -*- coding: utf-8 -*-
import torch
import torch_musa
from models.experimental import attempt_load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default="./yolov5m.pt", help="path to yolov5 weights")
opt = parser.parse_args()
path = opt.models

model = attempt_load(path, 'cpu')
model.to('musa')

example = torch.rand(1,3,224,224).to("musa")
for name, params in model.named_parameters():
    print(name, params.device, params.dtype)
print("load model and data success")
print("output:", model(example))

script_model = torch.jit.trace(model, example, check_trace=False)
print("jit success")
script_model.save('yolov5m_jit.pt')

print("save model successfully")
