# encoding=utf-8
import torch

from FlagEmbedding import BGEM3FlagModel
import time

 
model = BGEM3FlagModel('./bge-m3', use_fp16=True, device='musa')
 
sentences = []
with open('./test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        sentences.append(line.replace('\n', ''))

print("==warm up start==")
embeddings = model.encode(sentences, batch_size=512, max_length=8192)['dense_vecs']
print("==warm up end==")
 
start = time.time()
embeddings = model.encode(sentences, batch_size=512, max_length=8192)['dense_vecs']
total_time = time.time() - start
print("==total time:  ", total_time)
