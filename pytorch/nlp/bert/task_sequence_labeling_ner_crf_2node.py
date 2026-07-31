#! -*- coding:utf-8 -*-
# Multi-node extension of task_sequence_labeling_ner_crf.py (bert + CRF NER).
# Dataset: china-people-daily-ner-corpus (same as README)
# Model: bert-base-chinese under ./data/
#
# Launch via torchrun (see run_dist_train_2node.sh). Reads RANK/LOCAL_RANK/WORLD_SIZE from env.

import os
import sys

# Pin each process to a single MUSA device BEFORE importing torch_musa.
# Otherwise bert4torch/model init may touch all visible GPUs and multi-proc dies.
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_vis = os.environ.get("MUSA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
_devs = [d.strip() for d in _vis.split(",") if d.strip() != ""]
if _devs:
    os.environ["MUSA_VISIBLE_DEVICES"] = _devs[min(_local_rank, len(_devs) - 1)]
else:
    os.environ["MUSA_VISIBLE_DEVICES"] = str(_local_rank)

import argparse
import time
import shutil
import fcntl
import faulthandler

faulthandler.enable(all_threads=True)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model
import torch_musa  # noqa: F401


def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    print(f"[bert-crf-2node][rank{rank}] {msg}", flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=50, help="smoke demo steps (1 epoch capped)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--maxlen", type=int, default=256)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument(
        "--local-cache",
        type=str,
        default="/tmp/bert_ddp_2node_cache",
        help="per-node local dir for bert weights (avoids PVC thundering-herd before DDP)",
    )
    p.add_argument("--log-interval", type=int, default=5)
    return p.parse_args()


def ensure_local_bert(bert_src: str, bert_local: str, local_cache: str, local_rank: int) -> None:
    """One process per node copies weights to local disk; others wait. No dist barrier."""
    need = ("config.json", "pytorch_model.bin", "vocab.txt")
    os.makedirs(local_cache, exist_ok=True)
    if local_rank == 0:
        ok = all(os.path.isfile(os.path.join(bert_local, n)) for n in need)
        if not ok:
            log(f"copying bert weights -> {bert_local}")
            if os.path.isdir(bert_local):
                shutil.rmtree(bert_local)
            shutil.copytree(bert_src, bert_local)
        else:
            log(f"local bert cache hit: {bert_local}")
        open(os.path.join(local_cache, ".bert_ready"), "w").close()
    for _ in range(600):
        if all(os.path.isfile(os.path.join(bert_local, n)) for n in need):
            if local_rank == 0:
                log(f"local bert ready: {bert_local}")
            return
        time.sleep(0.5)
    raise RuntimeError(f"local bert cache not ready: {bert_local}")


def main():
    args = parse_args()

    # After pinning MUSA_VISIBLE_DEVICES, the only visible device is index 0.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    maxlen = args.maxlen
    batch_size = args.batch_size
    categories = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]
    categories_label2id = {k: i for i, k in enumerate(categories)}

    data_root = args.data_dir
    train_file = os.path.join(data_root, "china-people-daily-ner-corpus", "example.train")
    bert_src = os.path.join(data_root, "bert-base-chinese")
    bert_local = os.path.join(args.local_cache, "bert-base-chinese")

    for path in (
        os.path.join(bert_src, "pytorch_model.bin"),
        os.path.join(bert_src, "config.json"),
        os.path.join(bert_src, "vocab.txt"),
        train_file,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                "Follow README: prepare bert-base-chinese + china-people-daily-ner-corpus under ./data/"
            )

    torch.musa.set_device(0)
    device = "musa:0"
    log(f"MUSA_VISIBLE_DEVICES={os.environ.get('MUSA_VISIBLE_DEVICES')} device={device}")

    ensure_local_bert(bert_src, bert_local, args.local_cache, local_rank)
    config_path = os.path.join(bert_local, "config.json")
    checkpoint_path = os.path.join(bert_local, "pytorch_model.bin")
    dict_path = os.path.join(bert_local, "vocab.txt")

    log("init_process_group(mccl) ...")
    torch.distributed.init_process_group(backend="mccl", init_method="env://")
    log("init_process_group done")
    seed_everything(42)

    if rank == 0:
        log(
            f"world_size={world_size} max_steps={args.max_steps} "
            f"batch_size={batch_size} train_file={train_file}"
        )

    class MyDataset(ListDataset):
        @staticmethod
        def load_data(filename):
            D = []
            with open(filename, encoding="utf-8") as f:
                f = f.read()
                for l in f.split("\n\n"):
                    if not l:
                        continue
                    d = [""]
                    for i, c in enumerate(l.split("\n")):
                        char, flag = c.split(" ")
                        d[0] += char
                        if flag[0] == "B":
                            d.append([i, i, flag[2:]])
                        elif flag[0] == "I":
                            d[-1][1] = i
                    D.append(d)
            return D

    log("loading tokenizer + dataset ...")
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def collate_fn(batch):
        batch_token_ids, batch_labels = [], []
        for d in batch:
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories_label2id["B-" + label]
                    labels[start + 1 : end + 1] = categories_label2id["I-" + label]
            batch_token_ids.append(token_ids)
            batch_labels.append(labels)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long)
        batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long)
        return batch_token_ids, batch_labels

    dataset = MyDataset(train_file)
    if rank == 0:
        log(f"train samples={len(dataset)}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = build_transformer_model(
                config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0
            )
            self.fc = nn.Linear(768, len(categories))
            self.crf = CRF(len(categories))

        def forward(self, token_ids, labels=None):
            sequence_output = self.bert([token_ids])
            emission_score = self.fc(sequence_output)
            attention_mask = token_ids.gt(0).long()
            if labels is None:
                return emission_score, attention_mask
            return self.crf(emission_score, attention_mask, labels)

    log("building model ...")
    lock_path = os.path.join(args.local_cache, ".model_load.lock")
    os.makedirs(args.local_cache, exist_ok=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            model = Model().to(device)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
    log("model built; skip param broadcast (identical checkpoint load)")
    # Identical weights loaded from the same local checkpoint on every rank.
    # Extra broadcast of bert-size tensors has been observed to leave MUSA/MCCL
    # in a bad state where the next forward hangs.
    torch.distributed.barrier()
    log("barrier ok; enter train loop")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    t0 = time.time()
    epoch = 0

    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for token_ids, labels in train_dataloader:
            if step == 0:
                log(f"got batch shape={tuple(token_ids.shape)}")
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if step == 0:
                log("forward ...")
            loss = model(token_ids, labels)
            if step == 0:
                log(f"forward ok loss={float(loss.detach().cpu()):.4f}; backward ...")
            loss.backward()
            if step == 0:
                log("backward ok; allreduce grads ...")
            for p in model.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            if step == 0:
                log("allreduce ok; optimizer step ...")
            optimizer.step()

            step += 1
            if rank == 0 and (
                step == 1 or step % args.log_interval == 0 or step == args.max_steps
            ):
                elapsed = time.time() - t0
                print(
                    f"Step {step}/{args.max_steps}, Loss: {loss.item():.6f}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if step >= args.max_steps:
                break
        epoch += 1

    torch.distributed.barrier()
    if rank == 0:
        print(f"[bert-crf-2node] DONE in {time.time() - t0:.1f}s", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
