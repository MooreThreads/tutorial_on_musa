"""Demo of DistributedDataParall"""
import os
import torch
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_musa


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5,5)
    def forward(self, x):
        return self.linear(x)

def start(rank, world_size):
    if os.getenv("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"]= "192.168.24.32" # IP must be specified here
    if os.getenv("MASTER_PORT") is None:
        os.environ["MASTER_PORT"]= "23" # port must be specified here
    dist.init_process_group("mccl", rank=rank, world_size=world_size)

def clean():
    dist.destroy_process_group()

def runner(rank, world_size):
    torch_musa.set_device(rank)
    start(rank, world_size)
    model = Model().to('musa')
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for _ in range(5):
        input_tensor = torch.randn(5, dtype=torch.float, requires_grad=True).to('musa')
        target_tensor = torch.zeros(5, dtype=torch.float).to('musa')
        output_tensor = ddp_model(input_tensor)
        loss_f = nn.MSELoss()
        loss = loss_f(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()
    clean()

if __name__ == "__main__":
    mp.spawn(runner, args=(2,), nprocs=2, join=True)
