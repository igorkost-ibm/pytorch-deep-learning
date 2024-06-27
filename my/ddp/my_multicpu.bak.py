import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

device_type = "cuda" if torch.cuda.is_available() else "cpu"


def info():
    print(f'torch version: {torch.__version__}')
    print(f'device_type: {device_type}')


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        save_every: int,
    ) -> None:
        self.rank = rank
        self.model = model#.to('mps')
        self.model = DDP(model)
        self.device_type = next(model.parameters()).device.type
        self.device_name = f'{self.device_type}:{self.rank}:{os.getpid()}'
        self.train_data = train_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[{self.device_name}]: Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source#.to(self.rank)
            targets = targets#.to(self.rank)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"[{self.device_name}]: Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.save_every > 0 and self.rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def worker(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):

    # DDP setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)    
    # torch.cuda.set_device(rank)
    # torch.device('mps')

    # run training
    train_set = MyTrainDataset(10240)
    train_data = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_set))
    model = torch.nn.Linear(20, 1)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    Trainer(model, train_data, loss_fn, optimizer, rank, save_every).train(total_epochs)

    destroy_process_group() # DDP cleanup


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=1, type=int, help='Number of procresses')
    args = parser.parse_args()

    mp.spawn(worker, args=(args.world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=args.world_size)
