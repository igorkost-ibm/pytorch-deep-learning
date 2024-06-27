import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, ReduceOp
import os
from datetime import datetime
import time


device_type = "cuda" if torch.cuda.is_available() else "cpu"


def info():
    print(f'torch version: {torch.__version__}')
    print(f'device_type: {device_type}')


def create_data():
    X = torch.arange(0, 1, 0.001)
    y = -1.6 * X * X + 0.6 * X + 0.9
    # y = 0.5 * X + 0.2

    # Create train/test split
    train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    train_ds = TensorDataset(X_train, y_train)
    print(f'TRAIN dataset size: {len(train_ds)}')

    test_ds = TensorDataset(X_test, y_test)
    print(f'TEST  dataset size: {len(test_ds)}')

    return train_ds, test_ds


class PolynomialModel(nn.Module):
    def __init__(self):
        super().__init__() 
        # self.w3 = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.w1 = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.w0 = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2 * x * x + self.w1 * x + self.w0
    

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        train_ds: Dataset,
        test_ds: Dataset,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_every: int,
    ) -> None:
        self.rank_global = int(os.environ["RANK"])
        self.rank_local = int(os.environ["LOCAL_RANK"])
        self.model = model#.to('mps')
        self.model = DDP(model)
        self.device_type = next(model.parameters()).device.type
        self.device_name = f'{self.device_type}:{self.rank_global}:{self.rank_local}:{os.getpid()}'
        self.train_data = train_data
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        print(f'[{self.device_name}]: init with model {self.model.state_dict()}')

    def _run_batch(self, source, targets):
        batch_size = len(source)
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward() # AVG

        # for param in self.model.parameters():
        #     torch.distributed.all_reduce(param.grad.data, op=ReduceOp.SUM)

        self.optimizer.step()

    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        steps = len(self.train_data)
        # print(f"[{self.device_name}]: Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source#.to(self.rank)
            targets = targets#.to(self.rank)
            self._run_batch(source, targets)
        
        # Testing: Print out what's happening
        if epoch % 10 == 0:
            self.model.eval()
            with torch.inference_mode():
                train_pred = self.model(self.train_ds.tensors[0]) # forward pass
                train_loss = self.loss_fn(train_pred, self.train_ds.tensors[1].type(torch.float))
                test_pred = self.model(self.test_ds.tensors[0]) # forward pass
                test_loss = self.loss_fn(test_pred, self.test_ds.tensors[1].type(torch.float))
            # epoch_count.append(epoch)
            # train_loss_values.append(loss.detach().numpy())
            # test_loss_values.append(test_loss.detach().numpy())
            if epoch % 100 == 0:
                print(f"[{self.device_name}]: epoch: {epoch:4d} | batch_size: {batch_size} | steps: {steps} | train_loss: {train_loss:.5f} | test_loss: {test_loss:.5f} ")


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"[{self.device_name}]: Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs+1):
            self._run_epoch(epoch)
            if self.save_every > 0 and self.rank_local == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def worker(save_every: int, total_epochs: int, batch_size: int):

    # DDP setup
    init_process_group(backend="gloo")    
    # init_process_group(backend="gloo", rank=int(os.environ["LOCAL_RANK"]), world_size=int(os.environ["WORLD_SIZE"]))    
    # torch.cuda.set_device(rank)
    # torch.device('mps')

    # run training
    torch.manual_seed(2)
    train_ds, test_ds = create_data()
    train_data = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_ds))
    model = PolynomialModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=(0.1 * int(os.environ["WORLD_SIZE"])))
    Trainer(model, train_data, train_ds, test_ds, loss_fn, optimizer, save_every).train(total_epochs)

    destroy_process_group() # DDP cleanup


if __name__ == "__main__":
    import random
    time.sleep(random.uniform(100, 1000)/1000)
    print(f'STARTING on {os.environ["MASTER_ADDR"]} | WORLD_SIZE = {os.environ["WORLD_SIZE"]} | LOCAL_RANK = {os.environ["LOCAL_RANK"]} |  GLOBAL_RANK = {os.environ["RANK"]}')

    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=100, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    start_time = time.time()
    worker(args.save_every, args.total_epochs, args.batch_size)
    end_time = time.time()
    print(f'ALL DONE: {(end_time - start_time):.1f} secs')